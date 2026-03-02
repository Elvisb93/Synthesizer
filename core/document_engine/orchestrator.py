import json
import re
from typing import Callable, Dict, List, Optional, Tuple

from .models import (
    DocumentCheckpoint,
    DocumentChunk,
    DocumentGenerationOptions,
    DocumentMode,
    DocumentOutline,
    DocumentPosition,
    DocumentSection,
    DocumentState,
)
from .state_store import JsonCheckpointStore
from .validators import validate_chunk


class DocumentOrchestrator:
    def __init__(self, llm_client, rag_service=None, checkpoint_store: Optional[JsonCheckpointStore] = None):
        self.llm_client = llm_client
        self.rag_service = rag_service
        self.checkpoint_store = checkpoint_store or JsonCheckpointStore()
        self._rag_warning_emitted = False

    def run(
        self,
        *,
        job_id: str,
        options: DocumentGenerationOptions,
        on_log: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, object]:
        self._rag_warning_emitted = False
        checkpoint = self._load_or_initialize(job_id, options, on_log=on_log)
        if checkpoint.completed:
            return self._build_result(checkpoint)

        total_sections = len(checkpoint.outline.sections)
        current_section_start = int(checkpoint.state.position.section_index)
        hard_cap_hit = False

        for section_idx in range(current_section_start, total_sections):
            if should_stop and should_stop():
                if on_log:
                    on_log("Document generation stop requested.")
                self.checkpoint_store.save(checkpoint)
                return self._build_result(checkpoint, stopped=True)

            section = checkpoint.outline.sections[section_idx]
            existing_chunks = [
                c for c in checkpoint.chunks if c.section_index == section_idx
            ]
            words_written = sum(c.word_count for c in existing_chunks)
            chunk_index = len(existing_chunks)

            while words_written < section.target_words:
                if should_stop and should_stop():
                    if on_log:
                        on_log("Document generation stop requested.")
                    self.checkpoint_store.save(checkpoint)
                    return self._build_result(checkpoint, stopped=True)

                remaining = max(40, section.target_words - words_written)
                if remaining < options.min_chunk_words:
                    target_chunk_words = max(80, remaining)
                else:
                    target_chunk_words = min(options.max_chunk_words, remaining)
                    if target_chunk_words < options.min_chunk_words:
                        target_chunk_words = options.min_chunk_words

                chunk_text, citations = self._generate_chunk_with_retries(
                    checkpoint=checkpoint,
                    section_idx=section_idx,
                    chunk_index=chunk_index,
                    options=options,
                    target_chunk_words=target_chunk_words,
                    on_log=on_log,
                )

                chunk_id = f"{section_idx}-{chunk_index}"
                if any(c.chunk_id == chunk_id for c in checkpoint.chunks):
                    chunk_index += 1
                    continue

                new_chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    section_index=section_idx,
                    section_title=section.title,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    word_count=self._word_count(chunk_text),
                    citations=citations,
                )
                checkpoint.chunks.append(new_chunk)
                words_written += new_chunk.word_count
                chunk_index += 1
                total_written = sum(c.word_count for c in checkpoint.chunks)

                checkpoint.state = self._update_state(
                    checkpoint.state,
                    checkpoint.outline,
                    section_idx,
                    chunk_index,
                    new_chunk,
                    options,
                )
                self.checkpoint_store.save(checkpoint)

                if on_progress:
                    on_progress(section_idx + 1, total_sections)

                if on_log:
                    on_log(
                        f"Generated section {section_idx + 1}/{total_sections} chunk {chunk_index} "
                        f"({new_chunk.word_count} words)."
                    )

                if options.hard_max_words > 0 and total_written >= options.hard_max_words:
                    hard_cap_hit = True
                    if on_log:
                        on_log(
                            f"Reached document hard cap ({options.hard_max_words} words). "
                            "Stopping early to keep output bounded."
                        )
                    break

                if options.consistency_check_interval > 0 and len(checkpoint.chunks) % options.consistency_check_interval == 0:
                    issues = self._consistency_check(checkpoint)
                    checkpoint.state.consistency_patches = self._extract_patch_instructions(issues)
                    self.checkpoint_store.save(checkpoint)
                    if issues and on_log:
                        on_log(f"Consistency check warning: {issues}")

            if hard_cap_hit:
                break

        checkpoint.completed = True
        self.checkpoint_store.save(checkpoint)
        return self._build_result(checkpoint)

    def _load_or_initialize(
        self,
        job_id: str,
        options: DocumentGenerationOptions,
        *,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> DocumentCheckpoint:
        if options.resume:
            existing = self.checkpoint_store.load(job_id)
            if existing:
                if on_log:
                    on_log(f"Resuming document job '{job_id}' from checkpoint.")
                return existing

        outline = self._build_outline(options)
        initial_state = DocumentState(
            position=DocumentPosition(section_index=0, section_title=outline.sections[0].title if outline.sections else "", chunk_index=0),
        )
        checkpoint = DocumentCheckpoint(
            job_id=job_id,
            prompt=options.prompt,
            mode=options.mode,
            outline=outline,
            state=initial_state,
            chunks=[],
            completed=False,
        )
        self.checkpoint_store.save(checkpoint)
        if on_log:
            on_log(f"Started new document job '{job_id}'.")
        return checkpoint

    def _build_outline(self, options: DocumentGenerationOptions) -> DocumentOutline:
        prompt = (
            "You are a senior technical writer. Return ONLY JSON with this schema: "
            "{\"topic\": str, \"audience\": str, \"total_target_words\": int, "
            "\"sections\": [{\"title\": str, \"purpose\": str, \"target_words\": int}]}. "
            "No markdown.\n"
            f"Topic: {options.prompt}\n"
            f"Audience: {options.audience}\n"
            f"Target length words: {options.target_words}\n"
            "Create 4-10 sections with practical distribution and coherent flow."
        )
        raw = self.llm_client.generate_completion(prompt, system_prompt="You create strict JSON outlines.")
        parsed = self._parse_json(raw)
        if parsed and isinstance(parsed.get("sections"), list) and parsed["sections"]:
            try:
                safe_sections = []
                for sec in parsed.get("sections", []):
                    if not isinstance(sec, dict):
                        continue
                    safe_sections.append(
                        {
                            "title": str(sec.get("title", "Section")),
                            "purpose": str(sec.get("purpose", "")),
                            "target_words": int(sec.get("target_words", 200) or 200),
                        }
                    )
                outline = DocumentOutline(
                    topic=str(parsed.get("topic", options.prompt)),
                    audience=str(parsed.get("audience", options.audience)),
                    total_target_words=int(parsed.get("total_target_words", options.target_words) or options.target_words),
                    sections=[DocumentSection(**s) for s in safe_sections],
                )
                if outline.total_target_words <= 0:
                    outline.total_target_words = options.target_words
                return self._normalize_outline(outline, options.target_words)
            except Exception:
                pass

        fallback = DocumentOutline(
            topic=options.prompt,
            audience=options.audience,
            total_target_words=options.target_words,
            sections=[
                DocumentSection(title="Introduction", purpose="Set context and define goals.", target_words=max(150, options.target_words // 5)),
                DocumentSection(title="Core Analysis", purpose="Develop the key points in depth.", target_words=max(250, options.target_words // 2)),
                DocumentSection(title="Recommendations", purpose="Provide practical recommendations and actions.", target_words=max(180, options.target_words // 4)),
            ],
        )
        return self._normalize_outline(fallback, options.target_words)

    def _normalize_outline(self, outline: DocumentOutline, target_words: int) -> DocumentOutline:
        sections = outline.sections or []
        if not sections:
            return DocumentOutline(
                topic=outline.topic,
                audience=outline.audience,
                total_target_words=target_words,
                sections=[
                    DocumentSection(title="Document", purpose="Complete requested output.", target_words=target_words),
                ],
            )

        max_sections = max(2, min(10, target_words // 180))
        if len(sections) > max_sections:
            sections = sections[:max_sections]

        total = sum(max(60, int(s.target_words)) for s in sections)
        if total <= 0:
            total = 1
        scale = target_words / total
        min_per_section = max(60, min(100, target_words // max(1, len(sections))))
        adjusted: List[DocumentSection] = []
        for sec in sections:
            w = max(min_per_section, int(max(60, sec.target_words) * scale))
            adjusted.append(DocumentSection(title=sec.title, purpose=sec.purpose, target_words=w))

        current_total = sum(s.target_words for s in adjusted)
        if current_total != target_words and adjusted:
            adjusted[-1].target_words = max(min_per_section, adjusted[-1].target_words + (target_words - current_total))

        return DocumentOutline(
            topic=outline.topic,
            audience=outline.audience,
            total_target_words=target_words,
            sections=adjusted,
        )

    def _generate_chunk_with_retries(
        self,
        *,
        checkpoint: DocumentCheckpoint,
        section_idx: int,
        chunk_index: int,
        options: DocumentGenerationOptions,
        target_chunk_words: int,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> Tuple[str, List[Dict[str, object]]]:
        section = checkpoint.outline.sections[section_idx]
        next_title = checkpoint.outline.sections[section_idx + 1].title if section_idx + 1 < len(checkpoint.outline.sections) else ""
        context, citations = self._retrieve_context(checkpoint.prompt, options.mode, on_log=on_log)

        correction = ""
        text = ""
        max_retries = max(1, int(options.max_retries))
        for attempt in range(1, max_retries + 1):
            prompt = self._build_generation_prompt(
                checkpoint=checkpoint,
                section=section,
                next_title=next_title,
                target_chunk_words=target_chunk_words,
                mode=options.mode,
                retrieved_context=context,
                correction=correction,
            )
            text = self.llm_client.generate_completion(
                prompt,
                system_prompt=(
                    "You are an expert long-form writer. Write only the requested section chunk, "
                    "maintain coherence, and end on a complete paragraph boundary."
                ),
            )
            text = (text or "").strip()

            validation = validate_chunk(
                text,
                target_words=target_chunk_words,
                tail_content=checkpoint.state.tail_content,
                next_section_title=next_title,
                min_ratio=0.65 if options.fast_mode else 0.80,
                max_ratio=1.60 if options.fast_mode else 1.25,
                repetition_jaccard_threshold=0.45 if options.fast_mode else 0.30,
            )
            if validation.is_valid:
                bounded = self._enforce_chunk_bounds(text, target_chunk_words, options.fast_mode)
                return bounded, citations

            correction = "Validation failures: " + " ".join(validation.reasons)
            if on_log:
                on_log(f"Chunk retry {attempt}/{max_retries} due to: {correction}")

        fallback_text = text if text else f"{section.title}: {section.purpose}"
        bounded_fallback = self._enforce_chunk_bounds(fallback_text, target_chunk_words, options.fast_mode)
        return bounded_fallback, citations

    def _build_generation_prompt(
        self,
        *,
        checkpoint: DocumentCheckpoint,
        section: DocumentSection,
        next_title: str,
        target_chunk_words: int,
        mode: DocumentMode,
        retrieved_context: str,
        correction: str,
    ) -> str:
        outline_lines = [
            f"- {idx + 1}. {s.title} ({s.target_words} words): {s.purpose}"
            for idx, s in enumerate(checkpoint.outline.sections)
        ]
        fact_block = checkpoint.state.fact_registry.model_dump()

        grounding_rules = ""
        if mode == DocumentMode.STRICT_GROUNDED:
            grounding_rules = "Use only claims supported by retrieved context. If context is missing, state uncertainty."
        elif mode == DocumentMode.HYBRID:
            grounding_rules = "Use retrieved context when relevant, and synthesize additional narrative where safe."
        else:
            grounding_rules = "No external file grounding required; rely on outline and existing state."

        return (
            f"Document Topic: {checkpoint.outline.topic}\n"
            f"Audience: {checkpoint.outline.audience}\n"
            f"Tone: {checkpoint.state.style_signals.detected_tone}\n"
            f"Outline:\n" + "\n".join(outline_lines) + "\n\n"
            f"Current Section: {section.title}\n"
            f"Section Purpose: {section.purpose}\n"
            f"Section Target Words: {section.target_words}\n"
            f"Chunk Target Words: {target_chunk_words}\n"
            f"Rolling Summary:\n{checkpoint.state.rolling_summary or '(none)'}\n\n"
            f"Tail Content (verbatim):\n{checkpoint.state.tail_content or '(none)'}\n\n"
            f"Fact Registry:\n{fact_block}\n\n"
            f"Retrieved Context:\n{retrieved_context or '(none)'}\n\n"
            f"Rules:\n- {grounding_rules}\n"
            "- Do not repeat prior paragraphs verbatim.\n"
            "- Stay inside the current section; do not start the next section.\n"
            "- End at a complete paragraph boundary.\n"
            f"- Stop before beginning content related to: {next_title or 'END OF DOCUMENT'}.\n"
            f"{self._format_patch_rules(checkpoint.state.consistency_patches)}"
            f"{correction}\n"
            "Return only the chunk text."
        )

    def _retrieve_context(
        self,
        prompt: str,
        mode: DocumentMode,
        *,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> Tuple[str, List[Dict[str, object]]]:
        if mode == DocumentMode.PURE or not self.rag_service:
            return "", []
        try:
            top_k = getattr(self.rag_service, "top_k", 5)
            min_score = getattr(self.rag_service, "min_score", 0.25)
            hits = self.rag_service.search(prompt, top_k=top_k, min_score=min_score, source_filter=None)
            context = self.rag_service.format_hits(hits, max_context_chars=getattr(self.rag_service, "max_context_chars", 3000))

            citations = [
                {
                    "source": hit.metadata.get("source", "unknown"),
                    "page": hit.metadata.get("page", "?"),
                    "score": hit.score,
                }
                for hit in hits
            ]
            return context, citations
        except Exception as ex:
            if on_log and not self._rag_warning_emitted:
                on_log(f"RAG retrieval unavailable ({ex}). Continuing in non-RAG mode.")
                self._rag_warning_emitted = True
            return "", []

    def _update_state(
        self,
        state: DocumentState,
        outline: DocumentOutline,
        section_idx: int,
        chunk_index: int,
        chunk: DocumentChunk,
        options: DocumentGenerationOptions,
    ) -> DocumentState:
        updated = state.model_copy(deep=True)

        updated.position.section_index = section_idx
        updated.position.section_title = outline.sections[section_idx].title
        updated.position.chunk_index = chunk_index

        total_words = sum(s.target_words for s in outline.sections)
        written_words = self._word_count((updated.rolling_summary + "\n" + chunk.text).strip())
        updated.position.percent_complete = min(100.0, (written_words / max(1, total_words)) * 100.0)

        updated.tail_content = self._tail_words((updated.tail_content + "\n\n" + chunk.text).strip(), 480)
        updated.rolling_summary = self._summarize_text(
            current_summary=updated.rolling_summary,
            new_chunk=chunk.text,
            max_words=160,
        )

        entities = self._extract_named_entities(chunk.text)
        for item in entities:
            if item not in updated.fact_registry.named_entities:
                updated.fact_registry.named_entities.append(item)

        updated.style_signals.detected_tone = options.tone
        return updated

    def _consistency_check(self, checkpoint: DocumentCheckpoint) -> str:
        prompt = (
            "Review for contradictions, undefined terminology, or narrative inconsistency. "
            "Return either 'OK' or compact patch instructions (1-3 lines) for future sections.\n"
            f"Summary:\n{checkpoint.state.rolling_summary}\n\n"
            f"Fact Registry:\n{checkpoint.state.fact_registry.model_dump()}"
        )
        result = self.llm_client.generate_completion(prompt, system_prompt="You are a consistency auditor.")
        return (result or "").strip()

    @staticmethod
    def _extract_patch_instructions(issues_text: str) -> List[str]:
        text = (issues_text or "").strip()
        if not text:
            return []
        if text.upper() == "OK":
            return []
        lines = [line.strip(" -\t") for line in text.splitlines() if line.strip()]
        return lines[:3]

    @staticmethod
    def _format_patch_rules(patches: List[str]) -> str:
        if not patches:
            return ""
        lines = ""
        for patch in patches:
            lines += f"- Consistency patch: {patch}\n"
        return lines

    def _build_result(self, checkpoint: DocumentCheckpoint, stopped: bool = False) -> Dict[str, object]:
        ordered = sorted(checkpoint.chunks, key=lambda c: (c.section_index, c.chunk_index))
        section_blocks: Dict[int, List[str]] = {}
        citations: List[Dict[str, object]] = []
        for chunk in ordered:
            section_blocks.setdefault(chunk.section_index, []).append(chunk.text)
            citations.extend(chunk.citations)

        assembled_sections = []
        for idx, section in enumerate(checkpoint.outline.sections):
            body = "\n\n".join(section_blocks.get(idx, [])).strip()
            if not body:
                continue
            assembled_sections.append(f"{section.title}\n\n{body}")
        final_text = "\n\n".join(assembled_sections).strip()

        return {
            "job_id": checkpoint.job_id,
            "completed": checkpoint.completed,
            "stopped": stopped,
            "mode": checkpoint.mode.value,
            "title": checkpoint.outline.topic,
            "outline": checkpoint.outline.model_dump(),
            "state": checkpoint.state.model_dump(),
            "chunks": [c.model_dump() for c in ordered],
            "citations": citations,
            "text": final_text,
        }

    @staticmethod
    def _word_count(text: str) -> int:
        return len([w for w in (text or "").split() if w.strip()])

    @staticmethod
    def _tail_words(text: str, max_words: int) -> str:
        words = [w for w in text.split() if w.strip()]
        if len(words) <= max_words:
            return text
        return " ".join(words[-max_words:])

    def _summarize_text(self, current_summary: str, new_chunk: str, max_words: int = 160) -> str:
        prompt = (
            "Update this rolling summary with the new chunk in 2-3 sentences. Keep key decisions, entities, and conclusions.\n"
            f"Current summary:\n{current_summary or '(none)'}\n\n"
            f"New chunk:\n{new_chunk}\n"
        )
        summary = self.llm_client.generate_completion(prompt, system_prompt="You compress state for long-form writing.")
        summary = (summary or "").strip()
        words = [w for w in summary.split() if w.strip()]
        if len(words) <= max_words:
            return summary
        return " ".join(words[:max_words])

    @staticmethod
    def _extract_named_entities(text: str) -> List[str]:
        candidates = re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b", text or "")
        unique: List[str] = []
        for item in candidates:
            if item.lower() in {"the", "this", "that", "and"}:
                continue
            if item not in unique:
                unique.append(item)
        return unique[:30]

    @staticmethod
    def _enforce_chunk_bounds(text: str, target_chunk_words: int, fast_mode: bool) -> str:
        clean = (text or "").strip()
        if not clean:
            return clean

        words = [w for w in clean.split() if w.strip()]
        ratio = 1.35 if fast_mode else 1.20
        max_words = max(80, int(target_chunk_words * ratio))
        if len(words) <= max_words:
            return clean

        clipped = " ".join(words[:max_words]).strip()
        if clipped and clipped[-1] not in ".!?\"'":
            clipped += "."
        return clipped

    @staticmethod
    def _parse_json(raw: Optional[str]) -> Optional[Dict[str, object]]:
        if not raw:
            return None
        cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except Exception:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(cleaned[start : end + 1])
                except Exception:
                    return None
            return None
