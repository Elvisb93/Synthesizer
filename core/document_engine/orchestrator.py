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

    @staticmethod
    def _is_narrative_request(options: DocumentGenerationOptions) -> bool:
        return DocumentOrchestrator._is_narrative_request_from_parts(
            prompt=options.prompt,
            audience=options.audience,
            tone=options.tone,
            mode=options.mode,
        )

    @staticmethod
    def _is_narrative_request_from_parts(
        *,
        prompt: str,
        audience: str,
        tone: str,
        mode: DocumentMode,
    ) -> bool:
        if mode == DocumentMode.PURE:
            return True
        text = " ".join(
            part.strip().lower()
            for part in (prompt or "", audience or "", tone or "")
            if str(part or "").strip()
        )
        if not text:
            return False
        narrative_markers = {
            "story",
            "fiction",
            "novel",
            "narrative",
            "tale",
            "chapter",
            "scene",
            "character",
            "romance",
            "erotic",
            "sensual",
            "fantasy",
            "poem",
            "poetry",
            "screenplay",
            "script",
            "dialogue",
        }
        return any(marker in text for marker in narrative_markers)

    @staticmethod
    def _is_comparison_request_from_parts(
        *,
        prompt: str,
        audience: str,
        tone: str,
        mode: DocumentMode,
    ) -> bool:
        if mode == DocumentMode.PURE:
            return False
        text = " ".join(
            part.strip().lower()
            for part in (prompt or "", audience or "", tone or "")
            if str(part or "").strip()
        )
        if not text:
            return False
        comparison_markers = {
            "compare",
            "comparison",
            "best",
            "better",
            "recommend",
            "recommendation",
            "choose",
            "choice",
            "which",
            "option",
            "options",
            "rank",
            "ranking",
            "versus",
            "vs",
            "tradeoff",
            "trade-off",
            "evaluate",
            "selection",
        }
        return any(marker in text for marker in comparison_markers)

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
        narrative = self._is_narrative_request(options)
        comparison = self._is_comparison_request_from_parts(
            prompt=options.prompt,
            audience=options.audience,
            tone=options.tone,
            mode=options.mode,
        )
        if narrative:
            prompt = (
                "You are designing a narrative writing plan. Return ONLY JSON with this schema: "
                "{\"topic\": str, \"audience\": str, \"total_target_words\": int, "
                "\"sections\": [{\"title\": str, \"purpose\": str, \"target_words\": int}]}. "
                "No markdown.\n"
                f"Topic: {options.prompt}\n"
                f"Audience: {options.audience}\n"
                f"Target length words: {options.target_words}\n"
                "Create 3-8 scene or act beats with coherent emotional progression and varied pacing."
            )
            system_prompt = "You create strict JSON narrative outlines."
        elif comparison:
            prompt = (
                "You are planning a grounded comparison report. Return ONLY JSON with this schema: "
                "{\"topic\": str, \"audience\": str, \"total_target_words\": int, "
                "\"sections\": [{\"title\": str, \"purpose\": str, \"target_words\": int}]}. "
                "No markdown.\n"
                f"Topic: {options.prompt}\n"
                f"Audience: {options.audience}\n"
                f"Target length words: {options.target_words}\n"
                "Create 4-8 sections that: define the decision criteria, compare the relevant sources, "
                "surface tradeoffs or clauses that could change the answer, and end with a recommendation."
            )
            system_prompt = "You create strict JSON comparison outlines."
        else:
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
            system_prompt = "You create strict JSON outlines."
        raw = self.llm_client.generate_completion(prompt, system_prompt=system_prompt)
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

        if narrative:
            fallback = DocumentOutline(
                topic=options.prompt,
                audience=options.audience,
                total_target_words=options.target_words,
                sections=[
                    DocumentSection(title="Opening Scene", purpose="Establish the setting, mood, and characters.", target_words=max(150, options.target_words // 4)),
                    DocumentSection(title="Rising Tension", purpose="Deepen the conflict, chemistry, or intrigue.", target_words=max(220, options.target_words // 2)),
                    DocumentSection(title="Climax And Afterglow", purpose="Deliver the payoff and end on a satisfying emotional beat.", target_words=max(160, options.target_words // 4)),
                ],
            )
        elif comparison:
            fallback = DocumentOutline(
                topic=options.prompt,
                audience=options.audience,
                total_target_words=options.target_words,
                sections=[
                    DocumentSection(title="Decision Criteria", purpose="Define the user's requirements and the criteria that matter most.", target_words=max(150, options.target_words // 5)),
                    DocumentSection(title="Source-by-Source Findings", purpose="Summarize the strongest relevant evidence from each source.", target_words=max(220, options.target_words // 3)),
                    DocumentSection(title="Cross-Source Comparison", purpose="Compare tradeoffs, exclusions, edge clauses, and material differences.", target_words=max(220, options.target_words // 3)),
                    DocumentSection(title="Recommendation", purpose="Recommend the best-supported option and explain why the alternatives are weaker fits.", target_words=max(180, options.target_words // 5)),
                ],
            )
        else:
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
        retrieval_query = self._build_retrieval_query(
            checkpoint=checkpoint,
            section=section,
            next_title=next_title,
            mode=options.mode,
        )
        context, citations = self._retrieve_context(retrieval_query, options.mode, on_log=on_log)
        source_guidance = self._build_source_guidance(
            checkpoint=checkpoint,
            section=section,
            citations=citations,
            mode=options.mode,
        )

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
                source_guidance=source_guidance,
                correction=correction,
            )
            raw_text = self.llm_client.generate_completion(
                prompt,
                system_prompt=(
                    "You are an expert long-form writer. "
                    "Never output chain-of-thought, internal reasoning, analysis logs, or thinking steps. "
                    "Return only final content in the requested JSON schema."
                ),
            )
            text = self._extract_chunk_text(raw_text)
            text = self._sanitize_chunk_for_publish(text)
            if self._has_meta_artifacts(text):
                text = self._repair_chunk_with_llm(text, section.title, target_chunk_words)
                text = self._sanitize_chunk_for_publish(text)

            validation = validate_chunk(
                text,
                target_words=target_chunk_words,
                tail_content=checkpoint.state.tail_content,
                next_section_title=next_title,
                min_ratio=0.65 if options.fast_mode else 0.80,
                max_ratio=None,
                repetition_jaccard_threshold=0.45 if options.fast_mode else 0.30,
            )
            if validation.is_valid:
                bounded = self._enforce_chunk_bounds(text, target_chunk_words, options.fast_mode)
                return bounded, citations

            correction = "Validation failures: " + " ".join(validation.reasons)
            if on_log:
                on_log(f"Chunk retry {attempt}/{max_retries} due to: {correction}")

        fallback_text = text if text else f"{section.title}: {section.purpose}"
        fallback_text = self._sanitize_chunk_for_publish(fallback_text)
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
        source_guidance: str,
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

        narrative = self._is_narrative_request_from_parts(
            prompt=checkpoint.prompt,
            audience=checkpoint.outline.audience,
            tone=checkpoint.state.style_signals.detected_tone,
            mode=mode,
        )
        narrative_rule = (
            "- Prioritize vivid scene writing, character motion, and emotional continuity over report structure.\n"
            if narrative
            else ""
        )
        return (
            f"Document Topic: {checkpoint.outline.topic}\n"
            f"Audience: {checkpoint.outline.audience}\n"
            f"Tone: {checkpoint.state.style_signals.detected_tone}\n"
            f"Outline:\n" + "\n".join(outline_lines) + "\n\n"
            f"Current Section: {section.title}\n"
            f"Section Purpose: {section.purpose}\n"
            f"Section Target Words: {section.target_words}\n"
            f"Chunk Minimum Words: {target_chunk_words}\n"
            f"Rolling Summary:\n{checkpoint.state.rolling_summary or '(none)'}\n\n"
            f"Tail Content (verbatim):\n{checkpoint.state.tail_content or '(none)'}\n\n"
            f"Fact Registry:\n{fact_block}\n\n"
            f"Retrieved Context:\n{retrieved_context or '(none)'}\n\n"
            f"Rules:\n- {grounding_rules}\n"
            "- Treat the chunk minimum as a floor, not a ceiling; finish the thought naturally.\n"
            "- Do not repeat prior paragraphs verbatim.\n"
            "- Stay inside the current section; do not start the next section.\n"
            "- End at a complete paragraph boundary.\n"
            "- Do not include chain-of-thought, reasoning steps, or process commentary.\n"
            f"{source_guidance}"
            f"{narrative_rule}"
            f"- Stop before beginning content related to: {next_title or 'END OF DOCUMENT'}.\n"
            f"{self._format_patch_rules(checkpoint.state.consistency_patches)}"
            f"{correction}\n"
            "Return ONLY JSON with this exact schema: {\"chunk\": \"<final section prose>\"}"
        )

    def _build_retrieval_query(
        self,
        *,
        checkpoint: DocumentCheckpoint,
        section: DocumentSection,
        next_title: str,
        mode: DocumentMode,
    ) -> str:
        comparison = self._is_comparison_request_from_parts(
            prompt=checkpoint.prompt,
            audience=checkpoint.outline.audience,
            tone=checkpoint.state.style_signals.detected_tone,
            mode=mode,
        )
        query_lines = [
            checkpoint.prompt.strip(),
            f"Section focus: {section.title}. {section.purpose}",
        ]
        if checkpoint.state.rolling_summary:
            query_lines.append(f"Current document state: {checkpoint.state.rolling_summary}")
        if next_title:
            query_lines.append(f"Do not drift into the next section: {next_title}")
        if comparison:
            query_lines.append(
                "Find evidence that compares sources, highlights tradeoffs, and surfaces clauses or details that could materially change the final recommendation."
            )
        return "\n".join(line for line in query_lines if line.strip())

    def _build_source_guidance(
        self,
        *,
        checkpoint: DocumentCheckpoint,
        section: DocumentSection,
        citations: List[Dict[str, object]],
        mode: DocumentMode,
    ) -> str:
        sources: List[str] = []
        for citation in citations or []:
            source = str(citation.get("source", "") or "").strip()
            if source and source not in sources:
                sources.append(source)

        comparison = self._is_comparison_request_from_parts(
            prompt=checkpoint.prompt,
            audience=checkpoint.outline.audience,
            tone=checkpoint.state.style_signals.detected_tone,
            mode=mode,
        )
        if not sources and not comparison:
            return ""

        lines: List[str] = []
        if sources:
            joined = ", ".join(sources)
            if len(sources) > 1:
                lines.append(f"- Relevant source coverage in current evidence: {joined}.\n")
                lines.append("- Use more than one relevant source when the evidence supports it; do not let a single source dominate silently.\n")
            else:
                lines.append(f"- Primary source in current evidence: {joined}.\n")
        if comparison:
            lines.append("- Compare the relevant sources directly on the criteria implied by the request.\n")
            lines.append("- Surface tradeoffs, exclusions, caveats, and lower-ranked clauses when they materially affect the answer.\n")
            lines.append("- If you recommend an option, justify it against the strongest alternatives using cited evidence.\n")
            lines.append(f"- Keep this section focused on {section.title.lower()} rather than generic best-practice advice.\n")
        return "".join(lines)

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
            if hasattr(self.rag_service, "prepare_document_context"):
                prepared = self.rag_service.prepare_document_context(
                    prompt,
                    source_filter=None,
                    document_mode=mode.value,
                )
                if prepared is not None:
                    return (
                        str(prepared.get("context", "") or ""),
                        list(prepared.get("citations", []) or []),
                    )
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
        return clean

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
                    pass

        decoder = json.JSONDecoder()
        candidates: List[Dict[str, object]] = []
        for idx, ch in enumerate(cleaned):
            if ch != "{":
                continue
            try:
                parsed, _end = decoder.raw_decode(cleaned[idx:])
            except Exception:
                continue
            if isinstance(parsed, dict):
                candidates.append(parsed)

        if not candidates:
            return None

        for preferred_key in ("chunk", "charts", "sections", "body_markdown", "title"):
            for candidate in reversed(candidates):
                if preferred_key in candidate:
                    return candidate
        return candidates[-1]

    def _extract_chunk_text(self, raw: Optional[str]) -> str:
        text = (raw or "").strip()
        if not text:
            return ""

        parsed = self._parse_json(text)
        if parsed and isinstance(parsed.get("chunk"), str):
            text = str(parsed.get("chunk") or "").strip()

        # Strip common reasoning wrappers if model ignored instructions.
        text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
        lines = text.splitlines()
        if lines and re.match(r"^\s*(thinking process|reasoning|analysis)\b", lines[0], re.IGNORECASE):
            drop_idx = 0
            for idx, line in enumerate(lines):
                if line.strip() == "":
                    nxt = idx + 1
                    while nxt < len(lines) and lines[nxt].strip() == "":
                        nxt += 1
                    if nxt < len(lines):
                        probe = lines[nxt].strip()
                        if not re.match(r"^(\d+[\).\:]|[-*])\s+", probe) and not re.match(
                            r"^(thinking process|reasoning|analysis|step\b)", probe, re.IGNORECASE
                        ):
                            drop_idx = nxt
                            break
            if drop_idx > 0:
                text = "\n".join(lines[drop_idx:]).strip()

        # Last-resort cleanup: remove heading line if it is a reasoning marker.
        if re.match(r"^\s*(thinking process|reasoning|analysis)\b", text, re.IGNORECASE):
            text = re.sub(r"^\s*(thinking process|reasoning|analysis)\s*:?\s*", "", text, flags=re.IGNORECASE).strip()

        return text

    @staticmethod
    def _sanitize_chunk_for_publish(text: str) -> str:
        if not text:
            return ""
        cleaned = text
        cleaned = re.sub(r"(?is)<think>.*?</think>", "", cleaned)
        cleaned = cleaned.replace("```json", "").replace("```", "")
        cleaned = re.sub(r'(?is)^\s*\{\s*"chunk"\s*:\s*"(.*)"\s*\}\s*$', r"\1", cleaned)
        cleaned = re.sub(r"(?i)\*?\s*word count check\s*:?\*?", "", cleaned)

        # Drop common meta/prompt reflection lines.
        meta_line = re.compile(
            r"(?i)\b("
            r"thinking process|chain-of-thought|internal reasoning|reasoning steps|"
            r"constraint[s]?|output format|tail content|fact registry|retrieved context|"
            r"document topic|current section|section target words|chunk target words|"
            r"return only json|the prompt|the instruction says|i need to|i should|step-by-step"
            r")\b"
        )
        lines = cleaned.splitlines()
        kept: List[str] = []
        for ln in lines:
            s = ln.strip()
            if not s:
                kept.append("")
                continue
            if meta_line.search(s):
                continue
            # Remove list-style self-instruction bullets.
            if re.match(r"^\s*(\d+[\).\:]|[-*])\s+", s) and re.search(
                r"(?i)\b(analyze|constraint|output|instruction|prompt|reasoning|task)\b", s
            ):
                continue
            kept.append(ln)

        cleaned = "\n".join(kept).strip()
        numbered_pairs = re.findall(r"\b\d+\s+[A-Za-z][A-Za-z'-]*\b", cleaned[:5000])
        if len(numbered_pairs) >= 8:
            cleaned = re.sub(r"\b\d+\s+(?=[A-Za-z][A-Za-z'-]*\b)", "", cleaned)
            cleaned = re.sub(r"\s{2,}", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned

    @staticmethod
    def _has_meta_artifacts(text: str) -> bool:
        if not text:
            return False
        probes = (
            "thinking process",
            "chain-of-thought",
            "internal reasoning",
            "tail content",
            "fact registry",
            "retrieved context",
            "return only json",
            "the prompt",
            "the instruction says",
            "i need to",
            "i should",
            "output format",
            "constraint",
            "word count check",
        )
        lowered = text.lower()
        return any(p in lowered for p in probes)

    def _repair_chunk_with_llm(self, text: str, section_title: str, target_chunk_words: int) -> str:
        if not text or not self.llm_client:
            return text
        prompt = (
            "Clean the following draft section for publication.\n"
            "Remove all meta commentary, reasoning, prompt references, constraints, and instructions.\n"
            "Keep only polished final prose for the section.\n"
            f"Section: {section_title}\n"
            f"Target words: ~{target_chunk_words}\n"
            "Return ONLY JSON: {\"chunk\": \"...\"}\n\n"
            f"Draft:\n{text}\n"
        )
        repaired = self.llm_client.generate_completion(
            prompt,
            system_prompt="You are a strict editor. Output valid JSON only.",
        )
        out = self._extract_chunk_text(repaired)
        return out or text
