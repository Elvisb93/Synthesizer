"""
GeneratorController — Orchestrates synthetic data generation.

Manages the generation lifecycle: initialization, row generation (fresh + enrichment),
threading, and stop control. Delegates to specialized modules for:
- Prompt construction: core.prompt_builder
- Export: core.exporters
- Metrics: core.metrics
- Validation: core.validator
- Quality analysis: core.analytics
"""
import logging
import time
import threading
import hashlib
import json
import re
from typing import List, Callable, Optional, Dict, Any

from .models import GeneratorConfig, ColumnDefinition, RowData, ColumnType, AIProvider
from .llm_client import LLMClient, PROVIDER_BASE_URLS
from .charts import DocumentChartGenerator
from .validator import UniquenessValidator
from .analytics import QualityAnalyzer
from .exporters import (
    PDFReportGenerator,
    DocumentPDFExporter,
    DocumentDocxExporter,
    export_csv,
    export_json,
    export_power_bi_run,
    export_sql,
)
from .prompt_builder import get_dependencies, get_execution_order, construct_prompt
from .metrics import calculate_metrics
import pandas as pd
from .rag import create_rag_backend
from .document_engine import DocumentGenerationOptions, DocumentMode, DocumentOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneratorController:
    def __init__(self):
        self.config = GeneratorConfig(model_id="local-model")  # Default, will be updated
        self.columns: List[ColumnDefinition] = []
        self.execution_order: List[ColumnDefinition] = []  # Sorted columns
        self.generated_rows: List[RowData] = []
        
        self.llm_client: Optional[LLMClient] = None
        self.rag_service: Optional[RagService] = None
        self.validator: Optional[UniquenessValidator] = None
        self.analyzer = QualityAnalyzer()
        self.pdf_exporter = PDFReportGenerator()
        self.document_pdf_exporter = DocumentPDFExporter()
        self.document_docx_exporter = DocumentDocxExporter()
        self.document_orchestrator: Optional[DocumentOrchestrator] = None
        self.document_chart_generator: Optional[DocumentChartGenerator] = None
        self.document_result: Optional[Dict[str, Any]] = None
        self.json_template_result: Optional[Dict[str, Any]] = None
        
        # Metrics Tracking
        self.metrics_data = {
            "faker_cols": 0,
            "llm_cols": 0,
        }
        
        # Faker for deterministic columns
        try:
            from faker import Faker
            self.fake = Faker()
        except ImportError:
            self.fake = None
            self.log("Warning: Faker library not found. Deterministic columns will fail.")
        
        self.is_running = False
        self.stop_requested = False
        
        # Callbacks
        self.on_progress: Optional[Callable[[int, int], None]] = None
        self.on_log: Optional[Callable[[str], None]] = None
        self.on_finished: Optional[Callable[[], None]] = None
        
    def initialize(self, config: GeneratorConfig, columns: List[ColumnDefinition]):
        self.config = config
        self.columns = columns
        self.llm_client = LLMClient(config, on_log=self.log)
        self.initialize_rag()
        self.validator = UniquenessValidator(config)
        self.generated_rows = []
        self.document_result = None
        self.stop_requested = False
        
        # Reset run metrics
        self.metrics_data = {
            "faker_cols": 0,
            "llm_cols": 0,
        }
        
        # Calculate execution order based on dependencies
        try:
            self.execution_order = get_execution_order(columns, log_fn=self.log)
            self.log(f"Execution order resolved: {[c.name for c in self.execution_order]}")
        except Exception as e:
            self.log(f"Dependency Error: {e}")
            raise e

    def initialize_rag(self):
        rag_cfg = self.config.rag
        if not rag_cfg:
            self.rag_service = None
            if self.llm_client:
                self.llm_client.set_rag_service(None)
            return

        try:
            backend = rag_cfg.backend if hasattr(rag_cfg, "backend") else "Native"
            llm_base_url = PROVIDER_BASE_URLS.get(self.config.provider, self.config.api_base_url)
            llm_api_key = "lm-studio" if self.config.provider == AIProvider.LM_STUDIO else (self.config.api_key or "")
            self.rag_service = create_rag_backend(
                backend=backend,
                collection_name=rag_cfg.collection_name,
                qdrant_url=rag_cfg.qdrant_url,
                qdrant_api_key=rag_cfg.qdrant_api_key,
                embedding_model=rag_cfg.embedding_model,
                top_k=rag_cfg.top_k,
                min_score=rag_cfg.min_score,
                max_context_chars=rag_cfg.max_context_chars,
                ocr_mode=rag_cfg.ocr_mode.value if hasattr(rag_cfg.ocr_mode, "value") else str(rag_cfg.ocr_mode),
                ocr_dpi=rag_cfg.ocr_dpi,
                ocr_max_pages=rag_cfg.ocr_max_pages,
                ocr_max_regions_per_page=rag_cfg.ocr_max_regions_per_page,
                ocr_region_padding_px=rag_cfg.ocr_region_padding_px,
                ocr_gap_multiplier=rag_cfg.ocr_gap_multiplier,
                ocr_min_extracted_chars=rag_cfg.ocr_min_extracted_chars,
                ocr_timeout_ms_per_page=rag_cfg.ocr_timeout_ms_per_page,
                parser_mode=rag_cfg.parser_mode,
                hybrid_search_enabled=rag_cfg.hybrid_search_enabled,
                rerank_enabled=rag_cfg.rerank_enabled,
                summary_first_enabled=rag_cfg.summary_first_enabled,
                summary_top_k=rag_cfg.summary_top_k,
                dense_top_k=rag_cfg.dense_top_k,
                lexical_top_k=rag_cfg.lexical_top_k,
                parent_context_enabled=rag_cfg.parent_context_enabled,
                parent_context_max_chars=rag_cfg.parent_context_max_chars,
                graph_enabled=rag_cfg.graph_enabled,
                graph_hops=rag_cfg.graph_hops,
                graph_source_boost=rag_cfg.graph_source_boost,
                late_interaction_enabled=rag_cfg.late_interaction_enabled,
                late_interaction_weight=rag_cfg.late_interaction_weight,
                llm_model_name=self.config.model_id,
                llm_base_url=llm_base_url,
                llm_api_key=llm_api_key,
                llm_temperature=0.0,
                llm_context_window=16384,
                llm_num_output=768,
                llm_enabled=bool(self.config.model_id and llm_base_url),
            )
            if self.llm_client:
                self.llm_client.set_rag_service(self.rag_service)
            backend_name = backend.value if hasattr(backend, "value") else str(backend)
            self.log(f"RAG initialized successfully ({backend_name}).")
        except Exception as e:
            self.rag_service = None
            if self.llm_client:
                self.llm_client.set_rag_service(None)
            self.log(f"RAG initialization failed: {e}")

        if self.llm_client:
            self.document_orchestrator = DocumentOrchestrator(self.llm_client, self.rag_service)
            self.document_chart_generator = DocumentChartGenerator(self.llm_client)

    def ingest_documents(self, paths: List[str], force_reindex: bool = False) -> Dict[str, Any]:
        if not self.rag_service:
            return {"error": "RAG is not configured."}

        report = self.rag_service.ingest_documents(paths, force_reindex=force_reindex)
        return report.model_dump()

    def search_context(self, query: str, top_k: Optional[int] = None, source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.rag_service:
            return []
        hits = self.rag_service.search(query, top_k=top_k, source_filter=source_filter)
        return [h.model_dump() for h in hits]

    def clear_rag_collection(self) -> None:
        if self.rag_service:
            self.rag_service.clear_collection()

    def get_rag_status(self) -> Dict[str, Any]:
        if not self.rag_service:
            return {"enabled": False}
        status = self.rag_service.get_status()
        status["enabled"] = True
        status["collection_name"] = self.config.rag.collection_name if self.config.rag else ""
        status["source_filter"] = self.config.rag.source_filter if self.config.rag else None
        status["ocr_mode"] = self.config.rag.ocr_mode.value if self.config.rag else "off"
        status["ocr_dpi"] = self.config.rag.ocr_dpi if self.config.rag else 150
        return status

    def set_runtime_config(self, config: GeneratorConfig) -> None:
        self.config = config
        self.llm_client = LLMClient(config, on_log=self.log)
        self.initialize_rag()

    def _auto_document_target_words(self, prompt: str, mode: DocumentMode) -> int:
        prompt_l = (prompt or "").lower()
        prompt_words = len((prompt or "").split())

        base = 1100
        if mode == DocumentMode.STRICT_GROUNDED:
            base = 950
        elif mode == DocumentMode.PURE:
            base = 1300

        if prompt_words < 10:
            base -= 150
        elif prompt_words > 40:
            base += 300

        concise_markers = ("short", "brief", "quick", "summary", "one pager", "tl;dr")
        detailed_markers = ("detailed", "comprehensive", "deep dive", "in-depth", "full report", "long-form")
        if any(marker in prompt_l for marker in concise_markers):
            base -= 350
        if any(marker in prompt_l for marker in detailed_markers):
            base += 450

        if self.rag_service:
            try:
                status = self.get_rag_status()
                collection_size = int(status.get("collection_size", 0) or 0)
                if collection_size <= 0:
                    base -= 100
                elif collection_size < 200:
                    base += 100
                elif collection_size < 1000:
                    base += 250
                else:
                    base += 400
            except Exception:
                pass

        return max(500, min(3500, int(base)))

    @staticmethod
    def _word_count(text: str) -> int:
        return len([w for w in (text or "").split() if w.strip()])

    @staticmethod
    def _trim_text_to_words(text: str, max_words: int) -> str:
        words = [w for w in (text or "").split() if w.strip()]
        if len(words) <= max_words:
            return (text or "").strip()

        clipped = " ".join(words[:max_words]).strip()
        last_boundary = max(clipped.rfind("."), clipped.rfind("!"), clipped.rfind("?"))
        if last_boundary > int(len(clipped) * 0.55):
            clipped = clipped[: last_boundary + 1].strip()
        elif clipped and clipped[-1] not in ".!?\"'":
            clipped += "."
        return clipped

    def _expand_text_to_words(self, text: str, target_words: int, *, audience: str, tone: str) -> str:
        if not self.llm_client:
            return text

        current_words = self._word_count(text)
        needed = max(0, target_words - current_words)
        if needed < 80:
            return text

        add_prompt = (
            "Extend the document with additional content while preserving structure, tone, and factual consistency.\n"
            f"Audience: {audience}\n"
            f"Tone: {tone}\n"
            f"Current document:\n{text}\n\n"
            f"Add approximately {needed} words as continuation only.\n"
            "- Do not repeat existing paragraphs.\n"
            "- Do not add new top-level headings.\n"
            "- End on a complete sentence.\n"
            "Return only the continuation text."
        )
        add_text = self.llm_client.generate_completion(
            add_prompt,
            system_prompt="You are a concise editor extending a draft without duplication.",
        )
        add_text = (add_text or "").strip()
        if not add_text:
            return text
        return f"{text.strip()}\n\n{add_text}".strip()

    def _enforce_document_length_bounds(
        self,
        result: Dict[str, Any],
        *,
        target_words: int,
        audience: str,
        tone: str,
        tolerance_ratio: float = 0.10,
    ) -> Dict[str, Any]:
        text = (result.get("text") or "").strip()
        if not text or target_words <= 0:
            return result

        lower = max(350, int(target_words * (1.0 - tolerance_ratio)))
        current = self._word_count(text)
        adjusted = False

        if current < lower:
            expanded = self._expand_text_to_words(text, lower, audience=audience, tone=tone)
            if expanded != text:
                text = expanded
                adjusted = True

        if adjusted:
            result["text"] = text
            result["length_adjusted"] = True
        result["final_word_count"] = self._word_count(result.get("text", ""))
        return result

    @staticmethod
    def _parse_json_payload(text: str) -> Optional[Dict[str, Any]]:
        cleaned = (text or "").strip().replace("```json", "").replace("```", "").strip()
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(cleaned[start : end + 1])
                    return parsed if isinstance(parsed, dict) else None
                except Exception:
                    return None
            return None

    @staticmethod
    def _contains_numbered_word_artifact(text: str) -> bool:
        sample = (text or "")[:4000]
        if "word count check" in sample.lower():
            return True
        pairs = re.findall(r"\b\d+\s+[A-Za-z][A-Za-z'-]*\b", sample)
        return len(pairs) >= 8

    @staticmethod
    def _is_narrative_document_request(
        prompt: str,
        *,
        audience: str = "",
        tone: str = "",
        mode: DocumentMode | None = None,
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
    def _is_comparison_document_request(
        prompt: str,
        *,
        audience: str = "",
        tone: str = "",
        mode: DocumentMode | None = None,
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

    @staticmethod
    def _strip_numbered_word_artifact(text: str) -> str:
        if not text:
            return ""
        cleaned = text
        cleaned = re.sub(r"(?i)\*?\s*word count check\s*:?\*?", "", cleaned)
        cleaned = re.sub(r"\b\d+\s+(?=[A-Za-z][A-Za-z'-]*\b)", "", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @staticmethod
    def _looks_like_raw_prompt_title(title: str, prompt: str) -> bool:
        t = (title or "").strip().lower()
        p = (prompt or "").strip().lower()
        if not t:
            return True
        if p and t == p:
            return True
        if len(t) > 100:
            return True
        return t.startswith(("create ", "write ", "generate ", "build ", "draft "))

    @staticmethod
    def _fallback_title_from_prompt(prompt: str, *, narrative: bool = False) -> str:
        p = (prompt or "").strip()
        if not p:
            return "Untitled Story" if narrative else "Executive Report"
        tokens = re.findall(r"[A-Za-z0-9&/-]+", p)
        keep = [t for t in tokens if t.lower() not in {"create", "write", "generate", "build", "draft", "using", "include"}]
        if not keep:
            return "Untitled Story" if narrative else "Executive Report"
        title = " ".join(keep[:8]).strip()
        if not title:
            return "Untitled Story" if narrative else "Executive Report"
        if not narrative and "report" not in title.lower():
            title = f"{title} Report"
        return title[:80]

    def _polish_document_for_publish(
        self,
        *,
        title: str,
        prompt: str,
        text: str,
        audience: str,
        tone: str,
        target_words: int,
        mode: DocumentMode,
    ) -> Dict[str, str]:
        raw_title = (title or "").strip()
        raw_text = (text or "").strip()
        narrative = self._is_narrative_document_request(prompt, audience=audience, tone=tone, mode=mode)
        comparison = self._is_comparison_document_request(prompt, audience=audience, tone=tone, mode=mode)
        if not raw_text:
            return {"title": raw_title or self._fallback_title_from_prompt(prompt, narrative=narrative), "text": ""}

        needs_cleanup = self._contains_numbered_word_artifact(raw_text)
        bad_title = self._looks_like_raw_prompt_title(raw_title, prompt)
        if not (needs_cleanup or bad_title):
            return {"title": raw_title, "text": raw_text}

        cleaned_fallback_text = self._strip_numbered_word_artifact(raw_text) if needs_cleanup else raw_text
        fallback_title = self._fallback_title_from_prompt(prompt, narrative=narrative) if bad_title else raw_title

        if not self.llm_client:
            return {"title": fallback_title, "text": cleaned_fallback_text}

        if narrative:
            polish_prompt = (
                "Rewrite the draft into clean, user-facing narrative prose in markdown.\n"
                "Return ONLY JSON with schema: {\"title\": str, \"body_markdown\": str}\n"
                "Rules:\n"
                "- Remove any token-count traces, numbering artifacts, or prompt/instruction residue.\n"
                "- Preserve the creative premise, imagery, and narrative voice from the draft.\n"
                "- Do not force business/report structure, bullet summaries, risks, actions, or recommendations.\n"
                "- Keep the tone aligned with the requested audience and tone.\n"
                f"- Audience: {audience}\n"
                f"- Tone: {tone}\n"
                f"- Target length: ~{max(500, target_words)} words.\n\n"
                f"Requested task:\n{prompt}\n\n"
                f"Current title:\n{raw_title or '(none)'}\n\n"
                f"Draft text:\n{cleaned_fallback_text}\n"
            )
            system_prompt = "You are a careful narrative editor. Output valid JSON only."
        else:
            comparison_rule = (
                "- If the request is comparative or asks for a choice, preserve side-by-side contrasts, material tradeoffs, and a final recommendation justified against alternatives.\n"
                if comparison
                else ""
            )
            polish_prompt = (
                "Rewrite the draft into a clean, user-facing executive report in markdown.\n"
                "Return ONLY JSON with schema: {\"title\": str, \"body_markdown\": str}\n"
                "Rules:\n"
                "- Remove any token-count traces, numbering artifacts, or prompt/instruction residue.\n"
                "- Keep the report grounded to existing facts from the draft; do not invent new numbers.\n"
                "- Use a readable structure: executive summary, key findings, risks, and actions.\n"
                f"{comparison_rule}"
                "- Keep tone professional and concise.\n"
                f"- Audience: {audience}\n"
                f"- Tone: {tone}\n"
                f"- Target length: ~{max(500, target_words)} words.\n\n"
                f"Requested task:\n{prompt}\n\n"
                f"Current title:\n{raw_title or '(none)'}\n\n"
                f"Draft text:\n{cleaned_fallback_text}\n"
            )
            system_prompt = "You are a strict report editor. Output valid JSON only."
        polished = self.llm_client.generate_completion(
            polish_prompt,
            system_prompt=system_prompt,
        )
        parsed = self._parse_json_payload(polished or "")
        if not parsed:
            return {"title": fallback_title, "text": cleaned_fallback_text}

        new_title = str(parsed.get("title", "") or "").strip() or fallback_title
        new_text = str(parsed.get("body_markdown", "") or "").strip() or cleaned_fallback_text
        if self._contains_numbered_word_artifact(new_text):
            new_text = self._strip_numbered_word_artifact(new_text)
        if self._looks_like_raw_prompt_title(new_title, prompt):
            new_title = fallback_title
        return {"title": new_title[:90], "text": new_text.strip()}

    def _model_decide_document_target_words(self, prompt: str, mode: DocumentMode) -> Optional[int]:
        if not self.llm_client:
            return None

        rag_hint = "no_rag"
        if self.rag_service:
            try:
                status = self.get_rag_status()
                rag_hint = f"rag_collection_size={int(status.get('collection_size', 0) or 0)}"
            except Exception:
                rag_hint = "rag_unknown"

        planning_prompt = (
            "You are planning document length for a writing agent.\n"
            "Choose a practical output size that avoids filler and matches requested depth.\n"
            "Return ONLY JSON in this schema: "
            "{\"target_pages\": int, \"target_words\": int, \"reason\": str}\n"
            "Constraints:\n"
            "- target_pages must be between 1 and 7\n"
            "- target_words must be between 500 and 2800\n"
            "- For concise tasks choose lower values\n"
            "- For detailed strategic tasks choose moderate values\n"
            "Inputs:\n"
            f"- mode: {mode.value}\n"
            f"- context: {rag_hint}\n"
            f"- user_prompt: {prompt.strip()}\n"
        )
        raw = self.llm_client.generate_completion(
            planning_prompt,
            system_prompt="You are a strict planner. Output valid JSON only.",
        )

        parsed = self._parse_json_payload(raw or "")
        if parsed:
            target_words = parsed.get("target_words")
            target_pages = parsed.get("target_pages")

            try:
                words_int = int(target_words) if target_words is not None else 0
            except Exception:
                words_int = 0

            try:
                pages_int = int(target_pages) if target_pages is not None else 0
            except Exception:
                pages_int = 0

            if words_int <= 0 and pages_int > 0:
                words_int = pages_int * 500

            if words_int > 0:
                return max(500, min(2800, words_int))

        # Fallback parse from free text if model didn't follow JSON format.
        fallback_text = (raw or "").lower()
        words_match = re.search(r"(\d{3,4})\s*words?", fallback_text)
        if words_match:
            return max(500, min(2800, int(words_match.group(1))))
        pages_match = re.search(r"(\d{1,2})\s*pages?", fallback_text)
        if pages_match:
            return max(500, min(2800, int(pages_match.group(1)) * 500))
        return None

    @staticmethod
    def _build_document_runtime_tuning(
        *,
        target_words: int,
        requested_auto: bool,
        quality_mode: str,
        cfg_max_chunk_words: int,
        cfg_min_chunk_words: int,
        cfg_max_retries: int,
        cfg_consistency_check_interval: int,
    ) -> Dict[str, int | bool]:
        quality = (quality_mode or "Fast").strip().lower()
        is_thorough = quality == "thorough"
        fast_mode = (requested_auto or target_words <= 1200) and not is_thorough

        if is_thorough:
            consistency_interval = 4 if cfg_consistency_check_interval == 12 else max(1, min(cfg_consistency_check_interval, 8))
            return {
                "fast_mode": False,
                "max_chunk_words": max(cfg_max_chunk_words, 500),
                "min_chunk_words": max(cfg_min_chunk_words, 220),
                "max_retries": max(cfg_max_retries, 4),
                "consistency_check_interval": consistency_interval,
                "hard_max_words": 0,
            }

        if not fast_mode:
            return {
                "fast_mode": False,
                "max_chunk_words": cfg_max_chunk_words,
                "min_chunk_words": cfg_min_chunk_words,
                "max_retries": cfg_max_retries,
                "consistency_check_interval": cfg_consistency_check_interval,
                "hard_max_words": 0,
            }

        min_chunk_words = max(100, min(200, target_words // 4))
        max_chunk_words = max(min_chunk_words + 80, min(420, target_words // 2))
        max_retries = 1 if cfg_max_retries == 3 else max(1, cfg_max_retries)
        consistency_interval = 0 if cfg_consistency_check_interval == 12 else max(0, cfg_consistency_check_interval)

        return {
            "fast_mode": True,
            "max_chunk_words": max(160, min(cfg_max_chunk_words, max_chunk_words)),
            "min_chunk_words": min(cfg_min_chunk_words, min_chunk_words),
            "max_retries": max_retries,
            "consistency_check_interval": consistency_interval,
            "hard_max_words": 0,
        }

    def _build_document_charts(
        self,
        *,
        prompt: str,
        title: str,
        mode: DocumentMode,
        max_charts: int,
        include_flowchart: bool,
    ) -> List[Dict[str, Any]]:
        if max_charts <= 0:
            return []
        if mode == DocumentMode.PURE:
            return []
        if not self.document_chart_generator:
            return []
        if not self.rag_service:
            return []

        try:
            rag_top_k = getattr(self.rag_service, "top_k", 5)
            rag_min_score = getattr(self.rag_service, "min_score", 0.25)
            rag_max_chars = getattr(self.rag_service, "max_context_chars", 3000)
            hits = self.rag_service.search(
                prompt,
                top_k=max(rag_top_k, 8),
                min_score=max(0.0, min(rag_min_score, 0.2)),
                source_filter=None,
            )
            context = self.rag_service.format_hits(
                hits,
                max_context_chars=max(rag_max_chars, 5000),
            )
            if not context.strip():
                return []

            available_sources: List[str] = []
            for hit in hits:
                src = str(hit.metadata.get("source", "")).strip()
                if src and src not in available_sources:
                    available_sources.append(src)

            charts = self.document_chart_generator.generate(
                user_prompt=prompt,
                document_title=title,
                retrieved_context=context,
                available_sources=available_sources,
                max_charts=max_charts,
                include_flowchart=include_flowchart,
            )
            if not charts and self.document_chart_generator.last_error:
                self.log(f"Chart generation skipped: {self.document_chart_generator.last_error}")
            return charts
        except Exception as e:
            self.log(f"Chart generation failed: {e}")
            return []

    def generate_document(
        self,
        prompt: str,
        *,
        target_words: int = 1400,
        audience: str = "General",
        tone: str = "professional",
        mode: str = "hybrid",
        quality_mode: str = "Fast",
        resume: bool = True,
    ) -> Dict[str, Any]:
        if not prompt or not prompt.strip():
            return {"error": "Prompt is empty."}
        if not self.llm_client:
            return {"error": "LLM client is not initialized."}

        if not self.document_orchestrator:
            self.document_orchestrator = DocumentOrchestrator(self.llm_client, self.rag_service)

        mode_map = {
            "hybrid": DocumentMode.HYBRID,
            "strict_grounded": DocumentMode.STRICT_GROUNDED,
            "pure": DocumentMode.PURE,
        }
        safe_mode = mode_map.get((mode or "hybrid").strip().lower(), DocumentMode.HYBRID)

        doc_cfg = self.config.document_engine if self.config and self.config.document_engine else None
        configured_target_words = doc_cfg.target_words if doc_cfg else 1400
        configured_quality_mode = (doc_cfg.quality_mode if doc_cfg else "Fast")
        effective_quality_mode = (quality_mode or configured_quality_mode or "Fast").strip()
        requested_auto = target_words <= 0 and configured_target_words <= 0
        resolved_target_words = target_words if target_words > 0 else configured_target_words
        if resolved_target_words <= 0:
            model_selected_words = self._model_decide_document_target_words(prompt.strip(), safe_mode)
            if model_selected_words and model_selected_words > 0:
                resolved_target_words = model_selected_words
                self.log(f"Document length set to {resolved_target_words} words (AI-decided).")
            else:
                resolved_target_words = self._auto_document_target_words(prompt.strip(), safe_mode)
                self.log(f"Document length set to {resolved_target_words} words (auto fallback).")

        cfg_max_chunk_words = doc_cfg.max_chunk_words if doc_cfg else 500
        cfg_min_chunk_words = doc_cfg.min_chunk_words if doc_cfg else 220
        cfg_max_retries = doc_cfg.max_retries if doc_cfg else 3
        cfg_consistency_interval = doc_cfg.consistency_check_interval if doc_cfg else 12
        chart_enabled = bool(doc_cfg.chart_enabled) if doc_cfg else False
        max_charts = int(doc_cfg.max_charts) if doc_cfg else 3
        include_flowchart = bool(doc_cfg.include_flowchart) if doc_cfg else True
        tuning = self._build_document_runtime_tuning(
            target_words=resolved_target_words,
            requested_auto=requested_auto,
            quality_mode=effective_quality_mode,
            cfg_max_chunk_words=cfg_max_chunk_words,
            cfg_min_chunk_words=cfg_min_chunk_words,
            cfg_max_retries=cfg_max_retries,
            cfg_consistency_check_interval=cfg_consistency_interval,
        )
        self.log(f"Document quality mode: {effective_quality_mode}.")

        options = DocumentGenerationOptions(
            prompt=prompt.strip(),
            target_words=resolved_target_words,
            audience=audience or (doc_cfg.audience if doc_cfg else "General"),
            tone=tone or (doc_cfg.tone if doc_cfg else "professional"),
            mode=safe_mode,
            max_chunk_words=int(tuning["max_chunk_words"]),
            min_chunk_words=int(tuning["min_chunk_words"]),
            max_retries=int(tuning["max_retries"]),
            consistency_check_interval=int(tuning["consistency_check_interval"]),
            fast_mode=bool(tuning["fast_mode"]),
            hard_max_words=int(tuning["hard_max_words"]),
            resume=resume,
        )

        self.stop_requested = False
        basis = f"{prompt.strip()}|{options.mode.value}|{options.target_words}|{options.audience}|{options.tone}|{effective_quality_mode.lower()}"
        digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]
        job_id = f"doc_{digest}"

        try:
            result = self.document_orchestrator.run(
                job_id=job_id,
                options=options,
                on_log=self.log,
                on_progress=self.on_progress,
                should_stop=lambda: self.stop_requested,
            )
            result = self._enforce_document_length_bounds(
                result,
                target_words=resolved_target_words,
                audience=options.audience,
                tone=options.tone,
            )
            polished = self._polish_document_for_publish(
                title=str(result.get("title", "")),
                prompt=prompt.strip(),
                text=str(result.get("text", "")),
                audience=options.audience,
                tone=options.tone,
                target_words=resolved_target_words,
                mode=safe_mode,
            )
            result["title"] = polished["title"]
            result["text"] = polished["text"]
            result["final_word_count"] = self._word_count(result.get("text", ""))
            result["charts"] = []
            if chart_enabled:
                charts = self._build_document_charts(
                    prompt=prompt.strip(),
                    title=result.get("title", prompt.strip()),
                    mode=safe_mode,
                    max_charts=max(1, min(6, max_charts)),
                    include_flowchart=include_flowchart,
                )
                result["charts"] = charts
                if charts:
                    self.log(f"Generated {len(charts)} grounded chart(s) for document export.")
            self.document_result = result
            return result
        except Exception as e:
            self.log(f"Document generation failed: {e}")
            return {"error": str(e)}

    def stop_document_generation(self):
        self.stop_requested = True
        self.log("Stopping document generation...")

    def export_document_pdf(self, filepath: str):
        if not self.document_result:
            raise ValueError("No generated document available. Run document generation first.")
        self.document_pdf_exporter.export(
            title=self.document_result.get("title", "Generated Document"),
            outline=self.document_result.get("outline", {}),
            text=self.document_result.get("text", ""),
            output_path=filepath,
            chunks=self.document_result.get("chunks", []),
            charts=self.document_result.get("charts", []),
        )
        self.log(f"Document PDF exported to {filepath}")

    def export_document_docx(self, filepath: str):
        if not self.document_result:
            raise ValueError("No generated document available. Run document generation first.")
        self.document_docx_exporter.export(
            title=self.document_result.get("title", "Generated Document"),
            outline=self.document_result.get("outline", {}),
            text=self.document_result.get("text", ""),
            output_path=filepath,
            chunks=self.document_result.get("chunks", []),
            charts=self.document_result.get("charts", []),
        )
        self.log(f"Document DOCX exported to {filepath}")

    def ask_files(self, prompt: str) -> Dict[str, Any]:
        if not prompt.strip():
            return {"error": "Prompt is empty."}
        if not self.llm_client:
            return {"error": "LLM client is not initialized."}
        if not self.rag_service:
            return {"error": "RAG service is not configured."}

        if hasattr(self.rag_service, "answer_query"):
            try:
                top_k = self.config.rag.top_k if self.config.rag else 5
                min_score = self.config.rag.min_score if self.config.rag else 0.25
                source_filter = self.config.rag.source_filter if self.config.rag else None
                synthesized = self.rag_service.answer_query(
                    prompt,
                    top_k=top_k,
                    min_score=min_score,
                    source_filter=source_filter,
                )
                if synthesized and synthesized.get("answer"):
                    return synthesized
            except Exception as e:
                self.log(f"LlamaIndex answer synthesis unavailable, falling back to standard Q&A: {e}")

        context = self.llm_client.retrieve_context(prompt)
        if not context.strip():
            status = self.get_rag_status()
            self.log(
                "RAG query returned no context "
                f"(collection_size={status.get('collection_size', 0)}, "
                f"source_filter={status.get('source_filter')}, "
                f"min_score={status.get('min_score')})"
            )
            return {
                "answer": "I could not find relevant context in the imported files. Try a more specific query.",
                "context": "",
                "citations": [],
            }

        top_k = self.config.rag.top_k if self.config.rag else 5
        min_score = self.config.rag.min_score if self.config.rag else 0.25
        source_filter = self.config.rag.source_filter if self.config.rag else None

        hits = self.rag_service.search(
            prompt,
            top_k=top_k,
            min_score=min_score,
            source_filter=source_filter,
        )
        if not hits and source_filter:
            hits = self.rag_service.search(prompt, top_k=top_k, min_score=min_score, source_filter=None)
        if not hits and min_score > 0.0:
            hits = self.rag_service.search(prompt, top_k=top_k, min_score=0.0, source_filter=None)

        citations = []
        for hit in hits:
            citations.append(
                {
                    "source": hit.metadata.get("source", "unknown"),
                    "page": hit.metadata.get("page", "?"),
                    "score": hit.score,
                }
            )

        prompt_text = (
            "Use only the provided context from imported files. "
            "If context is insufficient, clearly say so.\n\n"
            f"Context:\n{context}\n\n"
            f"User Task:\n{prompt}\n\n"
            "Return a concise answer and keep it grounded in the context."
        )
        answer = self.llm_client.generate_completion(prompt_text, system_prompt="You are a file-grounded assistant.")

        return {
            "answer": answer or "No answer returned by model.",
            "context": context,
            "citations": citations,
        }
        
    def log(self, message: str):
        logger.info(message)
        if self.on_log:
            self.on_log(message)

    def generate_row(self, initial_context: Optional[Dict[str, Any]] = None) -> Optional[RowData]:
        from core.row_agent import create_row_generator_graph
        
        # Prepare initial data
        row_data: Dict[str, Any] = initial_context.copy() if initial_context else {}
        
        # Pre-fill deterministic logic (Faker, Auto-Increment) BEFORE Agent
        for col in self.execution_order:
            if col.name in row_data:
                continue
            
            if col.type == ColumnType.AUTO_INCREMENT:
                row_data[col.name] = len(self.generated_rows) + 1
                self.metrics_data["faker_cols"] += 1
            
            elif col.type == ColumnType.DETERMINISTIC and self.fake:
                provider = col.constraints.faker_provider or "name"
                try:
                    if hasattr(self.fake, provider):
                        func = getattr(self.fake, provider)
                        row_data[col.name] = str(func())
                    else:
                        row_data[col.name] = self.fake.name()
                    self.metrics_data["faker_cols"] += 1
                except:
                    pass

        # If all cols are filled (purely deterministic), skip agent
        if len(row_data) == len(self.columns):
            return RowData(data=row_data)

        # Filter out columns that are strictly "(Imported)"
        agent_cols = [
            col for col in self.execution_order 
            if col.prompt_instruction != "(Imported)" 
            and col.name not in row_data
        ]
        
        # Track LLM columns
        self.metrics_data["llm_cols"] += len(agent_cols)

        # Agentic Generation for LLM columns
        try:
            agent = create_row_generator_graph(self.llm_client)
            
            initial_state = {
                "row_data": row_data,
                "columns": agent_cols,
                "errors": [],
                "attempt_count": 0,
                "is_valid": False
            }
            
            final_state = agent.invoke(initial_state)
            
            result_data = final_state['row_data']
            is_valid = final_state['is_valid']
            
            if not is_valid and final_state['attempt_count'] >= 3:
                self.log(f"Row generation failed validation: {final_state.get('errors')}")
                return None

            enrichment_mode = bool(self.config.existing_data)

            # Post-Agent Guardrails: Uniqueness/Regex checks
            for col in self.columns:
                if col.prompt_instruction == "(Imported)":
                    continue

                val = result_data.get(col.name)
                if not val:
                    return None
                
                # Regex
                if not self.validator.validate_regex(val, col.constraints.regex_pattern):
                    self.log(f"Regex failed for {col.name}: {val}")
                    return None
                    
                # Uniqueness
                if not enrichment_mode and not col.constraints.allow_duplicates:
                    if not self.validator.is_unique(val, field_type=col.type.value):
                        self.log(f"Duplicate value for {col.name}: {val}")
                        return None
                        
            # Commit unique values
            for col in self.columns:
                if not enrichment_mode and not col.constraints.allow_duplicates:
                    val = result_data.get(col.name)
                    self.validator.commit(val, field_type=col.type.value)

            return RowData(data=result_data)

        except Exception as e:
            self.log(f"Agent error: {e}")
            return None

    def start_generation_thread(self):
        """Starts generation in a separate thread."""
        self.is_running = True
        t = threading.Thread(target=self._run_generation_loop)
        t.start()
        
    def stop_generation(self):
        print("DEBUG: Controller stop_generation called")
        self.stop_requested = True
        self.log("Stopping generation...")

    def _run_generation_loop(self):
        target_count = len(self.config.existing_data) if self.config.existing_data else self.config.num_rows
        self.log(f"Starting generation attempt for {target_count} rows...")
        
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 10
        
        self.metrics_data["start_time"] = time.time()
        self.metrics_data["end_time"] = None
        self.metrics_data["last_row_time"] = self.metrics_data["start_time"]
        self.metrics_data["total_attempts"] = 0
        self.metrics_data["failed_attempts"] = 0
        
        # Mode A: Enrichment (Existing Data)
        if self.config.existing_data:
            for i, context_row in enumerate(self.config.existing_data):
                if self.stop_requested:
                    break
                
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    self.log(f"CRITICAL: Aborting generation after {MAX_CONSECUTIVE_FAILURES} consecutive failures. Please check your constraints.")
                    break

                self.metrics_data["total_attempts"] += 1
                row = self.generate_row(initial_context=context_row)
                if self.stop_requested:
                    break
                
                if row:
                    self.generated_rows.append(row)
                    self.metrics_data["last_row_time"] = time.time()
                    self.log(f"Generated row {len(self.generated_rows)}/{target_count}")
                    consecutive_failures = 0
                else:
                    self.metrics_data["failed_attempts"] += 1
                    self.log(f"Row {i+1} enrichment failed. Skipping.")
                    consecutive_failures += 1
                
                if self.on_progress:
                    self.on_progress(len(self.generated_rows), target_count)
                    
        # Mode B: Fresh Generation
        else:
            while len(self.generated_rows) < target_count:
                if self.stop_requested:
                    break
                
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    self.log(f"CRITICAL: Aborting generation after {MAX_CONSECUTIVE_FAILURES} consecutive failures. Please check your constraints.")
                    break
                    
                self.metrics_data["total_attempts"] += 1
                row = self.generate_row()
                if self.stop_requested:
                    break
                
                if row:
                    self.generated_rows.append(row)
                    self.metrics_data["last_row_time"] = time.time()
                    self.log(f"Generated row {len(self.generated_rows)}/{target_count}")
                    consecutive_failures = 0
                    if self.on_progress:
                        self.on_progress(len(self.generated_rows), target_count)
                else:
                    self.metrics_data["failed_attempts"] += 1
                    self.log("Row generation failed. Retrying...")
                    consecutive_failures += 1
        
        self.metrics_data["end_time"] = time.time()
        self.is_running = False
        self.log("Generation finished.")
        if self.on_finished:
            self.on_finished()

    # --- Export Delegation ---
    # These thin wrappers maintain backward compatibility with FletApp
    
    def export_csv(self, filepath: str):
        export_csv(filepath, self.generated_rows, self.columns, log_fn=self.log)

    def export_json(self, filepath: str):
        export_json(filepath, self.generated_rows, log_fn=self.log)

    def export_sql(self, filepath: str, table_name: str = "synthetic_data"):
        export_sql(filepath, self.generated_rows, table_name=table_name, log_fn=self.log)

    def export_power_bi_run(
        self,
        destination_dir: str,
        *,
        dataset_name: str,
        privacy_export_mode: str = "Restored imported values",
        source_mode: str = "fresh_generation",
    ):
        return export_power_bi_run(
            destination_dir,
            self.generated_rows,
            self.columns,
            dataset_name=dataset_name,
            privacy_export_mode=privacy_export_mode,
            source_mode=source_mode,
            provider=self.config.provider,
            model=self.config.model_id,
            log_fn=self.log,
        )

    def analyze_quality(self) -> Dict[str, Any]:
        """Runs quality analysis on the generated data."""
        if not self.generated_rows:
            return {}
        try:
            df = pd.DataFrame([r.data for r in self.generated_rows])
            return self.analyzer.analyze(df)
        except Exception as e:
            self.log(f"Analysis failed: {e}")
            return {}

    def export_pdf_report(self, filepath: str):
        try:
            metrics = self.analyze_quality()
            self.pdf_exporter.generate_quality_report(metrics, filepath)
            self.log(f"PDF Quality Report exported to {filepath}")
        except Exception as e:
            self.log(f"PDF Export failed: {e}")

    def export_narrative_pdf(self, filepath: str):
        if not self.generated_rows:
            return
            
        try:
            df = pd.DataFrame([r.data for r in self.generated_rows])
            cols = df.columns.tolist()
            if not cols:
                return
                
            title_col = cols[0]
            body_cols = cols[1:]
            
            self.pdf_exporter.generate_narrative_export(df, title_col, body_cols, filepath)
            self.log(f"Narrative PDF exported to {filepath}")
        except Exception as e:
            self.log(f"Narrative Export failed: {e}")

    # --- JSON Template Generation ---

    def generate_json_batch(
        self,
        template_path: str,
        target_path: str,
        num_items: int = 10,
        *,
        clear_existing: bool = True,
    ) -> Dict[str, Any]:
        """Generate a batch of JSON objects and inject them into a template.

        Loads a JSON template, infers the schema from the target array,
        runs the json_agent LangGraph loop for each item, validates
        uniqueness via path-based flattening, and injects results.

        Args:
            template_path: Path to the JSON template file.
            target_path: Dot-notation path to the target array.
            num_items: Number of items to generate.
            clear_existing: If True, clear the target array before generating.

        Returns:
            The populated template dict, or a dict with 'error' key on failure.
        """
        from core.json_parser import (
            load_template,
            resolve_target_array,
            infer_item_schema,
            inject_item,
            clear_target_array,
        )
        from core.json_agent import create_json_generator_graph

        if not self.llm_client:
            return {"error": "LLM client is not initialized."}

        # 1. Load template
        try:
            template = load_template(template_path)
        except (FileNotFoundError, ValueError) as e:
            self.log(f"Template load failed: {e}")
            return {"error": str(e)}

        # 2. Resolve target array
        try:
            target_array = resolve_target_array(template, target_path)
        except ValueError as e:
            self.log(f"Target path resolution failed: {e}")
            return {"error": str(e)}

        # 3. Infer schema
        schema_model = infer_item_schema(target_array)
        schema_desc = "(no schema inferred — generate a diverse JSON object)"
        if schema_model is not None:
            try:
                schema_desc = json.dumps(schema_model.model_json_schema(), indent=2)
            except Exception:
                schema_desc = str(schema_model.model_fields)

        # 4. Clear existing items if requested
        if clear_existing:
            clear_target_array(template, target_path)
            self.log(f"Cleared existing items from '{target_path}'.")

        # 5. Build context from template (excluding target array for brevity)
        template_context = json.dumps(template, indent=2)[:2000]

        # 6. Reset validator for this batch
        if self.validator:
            self.validator.clear()

        # 7. Generation loop
        generated_count = 0
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 10

        self.log(f"Starting JSON template generation: {num_items} items into '{target_path}'...")

        for i in range(num_items):
            if self.stop_requested:
                break

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                self.log(
                    f"CRITICAL: Aborting after {MAX_CONSECUTIVE_FAILURES} "
                    "consecutive failures."
                )
                break

            try:
                agent = create_json_generator_graph(
                    self.llm_client,
                    schema_model=schema_model,
                )

                initial_state = {
                    "item_data": {},
                    "schema_description": schema_desc,
                    "template_context": template_context,
                    "errors": [],
                    "attempt_count": 0,
                    "is_valid": False,
                }

                final_state = agent.invoke(initial_state)
                item_data = final_state.get("item_data", {})
                is_valid = final_state.get("is_valid", False)

                if not is_valid or not item_data:
                    self.log(f"Item {i+1} generation failed validation.")
                    consecutive_failures += 1
                    continue

                # Uniqueness check
                if self.validator and not self.validator.is_unique_json(item_data):
                    self.log(f"Item {i+1} rejected: duplicate detected.")
                    consecutive_failures += 1
                    continue

                # Inject into template
                inject_item(template, target_path, item_data)

                # Commit to validator
                if self.validator:
                    self.validator.commit_json(item_data)

                generated_count += 1
                consecutive_failures = 0
                self.log(f"Generated item {generated_count}/{num_items}")

                if self.on_progress:
                    self.on_progress(generated_count, num_items)

            except Exception as e:
                self.log(f"Item {i+1} error: {e}")
                consecutive_failures += 1

        self.log(f"JSON template generation complete: {generated_count}/{num_items} items.")
        self.json_template_result = template
        return template

    def export_json_template(self, filepath: str) -> None:
        """Export the generated JSON template to a file."""
        from core.json_parser import export_template

        if not self.json_template_result:
            raise ValueError("No generated JSON template available. Run generate_json_batch first.")
        export_template(self.json_template_result, filepath)
        self.log(f"JSON template exported to {filepath}")

    def generate_exhaustive_extraction(
        self,
        template_path: str,
        target_path: str,
        *,
        source_filter: Optional[str] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """Extract grounded QA pairs from every RAG chunk into a JSON template.

        Processes all ingested document chunks through the chunk extraction
        agent (Self-Foveate + CoVe + LLM-as-a-Judge critique). The loop
        length is driven by total chunk count, not a hardcoded row target.

        Args:
            template_path: Path to the JSON template file.
            target_path: Dot-notation path to the target array.
            source_filter: Optional source filter for RAG chunks.
            on_progress: Progress callback (chunks_processed, total_chunks).

        Returns:
            The populated template dict, or a dict with 'error' key on failure.
        """
        from core.json_parser import (
            load_template,
            resolve_target_array,
            inject_item,
            clear_target_array,
        )
        from core.chunk_agent import create_chunk_extraction_graph

        if not self.llm_client:
            return {"error": "LLM client is not initialized."}
        if not self.rag_service:
            return {"error": "RAG service is not initialized. Ingest documents first."}

        # 1. Load template
        try:
            template = load_template(template_path)
        except (FileNotFoundError, ValueError) as e:
            self.log(f"Template load failed: {e}")
            return {"error": str(e)}

        # 2. Resolve target array
        try:
            resolve_target_array(template, target_path)
        except ValueError as e:
            self.log(f"Target path resolution failed: {e}")
            return {"error": str(e)}

        # 3. Clear existing items
        clear_target_array(template, target_path)

        # 4. Get ALL chunks from RAG
        all_chunks = self.rag_service.get_all_chunks(source_filter=source_filter)
        total_chunks = len(all_chunks)

        if total_chunks == 0:
            self.log("No chunks found in RAG. Ingest documents first.")
            return {"error": "No chunks found. Ingest documents first."}

        self.log(f"Exhaustive mode: processing {total_chunks} chunks...")

        # 5. Reset validator
        if self.validator:
            self.validator.clear()

        # 6. Extraction loop — length driven by chunk count
        total_pairs_injected = 0
        chunks_processed = 0

        for i, chunk in enumerate(all_chunks):
            if self.stop_requested:
                break

            try:
                agent = create_chunk_extraction_graph(self.llm_client)

                initial_state = {
                    "chunk_text": chunk.text,
                    "chunk_metadata": chunk.metadata,
                    "extracted_pairs": [],
                    "verified_pairs": [],
                    "errors": [],
                }

                final_state = agent.invoke(initial_state)
                verified_pairs = final_state.get("verified_pairs", [])

                for pair in verified_pairs:
                    # Add source metadata to each pair
                    enriched_pair = {
                        **pair,
                        "source": chunk.metadata.get("source", "unknown"),
                        "chunk_id": chunk.chunk_id,
                    }

                    # Uniqueness check
                    if self.validator and not self.validator.is_unique_json(enriched_pair):
                        continue

                    inject_item(template, target_path, enriched_pair)

                    if self.validator:
                        self.validator.commit_json(enriched_pair)

                    total_pairs_injected += 1

                chunks_processed += 1
                self.log(
                    f"Chunk {chunks_processed}/{total_chunks} — "
                    f"extracted {len(verified_pairs)} pairs "
                    f"(total: {total_pairs_injected})"
                )

                if on_progress:
                    on_progress(chunks_processed, total_chunks)
                if self.on_progress:
                    self.on_progress(chunks_processed, total_chunks)

            except Exception as e:
                self.log(f"Chunk {i+1} error: {e}")
                chunks_processed += 1

        self.log(
            f"Exhaustive extraction complete: {chunks_processed}/{total_chunks} chunks, "
            f"{total_pairs_injected} pairs injected."
        )
        self.json_template_result = template
        return template


    def get_metrics(self) -> Dict[str, Any]:
        """Calculate real-time metrics — delegates to core.metrics."""
        return calculate_metrics(
            config=self.config,
            generated_rows=self.generated_rows,
            llm_client=self.llm_client,
            run_metrics=self.metrics_data,
        )
