from __future__ import annotations

import copy
import re
from typing import Any, Dict, Optional

from .models import AIProvider, DocumentEngineConfig, GeneratorConfig, RagBackend, RagConfig


DEFAULT_UI_VALUES: Dict[str, Any] = {
    "model_id": "local-model",
    "provider": AIProvider.LM_STUDIO.value,
    "api_key": "",
    "azure_endpoint": "",
    "azure_deployment": "",
    "input_price_per_1m": 0.15,
    "output_price_per_1m": 0.60,
    "num_rows": 10,
    "similarity_threshold": 0.85,
    "max_retries": 50,
    "rag_backend": RagBackend.LLAMA_INDEX.value,
    "quick_qa_mode": "Broader Analysis",
    "collection_name": "synthesizer_default",
    "top_k": 5,
    "min_score": 0.25,
    "max_context_chars": 3000,
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "source_filter": "",
    "qdrant_url": ":memory:",
    "qdrant_api_key": "",
    "ocr_mode": "off",
    "ocr_dpi": 150,
    "ocr_max_pages": 20,
    "ocr_max_regions_per_page": 8,
    "ocr_region_padding_px": 18,
    "ocr_gap_multiplier": 2.5,
    "ocr_min_extracted_chars": 60,
    "ocr_timeout_ms_per_page": 4000,
    "parser_mode": "auto",
    "hybrid_search_enabled": True,
    "rerank_enabled": True,
    "summary_first_enabled": True,
    "summary_top_k": 3,
    "dense_top_k": 12,
    "lexical_top_k": 12,
    "parent_context_enabled": True,
    "parent_context_max_chars": 1200,
    "graph_enabled": True,
    "graph_hops": 1,
    "graph_source_boost": 0.08,
    "late_interaction_enabled": True,
    "late_interaction_weight": 0.2,
    "doc_mode": "Balanced",
    "doc_pages": "Let AI decide",
    "doc_quality": "Fast",
    "doc_audience": "General",
    "doc_tone": "professional",
    "doc_chart_enabled": False,
    "doc_flow_enabled": True,
    "doc_max_charts": 3,
}

PAGE_OPTIONS = [
    "Let AI decide",
    "1 page",
    "2 pages",
    "3 pages",
    "5 pages",
    "8 pages",
    "10 pages",
    "15 pages",
    "20 pages",
]


def default_ui_values() -> Dict[str, Any]:
    return copy.deepcopy(DEFAULT_UI_VALUES)


def resolve_document_mode(choice: str) -> str:
    selected = (choice or "Balanced").strip().lower()
    mode_map = {
        "balanced": "hybrid",
        "file-based": "strict_grounded",
        "creative": "pure",
        "hybrid": "hybrid",
        "strict_grounded": "strict_grounded",
        "factual by doc": "strict_grounded",
        "pure": "pure",
    }
    return mode_map.get(selected, "hybrid")


def document_mode_label(mode: str) -> str:
    selected = (mode or "Balanced").strip().lower()
    mode_map = {
        "balanced": "Balanced",
        "file-based": "File-based",
        "creative": "Creative",
        "hybrid": "Balanced",
        "strict_grounded": "File-based",
        "factual by doc": "File-based",
        "pure": "Creative",
    }
    return mode_map.get(selected, "Balanced")


def resolve_document_target_words(selection: str) -> int:
    selected = (selection or "Let AI decide").strip().lower()
    if selected == "let ai decide":
        return 0

    pages = 0
    digit_match = re.search(r"(\d+)", selected)
    if digit_match:
        pages = int(digit_match.group(1))

    if pages <= 0:
        return 0

    words_per_page = 500
    return max(350, pages * words_per_page)


def pages_label_from_target_words(target_words: int) -> str:
    if not target_words or target_words <= 0:
        return "Let AI decide"

    estimated_pages = max(1, round(target_words / 500))
    candidate = f"{estimated_pages} page" if estimated_pages == 1 else f"{estimated_pages} pages"
    return candidate


def build_generator_config(
    values: Dict[str, Any],
    *,
    include_existing_data: bool = True,
    existing_data: Optional[list[dict[str, Any]]] = None,
) -> GeneratorConfig:
    merged = default_ui_values()
    merged.update(values or {})

    rag_config = RagConfig(
        enabled=True,
        backend=RagBackend(str(merged.get("rag_backend", RagBackend.LLAMA_INDEX.value))),
        collection_name=str(merged.get("collection_name", "synthesizer_default")).strip() or "synthesizer_default",
        top_k=int(merged.get("top_k", 5) or 5),
        min_score=float(merged.get("min_score", 0.25) or 0.25),
        max_context_chars=int(merged.get("max_context_chars", 3000) or 3000),
        embedding_model=str(merged.get("embedding_model", "BAAI/bge-small-en-v1.5")).strip() or "BAAI/bge-small-en-v1.5",
        source_filter=str(merged.get("source_filter", "") or "").strip() or None,
        qdrant_url=str(merged.get("qdrant_url", ":memory:")).strip() or ":memory:",
        qdrant_api_key=str(merged.get("qdrant_api_key", "") or "").strip() or None,
        ocr_mode=str(merged.get("ocr_mode", "off")).strip().lower(),
        ocr_dpi=int(merged.get("ocr_dpi", 150) or 150),
        ocr_max_pages=int(merged.get("ocr_max_pages", 20) or 20),
        ocr_max_regions_per_page=int(merged.get("ocr_max_regions_per_page", 8) or 8),
        ocr_region_padding_px=int(merged.get("ocr_region_padding_px", 18) or 18),
        ocr_gap_multiplier=float(merged.get("ocr_gap_multiplier", 2.5) or 2.5),
        ocr_min_extracted_chars=int(merged.get("ocr_min_extracted_chars", 60) or 60),
        ocr_timeout_ms_per_page=int(merged.get("ocr_timeout_ms_per_page", 4000) or 4000),
        parser_mode=str(merged.get("parser_mode", "auto")).strip() or "auto",
        hybrid_search_enabled=bool(merged.get("hybrid_search_enabled", True)),
        rerank_enabled=bool(merged.get("rerank_enabled", True)),
        summary_first_enabled=bool(merged.get("summary_first_enabled", True)),
        summary_top_k=int(merged.get("summary_top_k", 3) or 3),
        dense_top_k=int(merged.get("dense_top_k", 12) or 12),
        lexical_top_k=int(merged.get("lexical_top_k", 12) or 12),
        parent_context_enabled=bool(merged.get("parent_context_enabled", True)),
        parent_context_max_chars=int(merged.get("parent_context_max_chars", 1200) or 1200),
        graph_enabled=bool(merged.get("graph_enabled", True)),
        graph_hops=int(merged.get("graph_hops", 1) or 1),
        graph_source_boost=float(merged.get("graph_source_boost", 0.08) or 0.08),
        late_interaction_enabled=bool(merged.get("late_interaction_enabled", True)),
        late_interaction_weight=float(merged.get("late_interaction_weight", 0.2) or 0.2),
    )

    document_config = DocumentEngineConfig(
        mode=resolve_document_mode(str(merged.get("doc_mode", "Balanced"))),
        target_words=resolve_document_target_words(str(merged.get("doc_pages", "Let AI decide"))),
        quality_mode=str(merged.get("doc_quality", "Fast")) or "Fast",
        audience=str(merged.get("doc_audience", "General")) or "General",
        tone=str(merged.get("doc_tone", "professional")) or "professional",
        chart_enabled=bool(merged.get("doc_chart_enabled", False)),
        include_flowchart=bool(merged.get("doc_flow_enabled", True)),
        max_charts=int(merged.get("doc_max_charts", 3) or 3),
    )

    provider = AIProvider(str(merged.get("provider", AIProvider.LM_STUDIO.value)))
    api_key = str(merged.get("api_key", "") or "").strip() or None

    return GeneratorConfig(
        model_id=str(merged.get("model_id", "local-model")).strip() or "local-model",
        provider=provider,
        api_key=api_key if provider != AIProvider.LM_STUDIO else None,
        azure_endpoint=str(merged.get("azure_endpoint", "") or "").strip() or None,
        azure_deployment=str(merged.get("azure_deployment", "") or "").strip() or None,
        input_price_per_1m=float(merged.get("input_price_per_1m", 0.15) or 0.15),
        output_price_per_1m=float(merged.get("output_price_per_1m", 0.60) or 0.60),
        num_rows=int(merged.get("num_rows", 10) or 10),
        similarity_threshold=float(merged.get("similarity_threshold", 0.85) or 0.85),
        max_retries=int(merged.get("max_retries", 50) or 50),
        existing_data=existing_data if include_existing_data else None,
        rag=rag_config,
        document_engine=document_config,
    )


def serialize_ui_config(values: Dict[str, Any], *, columns: Optional[list[dict[str, Any]]] = None) -> Dict[str, Any]:
    merged = default_ui_values()
    merged.update(values or {})
    runtime = build_generator_config(merged, include_existing_data=False)

    payload = {
        "model_id": runtime.model_id,
        "provider": runtime.provider.value,
        "api_key": merged.get("api_key", ""),
        "azure_endpoint": merged.get("azure_endpoint", ""),
        "azure_deployment": merged.get("azure_deployment", ""),
        "input_price_per_1m": runtime.input_price_per_1m,
        "output_price_per_1m": runtime.output_price_per_1m,
        "num_rows": runtime.num_rows,
        "similarity_threshold": runtime.similarity_threshold,
        "max_retries": runtime.max_retries,
        "rag": runtime.rag.model_dump() if runtime.rag else {},
        "document_engine": runtime.document_engine.model_dump() if runtime.document_engine else {},
        "columns": columns or [],
    }

    if payload["rag"]:
        payload["rag"]["backend"] = merged.get("rag_backend", payload["rag"].get("backend", RagBackend.LLAMA_INDEX.value))
        payload["rag"]["quick_qa_mode"] = merged.get("quick_qa_mode", "Broader Analysis")
        payload["rag"]["source_filter"] = merged.get("source_filter", "")
        payload["rag"]["qdrant_api_key"] = merged.get("qdrant_api_key", "")

    if payload["document_engine"]:
        payload["document_engine"]["mode"] = merged.get("doc_mode", "Balanced")

    return payload


def normalize_loaded_config(data: Dict[str, Any]) -> Dict[str, Any]:
    values = default_ui_values()
    if not isinstance(data, dict):
        return values

    for key in (
        "model_id",
        "provider",
        "api_key",
        "azure_endpoint",
        "azure_deployment",
        "input_price_per_1m",
        "output_price_per_1m",
        "num_rows",
        "similarity_threshold",
        "max_retries",
    ):
        if key in data:
            values[key] = data[key]

    rag = data.get("rag") or {}
    if isinstance(rag, dict):
        values.update(
            {
                "rag_backend": rag.get("backend", values["rag_backend"]),
                "quick_qa_mode": rag.get("quick_qa_mode", values["quick_qa_mode"]),
                "collection_name": rag.get("collection_name", values["collection_name"]),
                "top_k": rag.get("top_k", values["top_k"]),
                "min_score": rag.get("min_score", values["min_score"]),
                "max_context_chars": rag.get("max_context_chars", values["max_context_chars"]),
                "embedding_model": rag.get("embedding_model", values["embedding_model"]),
                "source_filter": rag.get("source_filter", values["source_filter"]),
                "qdrant_url": rag.get("qdrant_url", values["qdrant_url"]),
                "qdrant_api_key": rag.get("qdrant_api_key", values["qdrant_api_key"]),
                "ocr_mode": rag.get("ocr_mode", values["ocr_mode"]),
                "ocr_dpi": rag.get("ocr_dpi", values["ocr_dpi"]),
                "ocr_max_pages": rag.get("ocr_max_pages", values["ocr_max_pages"]),
                "ocr_max_regions_per_page": rag.get("ocr_max_regions_per_page", values["ocr_max_regions_per_page"]),
                "ocr_region_padding_px": rag.get("ocr_region_padding_px", values["ocr_region_padding_px"]),
                "ocr_gap_multiplier": rag.get("ocr_gap_multiplier", values["ocr_gap_multiplier"]),
                "ocr_min_extracted_chars": rag.get("ocr_min_extracted_chars", values["ocr_min_extracted_chars"]),
                "ocr_timeout_ms_per_page": rag.get("ocr_timeout_ms_per_page", values["ocr_timeout_ms_per_page"]),
                "parser_mode": rag.get("parser_mode", values["parser_mode"]),
                "hybrid_search_enabled": rag.get("hybrid_search_enabled", values["hybrid_search_enabled"]),
                "rerank_enabled": rag.get("rerank_enabled", values["rerank_enabled"]),
                "summary_first_enabled": rag.get("summary_first_enabled", values["summary_first_enabled"]),
                "summary_top_k": rag.get("summary_top_k", values["summary_top_k"]),
                "dense_top_k": rag.get("dense_top_k", values["dense_top_k"]),
                "lexical_top_k": rag.get("lexical_top_k", values["lexical_top_k"]),
                "parent_context_enabled": rag.get("parent_context_enabled", values["parent_context_enabled"]),
                "parent_context_max_chars": rag.get("parent_context_max_chars", values["parent_context_max_chars"]),
                "graph_enabled": rag.get("graph_enabled", values["graph_enabled"]),
                "graph_hops": rag.get("graph_hops", values["graph_hops"]),
                "graph_source_boost": rag.get("graph_source_boost", values["graph_source_boost"]),
                "late_interaction_enabled": rag.get("late_interaction_enabled", values["late_interaction_enabled"]),
                "late_interaction_weight": rag.get("late_interaction_weight", values["late_interaction_weight"]),
            }
        )

    doc = data.get("document_engine") or {}
    if isinstance(doc, dict):
        values.update(
            {
                "doc_mode": document_mode_label(str(doc.get("mode", values["doc_mode"]))),
                "doc_pages": pages_label_from_target_words(int(doc.get("target_words", 0) or 0)),
                "doc_quality": doc.get("quality_mode", values["doc_quality"]),
                "doc_audience": doc.get("audience", values["doc_audience"]),
                "doc_tone": doc.get("tone", values["doc_tone"]),
                "doc_chart_enabled": bool(doc.get("chart_enabled", values["doc_chart_enabled"])),
                "doc_flow_enabled": bool(doc.get("include_flowchart", values["doc_flow_enabled"])),
                "doc_max_charts": int(doc.get("max_charts", values["doc_max_charts"]) or values["doc_max_charts"]),
            }
        )

    return values
