from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from core.app_config import build_generator_config, default_ui_values, normalize_loaded_config, serialize_ui_config
from core.controller import GeneratorController
from core.llm_client import LLMClient
from web_ui.adapters import (
    field_records_to_columns,
    field_records_to_grid_dataframe,
    field_rows_markup,
    imported_columns_markup,
    normalize_field_record,
)
from web_ui.actions.data_actions import _bool_value, _grid_rows, _records_from_grid, _selected_grid_row_index
from web_ui.actions.files_actions import chat_markdown, files_status_text, source_selector_update, sources_dataframe
from web_ui.state import (
    WebSessionState,
    activity_markdown,
    append_activity,
    get_runtime_controller,
)


EXPORT_DIR = Path(".web_ui_exports")
DEFAULT_DATA_STATUS = "Start from scratch or import a CSV/JSON file to use existing columns as a base."
DEFAULT_GENERATION_PROGRESS = "Generation progress will appear here once you start a run."
DEFAULT_QUALITY_REPORT = "Quality review will appear here after generation."
DEFAULT_FILES_PROGRESS = "Files progress will appear here after you run a task."
DEFAULT_RAG_ADMIN_STATUS = "Search admin messages will appear here."

HELP_MARKDOWN = """
### Help & Docs

**Generate Sample Data**
- Describe the dataset you want.
- Use **Generate Fields** to draft a starter schema.
- Review the rows, then run **Generate Data**.

**Work With Files**
- Upload one or more files or add a web page.
- Choose whether you want a document, grounded answers, or structured JSON.
- Use saved prompts or document bundles to speed up common tasks.

**Technical Settings**
- Open them only when you need tuning, OCR, retrieval controls, or cost estimates.
- `Search Status` shows the current retrieval/index configuration.
- `Clear Search Index` clears the backing collection and resets the Files workspace sources.
"""


def _ui_values_from_args(
    model_id: str,
    provider: str,
    api_key: str,
    azure_endpoint: str,
    azure_deployment: str,
    input_price_per_1m: float,
    output_price_per_1m: float,
    num_rows: int,
    similarity_threshold: float,
    max_retries: int,
    rag_backend: str,
    collection_name: str,
    top_k: int,
    min_score: float,
    max_context_chars: int,
    embedding_model: str,
    source_filter: str,
    qdrant_url: str,
    qdrant_api_key: str,
    ocr_mode: str,
    ocr_dpi: int,
    ocr_max_pages: int,
    ocr_max_regions_per_page: int,
    ocr_region_padding_px: int,
    ocr_gap_multiplier: float,
    ocr_min_extracted_chars: int,
    ocr_timeout_ms_per_page: int,
    parser_mode: str,
    hybrid_search_enabled: bool,
    rerank_enabled: bool,
    summary_first_enabled: bool,
    summary_top_k: int,
    dense_top_k: int,
    lexical_top_k: int,
    parent_context_enabled: bool,
    parent_context_max_chars: int,
    graph_enabled: bool,
    graph_hops: int,
    graph_source_boost: float,
    late_interaction_enabled: bool,
    late_interaction_weight: float,
    quick_qa_mode: str,
    doc_mode: str,
    doc_pages: str,
    doc_quality: str,
    doc_audience: str,
    doc_tone: str,
    doc_chart_enabled: bool,
    doc_flow_enabled: bool,
    doc_max_charts: int,
    ) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "provider": provider,
        "api_key": api_key,
        "azure_endpoint": azure_endpoint,
        "azure_deployment": azure_deployment,
        "input_price_per_1m": input_price_per_1m,
        "output_price_per_1m": output_price_per_1m,
        "num_rows": num_rows,
        "similarity_threshold": similarity_threshold,
        "max_retries": max_retries,
        "rag_backend": rag_backend,
        "collection_name": collection_name,
        "top_k": top_k,
        "min_score": min_score,
        "max_context_chars": max_context_chars,
        "embedding_model": embedding_model,
        "source_filter": source_filter,
        "qdrant_url": qdrant_url,
        "qdrant_api_key": qdrant_api_key,
        "ocr_mode": ocr_mode,
        "ocr_dpi": ocr_dpi,
        "ocr_max_pages": ocr_max_pages,
        "ocr_max_regions_per_page": ocr_max_regions_per_page,
        "ocr_region_padding_px": ocr_region_padding_px,
        "ocr_gap_multiplier": ocr_gap_multiplier,
        "ocr_min_extracted_chars": ocr_min_extracted_chars,
        "ocr_timeout_ms_per_page": ocr_timeout_ms_per_page,
        "parser_mode": parser_mode,
        "hybrid_search_enabled": hybrid_search_enabled,
        "rerank_enabled": rerank_enabled,
        "summary_first_enabled": summary_first_enabled,
        "summary_top_k": summary_top_k,
        "dense_top_k": dense_top_k,
        "lexical_top_k": lexical_top_k,
        "parent_context_enabled": parent_context_enabled,
        "parent_context_max_chars": parent_context_max_chars,
        "graph_enabled": graph_enabled,
        "graph_hops": graph_hops,
        "graph_source_boost": graph_source_boost,
        "late_interaction_enabled": late_interaction_enabled,
        "late_interaction_weight": late_interaction_weight,
        "quick_qa_mode": quick_qa_mode,
        "doc_mode": doc_mode,
        "doc_pages": doc_pages,
        "doc_quality": doc_quality,
        "doc_audience": doc_audience,
        "doc_tone": doc_tone,
        "doc_chart_enabled": doc_chart_enabled,
        "doc_flow_enabled": doc_flow_enabled,
        "doc_max_charts": doc_max_charts,
    }


def _build_runtime_controller_from_values(values: dict[str, Any]) -> GeneratorController:
    config = build_generator_config(values, include_existing_data=False)
    controller = GeneratorController()
    controller.set_runtime_config(config)
    return controller


def debug_details_markdown(session: WebSessionState) -> str:
    lines = [
        "### Debug Details",
        f"- Active tab: **{session.active_tab}**",
        f"- Files mode: **{session.files_mode}**",
        f"- Imported rows in session: **{len(session.imported_data)}**",
        f"- Saved schema rows: **{len(session.fields)}**",
        f"- Generated rows in session: **{len(session.generated_rows)}**",
        f"- Files in session: **{len(session.rag_files)}**",
        f"- Activity entries stored: **{len(session.activity_log)}**",
    ]

    if session.latest_downloads:
        download_bits = ", ".join(sorted(session.latest_downloads.keys()))
        lines.append(f"- Latest downloads: {download_bits}")
    else:
        lines.append("- Latest downloads: none")

    for task_name in ("data", "files"):
        controller = get_runtime_controller(session, task_name)
        if controller is None:
            lines.append(f"- `{task_name}` controller: inactive")
            continue
        lines.append(f"- `{task_name}` controller: active")
        try:
            metrics = controller.get_metrics() or {}
            stats = metrics.get("stats") or {}
            generated = stats.get("generated", 0)
            target = stats.get("target", 0)
            elapsed = float(stats.get("elapsed", 0.0) or 0.0)
            lines.append(f"  Progress: {generated}/{target} | elapsed {elapsed:.1f}s")
        except Exception as exc:
            lines.append(f"  Metrics unavailable: {exc}")

    if session.activity_log:
        lines.append("")
        lines.append("**Recent activity**")
        lines.extend(f"- {entry}" for entry in session.activity_log[-5:])

    return "\n".join(lines)


def refresh_models(
    session: WebSessionState,
    current_model: str,
    provider: str,
    api_key: str,
    azure_endpoint: str,
    azure_deployment: str,
) -> tuple[WebSessionState, dict[str, Any], str]:
    try:
        config = build_generator_config(
            {
                "model_id": current_model or "local-model",
                "provider": provider,
                "api_key": api_key,
                "azure_endpoint": azure_endpoint,
                "azure_deployment": azure_deployment,
            },
            include_existing_data=False,
        )
        models = LLMClient(config).list_models()
        selected = models[0] if models else (current_model or "local-model")
        append_activity(session, f"Model refresh complete: {len(models)} model(s) found.")
        return session, gr.update(choices=models or [selected], value=selected), activity_markdown(session)
    except Exception as exc:
        append_activity(session, f"Model refresh failed: {exc}")
        return session, gr.update(), activity_markdown(session)


def test_connection(
    session: WebSessionState,
    current_model: str,
    provider: str,
    api_key: str,
    azure_endpoint: str,
    azure_deployment: str,
) -> tuple[WebSessionState, str]:
    try:
        config = build_generator_config(
            {
                "model_id": current_model or "local-model",
                "provider": provider,
                "api_key": api_key,
                "azure_endpoint": azure_endpoint,
                "azure_deployment": azure_deployment,
            },
            include_existing_data=False,
        )
        is_connected = LLMClient(config).check_connection()
        append_activity(
            session,
            f"Connection {'ready' if is_connected else 'failed'}: {provider} / {config.model_id}",
        )
    except Exception as exc:
        append_activity(session, f"Connection test error: {exc}")
    return session, activity_markdown(session)


def reset_config(session: WebSessionState):
    defaults = default_ui_values()
    defaults["collection_name"] = session.startup_collection_name or str(defaults["collection_name"])
    session.active_tab = "data"
    session.files_mode = "Document Engine"
    session.import_privacy_mode = "Mask likely personal values"
    session.import_mask_mappings = []
    session.raw_imported_data = []
    session.imported_data = []
    session.fields = []
    session.rag_files = []
    session.file_chat_history = []
    session.generated_rows = []
    session.latest_downloads = {}
    append_activity(session, "Settings reset to the default starting state.")

    return (
        session,
        gr.update(choices=[str(defaults["model_id"])], value=str(defaults["model_id"])),
        str(defaults["provider"]),
        str(defaults["api_key"]),
        str(defaults["azure_endpoint"]),
        str(defaults["azure_deployment"]),
        float(defaults["input_price_per_1m"]),
        float(defaults["output_price_per_1m"]),
        int(defaults["num_rows"]),
        float(defaults["similarity_threshold"]),
        int(defaults["max_retries"]),
        str(defaults["rag_backend"]),
        str(defaults["collection_name"]),
        int(defaults["top_k"]),
        float(defaults["min_score"]),
        int(defaults["max_context_chars"]),
        str(defaults["embedding_model"]),
        str(defaults["source_filter"]),
        str(defaults["qdrant_url"]),
        str(defaults["qdrant_api_key"]),
        str(defaults["ocr_mode"]),
        int(defaults["ocr_dpi"]),
        int(defaults["ocr_max_pages"]),
        int(defaults["ocr_max_regions_per_page"]),
        int(defaults["ocr_region_padding_px"]),
        float(defaults["ocr_gap_multiplier"]),
        int(defaults["ocr_min_extracted_chars"]),
        int(defaults["ocr_timeout_ms_per_page"]),
        str(defaults["parser_mode"]),
        bool(defaults["hybrid_search_enabled"]),
        bool(defaults["rerank_enabled"]),
        bool(defaults["summary_first_enabled"]),
        int(defaults["summary_top_k"]),
        int(defaults["dense_top_k"]),
        int(defaults["lexical_top_k"]),
        bool(defaults["parent_context_enabled"]),
        int(defaults["parent_context_max_chars"]),
        bool(defaults["graph_enabled"]),
        int(defaults["graph_hops"]),
        float(defaults["graph_source_boost"]),
        bool(defaults["late_interaction_enabled"]),
        float(defaults["late_interaction_weight"]),
        str(defaults["quick_qa_mode"]),
        str(defaults["doc_mode"]),
        str(defaults["doc_pages"]),
        str(defaults["doc_quality"]),
        str(defaults["doc_audience"]),
        str(defaults["doc_tone"]),
        bool(defaults["doc_chart_enabled"]),
        bool(defaults["doc_flow_enabled"]),
        int(defaults["doc_max_charts"]),
        session.import_privacy_mode,
        field_records_to_grid_dataframe([]),
        imported_columns_markup([], session.import_privacy_mode),
        "No imported data yet.",
        gr.update(value=pd.DataFrame(), visible=False),
        DEFAULT_DATA_STATUS,
        field_rows_markup([]),
        gr.update(value=pd.DataFrame(), visible=False),
        DEFAULT_GENERATION_PROGRESS,
        DEFAULT_QUALITY_REPORT,
        None,
        sources_dataframe([]),
        source_selector_update([]),
        chat_markdown(session),
        files_status_text("Document Engine", 0),
        DEFAULT_FILES_PROGRESS,
        None,
        None,
        None,
        DEFAULT_RAG_ADMIN_STATUS,
        activity_markdown(session),
        debug_details_markdown(session),
    )


def get_search_status(
    session: WebSessionState,
    model_id: str,
    provider: str,
    api_key: str,
    azure_endpoint: str,
    azure_deployment: str,
    input_price_per_1m: float,
    output_price_per_1m: float,
    num_rows: int,
    similarity_threshold: float,
    max_retries: int,
    rag_backend: str,
    collection_name: str,
    top_k: int,
    min_score: float,
    max_context_chars: int,
    embedding_model: str,
    source_filter: str,
    qdrant_url: str,
    qdrant_api_key: str,
    ocr_mode: str,
    ocr_dpi: int,
    ocr_max_pages: int,
    ocr_max_regions_per_page: int,
    ocr_region_padding_px: int,
    ocr_gap_multiplier: float,
    ocr_min_extracted_chars: int,
    ocr_timeout_ms_per_page: int,
    parser_mode: str,
    hybrid_search_enabled: bool,
    rerank_enabled: bool,
    summary_first_enabled: bool,
    summary_top_k: int,
    dense_top_k: int,
    lexical_top_k: int,
    parent_context_enabled: bool,
    parent_context_max_chars: int,
    graph_enabled: bool,
    graph_hops: int,
    graph_source_boost: float,
    late_interaction_enabled: bool,
    late_interaction_weight: float,
    quick_qa_mode: str,
    doc_mode: str,
    doc_pages: str,
    doc_quality: str,
    doc_audience: str,
    doc_tone: str,
    doc_chart_enabled: bool,
    doc_flow_enabled: bool,
    doc_max_charts: int,
):
    try:
        values = _ui_values_from_args(
            model_id,
            provider,
            api_key,
            azure_endpoint,
            azure_deployment,
            input_price_per_1m,
            output_price_per_1m,
            num_rows,
            similarity_threshold,
            max_retries,
            rag_backend,
            collection_name,
            top_k,
            min_score,
            max_context_chars,
            embedding_model,
            source_filter,
            qdrant_url,
            qdrant_api_key,
            ocr_mode,
            ocr_dpi,
            ocr_max_pages,
            ocr_max_regions_per_page,
            ocr_region_padding_px,
            ocr_gap_multiplier,
            ocr_min_extracted_chars,
            ocr_timeout_ms_per_page,
            parser_mode,
            hybrid_search_enabled,
            rerank_enabled,
            summary_first_enabled,
            summary_top_k,
            dense_top_k,
            lexical_top_k,
            parent_context_enabled,
            parent_context_max_chars,
            graph_enabled,
            graph_hops,
            graph_source_boost,
            late_interaction_enabled,
            late_interaction_weight,
            quick_qa_mode,
            doc_mode,
            doc_pages,
            doc_quality,
            doc_audience,
            doc_tone,
            doc_chart_enabled,
            doc_flow_enabled,
            doc_max_charts,
        )
        controller = _build_runtime_controller_from_values(values)
        status = controller.get_rag_status()
        append_activity(session, f"Fetched search status for collection {status.get('collection_name', collection_name)}.")
        markdown = (
            "### Search Status\n"
            f"- Collection: **{status.get('collection_name', '')}**\n"
            f"- Vectors: **{status.get('collection_size', 0)}**\n"
            f"- Top K: **{status.get('top_k', 0)}**\n"
            f"- Min score: **{status.get('min_score', 0)}**\n"
            f"- OCR mode: **{status.get('ocr_mode', 'off')}**\n"
            f"- Parser mode: **{status.get('parser_mode', 'auto')}**\n"
            f"- Hybrid: **{status.get('hybrid_search_enabled', True)}**\n"
            f"- Rerank: **{status.get('rerank_enabled', True)}**\n"
            f"- Graph retrieval: **{status.get('graph_enabled', True)}**\n"
            f"- Late interaction: **{status.get('late_interaction_enabled', True)}**"
        )
        return session, markdown, activity_markdown(session), debug_details_markdown(session)
    except Exception as exc:
        append_activity(session, f"Search status failed: {exc}")
        return session, "Search status could not be loaded.", activity_markdown(session), debug_details_markdown(session)


def clear_search_index(
    session: WebSessionState,
    model_id: str,
    provider: str,
    api_key: str,
    azure_endpoint: str,
    azure_deployment: str,
    input_price_per_1m: float,
    output_price_per_1m: float,
    num_rows: int,
    similarity_threshold: float,
    max_retries: int,
    rag_backend: str,
    collection_name: str,
    top_k: int,
    min_score: float,
    max_context_chars: int,
    embedding_model: str,
    source_filter: str,
    qdrant_url: str,
    qdrant_api_key: str,
    ocr_mode: str,
    ocr_dpi: int,
    ocr_max_pages: int,
    ocr_max_regions_per_page: int,
    ocr_region_padding_px: int,
    ocr_gap_multiplier: float,
    ocr_min_extracted_chars: int,
    ocr_timeout_ms_per_page: int,
    parser_mode: str,
    hybrid_search_enabled: bool,
    rerank_enabled: bool,
    summary_first_enabled: bool,
    summary_top_k: int,
    dense_top_k: int,
    lexical_top_k: int,
    parent_context_enabled: bool,
    parent_context_max_chars: int,
    graph_enabled: bool,
    graph_hops: int,
    graph_source_boost: float,
    late_interaction_enabled: bool,
    late_interaction_weight: float,
    quick_qa_mode: str,
    doc_mode: str,
    doc_pages: str,
    doc_quality: str,
    doc_audience: str,
    doc_tone: str,
    doc_chart_enabled: bool,
    doc_flow_enabled: bool,
    doc_max_charts: int,
):
    try:
        values = _ui_values_from_args(
            model_id,
            provider,
            api_key,
            azure_endpoint,
            azure_deployment,
            input_price_per_1m,
            output_price_per_1m,
            num_rows,
            similarity_threshold,
            max_retries,
            rag_backend,
            collection_name,
            top_k,
            min_score,
            max_context_chars,
            embedding_model,
            source_filter,
            qdrant_url,
            qdrant_api_key,
            ocr_mode,
            ocr_dpi,
            ocr_max_pages,
            ocr_max_regions_per_page,
            ocr_region_padding_px,
            ocr_gap_multiplier,
            ocr_min_extracted_chars,
            ocr_timeout_ms_per_page,
            parser_mode,
            hybrid_search_enabled,
            rerank_enabled,
            summary_first_enabled,
            summary_top_k,
            dense_top_k,
            lexical_top_k,
            parent_context_enabled,
            parent_context_max_chars,
            graph_enabled,
            graph_hops,
            graph_source_boost,
            late_interaction_enabled,
            late_interaction_weight,
            quick_qa_mode,
            doc_mode,
            doc_pages,
            doc_quality,
            doc_audience,
            doc_tone,
            doc_chart_enabled,
            doc_flow_enabled,
            doc_max_charts,
        )
        controller = _build_runtime_controller_from_values(values)
        controller.clear_rag_collection()
        session.rag_files = []
        session.file_chat_history = []
        append_activity(session, "Search index cleared.")
        return (
            session,
            sources_dataframe([]),
            source_selector_update([]),
            chat_markdown(session),
            files_status_text(session.files_mode, 0),
            "### Search Status\n- Search index cleared.",
            activity_markdown(session),
            debug_details_markdown(session),
        )
    except Exception as exc:
        append_activity(session, f"Failed to clear the search index: {exc}")
        return (
            session,
            sources_dataframe(session.rag_files),
            source_selector_update(session.rag_files),
            chat_markdown(session),
            files_status_text(session.files_mode, len(session.rag_files)),
            "Search index could not be cleared.",
            activity_markdown(session),
            debug_details_markdown(session),
        )


def refresh_debug_details(session: WebSessionState):
    append_activity(session, "Refreshed debug details.")
    return session, debug_details_markdown(session), activity_markdown(session)


def clear_debug_details(session: WebSessionState):
    session.activity_log = ["Web UI preview ready."]
    return session, debug_details_markdown(session), activity_markdown(session)


def save_config_file(
    session: WebSessionState,
    grid_value: Any,
    row_editor_choice: str | None,
    row_editor_name: str,
    row_editor_type: str,
    row_editor_prompt: str,
    row_editor_allow_duplicates: bool,
    model_id: str,
    provider: str,
    api_key: str,
    azure_endpoint: str,
    azure_deployment: str,
    input_price_per_1m: float,
    output_price_per_1m: float,
    num_rows: int,
    similarity_threshold: float,
    max_retries: int,
    rag_backend: str,
    collection_name: str,
    top_k: int,
    min_score: float,
    max_context_chars: int,
    embedding_model: str,
    source_filter: str,
    qdrant_url: str,
    qdrant_api_key: str,
    ocr_mode: str,
    ocr_dpi: int,
    ocr_max_pages: int,
    ocr_max_regions_per_page: int,
    ocr_region_padding_px: int,
    ocr_gap_multiplier: float,
    ocr_min_extracted_chars: int,
    ocr_timeout_ms_per_page: int,
    parser_mode: str,
    hybrid_search_enabled: bool,
    rerank_enabled: bool,
    summary_first_enabled: bool,
    summary_top_k: int,
    dense_top_k: int,
    lexical_top_k: int,
    parent_context_enabled: bool,
    parent_context_max_chars: int,
    graph_enabled: bool,
    graph_hops: int,
    graph_source_boost: float,
    late_interaction_enabled: bool,
    late_interaction_weight: float,
    quick_qa_mode: str,
    doc_mode: str,
    doc_pages: str,
    doc_quality: str,
    doc_audience: str,
    doc_tone: str,
    doc_chart_enabled: bool,
    doc_flow_enabled: bool,
    doc_max_charts: int,
    import_privacy_mode: str,
) -> tuple[WebSessionState, str, str]:
    try:
        rows = _grid_rows(grid_value)
        editor_name = str(row_editor_name or "").strip()
        editor_prompt = str(row_editor_prompt or "").strip()
        if rows and row_editor_choice and any([editor_name, editor_prompt]):
            selected_index = _selected_grid_row_index(rows, row_editor_choice)
            rows[selected_index] = {
                "row_id": f"Row {selected_index + 1}",
                "name": editor_name,
                "type": normalize_field_record({"type": row_editor_type}).get("type", "Short Text"),
                "prompt_instruction": editor_prompt,
                "allow_duplicates": _bool_value(row_editor_allow_duplicates),
            }
            grid_value = pd.DataFrame(rows, columns=["row_id", "name", "type", "prompt_instruction", "allow_duplicates"])

        records, error = _records_from_grid(grid_value, session.fields)
        if error:
            append_activity(session, f"Config save failed: {error}")
            return session, "", activity_markdown(session)
        session.fields = records
        values = _ui_values_from_args(
            model_id,
            provider,
            api_key,
            azure_endpoint,
            azure_deployment,
            input_price_per_1m,
            output_price_per_1m,
            num_rows,
            similarity_threshold,
            max_retries,
            rag_backend,
            collection_name,
            top_k,
            min_score,
            max_context_chars,
            embedding_model,
            source_filter,
            qdrant_url,
            qdrant_api_key,
            ocr_mode,
            ocr_dpi,
            ocr_max_pages,
            ocr_max_regions_per_page,
            ocr_region_padding_px,
            ocr_gap_multiplier,
            ocr_min_extracted_chars,
            ocr_timeout_ms_per_page,
            parser_mode,
            hybrid_search_enabled,
            rerank_enabled,
            summary_first_enabled,
            summary_top_k,
            dense_top_k,
            lexical_top_k,
            parent_context_enabled,
            parent_context_max_chars,
            graph_enabled,
            graph_hops,
            graph_source_boost,
            late_interaction_enabled,
            late_interaction_weight,
            quick_qa_mode,
            doc_mode,
            doc_pages,
            doc_quality,
            doc_audience,
            doc_tone,
            doc_chart_enabled,
            doc_flow_enabled,
            doc_max_charts,
        )
        values["import_privacy_mode"] = import_privacy_mode
        columns = [column.model_dump() for column in field_records_to_columns(records)]
        payload = serialize_ui_config(values, columns=columns)

        EXPORT_DIR.mkdir(exist_ok=True)
        filename = f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = EXPORT_DIR / filename
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        session.latest_downloads["config"] = str(path)
        append_activity(session, f"Saved config preview to {path}")
        return session, str(path), activity_markdown(session)
    except Exception as exc:
        append_activity(session, f"Config save failed: {exc}")
        return session, "", activity_markdown(session)


def load_config_file(
    session: WebSessionState,
    config_path: str,
) -> tuple[
    WebSessionState,
    dict[str, Any],
    str,
    str,
    str,
    str,
    float,
    float,
    int,
    float,
    int,
    str,
    str,
    int,
    float,
    int,
    str,
    str,
    str,
    int,
    int,
    int,
    int,
    float,
    int,
    int,
    str,
    bool,
    bool,
    bool,
    int,
    int,
    int,
    bool,
    int,
    bool,
    int,
    float,
    bool,
    float,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    bool,
    bool,
    int,
    str,
    Any,
    str,
    str,
]:
    if not config_path:
        append_activity(session, "No config file selected.")
        return (
            session,
            gr.update(),
            "",
            "",
            "",
            "",
            0.15,
            0.60,
            10,
            0.85,
            50,
            "LlamaIndex",
            "synthesizer_default",
            5,
            0.25,
            3000,
            "BAAI/bge-small-en-v1.5",
            "",
            ":memory:",
            "",
            "off",
            150,
            20,
            8,
            18,
            2.5,
            60,
            4000,
            "auto",
            True,
            True,
            True,
            3,
            12,
            12,
            True,
            1200,
            True,
            1,
            0.08,
            True,
            0.2,
            "Broader Analysis",
            "Balanced",
            "Let AI decide",
            "Fast",
            "General",
            "professional",
            False,
            True,
            3,
            "Mask likely personal values",
            field_records_to_grid_dataframe([]),
            "No config file selected.",
            activity_markdown(session),
        )

    with open(config_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    values = normalize_loaded_config(data)
    session.fields = [normalize_field_record(field) for field in (data.get("columns") or [])]
    session.import_privacy_mode = str(values.get("import_privacy_mode", session.import_privacy_mode or "Mask likely personal values"))
    model_value = str(values["model_id"])
    status = f"Loaded config with **{len(session.fields)}** schema row(s)."

    append_activity(session, f"Loaded config from {config_path}")
    return (
        session,
        gr.update(choices=[model_value], value=model_value),
        str(values["provider"]),
        str(values["api_key"]),
        str(values["azure_endpoint"]),
        str(values["azure_deployment"]),
        float(values["input_price_per_1m"]),
        float(values["output_price_per_1m"]),
        int(values["num_rows"]),
        float(values["similarity_threshold"]),
        int(values["max_retries"]),
        str(values["rag_backend"]),
        str(values["collection_name"]),
        int(values["top_k"]),
        float(values["min_score"]),
        int(values["max_context_chars"]),
        str(values["embedding_model"]),
        str(values["source_filter"]),
        str(values["qdrant_url"]),
        str(values["qdrant_api_key"]),
        str(values["ocr_mode"]),
        int(values["ocr_dpi"]),
        int(values["ocr_max_pages"]),
        int(values["ocr_max_regions_per_page"]),
        int(values["ocr_region_padding_px"]),
        float(values["ocr_gap_multiplier"]),
        int(values["ocr_min_extracted_chars"]),
        int(values["ocr_timeout_ms_per_page"]),
        str(values["parser_mode"]),
        bool(values["hybrid_search_enabled"]),
        bool(values["rerank_enabled"]),
        bool(values["summary_first_enabled"]),
        int(values["summary_top_k"]),
        int(values["dense_top_k"]),
        int(values["lexical_top_k"]),
        bool(values["parent_context_enabled"]),
        int(values["parent_context_max_chars"]),
        bool(values["graph_enabled"]),
        int(values["graph_hops"]),
        float(values["graph_source_boost"]),
        bool(values["late_interaction_enabled"]),
        float(values["late_interaction_weight"]),
        str(values["quick_qa_mode"]),
        str(values["doc_mode"]),
        str(values["doc_pages"]),
        str(values["doc_quality"]),
        str(values["doc_audience"]),
        str(values["doc_tone"]),
        bool(values["doc_chart_enabled"]),
        bool(values["doc_flow_enabled"]),
        int(values["doc_max_charts"]),
        session.import_privacy_mode,
        field_records_to_grid_dataframe(session.fields),
        status,
        activity_markdown(session),
    )
