from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr

from core.app_config import build_generator_config, normalize_loaded_config, serialize_ui_config
from core.llm_client import LLMClient
from web_ui.adapters import (
    field_records_to_columns,
    field_records_to_grid_dataframe,
    normalize_field_record,
)
from web_ui.state import WebSessionState, activity_markdown, append_activity


EXPORT_DIR = Path(".web_ui_exports")


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
    parser_mode: str,
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
        "parser_mode": parser_mode,
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


def save_config_file(
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
    parser_mode: str,
    quick_qa_mode: str,
    doc_mode: str,
    doc_pages: str,
    doc_quality: str,
    doc_audience: str,
    doc_tone: str,
    doc_chart_enabled: bool,
    doc_flow_enabled: bool,
    doc_max_charts: int,
) -> tuple[WebSessionState, str, str]:
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
            parser_mode,
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
        columns = [column.model_dump() for column in field_records_to_columns(session.fields)]
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
    str,
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
    Any,
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
            "auto",
            "Broader Analysis",
            "Balanced",
            "Let AI decide",
            "Fast",
            "General",
            "professional",
            False,
            True,
            3,
            field_records_to_grid_dataframe([]),
            activity_markdown(session),
        )

    with open(config_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    values = normalize_loaded_config(data)
    session.fields = [normalize_field_record(field) for field in (data.get("columns") or [])]
    model_value = str(values["model_id"])

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
        str(values["parser_mode"]),
        str(values["quick_qa_mode"]),
        str(values["doc_mode"]),
        str(values["doc_pages"]),
        str(values["doc_quality"]),
        str(values["doc_audience"]),
        str(values["doc_tone"]),
        bool(values["doc_chart_enabled"]),
        bool(values["doc_flow_enabled"]),
        int(values["doc_max_charts"]),
        field_records_to_grid_dataframe(session.fields),
        activity_markdown(session),
    )
