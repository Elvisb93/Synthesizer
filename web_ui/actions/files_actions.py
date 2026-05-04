from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from core.app_config import build_generator_config, resolve_document_mode
from core.controller import GeneratorController
from core.json_parser import resolve_target_array
from core.models import RagBackend
from web_ui.state import (
    WebSessionState,
    activity_markdown,
    append_activity,
    clear_runtime_controller,
    get_runtime_controller,
    register_runtime_controller,
)
from web_ui.runtime_cleanup import record_runtime_collection


EXPORT_DIR = Path(".web_ui_exports")
PRESETS_FILE = Path(".rag_task_presets.json")

FILES_MODE_HELPERS = {
    "Document Engine": "Upload files, describe the document you want, then generate grounded output with PDF and DOCX downloads.",
    "Quick Q&A": "Upload files, ask a question, and get a grounded answer with citations.",
    "Structured JSON": "Upload files or provide a template, then build or extract JSON into the target list.",
}

DEFAULT_FILE_PRESETS = {
    "Summarize": "Summarize the main points from the imported files in bullet points.",
    "Action Items": "Extract key action items, owners, and deadlines from the imported files.",
    "Draft Reply": "Draft a concise response email based on the imported file context.",
}

DOC_PRESET_BUNDLES = {
    "Executive Brief": {
        "mode": "Balanced",
        "pages": "2 pages",
        "quality": "Fast",
        "audience": "Executives",
        "tone": "professional",
        "prompt": "Create an executive brief with key findings, risks, and recommended next steps.",
    },
    "Policy Draft": {
        "mode": "File-based",
        "pages": "5 pages",
        "quality": "Thorough",
        "audience": "Policy stakeholders",
        "tone": "formal",
        "prompt": "Draft a policy document based on the imported files, including scope, requirements, and governance.",
    },
    "Action Plan": {
        "mode": "Balanced",
        "pages": "3 pages",
        "quality": "Fast",
        "audience": "Implementation team",
        "tone": "direct",
        "prompt": "Create an action plan with phases, owners, milestones, and measurable success criteria.",
    },
    "Meeting Summary": {
        "mode": "File-based",
        "pages": "1 page",
        "quality": "Fast",
        "audience": "Team",
        "tone": "concise",
        "prompt": "Summarize meeting outcomes, decisions, open questions, and immediate action items.",
    },
}


def files_mode_helper(mode: str) -> str:
    return FILES_MODE_HELPERS.get(mode or "Document Engine", FILES_MODE_HELPERS["Document Engine"])


def files_status_text(mode: str, source_count: int) -> str:
    return f"{files_mode_helper(mode)}\n\nCurrent files: **{source_count}**"


def _source_name(path: str) -> str:
    return path if str(path).lower().startswith(("http://", "https://")) else os.path.basename(path)


def sources_dataframe(paths: list[str]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame(columns=["name", "path"])
    return pd.DataFrame(
        [{"name": _source_name(path), "path": path} for path in paths],
        columns=["name", "path"],
    )


def source_selector_update(paths: list[str], selected_source: str | None = None):
    if not paths:
        return gr.update(choices=[], value=None)
    value = selected_source if selected_source in paths else paths[0]
    return gr.update(choices=paths, value=value)


def load_file_presets() -> dict[str, str]:
    presets = dict(DEFAULT_FILE_PRESETS)
    if PRESETS_FILE.exists():
        try:
            raw = json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                cleaned = {str(k): str(v) for k, v in raw.items() if str(k).strip() and str(v).strip()}
                if cleaned:
                    presets = cleaned
        except Exception:
            pass
    return presets


def save_file_presets(presets: dict[str, str]) -> None:
    PRESETS_FILE.write_text(json.dumps(presets, indent=2), encoding="utf-8")


def preset_choices() -> list[str]:
    return sorted(load_file_presets().keys())


def preset_dropdown_update(selected_name: str | None = None):
    choices = preset_choices()
    value = selected_name if selected_name in choices else (choices[0] if choices else None)
    return gr.update(choices=choices, value=value)


def files_mode_changed(mode: str):
    mode = (mode or "Document Engine").strip()
    is_doc = mode == "Document Engine"
    is_qa = mode == "Quick Q&A"
    is_json = mode == "Structured JSON"
    prompt_update = {
        "Document Engine": ("What document should the files help you create?", "e.g., Create an executive brief with findings, risks, and next steps.", True, True),
        "Quick Q&A": ("Ask The Uploaded Files", "Ask a question, then ask follow-ups in the same chat.", True, False),
        "Structured JSON": ("Structured JSON uses the template settings below.", "Select a template and target list, then run generation.", False, False),
    }[mode]
    return (
        files_mode_helper(mode),
        gr.update(label=prompt_update[0], placeholder=prompt_update[1], interactive=prompt_update[2], visible=prompt_update[3]),
        gr.update(visible=is_doc),
        gr.update(visible=is_qa),
        gr.update(visible=is_json),
        gr.update(visible=not is_json),
        gr.update(visible=not is_qa),
        gr.update(visible=not is_qa),
    )


def register_uploaded_files(session: WebSessionState, files: Any, mode: str):
    file_paths = files if isinstance(files, list) else ([files] if files else [])
    normalized = [str(path) for path in file_paths if path]
    merged = list(session.rag_files)
    for path in normalized:
        if path not in merged:
            merged.append(path)
    session.rag_files = merged
    clear_runtime_controller(session, "files_chat")
    session.files_mode = mode or "Document Engine"
    append_activity(session, f"Registered {len(normalized)} file(s) for Files workflow.")
    return (
        session,
        sources_dataframe(session.rag_files),
        source_selector_update(session.rag_files),
        files_status_text(mode, len(session.rag_files)),
        activity_markdown(session),
    )


def add_url_source(session: WebSessionState, url: str, mode: str):
    url = (url or "").strip()
    if not url:
        return session, sources_dataframe(session.rag_files), source_selector_update(session.rag_files), "Enter a URL first.", activity_markdown(session)
    if not url.lower().startswith(("http://", "https://")):
        return session, sources_dataframe(session.rag_files), source_selector_update(session.rag_files), "URL must start with http:// or https://", activity_markdown(session)
    if url not in session.rag_files:
        session.rag_files.append(url)
        clear_runtime_controller(session, "files_chat")
        append_activity(session, f"Added web source: {url}")
    return (
        session,
        sources_dataframe(session.rag_files),
        source_selector_update(session.rag_files, url),
        files_status_text(mode, len(session.rag_files)),
        activity_markdown(session),
    )


def clear_files(session: WebSessionState, mode: str):
    session.rag_files = []
    session.file_chat_history = []
    clear_runtime_controller(session, "files_chat")
    append_activity(session, "Cleared Files workspace sources.")
    return (
        session,
        sources_dataframe([]),
        source_selector_update([]),
        chat_markdown(session),
        files_status_text(mode, 0),
        "Files progress will appear here after you run a task.",
        activity_markdown(session),
    )


def apply_file_preset(session: WebSessionState, preset_name: str | None):
    name = str(preset_name or "").strip()
    presets = load_file_presets()
    prompt = presets.get(name, "")
    if name and prompt:
        append_activity(session, f"Loaded prompt preset: {name}")
    return session, prompt, name, activity_markdown(session)


def save_file_preset(session: WebSessionState, preset_name: str, prompt: str, current_selection: str | None):
    name = str(preset_name or "").strip()
    body = str(prompt or "").strip()
    if not name:
        append_activity(session, "Preset save skipped: missing preset name.")
        return session, preset_dropdown_update(current_selection), preset_name, activity_markdown(session)
    if not body:
        append_activity(session, "Preset save skipped: empty prompt.")
        return session, preset_dropdown_update(current_selection), name, activity_markdown(session)

    presets = load_file_presets()
    presets[name] = body
    save_file_presets(presets)
    append_activity(session, f"Saved prompt preset: {name}")
    return session, preset_dropdown_update(name), name, activity_markdown(session)


def delete_file_preset(session: WebSessionState, preset_name: str | None):
    name = str(preset_name or "").strip()
    if not name:
        append_activity(session, "Preset delete skipped: no preset selected.")
        return session, preset_dropdown_update(None), "", activity_markdown(session)

    presets = load_file_presets()
    if name in presets:
        del presets[name]
        save_file_presets(presets)
        append_activity(session, f"Deleted prompt preset: {name}")
    else:
        append_activity(session, f"Preset delete skipped: {name} was not found.")
    return session, preset_dropdown_update(None), "", activity_markdown(session)


def apply_doc_bundle(session: WebSessionState, bundle_name: str):
    bundle = DOC_PRESET_BUNDLES.get(bundle_name)
    if not bundle:
        append_activity(session, f"Unknown document bundle requested: {bundle_name}")
        return session, "", "Balanced", "Let AI decide", "Fast", "General", "professional", activity_markdown(session)

    append_activity(session, f"Applied document bundle: {bundle_name}")
    return (
        session,
        bundle["prompt"],
        bundle["mode"],
        bundle["pages"],
        bundle["quality"],
        bundle["audience"],
        bundle["tone"],
        activity_markdown(session),
    )


def reindex_selected_source(
    session: WebSessionState,
    selected_source: str | None,
    mode: str,
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
    selected = str(selected_source or "").strip()
    if not selected:
        append_activity(session, "Source re-index skipped: no source selected.")
        return session, sources_dataframe(session.rag_files), source_selector_update(session.rag_files), "Select a source first.", activity_markdown(session)

    controller, _ = _build_files_controller(
        session,
        mode=mode,
        model_id=model_id,
        provider=provider,
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        input_price_per_1m=input_price_per_1m,
        output_price_per_1m=output_price_per_1m,
        num_rows=num_rows,
        similarity_threshold=similarity_threshold,
        max_retries=max_retries,
        rag_backend=rag_backend,
        collection_name=collection_name,
        top_k=top_k,
        min_score=min_score,
        max_context_chars=max_context_chars,
        embedding_model=embedding_model,
        source_filter=source_filter,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        ocr_mode=ocr_mode,
        ocr_dpi=ocr_dpi,
        ocr_max_pages=ocr_max_pages,
        ocr_max_regions_per_page=ocr_max_regions_per_page,
        ocr_region_padding_px=ocr_region_padding_px,
        ocr_gap_multiplier=ocr_gap_multiplier,
        ocr_min_extracted_chars=ocr_min_extracted_chars,
        ocr_timeout_ms_per_page=ocr_timeout_ms_per_page,
        parser_mode=parser_mode,
        hybrid_search_enabled=hybrid_search_enabled,
        rerank_enabled=rerank_enabled,
        summary_first_enabled=summary_first_enabled,
        summary_top_k=summary_top_k,
        dense_top_k=dense_top_k,
        lexical_top_k=lexical_top_k,
        parent_context_enabled=parent_context_enabled,
        parent_context_max_chars=parent_context_max_chars,
        graph_enabled=graph_enabled,
        graph_hops=graph_hops,
        graph_source_boost=graph_source_boost,
        late_interaction_enabled=late_interaction_enabled,
        late_interaction_weight=late_interaction_weight,
        quick_qa_mode=quick_qa_mode,
        doc_mode=doc_mode,
        doc_pages=doc_pages,
        doc_quality=doc_quality,
        doc_audience=doc_audience,
        doc_tone=doc_tone,
        doc_chart_enabled=doc_chart_enabled,
        doc_flow_enabled=doc_flow_enabled,
        doc_max_charts=doc_max_charts,
    )
    report = controller.ingest_documents([selected], force_reindex=True)
    if report.get("error"):
        append_activity(session, f"Source re-index failed: {report['error']}")
        return session, sources_dataframe(session.rag_files), source_selector_update(session.rag_files, selected), "Re-index failed.", activity_markdown(session)

    append_activity(session, f"Re-indexed source: {_source_name(selected)}")
    clear_runtime_controller(session, "files_chat")
    status = (
        f"Re-indexed **{_source_name(selected)}**. "
        f"Files processed: **{report.get('files_processed', 0)}**, chunks: **{report.get('chunks_created', 0)}**, vectors: **{report.get('vectors_upserted', 0)}**."
    )
    return session, sources_dataframe(session.rag_files), source_selector_update(session.rag_files, selected), status, activity_markdown(session)


def remove_selected_source(session: WebSessionState, selected_source: str | None, mode: str):
    selected = str(selected_source or "").strip()
    if not selected:
        append_activity(session, "Source removal skipped: no source selected.")
        return session, sources_dataframe(session.rag_files), source_selector_update(session.rag_files), chat_markdown(session), "Select a source first.", activity_markdown(session)

    remaining = [path for path in session.rag_files if path != selected]
    if len(remaining) == len(session.rag_files):
        append_activity(session, f"Source removal skipped: {selected} was not in the current list.")
        return session, sources_dataframe(session.rag_files), source_selector_update(session.rag_files), chat_markdown(session), "Selected source was not found.", activity_markdown(session)

    session.rag_files = remaining
    clear_runtime_controller(session, "files_chat")
    if not remaining:
        session.file_chat_history = []
    append_activity(session, f"Removed source: {_source_name(selected)}")
    return (
        session,
        sources_dataframe(session.rag_files),
        source_selector_update(session.rag_files),
        chat_markdown(session),
        files_status_text(mode, len(session.rag_files)),
        activity_markdown(session),
    )


def resolve_effective_rag_backend(mode: str, rag_backend: str, quick_qa_mode: str) -> str:
    selected_backend = str(rag_backend or RagBackend.LLAMA_INDEX.value)
    active_mode = (mode or "Document Engine").strip()
    qa_style = (quick_qa_mode or "Broader Analysis").strip()
    if active_mode == "Quick Q&A" and qa_style == "Pinpoint Quick":
        return RagBackend.NATIVE.value
    return selected_backend


def _build_files_controller(
    session: WebSessionState,
    *,
    mode: str,
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
) -> tuple[GeneratorController, list[str]]:
    config = build_generator_config(
        {
            "model_id": model_id or "local-model",
            "provider": provider,
            "api_key": api_key,
            "azure_endpoint": azure_endpoint,
            "azure_deployment": azure_deployment,
            "input_price_per_1m": input_price_per_1m,
            "output_price_per_1m": output_price_per_1m,
            "num_rows": int(num_rows or 10),
            "similarity_threshold": similarity_threshold,
            "max_retries": int(max_retries or 50),
            "rag_backend": resolve_effective_rag_backend(mode, rag_backend, quick_qa_mode),
            "collection_name": collection_name,
            "top_k": int(top_k or 5),
            "min_score": min_score,
            "max_context_chars": int(max_context_chars or 3000),
            "embedding_model": embedding_model,
            "source_filter": source_filter,
            "qdrant_url": qdrant_url,
            "qdrant_api_key": qdrant_api_key,
            "ocr_mode": ocr_mode,
            "ocr_dpi": int(ocr_dpi or 150),
            "ocr_max_pages": int(ocr_max_pages or 20),
            "ocr_max_regions_per_page": int(ocr_max_regions_per_page or 8),
            "ocr_region_padding_px": int(ocr_region_padding_px or 18),
            "ocr_gap_multiplier": float(ocr_gap_multiplier or 2.5),
            "ocr_min_extracted_chars": int(ocr_min_extracted_chars or 60),
            "ocr_timeout_ms_per_page": int(ocr_timeout_ms_per_page or 4000),
            "parser_mode": parser_mode,
            "hybrid_search_enabled": bool(hybrid_search_enabled),
            "rerank_enabled": bool(rerank_enabled),
            "summary_first_enabled": bool(summary_first_enabled),
            "summary_top_k": int(summary_top_k or 3),
            "dense_top_k": int(dense_top_k or 12),
            "lexical_top_k": int(lexical_top_k or 12),
            "parent_context_enabled": bool(parent_context_enabled),
            "parent_context_max_chars": int(parent_context_max_chars or 1200),
            "graph_enabled": bool(graph_enabled),
            "graph_hops": int(graph_hops or 1),
            "graph_source_boost": float(graph_source_boost or 0.08),
            "late_interaction_enabled": bool(late_interaction_enabled),
            "late_interaction_weight": float(late_interaction_weight or 0.2),
            "quick_qa_mode": quick_qa_mode,
            "doc_mode": doc_mode,
            "doc_pages": doc_pages,
            "doc_quality": doc_quality,
            "doc_audience": doc_audience,
            "doc_tone": doc_tone,
            "doc_chart_enabled": doc_chart_enabled,
            "doc_flow_enabled": doc_flow_enabled,
            "doc_max_charts": int(doc_max_charts or 3),
        },
        include_existing_data=False,
    )
    controller = GeneratorController()
    logs: list[str] = []
    controller.on_log = lambda message: logs.append(str(message))
    controller.set_runtime_config(config)
    return controller, logs


def _append_chat(session: WebSessionState, role: str, content: str) -> None:
    session.file_chat_history.append({"role": role, "content": content})
    if len(session.file_chat_history) > 40:
        session.file_chat_history = session.file_chat_history[-40:]


def _format_citations(citations: list[dict[str, Any]]) -> str:
    if not citations:
        return ""
    citation_lines = [
        f"- {os.path.basename(str(c.get('source', 'unknown')))} | page {c.get('page', '?')} | score {float(c.get('score', 0.0)):.3f}"
        for c in citations[:5]
    ]
    return "Citations:\n" + "\n".join(citation_lines)


def _question_with_recent_chat(session: WebSessionState, question: str) -> str:
    recent = session.file_chat_history[-6:]
    if not recent:
        return question
    lines = ["Use the recent conversation only to understand follow-up wording. Answer from the uploaded files."]
    for item in recent:
        role = "User" if item.get("role") == "user" else "Assistant"
        content = str(item.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content[:700]}")
    lines.append(f"New question: {question}")
    return "\n".join(lines)


def chat_markdown(session: WebSessionState) -> str:
    if not session.file_chat_history:
        return "Results will appear here after you run a Files task."
    blocks = []
    for item in session.file_chat_history:
        role = "You" if item.get("role") == "user" else "Assistant"
        blocks.append(f"**{role}**\n\n{item.get('content', '')}")
    return "\n\n---\n\n".join(blocks)


def _combined_activity_markdown(session: WebSessionState, live_logs: list[str] | None = None) -> str:
    lines = list(session.activity_log[-8:])
    if live_logs:
        lines.extend(live_logs[-8:])
    if not lines:
        return "No activity yet."
    return "\n".join(f"- {line}" for line in lines[-12:])


def _files_progress_markdown(
    *,
    mode: str,
    backend: str,
    stage: str,
    done: int,
    target: int,
    last_event: str,
    started_at: float,
    live_logs: list[str],
    is_running: bool,
) -> str:
    elapsed = max(0, int(time.time() - started_at))
    state_label = "Running" if is_running else "Completed"
    progress_line = f"**{done}/{target}** step(s)" if target > 0 else "Working..."
    recent_lines = "\n".join(f"- {line}" for line in live_logs[-6:]) if live_logs else "- Waiting for the next update..."
    return (
        "### Files Progress\n"
        f"- Status: **{state_label}**\n"
        f"- Task: **{mode}**\n"
        f"- Backend: **{backend}**\n"
        f"- Stage: {stage}\n"
        f"- Progress: {progress_line}\n"
        f"- Last event: {last_event}\n"
        f"- Elapsed: **{elapsed}s**\n\n"
        f"**Recent events**\n{recent_lines}"
    )


def request_stop_files_task(session: WebSessionState):
    controller = get_runtime_controller(session, "files")
    if controller is None:
        append_activity(session, "No Files task is currently running.")
        return session, "No active Files task to stop.", "Files progress will appear here after you run a task.", activity_markdown(session)

    if hasattr(controller, "stop_document_generation"):
        controller.stop_document_generation()
    elif hasattr(controller, "stop_generation"):
        controller.stop_generation()
    else:
        controller.stop_requested = True
    append_activity(session, "Stop requested for Files workflow.")
    return (
        session,
        "Stop requested. Partial results will be kept when possible.",
        "### Files Progress\n- Status: **Stopping**\n- Finishing the current unit of work before stopping.",
        activity_markdown(session),
    )


def run_files_task(
    session: WebSessionState,
    mode: str,
    prompt: str,
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
    json_template_path: str,
    json_target_key: str,
    json_mode: str,
    json_clear_existing: bool,
):
    mode = (mode or "Document Engine").strip()
    prompt = (prompt or "").strip()
    effective_backend = resolve_effective_rag_backend(mode, rag_backend, quick_qa_mode)
    if mode != "Structured JSON" and not prompt:
        yield session, chat_markdown(session), "Describe what you want from the files first.", "Files progress will appear here after you run a task.", None, None, None, activity_markdown(session)
        return
    if mode in {"Document Engine", "Quick Q&A"} and not session.rag_files:
        yield session, chat_markdown(session), "Upload or add at least one file first.", "Files progress will appear here after you run a task.", None, None, None, activity_markdown(session)
        return
    if mode == "Structured JSON":
        if not json_template_path:
            yield session, chat_markdown(session), "Select a JSON template first.", "Files progress will appear here after you run a task.", None, None, None, activity_markdown(session)
            return
        if not json_target_key:
            yield session, chat_markdown(session), "Enter the target list key first.", "Files progress will appear here after you run a task.", None, None, None, activity_markdown(session)
            return
        if json_mode == "Exhaustive Extraction" and not session.rag_files:
            yield session, chat_markdown(session), "Import at least one file before running exhaustive extraction.", "Files progress will appear here after you run a task.", None, None, None, activity_markdown(session)
            return

    controller, logs = _build_files_controller(
        session,
        mode=mode,
        model_id=model_id,
        provider=provider,
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        input_price_per_1m=input_price_per_1m,
        output_price_per_1m=output_price_per_1m,
        num_rows=num_rows,
        similarity_threshold=similarity_threshold,
        max_retries=max_retries,
        rag_backend=rag_backend,
        collection_name=collection_name,
        top_k=top_k,
        min_score=min_score,
        max_context_chars=max_context_chars,
        embedding_model=embedding_model,
        source_filter=source_filter,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        ocr_mode=ocr_mode,
        ocr_dpi=ocr_dpi,
        ocr_max_pages=ocr_max_pages,
        ocr_max_regions_per_page=ocr_max_regions_per_page,
        ocr_region_padding_px=ocr_region_padding_px,
        ocr_gap_multiplier=ocr_gap_multiplier,
        ocr_min_extracted_chars=ocr_min_extracted_chars,
        ocr_timeout_ms_per_page=ocr_timeout_ms_per_page,
        parser_mode=parser_mode,
        hybrid_search_enabled=hybrid_search_enabled,
        rerank_enabled=rerank_enabled,
        summary_first_enabled=summary_first_enabled,
        summary_top_k=summary_top_k,
        dense_top_k=dense_top_k,
        lexical_top_k=lexical_top_k,
        parent_context_enabled=parent_context_enabled,
        parent_context_max_chars=parent_context_max_chars,
        graph_enabled=graph_enabled,
        graph_hops=graph_hops,
        graph_source_boost=graph_source_boost,
        late_interaction_enabled=late_interaction_enabled,
        late_interaction_weight=late_interaction_weight,
        quick_qa_mode=quick_qa_mode,
        doc_mode=doc_mode,
        doc_pages=doc_pages,
        doc_quality=doc_quality,
        doc_audience=doc_audience,
        doc_tone=doc_tone,
        doc_chart_enabled=doc_chart_enabled,
        doc_flow_enabled=doc_flow_enabled,
        doc_max_charts=doc_max_charts,
    )
    record_runtime_collection(collection_name, qdrant_url)
    register_runtime_controller(session, "files", controller)
    progress_state = {
        "stage": "Preparing files task...",
        "done": 0,
        "target": 0,
        "last_event": "Initializing file workflow...",
    }

    def handle_log(message: str):
        text = str(message)
        logs.append(text)
        progress_state["last_event"] = text

    def handle_progress(done: int, target: int):
        progress_state["done"] = int(done)
        progress_state["target"] = int(target)

    controller.on_log = handle_log
    controller.on_progress = handle_progress
    started_at = time.time()
    result_box: dict[str, Any] = {
        "status": None,
        "error": None,
        "pdf_path": None,
        "docx_path": None,
        "json_path": None,
        "exception": None,
        "stopped": False,
    }

    yield (
        session,
        chat_markdown(session),
        f"{files_mode_helper(mode)}\n\nCurrent files: **{len(session.rag_files)}**",
        _files_progress_markdown(
            mode=mode,
            backend=effective_backend,
            stage=progress_state["stage"],
            done=progress_state["done"],
            target=progress_state["target"],
            last_event=progress_state["last_event"],
            started_at=started_at,
            live_logs=logs,
            is_running=True,
        ),
        None,
        None,
        None,
        _combined_activity_markdown(session, logs),
    )


    def worker() -> None:
        try:
            if session.rag_files:
                progress_state["stage"] = "Indexing sources..."
                report = controller.ingest_documents(session.rag_files, force_reindex=True)
                if report.get("error"):
                    append_activity(session, f"Files ingest failed: {report['error']}")
                    result_box["error"] = report["error"]
                    return
                append_activity(
                    session,
                    f"Indexed {report.get('files_processed', 0)} file(s), {report.get('chunks_created', 0)} chunk(s), {report.get('vectors_upserted', 0)} vector(s).",
                )
                progress_state["last_event"] = (
                    f"Indexed {report.get('files_processed', 0)} file(s), "
                    f"{report.get('chunks_created', 0)} chunk(s), {report.get('vectors_upserted', 0)} vector(s)."
                )

            EXPORT_DIR.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if mode == "Document Engine":
                progress_state["stage"] = "Generating grounded document..."
                _append_chat(session, "user", prompt)
                result = controller.generate_document(
                    prompt,
                    target_words=controller.config.document_engine.target_words if controller.config.document_engine else 0,
                    audience=doc_audience,
                    tone=doc_tone,
                    mode=resolve_document_mode(doc_mode),
                    quality_mode=doc_quality,
                    resume=True,
                )
                if result.get("error"):
                    append_activity(session, f"Document generation failed: {result['error']}")
                    result_box["error"] = result["error"]
                    return
                result_box["stopped"] = bool(result.get("stopped"))
                preview = result.get("text", "")
                preview = preview[:2400] + ("\n\n... [truncated preview]" if len(preview) > 2400 else "")
                _append_chat(session, "assistant", preview or "No text was generated.")
                citations = result.get("citations", [])
                if citations:
                    citation_lines = [
                        f"- {os.path.basename(str(c.get('source', 'unknown')))} | page {c.get('page', '?')} | score {float(c.get('score', 0.0)):.3f}"
                        for c in citations[:5]
                    ]
                    _append_chat(session, "assistant", "Top references:\n" + "\n".join(citation_lines))
                pdf_path = EXPORT_DIR / f"document_{timestamp}.pdf"
                docx_path = EXPORT_DIR / f"document_{timestamp}.docx"
                controller.export_document_pdf(str(pdf_path))
                controller.export_document_docx(str(docx_path))
                session.latest_downloads["document_pdf"] = str(pdf_path)
                session.latest_downloads["document_docx"] = str(docx_path)
                result_box["pdf_path"] = str(pdf_path)
                result_box["docx_path"] = str(docx_path)
                if result_box["stopped"]:
                    result_box["status"] = "Stopped early. Partial document export is ready."
                else:
                    result_box["status"] = "Document generated. PDF and DOCX downloads are ready."
            elif mode == "Quick Q&A":
                progress_state["stage"] = "Retrieving context and drafting answer..."
                question_for_model = _question_with_recent_chat(session, prompt)
                _append_chat(session, "user", prompt)
                result = controller.ask_files(question_for_model)
                if result.get("error"):
                    append_activity(session, f"Quick Q&A failed: {result['error']}")
                    result_box["error"] = result["error"]
                    return
                _append_chat(session, "assistant", result.get("answer", "No answer returned."))
                citations = result.get("citations", [])
                formatted_citations = _format_citations(citations)
                if formatted_citations:
                    _append_chat(session, "assistant", formatted_citations)
                register_runtime_controller(session, "files_chat", controller)
                result_box["status"] = "Grounded answer ready."
            else:
                progress_state["stage"] = "Building structured JSON..."
                progress_state["target"] = int(num_rows or 10) if json_mode != "Exhaustive Extraction" else 0
                _append_chat(session, "user", f"Structured JSON -> {os.path.basename(json_template_path)} :: {json_target_key}")
                if json_mode == "Exhaustive Extraction":
                    result = controller.generate_exhaustive_extraction(
                        json_template_path,
                        json_target_key,
                        source_filter=source_filter or None,
                        on_progress=handle_progress,
                    )
                else:
                    result = controller.generate_json_batch(
                        json_template_path,
                        json_target_key,
                        num_items=int(num_rows or 10),
                        clear_existing=bool(json_clear_existing),
                    )
                if result.get("error"):
                    append_activity(session, f"Structured JSON failed: {result['error']}")
                    result_box["error"] = result["error"]
                    return
                result_box["stopped"] = bool(controller.stop_requested)
                preview = json.dumps(result, indent=2, ensure_ascii=False)
                preview = preview[:2400] + ("\n\n... [truncated preview]" if len(preview) > 2400 else "")
                _append_chat(session, "assistant", preview)
                try:
                    item_count = len(resolve_target_array(result, json_target_key))
                except Exception:
                    item_count = 0
                json_path = EXPORT_DIR / f"structured_json_{timestamp}.json"
                json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
                session.latest_downloads["structured_json"] = str(json_path)
                result_box["json_path"] = str(json_path)
                if result_box["stopped"]:
                    result_box["status"] = f"Stopped early. Partial structured JSON is ready with **{item_count}** item(s)."
                else:
                    result_box["status"] = f"Structured JSON ready with **{item_count}** item(s)."

            for line in logs[-8:]:
                append_activity(session, line)
        except Exception as exc:
            append_activity(session, f"Files task error: {exc}")
            result_box["exception"] = exc
        finally:
            clear_runtime_controller(session, "files", controller)
            if mode != "Quick Q&A" or result_box.get("error") or result_box.get("exception"):
                clear_runtime_controller(session, "files_chat", controller)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while thread.is_alive():
        yield (
            session,
            chat_markdown(session),
            f"{files_mode_helper(mode)}\n\nCurrent files: **{len(session.rag_files)}**",
            _files_progress_markdown(
                mode=mode,
                backend=effective_backend,
                stage=progress_state["stage"],
                done=progress_state["done"],
                target=progress_state["target"],
                last_event=progress_state["last_event"],
                started_at=started_at,
                live_logs=logs,
                is_running=True,
            ),
            result_box["pdf_path"],
            result_box["docx_path"],
            result_box["json_path"],
            _combined_activity_markdown(session, logs),
        )
        time.sleep(0.75)

    if result_box["exception"] is not None:
        yield (
            session,
            chat_markdown(session),
            "Files task failed. Check the activity log.",
            _files_progress_markdown(
                mode=mode,
                backend=effective_backend,
                stage="Failed",
                done=progress_state["done"],
                target=progress_state["target"],
                last_event=str(result_box["exception"]),
                started_at=started_at,
                live_logs=logs,
                is_running=False,
            ),
            None,
            None,
            None,
            activity_markdown(session),
        )
        return

    if result_box["error"]:
        yield (
            session,
            chat_markdown(session),
            result_box["error"],
            _files_progress_markdown(
                mode=mode,
                backend=effective_backend,
                stage="Failed",
                done=progress_state["done"],
                target=progress_state["target"],
                last_event=str(result_box["error"]),
                started_at=started_at,
                live_logs=logs,
                is_running=False,
            ),
            None,
            None,
            None,
            activity_markdown(session),
        )
        return

    yield (
        session,
        chat_markdown(session),
        result_box["status"] or "Files task finished.",
        _files_progress_markdown(
            mode=mode,
            backend=effective_backend,
            stage="Stopped" if result_box["stopped"] else "Completed",
            done=progress_state["done"],
            target=progress_state["target"],
            last_event=result_box["status"] or progress_state["last_event"],
            started_at=started_at,
            live_logs=logs,
            is_running=False,
        ),
        result_box["pdf_path"],
        result_box["docx_path"],
        result_box["json_path"],
        activity_markdown(session),
    )


def ask_quick_qa_chat(
    session: WebSessionState,
    question: str,
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
    question = str(question or "").strip()
    if not question:
        yield session, chat_markdown(session), "Ask a question about the uploaded files.", "Files progress will appear here after you run a task.", activity_markdown(session)
        return
    if not session.rag_files:
        yield session, chat_markdown(session), "Upload or add at least one file first.", "Files progress will appear here after you run a task.", activity_markdown(session)
        return

    effective_backend = resolve_effective_rag_backend("Quick Q&A", rag_backend, quick_qa_mode)
    controller = get_runtime_controller(session, "files_chat")
    logs: list[str] = []
    reused_index = controller is not None
    if controller is None:
        controller, logs = _build_files_controller(
            session,
            mode="Quick Q&A",
            model_id=model_id,
            provider=provider,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            input_price_per_1m=input_price_per_1m,
            output_price_per_1m=output_price_per_1m,
            num_rows=num_rows,
            similarity_threshold=similarity_threshold,
            max_retries=max_retries,
            rag_backend=rag_backend,
            collection_name=collection_name,
            top_k=top_k,
            min_score=min_score,
            max_context_chars=max_context_chars,
            embedding_model=embedding_model,
            source_filter=source_filter,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            ocr_mode=ocr_mode,
            ocr_dpi=ocr_dpi,
            ocr_max_pages=ocr_max_pages,
            ocr_max_regions_per_page=ocr_max_regions_per_page,
            ocr_region_padding_px=ocr_region_padding_px,
            ocr_gap_multiplier=ocr_gap_multiplier,
            ocr_min_extracted_chars=ocr_min_extracted_chars,
            ocr_timeout_ms_per_page=ocr_timeout_ms_per_page,
            parser_mode=parser_mode,
            hybrid_search_enabled=hybrid_search_enabled,
            rerank_enabled=rerank_enabled,
            summary_first_enabled=summary_first_enabled,
            summary_top_k=summary_top_k,
            dense_top_k=dense_top_k,
            lexical_top_k=lexical_top_k,
            parent_context_enabled=parent_context_enabled,
            parent_context_max_chars=parent_context_max_chars,
            graph_enabled=graph_enabled,
            graph_hops=graph_hops,
            graph_source_boost=graph_source_boost,
            late_interaction_enabled=late_interaction_enabled,
            late_interaction_weight=late_interaction_weight,
            quick_qa_mode=quick_qa_mode,
            doc_mode=doc_mode,
            doc_pages=doc_pages,
            doc_quality=doc_quality,
            doc_audience=doc_audience,
            doc_tone=doc_tone,
            doc_chart_enabled=doc_chart_enabled,
            doc_flow_enabled=doc_flow_enabled,
            doc_max_charts=doc_max_charts,
        )
        record_runtime_collection(collection_name, qdrant_url)

    started_at = time.time()
    question_for_model = _question_with_recent_chat(session, question)
    _append_chat(session, "user", question)
    stage = "Using indexed files..." if reused_index else "Indexing files for chat..."

    yield (
        session,
        chat_markdown(session),
        "Quick Q&A is working on your question.",
        _files_progress_markdown(
            mode="Quick Q&A",
            backend=effective_backend,
            stage=stage,
            done=0,
            target=1,
            last_event=stage,
            started_at=started_at,
            live_logs=logs,
            is_running=True,
        ),
        activity_markdown(session),
    )

    try:
        if not reused_index:
            report = controller.ingest_documents(session.rag_files, force_reindex=True)
            if report.get("error"):
                _append_chat(session, "assistant", f"I could not read the uploaded files: {report['error']}")
                append_activity(session, f"Quick Q&A ingest failed: {report['error']}")
                yield session, chat_markdown(session), "File indexing failed.", "Files progress will appear here after you run a task.", activity_markdown(session)
                return
            append_activity(
                session,
                f"Indexed {report.get('files_processed', 0)} file(s), {report.get('chunks_created', 0)} chunk(s), {report.get('vectors_upserted', 0)} vector(s).",
            )

        result = controller.ask_files(question_for_model)
        if result.get("error"):
            _append_chat(session, "assistant", f"I could not answer that from the uploaded files: {result['error']}")
            append_activity(session, f"Quick Q&A failed: {result['error']}")
            yield session, chat_markdown(session), "Quick Q&A failed.", "Files progress will appear here after you run a task.", activity_markdown(session)
            return

        _append_chat(session, "assistant", result.get("answer", "No answer returned."))
        formatted_citations = _format_citations(result.get("citations", []))
        if formatted_citations:
            _append_chat(session, "assistant", formatted_citations)
        register_runtime_controller(session, "files_chat", controller)
        append_activity(session, "Quick Q&A answer ready.")
        yield (
            session,
            chat_markdown(session),
            "Quick Q&A answer ready. Ask another question when ready.",
            _files_progress_markdown(
                mode="Quick Q&A",
                backend=effective_backend,
                stage="Completed",
                done=1,
                target=1,
                last_event="Answer ready.",
                started_at=started_at,
                live_logs=logs,
                is_running=False,
            ),
            activity_markdown(session),
        )
    except Exception as exc:
        clear_runtime_controller(session, "files_chat", controller)
        _append_chat(session, "assistant", f"Quick Q&A failed: {exc}")
        append_activity(session, f"Quick Q&A error: {exc}")
        yield session, chat_markdown(session), "Quick Q&A failed.", "Files progress will appear here after you run a task.", activity_markdown(session)
