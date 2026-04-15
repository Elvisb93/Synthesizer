from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from core.app_config import build_generator_config, resolve_document_mode
from core.controller import GeneratorController
from core.json_parser import resolve_target_array
from web_ui.state import WebSessionState, activity_markdown, append_activity


EXPORT_DIR = Path(".web_ui_exports")

FILES_MODE_HELPERS = {
    "Document Engine": "Upload files, describe the document you want, then generate grounded output with PDF and DOCX downloads.",
    "Quick Q&A": "Upload files, ask a question, and get a grounded answer with citations.",
    "Structured JSON": "Upload files or provide a template, then build or extract JSON into the target list.",
}


def files_mode_helper(mode: str) -> str:
    return FILES_MODE_HELPERS.get(mode or "Document Engine", FILES_MODE_HELPERS["Document Engine"])


def files_mode_changed(mode: str):
    mode = (mode or "Document Engine").strip()
    is_doc = mode == "Document Engine"
    is_qa = mode == "Quick Q&A"
    is_json = mode == "Structured JSON"
    prompt_update = {
        "Document Engine": ("What document should the files help you create?", "e.g., Create an executive brief with findings, risks, and next steps.", True),
        "Quick Q&A": ("What do you want to ask or summarize?", "e.g., Summarize the main requests and draft a reply.", True),
        "Structured JSON": ("Structured JSON uses the template settings below.", "Select a template and target list, then run generation.", False),
    }[mode]
    return (
        files_mode_helper(mode),
        gr.update(label=prompt_update[0], placeholder=prompt_update[1], interactive=prompt_update[2]),
        gr.update(visible=is_doc),
        gr.update(visible=is_qa),
        gr.update(visible=is_json),
    )


def register_uploaded_files(session: WebSessionState, files: Any, mode: str):
    file_paths = files if isinstance(files, list) else ([files] if files else [])
    normalized = [str(path) for path in file_paths if path]
    session.rag_files = normalized
    session.files_mode = mode or "Document Engine"
    listing = pd.DataFrame(
        [{"name": os.path.basename(path), "path": path} for path in normalized],
        columns=["name", "path"],
    ) if normalized else pd.DataFrame(columns=["name", "path"])
    status = f"{files_mode_helper(mode)}\n\nCurrent files: **{len(normalized)}**"
    append_activity(session, f"Registered {len(normalized)} file(s) for Files workflow.")
    return session, listing, status, activity_markdown(session)


def add_url_source(session: WebSessionState, url: str, mode: str):
    url = (url or "").strip()
    if not url:
        return session, pd.DataFrame([{"name": os.path.basename(path), "path": path} for path in session.rag_files], columns=["name", "path"]) if session.rag_files else pd.DataFrame(columns=["name", "path"]), "Enter a URL first.", activity_markdown(session)
    if not url.lower().startswith(("http://", "https://")):
        return session, pd.DataFrame([{"name": os.path.basename(path), "path": path} for path in session.rag_files], columns=["name", "path"]) if session.rag_files else pd.DataFrame(columns=["name", "path"]), "URL must start with http:// or https://", activity_markdown(session)
    if url not in session.rag_files:
        session.rag_files.append(url)
        append_activity(session, f"Added web source: {url}")
    listing = pd.DataFrame(
        [{"name": (path if path.startswith("http") else os.path.basename(path)), "path": path} for path in session.rag_files],
        columns=["name", "path"],
    )
    return session, listing, f"{files_mode_helper(mode)}\n\nCurrent files: **{len(session.rag_files)}**", activity_markdown(session)


def clear_files(session: WebSessionState, mode: str):
    session.rag_files = []
    session.file_chat_history = []
    append_activity(session, "Cleared Files workspace sources.")
    return session, pd.DataFrame(columns=["name", "path"]), chat_markdown(session), f"{files_mode_helper(mode)}\n\nCurrent files: **0**", activity_markdown(session)


def _build_files_controller(
    session: WebSessionState,
    *,
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
            "rag_backend": rag_backend,
            "collection_name": collection_name,
            "top_k": int(top_k or 5),
            "min_score": min_score,
            "max_context_chars": int(max_context_chars or 3000),
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


def chat_markdown(session: WebSessionState) -> str:
    if not session.file_chat_history:
        return "Results will appear here after you run a Files task."
    blocks = []
    for item in session.file_chat_history:
        role = "You" if item.get("role") == "user" else "Assistant"
        blocks.append(f"**{role}**\n\n{item.get('content', '')}")
    return "\n\n---\n\n".join(blocks)


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
    json_template_path: str,
    json_target_key: str,
    json_mode: str,
    json_clear_existing: bool,
):
    mode = (mode or "Document Engine").strip()
    prompt = (prompt or "").strip()
    if mode != "Structured JSON" and not prompt:
        return session, chat_markdown(session), "Describe what you want from the files first.", None, None, None, activity_markdown(session)
    if mode in {"Document Engine", "Quick Q&A"} and not session.rag_files:
        return session, chat_markdown(session), "Upload or add at least one file first.", None, None, None, activity_markdown(session)
    if mode == "Structured JSON":
        if not json_template_path:
            return session, chat_markdown(session), "Select a JSON template first.", None, None, None, activity_markdown(session)
        if not json_target_key:
            return session, chat_markdown(session), "Enter the target list key first.", None, None, None, activity_markdown(session)
        if json_mode == "Exhaustive Extraction" and not session.rag_files:
            return session, chat_markdown(session), "Import at least one file before running exhaustive extraction.", None, None, None, activity_markdown(session)

    controller, logs = _build_files_controller(
        session,
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
        parser_mode=parser_mode,
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

    if session.rag_files:
        report = controller.ingest_documents(session.rag_files, force_reindex=True)
        if report.get("error"):
            append_activity(session, f"Files ingest failed: {report['error']}")
            return session, chat_markdown(session), report["error"], None, None, None, activity_markdown(session)
        append_activity(
            session,
            f"Indexed {report.get('files_processed', 0)} file(s), {report.get('chunks_created', 0)} chunk(s), {report.get('vectors_upserted', 0)} vector(s).",
        )

    EXPORT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = None
    docx_path = None
    json_path = None

    try:
        if mode == "Document Engine":
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
                return session, chat_markdown(session), result["error"], None, None, None, activity_markdown(session)
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
            status = "Document generated. PDF and DOCX downloads are ready."
        elif mode == "Quick Q&A":
            _append_chat(session, "user", prompt)
            result = controller.ask_files(prompt)
            if result.get("error"):
                append_activity(session, f"Quick Q&A failed: {result['error']}")
                return session, chat_markdown(session), result["error"], None, None, None, activity_markdown(session)
            _append_chat(session, "assistant", result.get("answer", "No answer returned."))
            citations = result.get("citations", [])
            if citations:
                citation_lines = [
                    f"- {os.path.basename(str(c.get('source', 'unknown')))} | page {c.get('page', '?')} | score {float(c.get('score', 0.0)):.3f}"
                    for c in citations[:5]
                ]
                _append_chat(session, "assistant", "Citations:\n" + "\n".join(citation_lines))
            status = "Grounded answer ready."
        else:
            _append_chat(session, "user", f"Structured JSON -> {os.path.basename(json_template_path)} :: {json_target_key}")
            if json_mode == "Exhaustive Extraction":
                result = controller.generate_exhaustive_extraction(
                    json_template_path,
                    json_target_key,
                    source_filter=source_filter or None,
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
                return session, chat_markdown(session), result["error"], None, None, None, activity_markdown(session)
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
            status = f"Structured JSON ready with **{item_count}** item(s)."

        for line in logs[-8:]:
            append_activity(session, line)
        return session, chat_markdown(session), status, str(pdf_path) if pdf_path else None, str(docx_path) if docx_path else None, str(json_path) if json_path else None, activity_markdown(session)
    except Exception as exc:
        append_activity(session, f"Files task error: {exc}")
        return session, chat_markdown(session), "Files task failed. Check the activity log.", None, None, None, activity_markdown(session)
