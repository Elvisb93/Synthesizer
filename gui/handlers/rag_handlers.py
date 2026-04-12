"""RAG-related event handlers for FletApp."""
import asyncio
import json
import os
import queue

import flet as ft

from core.models import OcrMode, RagBackend, RagConfig
from gui.utils import Dialogs, pick_file, pick_files, save_file


class RagHandlersMixin:
    _PRESETS_FILE = ".rag_task_presets.json"
    _DOC_PRESET_BUNDLES = {
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

    def _load_rag_presets(self) -> None:
        default_presets = {
            "Summarize": "Summarize the main points from the imported files in bullet points.",
            "Action Items": "Extract key action items, owners, and deadlines from the imported files.",
            "Draft Reply": "Draft a concise response email based on the imported file context.",
        }
        self.rag_task_presets = default_presets

        if os.path.exists(self._PRESETS_FILE):
            try:
                with open(self._PRESETS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    cleaned = {str(k): str(v) for k, v in data.items() if str(k).strip() and str(v).strip()}
                    if cleaned:
                        self.rag_task_presets = cleaned
            except Exception:
                pass

        self._refresh_preset_options()

    def _save_rag_presets(self) -> None:
        with open(self._PRESETS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.rag_task_presets, f, indent=2)

    def _refresh_preset_options(self) -> None:
        names = sorted(self.rag_task_presets.keys())
        self.preset_dropdown.options = [ft.dropdown.Option(name) for name in names]
        if names and not self.preset_dropdown.value:
            self.preset_dropdown.value = names[0]

    def _drain_progress_queue(self) -> None:
        while not self.progress_queue.empty():
            try:
                self.progress_queue.get_nowait()
            except queue.Empty:
                break

    def _set_structured_json_status(self, message: str, *, color=ft.Colors.BLUE_400, progress: float | None = None, visible: bool = True) -> None:
        self.status_text.value = message
        self.status_text.color = color
        self.progress_bar.visible = visible
        if progress is not None:
            self.progress_bar.value = progress
        self.page.update()

    def _structured_json_button_label(self) -> str:
        json_mode = (self.json_mode_dropdown.value or "Standard Generation").strip()
        if json_mode == "Exhaustive Extraction":
            return "Extract JSON"
        return "Build JSON"

    def _on_json_mode_change(self, e) -> None:
        mode = (self.json_mode_dropdown.value or "Standard Generation").strip()
        exhaustive = mode == "Exhaustive Extraction"
        self.json_clear_existing_switch.disabled = exhaustive
        if exhaustive:
            self.json_clear_existing_switch.value = True
            self.json_template_helper.value = (
                "Exhaustive extraction processes every imported chunk into the selected target list. "
                "Import files first; the row count is ignored."
            )
        else:
            self.json_template_helper.value = (
                "Standard generation uses the row count and follows the inferred shape of the selected target list."
            )
        if (self.files_mode_dropdown.value or "").strip() == "Structured JSON":
            self.files_magic_btn.content = ft.Row([ft.Icon(ft.Icons.SMART_TOY), ft.Text(self._structured_json_button_label())], spacing=6)
        self.page.update()

    def _on_pick_json_template(self, e):
        async def task():
            path = await pick_file(
                title="Select JSON Template",
                filter_pairs=("JSON files", "*.json", "All files", "*.*"),
            )
            if path:
                self.json_template_path_field.value = path
                self.page.update()

        self.page.run_task(task)

    def _apply_doc_bundle(self, bundle_name: str) -> None:
        bundle = self._DOC_PRESET_BUNDLES.get(bundle_name)
        if not bundle:
            Dialogs.show_snackbar(self.page, f"Unknown preset bundle: {bundle_name}")
            return

        if hasattr(self, "doc_mode_dropdown"):
            self.doc_mode_dropdown.value = bundle.get("mode", "Balanced")
        if hasattr(self, "doc_pages_dropdown"):
            self.doc_pages_dropdown.value = bundle.get("pages", "Let AI decide")
        if hasattr(self, "doc_quality_dropdown"):
            self.doc_quality_dropdown.value = bundle.get("quality", "Fast")
        if hasattr(self, "doc_audience_field"):
            self.doc_audience_field.value = bundle.get("audience", "General")
        if hasattr(self, "doc_tone_field"):
            self.doc_tone_field.value = bundle.get("tone", "professional")
        if hasattr(self, "files_prompt") and bundle.get("prompt"):
            self.files_prompt.value = bundle["prompt"]

        Dialogs.show_snackbar(self.page, f"Applied preset: {bundle_name}")
        self.page.update()

    def _on_file_preset_change(self, e):
        name = self.preset_dropdown.value
        if name and name in self.rag_task_presets:
            self.files_prompt.value = self.rag_task_presets[name]
            self.preset_name_field.value = name
            self.page.update()

    def _on_save_file_preset(self, e):
        name = (self.preset_name_field.value or "").strip()
        body = (self.files_prompt.value or "").strip()
        if not name:
            Dialogs.show_snackbar(self.page, "Preset name is required.")
            return
        if not body:
            Dialogs.show_snackbar(self.page, "Preset text is empty.")
            return

        self.rag_task_presets[name] = body
        self._save_rag_presets()
        self._refresh_preset_options()
        self.preset_dropdown.value = name
        Dialogs.show_snackbar(self.page, f"Saved preset: {name}")
        self.page.update()

    def _on_delete_file_preset(self, e):
        name = (self.preset_dropdown.value or "").strip()
        if not name:
            Dialogs.show_snackbar(self.page, "Select a preset to delete.")
            return
        if name in self.rag_task_presets:
            del self.rag_task_presets[name]
            self._save_rag_presets()
            self.preset_dropdown.value = None
            self.preset_name_field.value = ""
            self._refresh_preset_options()
            Dialogs.show_snackbar(self.page, f"Deleted preset: {name}")
            self.page.update()

    def _build_rag_config(self) -> RagConfig:
        source_filter = (self.rag_source_filter_field.value or "").strip() or None
        effective_backend = self._resolve_effective_rag_backend()
        return RagConfig(
            enabled=True,
            backend=effective_backend,
            collection_name=(self.rag_collection_field.value or "synthesizer_default").strip(),
            top_k=int(self.rag_top_k_field.value or 5),
            min_score=float(self.rag_min_score_field.value or 0.25),
            max_context_chars=int(self.rag_max_context_chars_field.value or 3000),
            embedding_model=(self.rag_embedding_model_field.value or "BAAI/bge-small-en-v1.5").strip(),
            source_filter=source_filter,
            qdrant_url=(self.rag_qdrant_url_field.value or ":memory:").strip(),
            qdrant_api_key=(self.rag_qdrant_api_key_field.value or "").strip() or None,
            ocr_mode=OcrMode((self.rag_ocr_mode_dropdown.value or "off").lower()),
            ocr_dpi=int(self.rag_ocr_dpi_field.value or 150),
            ocr_max_pages=int(self.rag_ocr_max_pages_field.value or 20),
            ocr_max_regions_per_page=int(self.rag_ocr_max_regions_field.value or 8),
            ocr_region_padding_px=int(self.rag_ocr_padding_field.value or 18),
            ocr_gap_multiplier=float(self.rag_ocr_gap_multiplier_field.value or 2.5),
            ocr_min_extracted_chars=int(self.rag_ocr_min_chars_field.value or 60),
            ocr_timeout_ms_per_page=int(self.rag_ocr_timeout_field.value or 4000),
            parser_mode=(self.rag_parser_mode_dropdown.value or "auto").strip(),
            hybrid_search_enabled=bool(self.rag_hybrid_switch.value),
            rerank_enabled=bool(self.rag_rerank_switch.value),
            summary_first_enabled=bool(self.rag_summary_switch.value),
            summary_top_k=int(self.rag_summary_top_k_field.value or 3),
            dense_top_k=int(self.rag_dense_top_k_field.value or 12),
            lexical_top_k=int(self.rag_lexical_top_k_field.value or 12),
            parent_context_enabled=bool(self.rag_parent_ctx_switch.value),
            parent_context_max_chars=int(self.rag_parent_ctx_max_chars_field.value or 1200),
            graph_enabled=bool(self.rag_graph_switch.value),
            graph_hops=int(self.rag_graph_hops_field.value or 1),
            graph_source_boost=float(self.rag_graph_boost_field.value or 0.08),
            late_interaction_enabled=bool(self.rag_late_interaction_switch.value),
            late_interaction_weight=float(self.rag_late_interaction_weight_field.value or 0.2),
        )

    def _resolve_effective_rag_backend(self) -> RagBackend:
        selected = RagBackend(self.rag_backend_dropdown.value or RagBackend.LLAMA_INDEX.value)
        current_mode = (self.files_mode_dropdown.value or "").strip() if hasattr(self, "files_mode_dropdown") else ""
        active_tab = getattr(self, "active_workspace_tab", "")
        quick_qa_style = (self.quick_qa_backend_dropdown.value or "Broader Analysis").strip() if hasattr(self, "quick_qa_backend_dropdown") else "Broader Analysis"
        if active_tab == "files" and current_mode == "Quick Q&A" and quick_qa_style == "Pinpoint Quick":
            return RagBackend.NATIVE
        return selected

    def _build_runtime_signature(self):
        rag = self._build_rag_config()
        return (
            getattr(self, "active_workspace_tab", ""),
            (self.files_mode_dropdown.value or "Document Engine") if hasattr(self, "files_mode_dropdown") else "Document Engine",
            (self.quick_qa_backend_dropdown.value or "Broader Analysis") if hasattr(self, "quick_qa_backend_dropdown") else "Broader Analysis",
            self.model_dropdown.value or "local-model",
            self.provider_dropdown.value,
            self.api_key_field.value or "",
            self.azure_endpoint.value or "",
            self.azure_deployment.value or "",
            (self.doc_mode_dropdown.value or "Balanced") if hasattr(self, "doc_mode_dropdown") else "Balanced",
            (self.doc_pages_dropdown.value or "Let AI decide") if hasattr(self, "doc_pages_dropdown") else "Let AI decide",
            (self.doc_quality_dropdown.value or "Fast") if hasattr(self, "doc_quality_dropdown") else "Fast",
            (self.doc_audience_field.value or "General") if hasattr(self, "doc_audience_field") else "General",
            (self.doc_tone_field.value or "professional") if hasattr(self, "doc_tone_field") else "professional",
            bool(self.doc_chart_switch.value) if hasattr(self, "doc_chart_switch") else False,
            bool(self.doc_flow_switch.value) if hasattr(self, "doc_flow_switch") else True,
            (self.doc_max_charts_field.value or "3") if hasattr(self, "doc_max_charts_field") else "3",
            rag.backend.value,
            rag.collection_name,
            rag.top_k,
            rag.min_score,
            rag.max_context_chars,
            rag.embedding_model,
            rag.source_filter or "",
            rag.qdrant_url,
            rag.qdrant_api_key or "",
            rag.ocr_mode.value,
            rag.ocr_dpi,
            rag.ocr_max_pages,
            rag.ocr_max_regions_per_page,
            rag.ocr_region_padding_px,
            rag.ocr_gap_multiplier,
            rag.ocr_min_extracted_chars,
            rag.ocr_timeout_ms_per_page,
            rag.parser_mode,
            rag.hybrid_search_enabled,
            rag.rerank_enabled,
            rag.summary_first_enabled,
            rag.summary_top_k,
            rag.dense_top_k,
            rag.lexical_top_k,
            rag.parent_context_enabled,
            rag.parent_context_max_chars,
            rag.graph_enabled,
            rag.graph_hops,
            rag.graph_source_boost,
            rag.late_interaction_enabled,
            rag.late_interaction_weight,
        )

    def _sync_runtime_clients(self, force: bool = False) -> None:
        signature = self._build_runtime_signature()
        needs_init = (
            force
            or self.controller.llm_client is None
            or self.controller.rag_service is None
            or self._runtime_config_signature != signature
        )
        if not needs_init:
            return
        config = self._build_runtime_config(include_existing_data=True)
        self.controller.set_runtime_config(config)
        self._runtime_config_signature = signature

    def _refresh_files_view(self) -> None:
        self.files_list_view.controls.clear()
        if not self.rag_files:
            self.files_list_view.controls.append(ft.Text("No files imported yet.", color=ft.Colors.GREY_500))
            self.files_list_view.controls.append(
                ft.Text("Use 'Import File' to add PDFs, spreadsheets, text, HTML, docs, and images for search and tasks.", size=11, color=ft.Colors.GREY_500)
            )
            if hasattr(self, "files_intro_text"):
                self.files_intro_text.value = "Add files first, then choose whether you want a document, grounded answers, or structured JSON."
        else:
            for entry in self.rag_files:
                is_url = bool(str(entry.get("path", "")).lower().startswith(("http://", "https://")))
                self.files_list_view.controls.append(
                    ft.Row(
                        [
                            ft.Icon(ft.Icons.LINK if is_url else ft.Icons.DESCRIPTION_OUTLINED, size=16, color=ft.Colors.BLUE_300),
                            ft.Text(entry["name"], expand=True),
                            ft.Text(entry["status"], size=11, color=ft.Colors.GREY_400),
                            ft.IconButton(
                                icon=ft.Icons.REFRESH,
                                tooltip="Re-index file",
                                icon_size=16,
                                on_click=lambda e, p=entry["path"]: self._on_reindex_file(p),
                            ),
                            ft.IconButton(
                                icon=ft.Icons.DELETE_OUTLINE,
                                tooltip="Remove file",
                                icon_size=16,
                                on_click=lambda e, p=entry["path"]: self._on_remove_file(p),
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    )
                )
            if hasattr(self, "files_intro_text"):
                self.files_intro_text.value = (
                    f"{len(self.rag_files)} file(s) ready. Choose a task below, then describe the result you want."
                )

        self.files_count_text.value = f"Files ready: {len(self.rag_files)}"
        self.page.update()

    def _is_source_indexed(self, path: str) -> bool:
        try:
            if not self.controller.rag_service:
                return False
            return bool(self.controller.rag_service.store.has_source(path))
        except Exception:
            return False

    async def _import_file_for_rag(self):
        paths = await pick_files(
            title="Import Files for RAG",
            filter_pairs=(
                "Supported files",
                "*.pdf;*.txt;*.md;*.csv;*.json;*.xlsx;*.xls;*.html;*.htm;*.docx;*.png;*.jpg;*.jpeg;*.webp",
                "All files",
                "*.*",
            ),
        )
        if not paths:
            return

        self._sync_runtime_clients()
        Dialogs.show_snackbar(self.page, f"Adding {len(paths)} file(s) to the search workspace...")

        def run_ingest(selected_paths):
            return self.controller.ingest_documents(selected_paths)

        report = await asyncio.to_thread(run_ingest, paths)
        if report.get("error"):
            Dialogs.show_snackbar(self.page, report["error"])
            return

        errors = report.get("errors") or []
        if errors and not report.get("vectors_upserted"):
            Dialogs.show_snackbar(self.page, f"Could not add the files to search: {errors[0]}")
            return

        existing = {item["path"] for item in self.rag_files}
        for path in paths:
            if path not in existing and self._is_source_indexed(path):
                self.rag_files.append({"path": path, "name": os.path.basename(path), "status": "Indexed"})
        self._refresh_files_view()

        msg = (
            f"RAG ingest done: files={report.get('files_processed', 0)}, "
            f"chunks={report.get('chunks_created', 0)}, vectors={report.get('vectors_upserted', 0)}, "
            f"ocr_pages={report.get('ocr_pages_total', 0)}, ocr_regions={report.get('ocr_regions_total', 0)}"
        )
        Dialogs.show_snackbar(self.page, msg.replace("RAG ingest done", "Files added"))

    def _on_add_rag_url(self, e):
        async def task():
            url = (self.rag_url_field.value or "").strip()
            if not url:
                Dialogs.show_snackbar(self.page, "Enter a URL first.")
                return
            if not url.lower().startswith(("http://", "https://")):
                Dialogs.show_snackbar(self.page, "URL must start with http:// or https://")
                return

            self._sync_runtime_clients()
            Dialogs.show_snackbar(self.page, "Adding the web page to the search workspace...")

            def run_ingest():
                return self.controller.ingest_documents([url], force_reindex=True)

            report = await asyncio.to_thread(run_ingest)
            if report.get("error"):
                Dialogs.show_snackbar(self.page, report["error"])
                return
            errors = report.get("errors") or []
            if errors:
                Dialogs.show_snackbar(self.page, f"Could not add the web page: {errors[0]}")
                return

            existing = {item["path"] for item in self.rag_files}
            if url not in existing and self._is_source_indexed(url):
                self.rag_files.append({"path": url, "name": url, "status": "Indexed"})
            self._refresh_files_view()
            self.rag_url_field.value = ""
            self.page.update()
            Dialogs.show_snackbar(
                self.page,
                f"URL indexed: chunks={report.get('chunks_created', 0)}, vectors={report.get('vectors_upserted', 0)}",
            )

        self.page.run_task(task)

    def _append_file_chat(self, role: str, text: str) -> None:
        if (
            self.file_chat_view.controls
            and isinstance(self.file_chat_view.controls[0], ft.Text)
            and str(self.file_chat_view.controls[0].value).startswith("Results will appear here")
        ):
            self.file_chat_view.controls.clear()
        color = ft.Colors.CYAN_300 if role == "user" else ft.Colors.GREEN_300
        prefix = "You" if role == "user" else "Assistant"
        self.file_chat_view.controls.append(ft.Text(f"{prefix}: {text}", size=12, color=color))
        if len(self.file_chat_view.controls) > 150:
            self.file_chat_view.controls = self.file_chat_view.controls[-150:]
        self.file_chat_view.update()

    def _on_reindex_file(self, path: str):
        async def task():
            self._sync_runtime_clients()
            Dialogs.show_snackbar(self.page, f"Re-indexing {os.path.basename(path)}...")

            def run_reindex():
                return self.controller.ingest_documents([path], force_reindex=True)

            report = await asyncio.to_thread(run_reindex)
            if report.get("error"):
                Dialogs.show_snackbar(self.page, report["error"])
                return
            Dialogs.show_snackbar(self.page, f"Re-indexed {os.path.basename(path)}")
            self._refresh_files_view()

        self.page.run_task(task)

    def _on_remove_file(self, path: str):
        async def task():
            remaining = [f for f in self.rag_files if f["path"] != path]
            self.rag_files = remaining
            self._refresh_files_view()
            self._sync_runtime_clients()

            if not remaining:
                self.controller.clear_rag_collection()
                if hasattr(self, "_reset_file_chat_placeholder"):
                    self._reset_file_chat_placeholder()
                Dialogs.show_snackbar(self.page, "Removed the file and cleared the search index.")
                return

            Dialogs.show_snackbar(self.page, "Rebuilding the search index from the remaining files...")
            self.controller.clear_rag_collection()

            def run_rebuild(paths):
                return self.controller.ingest_documents(paths, force_reindex=True)

            report = await asyncio.to_thread(run_rebuild, [f["path"] for f in remaining])
            if report.get("error"):
                Dialogs.show_snackbar(self.page, report["error"])
                return

            Dialogs.show_snackbar(self.page, "File removed and search index rebuilt.")
            self._refresh_files_view()

        self.page.run_task(task)

    def _on_files_magic_task(self, e):
        prompt = (self.files_prompt.value or "").strip()
        mode = (self.files_mode_dropdown.value or "Document Engine").strip()

        if mode != "Structured JSON" and not prompt:
            Dialogs.show_snackbar(self.page, "Describe what you want the files to help with.")
            return

        if mode == "Quick Q&A" and not self.rag_files:
            Dialogs.show_snackbar(self.page, "Import at least one file before asking questions.")
            return
        if mode == "Structured JSON":
            template_path = (self.json_template_path_field.value or "").strip()
            target_path = (self.json_target_key_field.value or "").strip()
            if not template_path:
                Dialogs.show_snackbar(self.page, "Select a JSON template first.")
                return
            if not target_path:
                Dialogs.show_snackbar(self.page, "Enter the list key you want to fill first.")
                return
            if (self.json_mode_dropdown.value or "Standard Generation").strip() == "Exhaustive Extraction" and not self.rag_files:
                Dialogs.show_snackbar(self.page, "Import at least one file before running exhaustive extraction.")
                return

        self.files_magic_btn.disabled = True
        if mode == "Document Engine":
            self.files_magic_btn.content = ft.Row([ft.Icon(ft.Icons.HOURGLASS_TOP), ft.Text("Generating Document...")], spacing=6)
        elif mode == "Structured JSON":
            if (self.json_mode_dropdown.value or "Standard Generation").strip() == "Exhaustive Extraction":
                self.files_magic_btn.content = ft.Row([ft.Icon(ft.Icons.HOURGLASS_TOP), ft.Text("Extracting JSON...")], spacing=6)
            else:
                self.files_magic_btn.content = ft.Row([ft.Icon(ft.Icons.HOURGLASS_TOP), ft.Text("Building JSON...")], spacing=6)
        else:
            self.files_magic_btn.content = ft.Row([ft.Icon(ft.Icons.HOURGLASS_TOP), ft.Text("Searching Files...")], spacing=6)
        if mode != "Structured JSON":
            self._append_file_chat("user", prompt)
        else:
            template_name = os.path.basename((self.json_template_path_field.value or "").strip())
            target_path = (self.json_target_key_field.value or "").strip()
            self._append_file_chat("user", f"Structured JSON -> {template_name} :: {target_path}")
            self.progress_bar.visible = True
            self.progress_bar.value = 0
            self.status_text.value = "Status: Building structured JSON..."
            self.status_text.color = ft.Colors.BLUE_400
        self.page.update()

        async def task():
            try:
                self._sync_runtime_clients()
                self.controller.stop_requested = False

                if mode == "Document Engine":
                    target_words = self._resolve_document_target_words() if hasattr(self, "_resolve_document_target_words") else 1400
                    audience = (self.doc_audience_field.value or "General").strip()
                    tone = (self.doc_tone_field.value or "professional").strip()
                    strategy = self._resolve_document_mode() if hasattr(self, "_resolve_document_mode") else "hybrid"
                    quality_mode = (self.doc_quality_dropdown.value or "Fast").strip() if hasattr(self, "doc_quality_dropdown") else "Fast"

                    def run_document_generation():
                        return self.controller.generate_document(
                            prompt,
                            target_words=target_words,
                            audience=audience,
                            tone=tone,
                            mode=strategy,
                            quality_mode=quality_mode,
                            resume=True,
                        )

                    result = await asyncio.to_thread(run_document_generation)
                    if result.get("error"):
                        Dialogs.show_snackbar(self.page, result["error"])
                        return

                    full_text = result.get("text", "")
                    preview = full_text[:1800] + ("\n\n... [truncated preview]" if len(full_text) > 1800 else "")
                    self._append_file_chat("assistant", preview or "No text was generated.")
                    citations = result.get("citations", [])
                    if citations:
                        top = citations[:5]
                        lines = [
                            f"- {os.path.basename(str(c.get('source', 'unknown')))} | page {c.get('page', '?')} | score {float(c.get('score', 0.0)):.3f}"
                            for c in top
                        ]
                        self._append_file_chat("assistant", "Top References:\n" + "\n".join(lines))

                    done_msg = "Document generated. Use Export PDF or Export DOCX."
                    if result.get("stopped"):
                        done_msg = "Document generation stopped. You can run again to resume."
                    Dialogs.show_snackbar(self.page, done_msg)
                elif mode == "Quick Q&A":
                    def run_query():
                        return self.controller.ask_files(prompt)

                    result = await asyncio.to_thread(run_query)
                    if result.get("error"):
                        Dialogs.show_snackbar(self.page, result["error"])
                        return

                    answer = result.get("answer", "")
                    citations = result.get("citations", [])
                    self._append_file_chat("assistant", answer)
                    if citations:
                        top = citations[:5]
                        lines = [
                            f"- {os.path.basename(str(c.get('source', 'unknown')))} | page {c.get('page', '?')} | score {float(c.get('score', 0.0)):.3f}"
                            for c in top
                        ]
                        self._append_file_chat("assistant", "Citations:\n" + "\n".join(lines))
                else:
                    from core.json_parser import resolve_target_array

                    template_path = (self.json_template_path_field.value or "").strip()
                    target_path = (self.json_target_key_field.value or "").strip()
                    json_mode = (self.json_mode_dropdown.value or "Standard Generation").strip()
                    source_filter = (self.rag_source_filter_field.value or "").strip() or None

                    def run_structured_json():
                        if json_mode == "Exhaustive Extraction":
                            return self.controller.generate_exhaustive_extraction(
                                template_path,
                                target_path,
                                source_filter=source_filter,
                            )
                        return self.controller.generate_json_batch(
                            template_path,
                            target_path,
                            num_items=int(self.rows_field.value or 10),
                            clear_existing=bool(self.json_clear_existing_switch.value),
                        )

                    result = await asyncio.to_thread(run_structured_json)
                    self._drain_progress_queue()

                    if result.get("error"):
                        self.progress_bar.visible = False
                        self.status_text.value = "Status: Structured JSON needs attention"
                        self.status_text.color = ft.Colors.RED_400
                        Dialogs.show_snackbar(self.page, result["error"])
                        return

                    item_count = 0
                    try:
                        item_count = len(resolve_target_array(result, target_path))
                    except Exception:
                        item_count = 0

                    preview = json.dumps(result, indent=2, ensure_ascii=False)
                    if len(preview) > 1800:
                        preview = preview[:1800] + "\n\n... [truncated preview]"
                    self._append_file_chat("assistant", preview)
                    self.json_export_btn.visible = True
                    self.progress_bar.visible = False
                    self.progress_bar.value = 0
                    self.status_text.value = "Status: Structured JSON is ready"
                    self.status_text.color = ft.Colors.GREEN_400

                    if json_mode == "Exhaustive Extraction":
                        Dialogs.show_snackbar(self.page, f"Structured JSON extraction finished: {item_count} item(s).")
                    else:
                        Dialogs.show_snackbar(self.page, f"Structured JSON generation finished: {item_count} item(s).")
            except Exception as ex:
                if mode == "Structured JSON":
                    self.progress_bar.visible = False
                    self.status_text.value = "Status: Structured JSON needs attention"
                    self.status_text.color = ft.Colors.RED_400
                Dialogs.show_snackbar(self.page, f"File task error: {ex}")
            finally:
                self.files_magic_btn.disabled = False
                current_mode = (self.files_mode_dropdown.value or "").strip()
                if current_mode == "Document Engine":
                    label = "Generate Document"
                elif current_mode == "Quick Q&A":
                    label = "Ask About Files"
                else:
                    label = self._structured_json_button_label()
                self.files_magic_btn.content = ft.Row([ft.Icon(ft.Icons.SMART_TOY), ft.Text(label)], spacing=6)
                self.page.update()

        self.page.run_task(task)

    def _on_files_mode_change(self, e):
        mode = (self.files_mode_dropdown.value or "Document Engine").strip()
        self.file_assistant_mode = mode
        self.files_doc_mode_btn.style = ft.ButtonStyle(bgcolor=ft.Colors.BLUE_700 if mode == "Document Engine" else None)
        self.files_qa_mode_btn.style = ft.ButtonStyle(bgcolor=ft.Colors.BLUE_700 if mode == "Quick Q&A" else None)
        self.files_json_mode_btn.style = ft.ButtonStyle(bgcolor=ft.Colors.BLUE_700 if mode == "Structured JSON" else None)
        if mode == "Document Engine":
            self.files_mode_helper.value = "Draft reports and summaries that stay grounded in the files you imported."
            self.files_prompt.label = "What document should the files help you create?"
            self.files_prompt.hint_text = "e.g., Create a market analysis memo with recommendations and risks."
            self.files_prompt.disabled = False
            self.files_magic_btn.content = ft.Row([ft.Icon(ft.Icons.SMART_TOY), ft.Text("Generate Document")], spacing=6)
            self.document_settings_container.visible = True
            self.quick_qa_helper_container.visible = False
            self.json_template_container.visible = False
            self.json_export_btn.visible = False
            self.preset_dropdown.visible = True
            self.preset_name_field.visible = True
            self.preset_save_btn.visible = True
            self.preset_delete_btn.visible = True
        elif mode == "Quick Q&A":
            self.files_mode_helper.value = "Ask questions, summarize key points, or draft grounded responses from the imported files."
            self.files_prompt.label = "What do you want to ask or summarize?"
            self.files_prompt.hint_text = "e.g., Summarize key benefit requests and draft a response email."
            self.files_prompt.disabled = False
            self.files_magic_btn.content = ft.Row([ft.Icon(ft.Icons.SMART_TOY), ft.Text("Ask About Files")], spacing=6)
            self.document_settings_container.visible = False
            self.quick_qa_helper_container.visible = True
            self.json_template_container.visible = False
            self.json_export_btn.visible = False
            self.preset_dropdown.visible = True
            self.preset_name_field.visible = True
            self.preset_save_btn.visible = True
            self.preset_delete_btn.visible = True
        else:
            self.files_mode_helper.value = "Use a template to generate or extract a list of JSON items from your imported files."
            self.files_prompt.label = "Structured JSON is controlled by the template settings below."
            self.files_prompt.hint_text = "Select a template, choose the list key, then run generation or exhaustive extraction."
            self.files_prompt.disabled = True
            self.files_magic_btn.content = ft.Row([ft.Icon(ft.Icons.SMART_TOY), ft.Text(self._structured_json_button_label())], spacing=6)
            self.document_settings_container.visible = False
            self.quick_qa_helper_container.visible = False
            self.json_template_container.visible = True
            self.preset_dropdown.visible = False
            self.preset_name_field.visible = False
            self.preset_save_btn.visible = False
            self.preset_delete_btn.visible = False
            self._on_json_mode_change(None)
        if hasattr(self, "_refresh_quick_start_content"):
            self._refresh_quick_start_content()
        self.page.update()

    def _on_export_json_template(self, e):
        async def task():
            if not self.controller.json_template_result:
                Dialogs.show_snackbar(self.page, "Generate structured JSON first.")
                return
            path = await save_file(
                title="Export Structured JSON",
                default_name="generated_template.json",
                filter_pairs=("JSON files", "*.json", "All files", "*.*"),
            )
            if not path:
                return
            try:
                self.controller.export_json_template(path)
                Dialogs.show_snackbar(self.page, f"Structured JSON exported to {path}")
            except Exception as ex:
                Dialogs.show_snackbar(self.page, f"Export error: {ex}")

        self.page.run_task(task)

    def _on_export_document_pdf(self, e):
        async def task():
            if not self.controller.document_result:
                Dialogs.show_snackbar(self.page, "Generate a document first.")
                return
            path = await save_file(
                title="Export Document PDF",
                default_name="generated_document.pdf",
                filter_pairs=("PDF files", "*.pdf", "All files", "*.*"),
            )
            if not path:
                return
            try:
                self.controller.export_document_pdf(path)
                Dialogs.show_snackbar(self.page, f"Document exported to {path}")
            except Exception as ex:
                Dialogs.show_snackbar(self.page, f"Export error: {ex}")

        self.page.run_task(task)

    def _on_export_document_docx(self, e):
        async def task():
            if not self.controller.document_result:
                Dialogs.show_snackbar(self.page, "Generate a document first.")
                return
            path = await save_file(
                title="Export Document DOCX",
                default_name="generated_document.docx",
                filter_pairs=("DOCX files", "*.docx", "All files", "*.*"),
            )
            if not path:
                return
            try:
                self.controller.export_document_docx(path)
                Dialogs.show_snackbar(self.page, f"Document exported to {path}")
            except Exception as ex:
                Dialogs.show_snackbar(self.page, f"Export error: {ex}")

        self.page.run_task(task)

    def _on_rag_status(self, e):
        self._sync_runtime_clients()
        status = self.controller.get_rag_status()
        if not status.get("enabled"):
            Dialogs.show_snackbar(self.page, "File search is not configured.")
            return

        msg = (
            f"Search status: collection={status.get('collection_name', '')}, "
            f"vectors={status.get('collection_size', 0)}, top_k={status.get('top_k', 0)}, "
            f"min_score={status.get('min_score', 0)}, "
            f"ocr_mode={status.get('ocr_mode', 'off')}, parser_mode={status.get('parser_mode', 'auto')}, "
            f"hybrid={status.get('hybrid_search_enabled', True)}, rerank={status.get('rerank_enabled', True)}, "
            f"graph={status.get('graph_enabled', True)}({status.get('graph_sources', 0)} docs), "
            f"late_interaction={status.get('late_interaction_enabled', True)}, "
            f"dpi={status.get('ocr_dpi', 150)}"
        )
        Dialogs.show_snackbar(self.page, msg)

    def _on_rag_clear(self, e):
        self._sync_runtime_clients()
        try:
            self.controller.clear_rag_collection()
            self.rag_files.clear()
            if hasattr(self, "_reset_file_chat_placeholder"):
                self._reset_file_chat_placeholder()
            self._refresh_files_view()
            Dialogs.show_snackbar(self.page, "Search index cleared.")
        except Exception as ex:
            Dialogs.show_snackbar(self.page, f"Failed to clear the search index: {ex}")
