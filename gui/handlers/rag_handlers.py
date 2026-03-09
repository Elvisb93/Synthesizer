"""RAG-related event handlers for FletApp."""
import asyncio
import json
import os

import flet as ft

from core.models import OcrMode, RagConfig
from gui.utils import Dialogs, pick_files, save_file


class RagHandlersMixin:
    _PRESETS_FILE = ".rag_task_presets.json"
    _DOC_PRESET_BUNDLES = {
        "Executive Brief": {
            "mode": "hybrid",
            "pages": "2 pages",
            "quality": "Fast",
            "audience": "Executives",
            "tone": "professional",
            "prompt": "Create an executive brief with key findings, risks, and recommended next steps.",
        },
        "Policy Draft": {
            "mode": "factual by doc",
            "pages": "5 pages",
            "quality": "Thorough",
            "audience": "Policy stakeholders",
            "tone": "formal",
            "prompt": "Draft a policy document based on the imported files, including scope, requirements, and governance.",
        },
        "Action Plan": {
            "mode": "hybrid",
            "pages": "3 pages",
            "quality": "Fast",
            "audience": "Implementation team",
            "tone": "direct",
            "prompt": "Create an action plan with phases, owners, milestones, and measurable success criteria.",
        },
        "Meeting Summary": {
            "mode": "factual by doc",
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

    def _apply_doc_bundle(self, bundle_name: str) -> None:
        bundle = self._DOC_PRESET_BUNDLES.get(bundle_name)
        if not bundle:
            Dialogs.show_snackbar(self.page, f"Unknown preset bundle: {bundle_name}")
            return

        if hasattr(self, "doc_mode_dropdown"):
            self.doc_mode_dropdown.value = bundle.get("mode", "hybrid")
        if hasattr(self, "doc_pages_dropdown"):
            self.doc_pages_dropdown.value = bundle.get("pages", "Let AI decide")
        if hasattr(self, "doc_quality_dropdown"):
            self.doc_quality_dropdown.value = bundle.get("quality", "Fast")
        if hasattr(self, "doc_audience_field"):
            self.doc_audience_field.value = bundle.get("audience", "General")
        if hasattr(self, "doc_tone_field"):
            self.doc_tone_field.value = bundle.get("tone", "professional")
        if hasattr(self, "magic_prompt") and bundle.get("prompt"):
            self.magic_prompt.value = bundle["prompt"]

        Dialogs.show_snackbar(self.page, f"Applied preset: {bundle_name}")
        self.page.update()

    def _on_file_preset_change(self, e):
        name = self.preset_dropdown.value
        if name and name in self.rag_task_presets:
            self.magic_prompt.value = self.rag_task_presets[name]
            self.preset_name_field.value = name
            self.page.update()

    def _on_save_file_preset(self, e):
        name = (self.preset_name_field.value or "").strip()
        body = (self.magic_prompt.value or "").strip()
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
        return RagConfig(
            enabled=True,
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

    def _build_runtime_signature(self):
        rag = self._build_rag_config()
        return (
            self.model_dropdown.value or "local-model",
            self.provider_dropdown.value,
            self.api_key_field.value or "",
            self.azure_endpoint.value or "",
            self.azure_deployment.value or "",
            (self.doc_mode_dropdown.value or "hybrid") if hasattr(self, "doc_mode_dropdown") else "hybrid",
            (self.doc_pages_dropdown.value or "Let AI decide") if hasattr(self, "doc_pages_dropdown") else "Let AI decide",
            (self.doc_quality_dropdown.value or "Fast") if hasattr(self, "doc_quality_dropdown") else "Fast",
            (self.doc_audience_field.value or "General") if hasattr(self, "doc_audience_field") else "General",
            (self.doc_tone_field.value or "professional") if hasattr(self, "doc_tone_field") else "professional",
            bool(self.doc_chart_switch.value) if hasattr(self, "doc_chart_switch") else False,
            bool(self.doc_flow_switch.value) if hasattr(self, "doc_flow_switch") else True,
            (self.doc_max_charts_field.value or "3") if hasattr(self, "doc_max_charts_field") else "3",
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
                ft.Text("Use 'Import File' to add PDFs, spreadsheets, text, HTML, docs, and images for retrieval and tasks.", size=11, color=ft.Colors.GREY_500)
            )
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

        self.files_count_text.value = f"Indexed files: {len(self.rag_files)}"
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
        Dialogs.show_snackbar(self.page, f"Indexing {len(paths)} file(s) for RAG...")

        def run_ingest(selected_paths):
            return self.controller.ingest_documents(selected_paths)

        report = await asyncio.to_thread(run_ingest, paths)
        if report.get("error"):
            Dialogs.show_snackbar(self.page, report["error"])
            return

        errors = report.get("errors") or []
        if errors and not report.get("vectors_upserted"):
            Dialogs.show_snackbar(self.page, f"RAG ingest failed: {errors[0]}")
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
        Dialogs.show_snackbar(self.page, msg)

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
            Dialogs.show_snackbar(self.page, "Indexing URL for RAG...")

            def run_ingest():
                return self.controller.ingest_documents([url], force_reindex=True)

            report = await asyncio.to_thread(run_ingest)
            if report.get("error"):
                Dialogs.show_snackbar(self.page, report["error"])
                return
            errors = report.get("errors") or []
            if errors:
                Dialogs.show_snackbar(self.page, f"RAG ingest failed: {errors[0]}")
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
                Dialogs.show_snackbar(self.page, "Removed file and cleared index.")
                return

            Dialogs.show_snackbar(self.page, "Rebuilding index from remaining files...")
            self.controller.clear_rag_collection()

            def run_rebuild(paths):
                return self.controller.ingest_documents(paths, force_reindex=True)

            report = await asyncio.to_thread(run_rebuild, [f["path"] for f in remaining])
            if report.get("error"):
                Dialogs.show_snackbar(self.page, report["error"])
                return

            Dialogs.show_snackbar(self.page, "File removed and index rebuilt.")
            self._refresh_files_view()

        self.page.run_task(task)

    def _on_files_magic_task(self, e):
        prompt = (self.magic_prompt.value or "").strip()
        if not prompt:
            Dialogs.show_snackbar(self.page, "Enter a file task or question.")
            return

        mode = (self.files_mode_dropdown.value or "Document Engine").strip()
        if mode == "Quick Q&A" and not self.rag_files:
            Dialogs.show_snackbar(self.page, "Import at least one file for Quick Q&A mode.")
            return

        self.magic_btn.disabled = True
        if mode == "Document Engine":
            self.magic_btn.content = ft.Row([ft.Icon(ft.Icons.HOURGLASS_TOP), ft.Text("Generating Document...")], spacing=6)
        else:
            self.magic_btn.content = ft.Row([ft.Icon(ft.Icons.HOURGLASS_TOP), ft.Text("Querying Files...")], spacing=6)
        self._append_file_chat("user", prompt)
        self.page.update()

        async def task():
            try:
                self._sync_runtime_clients()

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
                else:
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
            except Exception as ex:
                Dialogs.show_snackbar(self.page, f"File task error: {ex}")
            finally:
                self.magic_btn.disabled = False
                label = "Generate Document" if (self.files_mode_dropdown.value or "") == "Document Engine" else "Run File Task"
                self.magic_btn.content = ft.Row([ft.Icon(ft.Icons.SMART_TOY), ft.Text(label)], spacing=6)
                self.page.update()

        self.page.run_task(task)

    def _on_files_mode_change(self, e):
        mode = (self.files_mode_dropdown.value or "Document Engine").strip()
        self.file_assistant_mode = mode
        if mode == "Document Engine":
            self.magic_prompt.label = "Generate a long document or run file Q&A..."
            self.magic_prompt.hint_text = "e.g., 'Create a market analysis memo with recommendations and risks.'"
            self.magic_btn.content = ft.Row([ft.Icon(ft.Icons.SMART_TOY), ft.Text("Generate Document")], spacing=6)
            self.doc_export_pdf_btn.visible = True
            self.doc_export_docx_btn.visible = True
            if hasattr(self, "doc_chart_switch"):
                self.doc_chart_switch.visible = True
            if hasattr(self, "doc_flow_switch"):
                self.doc_flow_switch.visible = True
            if hasattr(self, "doc_max_charts_field"):
                self.doc_max_charts_field.visible = True
        else:
            self.magic_prompt.label = "Ask questions about imported files..."
            self.magic_prompt.hint_text = "e.g., 'Summarize key benefit requests and draft a response email.'"
            self.magic_btn.content = ft.Row([ft.Icon(ft.Icons.SMART_TOY), ft.Text("Run File Task")], spacing=6)
            self.doc_export_pdf_btn.visible = False
            self.doc_export_docx_btn.visible = False
            if hasattr(self, "doc_chart_switch"):
                self.doc_chart_switch.visible = False
            if hasattr(self, "doc_flow_switch"):
                self.doc_flow_switch.visible = False
            if hasattr(self, "doc_max_charts_field"):
                self.doc_max_charts_field.visible = False
        self.page.update()

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
            Dialogs.show_snackbar(self.page, "RAG is not configured.")
            return

        msg = (
            f"RAG status: collection={status.get('collection_name', '')}, "
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
            self.file_chat_view.controls.clear()
            self._refresh_files_view()
            Dialogs.show_snackbar(self.page, "RAG index cleared.")
        except Exception as ex:
            Dialogs.show_snackbar(self.page, f"Failed to clear RAG index: {ex}")
