"""
RAG-related event handlers for FletApp.

Handles: PDF ingestion and RAG status checks.
"""
import asyncio

from core.models import RagConfig
from gui.utils import Dialogs, pick_file


class RagHandlersMixin:
    def _build_rag_config(self) -> RagConfig:
        return RagConfig(
            enabled=bool(self.rag_enabled_switch.value),
            collection_name=(self.rag_collection_field.value or "synthesizer_default").strip(),
            top_k=int(self.rag_top_k_field.value or 5),
            min_score=float(self.rag_min_score_field.value or 0.25),
            max_context_chars=int(self.rag_max_context_chars_field.value or 3000),
            embedding_model=(self.rag_embedding_model_field.value or "BAAI/bge-small-en-v1.5").strip(),
            source_filter=(self.rag_source_filter_field.value or "").strip() or None,
            qdrant_url=(self.rag_qdrant_url_field.value or "http://localhost:6333").strip(),
            qdrant_api_key=(self.rag_qdrant_api_key_field.value or "").strip() or None,
        )

    def _sync_controller_rag_config(self) -> None:
        self.controller.config.rag = self._build_rag_config()
        self.controller.initialize_rag()

    def _on_rag_toggle(self, e):
        enabled = bool(self.rag_enabled_switch.value)
        self.rag_config_block.visible = enabled
        self._sync_controller_rag_config()
        self.page.update()

    def _on_rag_ingest(self, e):
        async def task():
            if not self.rag_enabled_switch.value:
                Dialogs.show_snackbar(self.page, "Enable RAG first.")
                return

            self._sync_controller_rag_config()

            path = await pick_file(
                title="Ingest PDF for RAG",
                filter_pairs=("PDF files", "*.pdf", "All files", "*.*"),
            )
            if not path:
                return

            Dialogs.show_snackbar(self.page, "Ingesting PDF. This may take a moment...")

            def run_ingest():
                return self.controller.ingest_documents([path])

            report = await asyncio.to_thread(run_ingest)
            if report.get("error"):
                Dialogs.show_snackbar(self.page, report["error"])
                return

            msg = (
                f"RAG ingest done: files={report.get('files_processed', 0)}, "
                f"skipped={report.get('files_skipped', 0)}, "
                f"chunks={report.get('chunks_created', 0)}, "
                f"vectors={report.get('vectors_upserted', 0)}"
            )
            Dialogs.show_snackbar(self.page, msg)

        self.page.run_task(task)

    def _on_rag_status(self, e):
        self._sync_controller_rag_config()
        status = self.controller.get_rag_status()
        if not status.get("enabled"):
            Dialogs.show_snackbar(self.page, "RAG is disabled.")
            return

        msg = (
            f"RAG status: collection={status.get('collection_name', '')}, "
            f"vectors={status.get('collection_size', 0)}, "
            f"top_k={status.get('top_k', 0)}, "
            f"min_score={status.get('min_score', 0)}, "
            f"source_filter={status.get('source_filter') or 'none'}"
        )
        Dialogs.show_snackbar(self.page, msg)

    def _on_rag_clear(self, e):
        self._sync_controller_rag_config()
        status = self.controller.get_rag_status()
        if not status.get("enabled"):
            Dialogs.show_snackbar(self.page, "RAG is disabled.")
            return

        try:
            self.controller.clear_rag_collection()
            Dialogs.show_snackbar(self.page, "RAG index cleared.")
        except Exception as ex:
            Dialogs.show_snackbar(self.page, f"Failed to clear RAG index: {ex}")
