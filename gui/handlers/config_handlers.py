"""
Configuration-related event handlers for FletApp.

Handles: save config, load config, reset config, provider change.
"""
import json
from core.models import ColumnDefinition, AIProvider
from gui.utils import Dialogs, pick_file, save_file


class ConfigHandlersMixin:
    """Mixin providing configuration I/O handlers for FletApp."""

    def _on_provider_change(self, e):
        val = self.provider_dropdown.value
        if val == AIProvider.LM_STUDIO.value:
            self.api_key_field.disabled = True
            self.api_key_field.value = ""
            self.azure_row.visible = False
        elif val == AIProvider.AZURE_OPENAI.value:
            self.api_key_field.disabled = False
            self.azure_row.visible = True
        else:
            self.api_key_field.disabled = False
            self.azure_row.visible = False
        self.page.update()

    async def _on_save_config(self, e):
        path = await save_file(
            title="Save Configuration",
            default_name="config.json",
            filter_pairs=("JSON files", "*.json", "All files", "*.*")
        )
        if path:
            try:
                config_data = {
                    "model_id": self.model_dropdown.value,
                    "provider": self.provider_dropdown.value,
                    "api_key": self.api_key_field.value,
                    "azure_endpoint": self.azure_endpoint.value,
                    "azure_deployment": self.azure_deployment.value,
                    "input_price_per_1m": float(self.input_price_field.value),
                    "output_price_per_1m": float(self.output_price_field.value),
                    "num_rows": int(self.rows_field.value),
                    "similarity_threshold": float(self.sim_threshold_field.value),
                    "max_retries": int(self.max_retries_field.value),
                    "rag": {
                        "enabled": True,
                        "collection_name": self.rag_collection_field.value,
                        "top_k": int(self.rag_top_k_field.value),
                        "min_score": float(self.rag_min_score_field.value),
                        "max_context_chars": int(self.rag_max_context_chars_field.value),
                        "embedding_model": self.rag_embedding_model_field.value,
                        "source_filter": self.rag_source_filter_field.value,
                        "qdrant_url": self.rag_qdrant_url_field.value,
                        "qdrant_api_key": self.rag_qdrant_api_key_field.value,
                        "ocr_mode": self.rag_ocr_mode_dropdown.value,
                        "ocr_dpi": int(self.rag_ocr_dpi_field.value),
                        "ocr_max_pages": int(self.rag_ocr_max_pages_field.value),
                        "ocr_max_regions_per_page": int(self.rag_ocr_max_regions_field.value),
                        "ocr_region_padding_px": int(self.rag_ocr_padding_field.value),
                        "ocr_gap_multiplier": float(self.rag_ocr_gap_multiplier_field.value),
                        "ocr_min_extracted_chars": int(self.rag_ocr_min_chars_field.value),
                        "ocr_timeout_ms_per_page": int(self.rag_ocr_timeout_field.value),
                    },
                    "columns": [col.get_definition().model_dump() for col in self.columns]
                }
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=4)
                Dialogs.show_snackbar(self.page, f"Configuration saved to {path}")
            except Exception as ex:
                Dialogs.show_snackbar(self.page, f"Error saving config: {ex}")

    async def _on_load_config(self, e):
        path = await pick_file(
            title="Load Configuration",
            filter_pairs=("JSON files", "*.json", "All files", "*.*")
        )
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Restore config
                if "model_id" in data:
                    self.model_dropdown.value = data["model_id"]
                if "provider" in data:
                    self.provider_dropdown.value = data["provider"]
                    self._on_provider_change(None)
                if "api_key" in data:
                    self.api_key_field.value = data["api_key"]
                if "azure_endpoint" in data:
                    self.azure_endpoint.value = data["azure_endpoint"]
                if "azure_deployment" in data:
                    self.azure_deployment.value = data["azure_deployment"]
                if "input_price_per_1m" in data:
                    self.input_price_field.value = str(data["input_price_per_1m"])
                if "output_price_per_1m" in data:
                    self.output_price_field.value = str(data["output_price_per_1m"])
                if "num_rows" in data:
                    self.rows_field.value = str(data["num_rows"])
                if "similarity_threshold" in data:
                    self.sim_threshold_field.value = str(data["similarity_threshold"])
                if "max_retries" in data:
                    self.max_retries_field.value = str(data["max_retries"])

                rag = data.get("rag") or {}
                if rag:
                    self.rag_collection_field.value = rag.get("collection_name", "synthesizer_default")
                    self.rag_top_k_field.value = str(rag.get("top_k", 5))
                    self.rag_min_score_field.value = str(rag.get("min_score", 0.25))
                    self.rag_max_context_chars_field.value = str(rag.get("max_context_chars", 3000))
                    self.rag_embedding_model_field.value = rag.get("embedding_model", "BAAI/bge-small-en-v1.5")
                    self.rag_source_filter_field.value = rag.get("source_filter", "")
                    self.rag_qdrant_url_field.value = rag.get("qdrant_url", ":memory:")
                    self.rag_qdrant_api_key_field.value = rag.get("qdrant_api_key") or ""
                    self.rag_ocr_mode_dropdown.value = rag.get("ocr_mode", "off")
                    self.rag_ocr_dpi_field.value = str(rag.get("ocr_dpi", 150))
                    self.rag_ocr_max_pages_field.value = str(rag.get("ocr_max_pages", 20))
                    self.rag_ocr_max_regions_field.value = str(rag.get("ocr_max_regions_per_page", 8))
                    self.rag_ocr_padding_field.value = str(rag.get("ocr_region_padding_px", 18))
                    self.rag_ocr_gap_multiplier_field.value = str(rag.get("ocr_gap_multiplier", 2.5))
                    self.rag_ocr_min_chars_field.value = str(rag.get("ocr_min_extracted_chars", 60))
                    self.rag_ocr_timeout_field.value = str(rag.get("ocr_timeout_ms_per_page", 4000))

                self.rag_files = []
                self._refresh_files_view()

                # Restore columns
                if "columns" in data:
                    self.columns.clear()
                    self.columns_list.controls.clear()
                    for col_data in data["columns"]:
                        self._add_column(ColumnDefinition(**col_data))

                Dialogs.show_snackbar(self.page, "Configuration loaded!")
                self.page.update()
            except Exception as ex:
                Dialogs.show_snackbar(self.page, f"Error loading config: {ex}")

    def _on_reset_config(self, e):
        """Resets the configuration to defaults, keeping the model connection."""
        try:
            self.rows_field.value = "10"
            self.sim_threshold_field.value = "0.85"
            self.max_retries_field.value = "50"
            self.input_price_field.value = "0.15"
            self.output_price_field.value = "0.60"
            self.magic_prompt.value = ""
            self.rag_collection_field.value = "synthesizer_default"
            self.rag_top_k_field.value = "5"
            self.rag_min_score_field.value = "0.25"
            self.rag_max_context_chars_field.value = "3000"
            self.rag_embedding_model_field.value = "BAAI/bge-small-en-v1.5"
            self.rag_source_filter_field.value = ""
            self.rag_qdrant_url_field.value = ":memory:"
            self.rag_qdrant_api_key_field.value = ""
            self.rag_ocr_mode_dropdown.value = "off"
            self.rag_ocr_dpi_field.value = "150"
            self.rag_ocr_max_pages_field.value = "20"
            self.rag_ocr_max_regions_field.value = "8"
            self.rag_ocr_padding_field.value = "18"
            self.rag_ocr_gap_multiplier_field.value = "2.5"
            self.rag_ocr_min_chars_field.value = "60"
            self.rag_ocr_timeout_field.value = "4000"
            self.rag_files = []
            if hasattr(self, "file_chat_view"):
                self.file_chat_view.controls.clear()
            self._refresh_files_view()
            self.imported_data = None
            self.columns.clear()
            self.columns_list.controls.clear()
            self._add_column()
            Dialogs.show_snackbar(self.page, "Configuration reset to defaults.")
            self.page.update()
        except Exception as ex:
            Dialogs.show_snackbar(self.page, f"Error resetting config: {ex}")
