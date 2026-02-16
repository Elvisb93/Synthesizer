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
            self.imported_data = None
            self.columns.clear()
            self.columns_list.controls.clear()
            self._add_column()
            Dialogs.show_snackbar(self.page, "Configuration reset to defaults.")
            self.page.update()
        except Exception as ex:
            Dialogs.show_snackbar(self.page, f"Error resetting config: {ex}")
