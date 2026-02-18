"""
Generation-related event handlers for FletApp.

Handles: magic schema generation, start/stop generation, model refresh, connection test.
"""
import asyncio
import flet as ft

from core.models import (
    GeneratorConfig,
    ColumnDefinition,
    ColumnType,
    ColumnConstraints,
    AIProvider,
    RagConfig,
    DocumentEngineConfig,
)
from gui.controls.column_card import ColumnControl
from gui.utils import Dialogs


class GenerationHandlersMixin:
    """Mixin providing generation handlers for FletApp."""

    def _build_runtime_config(self, include_existing_data: bool = True) -> GeneratorConfig:
        rag_config = self._build_rag_config() if hasattr(self, "_build_rag_config") else RagConfig()
        doc_cfg = DocumentEngineConfig(
            mode=(self.doc_mode_dropdown.value or "hybrid") if hasattr(self, "doc_mode_dropdown") else "hybrid",
            target_words=int(self.doc_target_words_field.value or 1400) if hasattr(self, "doc_target_words_field") else 1400,
            audience=(self.doc_audience_field.value or "General") if hasattr(self, "doc_audience_field") else "General",
            tone=(self.doc_tone_field.value or "professional") if hasattr(self, "doc_tone_field") else "professional",
        )
        return GeneratorConfig(
            model_id=self.model_dropdown.value or "local-model",
            provider=AIProvider(self.provider_dropdown.value),
            api_key=self.api_key_field.value if self.provider_dropdown.value != AIProvider.LM_STUDIO.value else None,
            input_price_per_1m=float(self.input_price_field.value or 0),
            output_price_per_1m=float(self.output_price_field.value or 0),
            num_rows=int(self.rows_field.value),
            similarity_threshold=float(self.sim_threshold_field.value),
            max_retries=int(self.max_retries_field.value),
            existing_data=self.imported_data if include_existing_data else None,
            rag=rag_config,
            document_engine=doc_cfg,
        )

    def _refresh_models(self, e):
        async def task():
            try:
                from core.llm_client import LLMClient
                def fetch_models():
                    temp_config = GeneratorConfig(model_id="temp")
                    client = LLMClient(temp_config)
                    return client.list_models()
                
                models = await asyncio.to_thread(fetch_models)
                
                options = [ft.dropdown.Option(m) for m in models] if models else []
                self.model_dropdown.options = options
                if options:
                    self.model_dropdown.value = options[0].key
                Dialogs.show_snackbar(self.page, f"Found {len(models)} models")
                self.page.update()
            except Exception as ex:
                Dialogs.show_snackbar(self.page, f"Error: {ex}")
                self.page.update()
                
        self.page.run_task(task)

    def _test_connection(self, e):
        Dialogs.show_snackbar(self.page, "Connection test not implemented yet.")

    def _on_magic_generate(self, e):
        if hasattr(self, "active_workspace_tab") and self.active_workspace_tab == "files":
            if hasattr(self, "_on_files_magic_task"):
                self._on_files_magic_task(e)
            return

        prompt = self.magic_prompt.value
        if not prompt:
            Dialogs.show_snackbar(self.page, "Please describe your dataset first.")
            return
        
        self.magic_btn.disabled = True
        self.magic_btn.content = ft.Row([ft.Icon(ft.Icons.HOURGLASS_TOP), ft.Text("Generating... (10-20s)")], spacing=6)
        self.page.update()

        async def task():
            try:
                # Gather config values before entering the thread
                model_id = self.model_dropdown.value or "local-model"
                provider_val = self.provider_dropdown.value
                api_key_val = self.api_key_field.value
                az_ep = self.azure_endpoint.value
                az_dep = self.azure_deployment.value
                
                def run_magic():
                    from core.llm_client import LLMClient
                    config = GeneratorConfig(
                        model_id=model_id,
                        provider=AIProvider(provider_val),
                        api_key=api_key_val if provider_val != AIProvider.LM_STUDIO.value else None,
                        azure_endpoint=az_ep,
                        azure_deployment=az_dep
                    )
                    client = LLMClient(config)
                    
                    if not client.check_connection():
                        return None, "Error: Could not connect to LLM Provider."

                    # Prepare Context (Type Hinting & Sample Value)
                    context_str = None
                    if self.imported_data and len(self.imported_data) > 0:
                        sample_row = self.imported_data[0]
                        headers = list(sample_row.keys())
                        
                        context_lines = []
                        for h in headers:
                            val = sample_row[h]
                            type_hint = "String"
                            if isinstance(val, int): type_hint = "Integer"
                            elif isinstance(val, float): type_hint = "Float"
                            elif isinstance(val, bool): type_hint = "Boolean"
                            
                            context_lines.append(f"Column: {h} ({type_hint}) | Sample: {val}")
                        
                        context_str = "\n".join(context_lines)

                    schema_list = client.generate_schema(prompt, context=context_str)
                    return schema_list, None

                schema_list, error = await asyncio.to_thread(run_magic)
                
                if error:
                    self.controller.log(error)
                    Dialogs.show_snackbar(self.page, error)
                    return

                if not schema_list:
                    Dialogs.show_snackbar(self.page, "Magic Generator returned no columns. Check logs.")
                    return

                self.controller.log(f"Received {len(schema_list)} columns from LLM.")

                # Prepare Column Definitions
                new_col_defs = []
                for col_data in schema_list:
                    try:
                        type_str = col_data.get("type", "Short Text")
                        try:
                            col_type = ColumnType(type_str)
                        except ValueError:
                            col_type = ColumnType.SHORT_TEXT
                            
                        constraints_data = col_data.get("constraints", {})
                        
                        const_kwargs = {
                            "min_value": constraints_data.get("min_value"),
                            "max_value": constraints_data.get("max_value"),
                            "options": constraints_data.get("options", []),
                            "regex_pattern": constraints_data.get("regex_pattern"),
                            "faker_provider": constraints_data.get("faker_provider"),
                            "allow_duplicates": constraints_data.get("allow_duplicates", False)
                        }
                        
                        if constraints_data.get("min_length") is not None:
                            const_kwargs["min_length"] = int(constraints_data["min_length"])
                        if constraints_data.get("max_length") is not None:
                            const_kwargs["max_length"] = int(constraints_data["max_length"])
                            
                        constraints = ColumnConstraints(**const_kwargs)
                        
                        col_def = ColumnDefinition(
                            name=col_data.get("name", "untitled"),
                            type=col_type,
                            prompt_instruction=col_data.get("prompt_instruction", ""),
                            constraints=constraints
                        )
                        new_col_defs.append(col_def)
                    except Exception as e:
                        self.controller.log(f"Skipping invalid column spec: {e}")

                # Update UI — append new columns, don't clear existing
                added_count = 0
                for col_def in new_col_defs:
                    if any(c.name_field.value == col_def.name for c in self.columns):
                        continue
                        
                    col_ctrl = ColumnControl(self, index=len(self.columns), on_remove=self._remove_column, col_def=col_def)
                    self.columns.append(col_ctrl)
                    self.columns_list.controls.append(col_ctrl)
                    added_count += 1
                
                if added_count > 0:
                    Dialogs.show_snackbar(self.page, f"Magic! Added {added_count} new columns.")
                else:
                    Dialogs.show_snackbar(self.page, "No new columns added (duplicates or empty).")
                
                self.page.update()
                
            except Exception as ex:
                self.controller.log(f"Magic Error: {ex}")
                Dialogs.show_snackbar(self.page, f"Magic Error: {ex}")
            finally:
                self.magic_btn.disabled = False
                self.magic_btn.content = ft.Row([ft.Icon(ft.Icons.AUTO_AWESOME), ft.Text("Auto-Generate Schema")], spacing=6)
                self.page.update()
        
        self.page.run_task(task)

    def toggle_generation(self, e):
        print(f"DEBUG: toggle_generation called. is_generating={self.is_generating}")
        if self.is_generating:
            self.controller.stop_generation()
            self.start_btn.content = ft.Row(
                [ft.ProgressRing(width=16, height=16, stroke_width=2, color=ft.Colors.WHITE), ft.Text("Stopping...", color=ft.Colors.WHITE)],
                alignment=ft.MainAxisAlignment.CENTER
            )
            self.start_btn.disabled = True
            self.start_btn.update()
            self.page.update()
        else:
            try:
                columns = [c.get_definition() for c in self.columns]
                if not columns:
                    Dialogs.show_snackbar(self.page, "Add at least one column.")
                    return

                config = self._build_runtime_config(include_existing_data=True)
                self.controller.initialize(config, columns)
                self.controller.start_generation_thread()
                
                self.start_btn.style = ft.ButtonStyle(bgcolor=ft.Colors.RED_700, color=ft.Colors.WHITE)
                self.start_btn.content = ft.Row(
                    [ft.Icon(ft.Icons.STOP, color=ft.Colors.WHITE), ft.Text("STOP", color=ft.Colors.WHITE)],
                    alignment=ft.MainAxisAlignment.CENTER
                )
                self.start_btn.update()
                
                self.export_btn.disabled = True
                self.analyze_btn.disabled = True
                
                self.progress_bar.visible = True
                self.status_text.value = "Status: Generating..."
                self.status_text.color = ft.Colors.BLUE_400
                self.log_view.controls.clear()
                self.is_generating = True
            except Exception as ex:
                Dialogs.show_snackbar(self.page, f"Error: {ex}")
                self.status_text.value = "Status: Error"
                self.status_text.color = ft.Colors.RED_400
        self.page.update()
