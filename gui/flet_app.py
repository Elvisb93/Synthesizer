import flet as ft
import asyncio
import json
import queue
import time
from typing import List, Any, Dict

# Need pandas for import
try:
    import pandas as pd
except ImportError:
    pd = None

from core.models import GeneratorConfig, ColumnDefinition, ColumnType, ColumnConstraints, AIProvider
from core.controller import GeneratorController
from gui.controls.column_card import ColumnControl
from gui.utils import Dialogs

class FletApp:
    def __init__(self, page: ft.Page, controller: GeneratorController):
        self.page = page
        self.controller = controller
        self.page.title = "Synthetic Data Generator"
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.padding = 20
        self.page.scroll = ft.ScrollMode.AUTO

        # State
        self.columns: List['ColumnControl'] = []
        self.log_queue = queue.Queue()
        self.progress_queue = queue.Queue()
        self.is_generating = False
        self.imported_data: List[Dict[str, Any]] = None

        # Init UI
        self._setup_ui()
        self._init_controller_callbacks()
        
        # Initial Log
        self.log_queue.put("System Ready. Logs will appear here...")

    def _setup_ui(self):
        # === TOOLBAR / ACTIONS ===
        self.save_btn = ft.ElevatedButton("Save Config", icon=ft.Icons.SAVE, on_click=self._on_save_config)
        self.load_btn = ft.ElevatedButton("Load Config", icon=ft.Icons.UPLOAD_FILE, on_click=self._on_load_config)
        self.import_btn = ft.ElevatedButton("Import Data", icon=ft.Icons.TABLE_CHART, on_click=self._on_import_data)
        self.reset_btn = ft.ElevatedButton("Reset Config", icon=ft.Icons.RESTART_ALT, on_click=self._on_reset_config, style=ft.ButtonStyle(color=ft.Colors.RED_400))
        self.help_btn = ft.IconButton(icon=ft.Icons.HELP_OUTLINE, tooltip="Help / Docs", on_click=self._show_help)

        # === MODEL CONFIGURATION ===
        self.model_dropdown = ft.Dropdown(
            label="Model ID",
            options=[ft.dropdown.Option("local-model")],
            width=280,
            value="local-model",
            dense=True
        )
        self.refresh_models_btn = ft.IconButton(
            icon=ft.Icons.REFRESH,
            on_click=self._refresh_models,
            tooltip="Refresh Models"
        )

        # === GENERATION SETTINGS ===
        self.rows_field = ft.TextField(label="Rows", value="10", width=100, dense=True)
        self.sim_threshold_field = ft.TextField(label="Similarity Threshold", value="0.85", width=140, dense=True)
        self.max_retries_field = ft.TextField(label="Max Retries", value="50", width=120, dense=True)

        # === AI PROVIDER CONFIG ===
        self.provider_dropdown = ft.Dropdown(
            label="Provider",
            options=[ft.dropdown.Option(p.value) for p in AIProvider],
            value=AIProvider.LM_STUDIO.value,
            width=180,
            dense=True
        )
        self.provider_dropdown.on_change = self._on_provider_change

        self.api_key_field = ft.TextField(label="API Key", password=True, width=280, disabled=True, dense=True)
        self.test_connection_btn = ft.ElevatedButton("Test Connection", on_click=self._test_connection)

        # Azure-specific fields
        self.azure_endpoint = ft.TextField(label="Azure Endpoint", width=280, dense=True)
        self.azure_deployment = ft.TextField(label="Deployment Name", width=180, dense=True)
        
        self.azure_hints = ft.Column([
            ft.Text("(e.g., https://your-resource.openai.azure.com)", size=12, color=ft.Colors.GREY_500),
            ft.Text("(e.g., gpt-4, gpt-35-turbo)", size=12, color=ft.Colors.GREY_500)
        ], visible=False)

        self.azure_row = ft.Row([
            ft.Column([self.azure_endpoint, ft.Text("(e.g., https://your-resource.openai.azure.com)", size=12, color=ft.Colors.GREY_500)]),
            ft.Column([self.azure_deployment, ft.Text("(e.g., gpt-4)", size=12, color=ft.Colors.GREY_500)])
        ], visible=False, wrap=True)
        
        # === COST & METRICS CONFIG ===
        self.input_price_field = ft.TextField(label="Input Price ($/1M)", value="0.15", width=120, dense=True)
        self.output_price_field = ft.TextField(label="Output Price ($/1M)", value="0.60", width=120, dense=True)

        self.cost_config_row = ft.Row([
            self.input_price_field,
            self.output_price_field,
            ft.Text("Approx prices for GPT-4o-mini", size=10, color=ft.Colors.GREY_500, italic=True)
        ], alignment=ft.MainAxisAlignment.START, visible=True)

        # === MAGIC GENERATOR ===
        self.magic_prompt = ft.TextField(
            label="Describe your dataset...",
            hint_text="e.g., 'Customer database with names, emails, phone numbers, and purchase history'",
            expand=True,
            multiline=True,
            min_lines=1,
            max_lines=2,
            dense=True
        )
        self.magic_btn = ft.ElevatedButton(
            "Auto-Generate Schema",
            icon=ft.Icons.AUTO_AWESOME,
            on_click=self._on_magic_generate
        )

        # === COLUMNS ===
        self.columns_list = ft.Column(spacing=8)
        self.add_col_btn = ft.OutlinedButton("+ Add Column", on_click=lambda e: self._add_column())

        # === ACTION BAR ===
        self.start_btn = ft.ElevatedButton(
            content=ft.Row(
                [ft.Icon(ft.Icons.PLAY_ARROW, color=ft.Colors.WHITE), ft.Text("Start Generation", color=ft.Colors.WHITE)],
                alignment=ft.MainAxisAlignment.CENTER
            ),
            style=ft.ButtonStyle(bgcolor=ft.Colors.GREEN_700, color=ft.Colors.WHITE),
            on_click=self.toggle_generation
        )
        self.export_btn = ft.PopupMenuButton(
            content=ft.Row([ft.Icon(ft.Icons.DOWNLOAD, size=18), ft.Text("Export")], spacing=5),
            items=[
                ft.PopupMenuItem(content=ft.Text("Export CSV"), on_click=lambda e: self.export_data("csv")),
                ft.PopupMenuItem(content=ft.Text("Export JSON"), on_click=lambda e: self.export_data("json")),
                ft.PopupMenuItem(content=ft.Text("Export SQL"), on_click=lambda e: self.export_data("sql")),
                ft.PopupMenuItem(content=ft.Text("Export PDF Report"), on_click=lambda e: self.export_data("pdf_report")),
                ft.PopupMenuItem(content=ft.Text("Export PDF Narrative"), on_click=lambda e: self.export_data("pdf_narrative")),
            ],
            disabled=False,
            tooltip="Export Data (Generate first)"
        )
        self.analyze_btn = ft.ElevatedButton("Analyze Quality", icon=ft.Icons.ANALYTICS, on_click=self._on_analyze, disabled=True)
        
        # Status & Progress
        self.status_text = ft.Text("Status: Ready", color=ft.Colors.GREY_400, size=12)
        self.progress_bar = ft.ProgressBar(value=0, width=200, visible=False, color=ft.Colors.BLUE_400)

        # === METRICS DISPLAY ===
        self.metrics_text = ft.Text("", size=11, color=ft.Colors.CYAN_200, font_family="monospace")

        # === LOGS ===
        self.clear_logs_btn = ft.TextButton("Clear Logs", on_click=self._on_clear_logs, style=ft.ButtonStyle(color=ft.Colors.GREY_400))
        self.log_view = ft.ListView(spacing=2, auto_scroll=True, height=150)

        # === BUILD LAYOUT ===
        self.page.add(
            # Header
            ft.Row([
                ft.Text("Synthetic Data Generator", size=24, weight=ft.FontWeight.BOLD),
                ft.Container(expand=True),
                self.help_btn
            ]),
            
            # Toolbar
            ft.Card(
                content=ft.Container(
                    content=ft.Row([
                        self.save_btn, self.load_btn, self.import_btn, self.reset_btn
                    ], scroll=ft.ScrollMode.AUTO),
                    padding=10
                )
            ),

            # Configuration Section
            ft.Text("Configuration", size=18, weight=ft.FontWeight.BOLD),
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Row([
                            self.model_dropdown,
                            self.refresh_models_btn,
                            ft.VerticalDivider(width=20),
                            self.rows_field,
                            self.sim_threshold_field,
                            self.max_retries_field,
                        ], spacing=10, wrap=True),
                    ]),
                    padding=15
                )
            ),
            
            # AI Configuration Section
            ft.Text("AI Configuration", size=18, weight=ft.FontWeight.BOLD),
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Row([
                            self.provider_dropdown,
                            self.api_key_field,
                            self.test_connection_btn,
                        ], spacing=10, wrap=True),
                        self.azure_row,
                        ft.Divider(height=10, color=ft.Colors.TRANSPARENT),
                        ft.Text("Cost Estimation Settings", size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_400),
                        self.cost_config_row,
                    ]),
                    padding=15
                )
            ),

            # Magic Generator Section
            ft.Row([
                ft.Icon(ft.Icons.AUTO_AWESOME, color=ft.Colors.AMBER_400, size=20),
                ft.Text("Magic Generator", size=18, weight=ft.FontWeight.BOLD),
            ], spacing=8),
            ft.Card(
                content=ft.Container(
                    content=ft.Row([self.magic_prompt, self.magic_btn], spacing=10),
                    padding=15
                )
            ),

            # Columns Section
            ft.Row([
                ft.Text("Columns", size=18, weight=ft.FontWeight.BOLD),
                ft.Container(expand=True),
                self.add_col_btn,
            ]),
            ft.Container(
                content=self.columns_list,
                bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.WHITE),
                border_radius=8,
                padding=10,
                border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
            ),

            # Action Bar
            ft.Row([
                self.start_btn, 
                self.export_btn, 
                self.analyze_btn,

            ], alignment=ft.MainAxisAlignment.START, wrap=True),

            # Logs Section
            ft.Container(
                content=ft.Row([
                    ft.Text("Logs & Status", size=16, weight=ft.FontWeight.BOLD),
                    ft.Container(width=10),
                    self.status_text,
                    ft.Container(width=10),
                    self.progress_bar,
                    ft.Container(expand=True),
                    self.clear_logs_btn
                ], alignment=ft.MainAxisAlignment.CENTER),
                padding=ft.padding.only(top=10, bottom=5)
            ),
            
            # Metrics Row (Above logs)
            ft.Container(
                content=self.metrics_text,
                padding=ft.padding.only(left=10, bottom=5)
            ),
            
            ft.Container(
                content=self.log_view,
                bgcolor=ft.Colors.GREY_900,
                border_radius=5,
                padding=10,
                border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
            ),
        )

        # Add one default column
        self._add_column()

    def _add_column(self, col_def: ColumnDefinition = None):
        col_ctrl = ColumnControl(self, index=len(self.columns), on_remove=self._remove_column, col_def=col_def)
        self.columns.append(col_ctrl)
        self.columns_list.controls.append(col_ctrl)
        self.page.update()

    def _remove_column(self, ctrl):
        if ctrl in self.columns:
            self.columns.remove(ctrl)
            self.columns_list.controls.remove(ctrl)
            self.page.update()

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

    # --- FILE DIALOG EVENTS (Via Dialogs helper) ---
    
    def _on_save_config(self, e):
        path = Dialogs.get_file_save_path(self.page, "Save Configuration", [("JSON", "*.json")], ".json")
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

    def _on_load_config(self, e):
        path = Dialogs.get_file_open_path(self.page, "Load Configuration", [("JSON", "*.json")])
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Restore config
                if "model_id" in data: self.model_dropdown.value = data["model_id"]
                if "provider" in data: 
                    self.provider_dropdown.value = data["provider"]
                    self._on_provider_change(None)
                if "api_key" in data: self.api_key_field.value = data["api_key"]
                if "azure_endpoint" in data: self.azure_endpoint.value = data["azure_endpoint"]
                if "azure_deployment" in data: self.azure_deployment.value = data["azure_deployment"]
                if "input_price_per_1m" in data: self.input_price_field.value = str(data["input_price_per_1m"])
                if "output_price_per_1m" in data: self.output_price_field.value = str(data["output_price_per_1m"])
                if "num_rows" in data: self.rows_field.value = str(data["num_rows"])
                if "similarity_threshold" in data: self.sim_threshold_field.value = str(data["similarity_threshold"])
                if "max_retries" in data: self.max_retries_field.value = str(data["max_retries"])

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

    def _on_import_data(self, e):
        if not pd:
            Dialogs.show_snackbar(self.page, "Error: pandas not installed.")
            return
            
        path = Dialogs.get_file_open_path(self.page, "Import Data", [("Data Files", "*.csv *.json")])
        if path:
            try:
                if path.endswith('.csv'):
                    df = pd.read_csv(path)
                else:
                    df = pd.read_json(path)
                
                self.imported_data = df.to_dict(orient='records')
                count = len(self.imported_data)
                self.rows_field.value = str(count)
                
                # Clear and create schema
                self.columns.clear()
                self.columns_list.controls.clear()
                
                for col_name in df.columns:
                    # Very basic type inference check
                    dtype = str(df[col_name].dtype)
                    col_type = ColumnType.NUMERIC if 'int' in dtype or 'float' in dtype else ColumnType.SHORT_TEXT
                    
                    self._add_column(ColumnDefinition(
                        name=col_name,
                        type=col_type,
                        prompt_instruction="(Imported)"
                    ))
                
                Dialogs.show_snackbar(self.page, f"Imported {count} rows. Schema updated.")
                self.page.update()
            except Exception as ex:
                Dialogs.show_snackbar(self.page, f"Import error: {ex}")

    def _on_reset_config(self, e):
        """Resets the configuration to defaults, keeping the model connection."""
        try:
            # Reset Generation Settings to defaults
            self.rows_field.value = "10"
            self.sim_threshold_field.value = "0.85"
            self.max_retries_field.value = "50"
            self.input_price_field.value = "0.15"
            self.output_price_field.value = "0.60"
            
            # Reset Magic Prompt
            self.magic_prompt.value = ""
            
            # Clear imported data
            self.imported_data = None
            
            # Reset Columns
            self.columns.clear()
            self.columns_list.controls.clear()
            
            # Add default column
            self._add_column()
            
            Dialogs.show_snackbar(self.page, "Configuration reset to defaults.")
            self.page.update()
        except Exception as ex:
            Dialogs.show_snackbar(self.page, f"Error resetting config: {ex}")

    def export_data(self, format_type):
        ext = format_type.split('_')[0] # pdf_report -> pdf
        file_types = [("CSV", "*.csv")] if ext == "csv" else \
                     [("JSON", "*.json")] if ext == "json" else \
                     [("SQL", "*.sql")] if ext == "sql" else \
                     [("PDF", "*.pdf")]
                     
        if not self.controller.generated_rows:
            Dialogs.show_snackbar(self.page, "Please generate data first (0 rows).")
            return

        path = Dialogs.get_file_save_path(self.page, f"Export {format_type.upper()}", file_types, f".{ext}")
        
        if path:
            try:
                if format_type == "csv":
                    self.controller.export_csv(path)
                elif format_type == "json":
                    self.controller.export_json(path)
                elif format_type == "sql":
                    self.controller.export_sql(path)
                elif format_type == "pdf_report":
                    self.controller.export_pdf_report(path)
                elif format_type == "pdf_narrative":
                    self.controller.export_narrative_pdf(path)
                
                Dialogs.show_snackbar(self.page, f"Exported to {path}")
            except Exception as ex:
                Dialogs.show_snackbar(self.page, f"Export error: {ex}")

    def _on_analyze(self, e):
        metrics = self.controller.analyze_quality()
        if not metrics:
            Dialogs.show_snackbar(self.page, "No data to analyze.")
            return
        
        # Build report text
        msg = "=== DATA QUALITY REPORT ===\n\n"
        for col, data in metrics.items():
            msg += f"COLUMN: {col}\n"
            msg += f"{'-'*30}\n"
            msg += f"  • Diversity Score: {data.get('diversity_score', 0):.1%}\n"
            msg += f"  • Null Count:      {data.get('null_count', 0)}\n"
            msg += f"  • Top Frequent Values:\n"
            for val, count in data.get('top_frequent', {}).items():
                msg += f"      - {val}: {count}\n"
            msg += "\n"

        dlg = ft.AlertDialog(
            title=ft.Text("Data Quality Analysis"),
            content=ft.Text(msg, font_family="monospace"),
            on_dismiss=lambda e: print("Dialog dismissed")
        )
        self.page.overlay.append(dlg)
        dlg.open = True
        self.page.update()

    def _show_help(self, e):
        help_text = """
        **Synthetic Data Generator Help**

        **Configuration:**
        - **Model ID:** Select the LLM model (e.g., from LM Studio).
        - **Rows:** Number of rows to generate.
        - **Sim. Threshold:** 0-1 (higher = stricter uniqueness).
        
        **Columns:**
        - **Format:** Define columns with name, type, and instruction.
        - **Logic:** Use `after @[ColName]` for dependencies.
        - **Regex:** Enforce patterns (e.g., `^\\d{5}$` for zip).
        
        **Import/Data Enrichment:**
        Import existing CSV/JSON to enrich it with new synthetic columns.
        """
        dlg = ft.AlertDialog(
            title=ft.Text("Help & Documentation"),
            content=ft.Markdown(help_text),
        )
        self.page.overlay.append(dlg)
        dlg.open = True
        self.page.update()

    def _refresh_models(self, e):
        async def task():
            try:
                from core.llm_client import LLMClient
                # Run potentially blocking I/O in a thread
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
                
        # Fire and forget async task
        asyncio.create_task(task())

    def _test_connection(self, e):
        Dialogs.show_snackbar(self.page, "Connection test not implemented yet.")

    def _on_magic_generate(self, e):
        prompt = self.magic_prompt.value
        if not prompt:
            Dialogs.show_snackbar(self.page, "Please describe your dataset first.")
            return
        
        self.magic_btn.disabled = True
        self.magic_btn.text = "Generating... (This may take 10-20s)"
        self.page.update()

        async def task():
            try:
                # 1. Gather config
                # We need to grab values here before entering the thread
                model_id = self.model_dropdown.value or "local-model"
                provider_val = self.provider_dropdown.value
                api_key_val = self.api_key_field.value
                az_ep = self.azure_endpoint.value
                az_dep = self.azure_deployment.value
                
                def run_magic():
                    # 1. Init Client from UI Config
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

                    # 2. Prepare Context (Type Hinting & Sample Value)
                    context_str = None
                    if self.imported_data and len(self.imported_data) > 0:
                        sample_row = self.imported_data[0]
                        headers = list(sample_row.keys())
                        
                        context_lines = []
                        for h in headers:
                            val = sample_row[h]
                            # Basic Type Inference
                            type_hint = "String"
                            if isinstance(val, int): type_hint = "Integer"
                            elif isinstance(val, float): type_hint = "Float"
                            elif isinstance(val, bool): type_hint = "Boolean"
                            
                            context_lines.append(f"Column: {h} ({type_hint}) | Sample: {val}")
                        
                        context_str = "\n".join(context_lines)
                        # self.controller.log(f"Context provided to AI:\n{context_str}")

                    # 3. Generate Schema
                    # self.controller.log(f"Magic Generating for prompt: {prompt}...") # cant call controller log safely from thread?
                    # actually controller log puts to queue, so it IS safe.
                    schema_list = client.generate_schema(prompt, context=context_str)
                    return schema_list, None

                # Run blocking LLM call in executor
                schema_list, error = await asyncio.to_thread(run_magic)
                
                if error:
                     self.controller.log(error)
                     Dialogs.show_snackbar(self.page, error)
                     return

                if not schema_list:
                    Dialogs.show_snackbar(self.page, "Magic Generator returned no columns. Check logs.")
                    return

                self.controller.log(f"Received {len(schema_list)} columns from LLM.")

                # 3. Prepare Column Definitions (Data Logic only)
                new_col_defs = []
                for col_data in schema_list:
                    try:
                        # Map string type to Enum
                        type_str = col_data.get("type", "Short Text")
                        try:
                            col_type = ColumnType(type_str)
                        except ValueError:
                            col_type = ColumnType.SHORT_TEXT
                            
                        # Extract constraints logic
                        constraints_data = col_data.get("constraints", {})
                        
                        # Build kwargs dict to ignore None values for strict fields (min_length, max_length)
                        const_kwargs = {
                            "min_value": constraints_data.get("min_value"),
                            "max_value": constraints_data.get("max_value"),
                            "options": constraints_data.get("options", []),
                            "regex_pattern": constraints_data.get("regex_pattern"),
                            "faker_provider": constraints_data.get("faker_provider"),
                            "allow_duplicates": constraints_data.get("allow_duplicates", False)
                        }
                        
                        # Only include length constraints if they exist and are not None
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

                # 4. Update UI (Batch Update)
                # FIX: Do not clear existing columns! Append new ones.
                # self.columns.clear()
                # self.columns_list.controls.clear()
                
                added_count = 0
                for col_def in new_col_defs:
                    # Double check existence in current UI list to be safe
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
                self.magic_btn.text = "Auto-Generate Schema"
                self.page.update()
        
        asyncio.create_task(task())

    def toggle_generation(self, e):
        print(f"DEBUG: toggle_generation called. is_generating={self.is_generating}")
        if self.is_generating:
            self.controller.stop_generation()
            self.start_btn.content = ft.Row(
                [ft.ProgressRing(width=16, height=16, stroke_width=2, color=ft.Colors.WHITE), ft.Text("Stopping...", color=ft.Colors.WHITE)],
                alignment=ft.MainAxisAlignment.CENTER
            )
            self.start_btn.disabled = True
            # The loop will reset this when it exits
            self.start_btn.update()
            self.page.update()
        else:
            try:
                columns = [c.get_definition() for c in self.columns]
                if not columns:
                    Dialogs.show_snackbar(self.page, "Add at least one column.")
                    return

                config = GeneratorConfig(
                    model_id=self.model_dropdown.value or "local-model",
                    provider=AIProvider(self.provider_dropdown.value),
                    api_key=self.api_key_field.value if self.provider_dropdown.value != AIProvider.LM_STUDIO.value else None,
                    input_price_per_1m=float(self.input_price_field.value or 0),
                    output_price_per_1m=float(self.output_price_field.value or 0),
                    num_rows=int(self.rows_field.value),
                    similarity_threshold=float(self.sim_threshold_field.value),
                    max_retries=int(self.max_retries_field.value),
                    existing_data = self.imported_data
                )
                self.controller.initialize(config, columns)
                self.controller.start_generation_thread()
                
                self.start_btn.style = ft.ButtonStyle(bgcolor=ft.Colors.RED_700, color=ft.Colors.WHITE)
                # Use content to force update if text property is stuck
                self.start_btn.content = ft.Row(
                    [ft.Icon(ft.Icons.STOP, color=ft.Colors.WHITE), ft.Text("STOP", color=ft.Colors.WHITE)],
                    alignment=ft.MainAxisAlignment.CENTER
                )
                self.start_btn.update()
                
                self.export_btn.disabled = True
                self.analyze_btn.disabled = True
                
                self.progress_bar.visible = True # Indeterminate until first progress
                self.status_text.value = "Status: Generating..."
                self.status_text.color = ft.Colors.BLUE_400
                self.log_view.controls.clear()
                self.is_generating = True
            except Exception as ex:
                Dialogs.show_snackbar(self.page, f"Error: {ex}")
                self.status_text.value = "Status: Error"
                self.status_text.color = ft.Colors.RED_400
        self.page.update()

    def _init_controller_callbacks(self):
        self.controller.on_log = lambda msg: self.log_queue.put(msg)
        self.controller.on_progress = lambda curr, total: self.progress_queue.put((curr, total))
        self.controller.on_finished = lambda: self.progress_queue.put("DONE")

    def _on_clear_logs(self, e):
        self.log_view.controls.clear()
        self.page.update()

    async def start_async_loop(self):
        """Async loop to process queues and update UI."""
        while True:
            # 1. Process Logs (Batching)
            logs_to_add = []
            while not self.log_queue.empty():
                try:
                    msg = self.log_queue.get_nowait()
                    logs_to_add.append(msg)
                except queue.Empty:
                    break
            
            if logs_to_add:
                for msg in logs_to_add:
                    color = ft.Colors.WHITE
                    if "Error" in msg or "Failed" in msg:
                        color = ft.Colors.RED_400
                    elif "Warning" in msg:
                        color = ft.Colors.AMBER_400
                    elif "Success" in msg or "Generated" in msg:
                        color = ft.Colors.GREEN_400
                        
                    self.log_view.controls.append(ft.Text(msg, size=12, font_family="monospace", color=color))
                
                if len(self.log_view.controls) > 500:
                    self.log_view.controls = self.log_view.controls[-500:]
                    
                self.log_view.update()

            # 2. Process Progress
            progress_updated = False
            while not self.progress_queue.empty():
                try:
                    data = self.progress_queue.get_nowait()
                    if data == "DONE":
                        self.is_generating = False
                        self.start_btn.content = ft.Row(
                            [ft.Icon(ft.Icons.PLAY_ARROW, color=ft.Colors.WHITE), ft.Text("Start Generation", color=ft.Colors.WHITE)],
                            alignment=ft.MainAxisAlignment.CENTER
                        )
                        self.start_btn.disabled = False # Re-enable if disabled
                        self.start_btn.style = ft.ButtonStyle(bgcolor=ft.Colors.GREEN_700, color=ft.Colors.WHITE)
                        self.start_btn.update()
                        
                        self.export_btn.disabled = False
                        self.analyze_btn.disabled = False
                        self.export_btn.disabled = False
                        self.analyze_btn.disabled = False
                        self.progress_bar.visible = False
                        self.progress_bar.value = 0
                        self.status_text.value = "Status: Complete"
                        self.status_text.color = ft.Colors.GREEN_400
                        Dialogs.show_snackbar(self.page, "Generation Complete!")
                        progress_updated = True
                    else:
                        curr, total = data
                        if total > 0:
                            self.progress_bar.visible = True
                            self.progress_bar.value = curr / total
                            self.status_text.value = f"Status: Generating ({int((curr/total)*100)}%)"
                            progress_updated = True
                except queue.Empty:
                    break
            
            if progress_updated:
                self.progress_bar.update()
                self.status_text.update()
                self.start_btn.update() 
                self.export_btn.update()
                self.analyze_btn.update()

            # 3. Update Metrics (Always)
            m = self.controller.get_metrics()
            if m:
                t = m['total']
                a = m['avg_row']
                s = m['stats']
                
                def fmt(n): return f"{n:,}"
                def fmt_float(n): return f"{n:,.2f}"

                line1 = f"TOTALS:  {fmt(t['in'])} In / {fmt(t['out'])} Out = {fmt(t['used'])} Used  |  COST: ${t['cost']:.4f}  |  SAVED: {fmt(t['saved_tokens'])} Toks (~${t['saved_cost']:.4f})"
                line2 = f"AVG/ROW: {fmt_float(a['in'])} In / {fmt_float(a['out'])} Out = {fmt_float(a['used'])} Used  |  COST: ${a['cost']:.4f}  |  SAVED: {fmt_float(a['saved_tokens'])} Toks (~${a['saved_cost']:.4f})"
                # Format elapsed time as MM:SS
                elapsed_min, elapsed_sec = divmod(int(s['elapsed']), 60)
                elapsed_str = f"{elapsed_min:02}:{elapsed_sec:02}"

                line3 = f"STATS:   {s['generated']}/{s['target']} Rows  |  {elapsed_str} Elapsed  |  {fmt_float(s['throughput'])} Rows/Sec  |  RETRY RATE: {fmt_float(s['retry_rate'])}%"
                
                etr_str = "Calculating..."
                if s['etr'] > 0:
                    m_time, sec = divmod(int(s['etr']), 60)
                    etr_str = f"{m_time}m {sec}s"
                elif s['throughput'] > 0:
                        etr_str = "< 1s"
                elif not self.is_generating and s['generated'] > 0:
                    etr_str = "Done"

                line4 = f"PERF:    AVG LATENCY: {fmt_float(s['latency'])}s  |  EST. TIME REMAINING: {etr_str}"

                new_val = f"{line1}\n{line2}\n{line3}\n{line4}"
                if self.metrics_text.value != new_val:
                    self.metrics_text.value = new_val
                    self.metrics_text.update()

            await asyncio.sleep(0.1)

async def main(page: ft.Page):
    controller = GeneratorController()
    app = FletApp(page, controller)
    
    # Start the async loop
    await app.start_async_loop()

if __name__ == "__main__":
    ft.app(target=main)
