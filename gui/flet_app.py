import flet as ft
import threading
import queue
import time
import json
import os
import tkinter as tk
from tkinter import filedialog
from typing import List, Any, Dict
# Need pandas for import
try:
    import pandas as pd
except ImportError:
    pd = None

from core.models import GeneratorConfig, ColumnDefinition, ColumnType, ColumnConstraints, AIProvider
from core.controller import GeneratorController


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
        # Init UI
        self._setup_ui()
        self._init_controller_callbacks()
        
        # Initial Log
        self.log_queue.put("System Ready. Logs will appear here...")
        
        self._start_background_tasks()

    def _show_snackbar(self, message: str):
        """Show a snackbar message using overlay."""
        sb = ft.SnackBar(ft.Text(message), open=True)
        self.page.overlay.append(sb)
        self.page.update()

    def _get_file_save_path(self, title: str, types: List[tuple], default_ext: str) -> str:
        """Helper to use Tkinter file dialog in a thread-safe way."""
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            path = filedialog.asksaveasfilename(title=title, filetypes=types, defaultextension=default_ext)
            root.destroy()
            return path
        except Exception as e:
            self._show_snackbar(f"Error opening file dialog: {e}")
            return ""

    def _get_file_open_path(self, title: str, types: List[tuple]) -> str:
        """Helper to use Tkinter file dialog in a thread-safe way."""
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            path = filedialog.askopenfilename(title=title, filetypes=types)
            root.destroy()
            return path
        except Exception as e:
            self._show_snackbar(f"Error opening file dialog: {e}")
            return ""

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
            "Start Generation",
            icon=ft.Icons.PLAY_ARROW,
            on_click=self.toggle_generation,
            style=ft.ButtonStyle(bgcolor=ft.Colors.GREEN_700, color=ft.Colors.WHITE)
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

    # --- FILE DIALOG EVENTS (TKINTER FALLBACK) ---
    
    def _on_save_config(self, e):
        path = self._get_file_save_path("Save Configuration", [("JSON", "*.json")], ".json")
        if path:
            try:
                config_data = {
                    "model_id": self.model_dropdown.value,
                    "provider": self.provider_dropdown.value,
                    "api_key": self.api_key_field.value,
                    "azure_endpoint": self.azure_endpoint.value,
                    "azure_deployment": self.azure_deployment.value,
                    "num_rows": int(self.rows_field.value),
                    "similarity_threshold": float(self.sim_threshold_field.value),
                    "max_retries": int(self.max_retries_field.value),
                    "columns": [col.get_definition().model_dump() for col in self.columns]
                }
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=4)
                self._show_snackbar(f"Configuration saved to {path}")
            except Exception as ex:
                self._show_snackbar(f"Error saving config: {ex}")

    def _on_load_config(self, e):
        path = self._get_file_open_path("Load Configuration", [("JSON", "*.json")])
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
                if "num_rows" in data: self.rows_field.value = str(data["num_rows"])
                if "similarity_threshold" in data: self.sim_threshold_field.value = str(data["similarity_threshold"])
                if "max_retries" in data: self.max_retries_field.value = str(data["max_retries"])

                # Restore columns
                if "columns" in data:
                    self.columns.clear()
                    self.columns_list.controls.clear()
                    for col_data in data["columns"]:
                        self._add_column(ColumnDefinition(**col_data))
                
                self._show_snackbar("Configuration loaded!")
                self.page.update()
            except Exception as ex:
                self._show_snackbar(f"Error loading config: {ex}")

    def _on_import_data(self, e):
        if not pd:
            self._show_snackbar("Error: pandas not installed.")
            return
            
        path = self._get_file_open_path("Import Data", [("Data Files", "*.csv *.json")])
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
                
                self._show_snackbar(f"Imported {count} rows. Schema updated.")
                self.page.update()
            except Exception as ex:
                self._show_snackbar(f"Import error: {ex}")

    def _on_reset_config(self, e):
        """Resets the configuration to defaults, keeping the model connection."""
        try:
            # Reset Generation Settings to defaults
            self.rows_field.value = "10"
            self.sim_threshold_field.value = "0.85"
            self.max_retries_field.value = "50"
            
            # Reset Magic Prompt
            self.magic_prompt.value = ""
            
            # Clear imported data
            self.imported_data = None
            
            # Reset Columns
            self.columns.clear()
            self.columns_list.controls.clear()
            
            # Add default column
            self._add_column()
            
            self._show_snackbar("Configuration reset to defaults.")
            self.page.update()
        except Exception as ex:
            self._show_snackbar(f"Error resetting config: {ex}")

    def export_data(self, format_type):
        ext = format_type.split('_')[0] # pdf_report -> pdf
        file_types = [("CSV", "*.csv")] if ext == "csv" else \
                     [("JSON", "*.json")] if ext == "json" else \
                     [("SQL", "*.sql")] if ext == "sql" else \
                     [("PDF", "*.pdf")]
                     
        if not self.controller.generated_rows:
            self._show_snackbar("Please generate data first (0 rows).")
            return

        path = self._get_file_save_path(f"Export {format_type.upper()}", file_types, f".{ext}")
        
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
                
                self._show_snackbar(f"Exported to {path}")
            except Exception as ex:
                self._show_snackbar(f"Export error: {ex}")

    def _on_analyze(self, e):
        metrics = self.controller.analyze_quality()
        if not metrics:
            self._show_snackbar("No data to analyze.")
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
        def task():
            try:
                from core.llm_client import LLMClient
                from core.models import GeneratorConfig
                temp_config = GeneratorConfig(model_id="temp")
                client = LLMClient(temp_config)
                models = client.list_models()
                options = [ft.dropdown.Option(m) for m in models] if models else []
                self.model_dropdown.options = options
                if options:
                    self.model_dropdown.value = options[0].key
                self._show_snackbar(f"Found {len(models)} models")
                self.page.update()
            except Exception as ex:
                self._show_snackbar(f"Error: {ex}")
                self.page.update()
        threading.Thread(target=task).start()

    def _test_connection(self, e):
        self._show_snackbar("Connection test not implemented yet.")

    def _on_magic_generate(self, e):
        prompt = self.magic_prompt.value
        if not prompt:
            self._show_snackbar("Please describe your dataset first.")
            return
        
        self.magic_btn.disabled = True
        self.magic_btn.text = "Generating... (This may take 10-20s)"
        self.page.update()

        def task():
            try:
                # 1. Init Client from UI Config
                from core.llm_client import LLMClient
                config = GeneratorConfig(
                    model_id=self.model_dropdown.value or "local-model",
                    provider=AIProvider(self.provider_dropdown.value),
                    api_key=self.api_key_field.value if self.provider_dropdown.value != AIProvider.LM_STUDIO.value else None,
                    azure_endpoint=self.azure_endpoint.value,
                    azure_deployment=self.azure_deployment.value
                )
                client = LLMClient(config)
                
                # Check connection first
                if not client.check_connection():
                     self.controller.log("Error: Could not connect to LLM Provider.")
                     self._show_snackbar("Error: Could not connect to LLM. Check settings/server.")
                     return

                # 2. Generate Schema
                self.controller.log(f"Magic Generating for prompt: {prompt}...")
                schema_list = client.generate_schema(prompt)
                
                if not schema_list:
                    self._show_snackbar("Magic Generator returned no columns. Check logs.")
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
                # clear existing
                self.columns.clear()
                self.columns_list.controls.clear()
                
                # Add new controls
                for col_def in new_col_defs:
                    col_ctrl = ColumnControl(self, index=len(self.columns), on_remove=self._remove_column, col_def=col_def)
                    self.columns.append(col_ctrl)
                    self.columns_list.controls.append(col_ctrl)
                
                self._show_snackbar(f"Magic! Generated {len(new_col_defs)} columns.")
                self.page.update()
                
            except Exception as ex:
                self.controller.log(f"Magic Error: {ex}")
                self._show_snackbar(f"Magic Error: {ex}")
            finally:
                self.magic_btn.disabled = False
                self.magic_btn.text = "Auto-Generate Schema"
                self.page.update()
        
        threading.Thread(target=task).start()

    def toggle_generation(self, e):
        if self.is_generating:
            self.controller.stop_generation()
            self.start_btn.text = "Start Generation"
            self.start_btn.style = ft.ButtonStyle(bgcolor=ft.Colors.GREEN_700, color=ft.Colors.WHITE)
            self.is_generating = False
        else:
            try:
                columns = [c.get_definition() for c in self.columns]
                if not columns:
                    self._show_snackbar("Add at least one column.")
                    return

                config = GeneratorConfig(
                    model_id=self.model_dropdown.value or "local-model",
                    provider=AIProvider(self.provider_dropdown.value),
                    api_key=self.api_key_field.value if self.provider_dropdown.value != AIProvider.LM_STUDIO.value else None,
                    num_rows=int(self.rows_field.value),
                    similarity_threshold=float(self.sim_threshold_field.value),
                    max_retries=int(self.max_retries_field.value),
                    existing_data = self.imported_data
                )
                self.controller.initialize(config, columns)
                self.controller.start_generation_thread()
                
                self.start_btn.text = "Stop Generation"
                self.start_btn.style = ft.ButtonStyle(bgcolor=ft.Colors.RED_700, color=ft.Colors.WHITE)
                
                self.export_btn.disabled = True
                self.analyze_btn.disabled = True
                
                self.progress_bar.visible = True
                self.progress_bar.value = None # Indeterminate until first progress
                self.status_text.value = "Status: Generating..."
                self.status_text.color = ft.Colors.BLUE_400
                self.log_view.controls.clear()
                self.is_generating = True
            except Exception as ex:
                self._show_snackbar(f"Error: {ex}")
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

    def _start_background_tasks(self):
        def loop():
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
                        # Optional: Color coding based on message content
                        color = ft.Colors.WHITE
                        if "Error" in msg or "Failed" in msg:
                            color = ft.Colors.RED_400
                        elif "Warning" in msg:
                            color = ft.Colors.AMBER_400
                        elif "Success" in msg or "Generated" in msg:
                            color = ft.Colors.GREEN_400
                            
                        self.log_view.controls.append(ft.Text(msg, size=12, font_family="monospace", color=color))
                    
                    # Buffer Limit (Keep last 500 lines)
                    if len(self.log_view.controls) > 500:
                        self.log_view.controls = self.log_view.controls[-500:]
                        
                    self.page.update()

                # 2. Process Progress
                progress_updated = False
                while not self.progress_queue.empty():
                    try:
                        data = self.progress_queue.get_nowait()
                        if data == "DONE":
                            self.is_generating = False
                            self.start_btn.text = "Start Generation"
                            self.start_btn.style = ft.ButtonStyle(bgcolor=ft.Colors.GREEN_700, color=ft.Colors.WHITE)
                            self.export_btn.disabled = False
                            self.analyze_btn.disabled = False
                            self.progress_bar.visible = False
                            self.progress_bar.value = 0
                            self.status_text.value = "Status: Complete"
                            self.status_text.color = ft.Colors.GREEN_400
                            self._show_snackbar("Generation Complete!")
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
                    self.page.update()
                    
                time.sleep(0.1)
        threading.Thread(target=loop, daemon=True).start()


class ColumnControl(ft.Card):
    """A single column definition card with expandable advanced options."""

    def __init__(self, app, index, on_remove, col_def=None):
        super().__init__()
        self.app = app
        self.index = index
        self.on_remove_callback = on_remove
        self.col_def = col_def
        self.show_advanced = False

        # Main fields
        self.name_field = ft.TextField(
            label="Name",
            value=col_def.name if col_def else f"column_{index}",
            width=140,
            dense=True
        )
        self.type_dropdown = ft.Dropdown(
            label="Type",
            options=[ft.dropdown.Option(t.value) for t in ColumnType],
            value=col_def.type.value if col_def else ColumnType.SHORT_TEXT.value,
            width=150,
            dense=True,
            on_change=self._on_type_change
        )
        # Initialize prompt value smart logic
        initial_prompt = ""
        if col_def:
            if col_def.type == ColumnType.DETERMINISTIC and col_def.constraints.faker_provider:
                initial_prompt = col_def.constraints.faker_provider
            else:
                initial_prompt = col_def.prompt_instruction

        self.prompt_field = ft.TextField(
            label="Prompt / Instructions",
            value=initial_prompt,
            expand=True,
            dense=True
        )

        # Advanced fields
        self.regex_field = ft.TextField(
            label="Regex Pattern",
            value=col_def.constraints.regex_pattern if col_def and col_def.constraints and col_def.constraints.regex_pattern else "",
            width=140,
            dense=True
        )
        self.logic_field = ft.TextField(
            label="Logic Expression",
            value=col_def.constraints.expression if col_def and col_def.constraints and col_def.constraints.expression else "",
            width=160,
            dense=True
        )
        self.sim_field = ft.TextField(
            label="Similarity Threshold",
            value=str(col_def.constraints.similarity_threshold) if col_def and col_def.constraints and col_def.constraints.similarity_threshold else "",
            width=130,
            dense=True
        )
        self.dupes_chk = ft.Checkbox(
            label="Allow Duplicates",
            value=col_def.constraints.allow_duplicates if col_def and col_def.constraints else False
        )
        
        # New Constraints (Min/Max Value & Length)
        self.min_val_field = ft.TextField(
            label="Min Value", 
            width=100, 
            dense=True, 
            visible=False,
            value=str(col_def.constraints.min_value) if col_def and col_def.constraints.min_value is not None else ""
        )
        self.max_val_field = ft.TextField(
            label="Max Value", 
            width=100, 
            dense=True, 
            visible=False,
            value=str(col_def.constraints.max_value) if col_def and col_def.constraints.max_value is not None else ""
        )
        self.min_len_field = ft.TextField(
            label="Min Len", 
            width=100, 
            dense=True, 
            visible=False,
            value=str(col_def.constraints.min_length) if col_def else ""
        )
        self.max_len_field = ft.TextField(
            label="Max Len", 
            width=100, 
            dense=True, 
            visible=False,
            value=str(col_def.constraints.max_length) if col_def else ""
        )

        self.advanced_row = ft.Row([
            self.regex_field,
            self.logic_field,
            self.sim_field,
            self.dupes_chk,
            self.min_val_field,
            self.max_val_field,
            self.min_len_field,
            self.max_len_field,
        ], spacing=10, visible=False, wrap=True)

        self.toggle_advanced_btn = ft.TextButton(
            "▶ Advanced",
            on_click=self._toggle_advanced,
            style=ft.ButtonStyle(color=ft.Colors.GREY_500)
        )

        self.content = ft.Container(
            content=ft.Column([
                ft.Row([
                    self.name_field,
                    ft.Container(
                        content=self.type_dropdown,
                        width=170, # Fixed width to ensure separation
                        padding=ft.padding.only(right=10)
                    ),
                    self.prompt_field,
                    ft.IconButton(
                        icon=ft.Icons.DELETE_OUTLINE,
                        on_click=lambda e: self.on_remove_callback(self),
                        icon_color=ft.Colors.RED_400,
                        tooltip="Remove column"
                    ),
                ], spacing=15),
                ft.Row([self.toggle_advanced_btn]),
                self.advanced_row,
            ], spacing=5),
            padding=10
        )
        
        # Trigger initial state
        self._on_type_change(None)

    def _on_type_change(self, e):
        val = self.type_dropdown.value
        if val == ColumnType.AUTO_INCREMENT.value:
            self.prompt_field.disabled = True
            self.prompt_field.value = "N/A"
            self.prompt_field.label = "Prompt (Not used)"
        elif val == ColumnType.DETERMINISTIC.value:
            self.prompt_field.disabled = False
            self.prompt_field.label = "Faker Provider (e.g. name, email)"
            self.prompt_field.hint_text = "name"
            # If switching TO deterministic, maybe clear if it was a prompt?
            # Or assume user might want to keep "name" if they typed it.
        else:
            self.prompt_field.disabled = False
            self.prompt_field.label = "Prompt / Instructions"
            self.prompt_field.hint_text = "e.g. A creative sci-fi name"
            if self.prompt_field.value == "N/A":
                 self.prompt_field.value = ""
        
        # Visibility Logic for new constraints
        if val == ColumnType.NUMERIC.value:
            self.min_val_field.visible = True
            self.max_val_field.visible = True
            self.min_len_field.visible = False
            self.max_len_field.visible = False
        elif val in [ColumnType.SHORT_TEXT.value, ColumnType.LONG_TEXT.value]:
            self.min_val_field.visible = False
            self.max_val_field.visible = False
            self.min_len_field.visible = True
            self.max_len_field.visible = True
        else:
            self.min_val_field.visible = False
            self.max_val_field.visible = False
            self.min_len_field.visible = False
            self.max_len_field.visible = False

        # Only update if mounted to page
        if self.prompt_field.page:
            self.prompt_field.update()
            self.min_val_field.update()
            self.max_val_field.update()
            self.min_len_field.update()
            self.max_len_field.update()

    def _toggle_advanced(self, e):
        self.show_advanced = not self.show_advanced
        self.advanced_row.visible = self.show_advanced
        self.toggle_advanced_btn.text = "▼ Advanced" if self.show_advanced else "▶ Advanced"
        self.app.page.update()

    def get_definition(self) -> ColumnDefinition:
        sim = float(self.sim_field.value) if self.sim_field.value else None
        col_type = ColumnType(self.type_dropdown.value)
        
        # Base constraints
            regex_pattern=self.regex_field.value if self.regex_field.value else None,
            expression=self.logic_field.value if self.logic_field.value else None,
            similarity_threshold=sim,
            allow_duplicates=self.dupes_chk.value,
            # New constraints
            min_value=float(self.min_val_field.value) if self.min_val_field.value else None,
            max_value=float(self.max_val_field.value) if self.max_val_field.value else None,
            min_length=int(self.min_len_field.value) if self.min_len_field.value else 10, # Defaults from models.py
            max_length=int(self.max_len_field.value) if self.max_len_field.value else 2000
        )
        
        # Handle special cases
        prompt_val = self.prompt_field.value
        if col_type == ColumnType.DETERMINISTIC:
            constraints.faker_provider = prompt_val
            prompt_val = "" # Clear prompt for faker
        elif col_type == ColumnType.AUTO_INCREMENT:
            prompt_val = ""

        return ColumnDefinition(
            name=self.name_field.value,
            type=col_type,
            prompt_instruction=prompt_val,
            constraints=constraints
        )


def main(page: ft.Page):
    controller = GeneratorController()
    app = FletApp(page, controller)


if __name__ == "__main__":
    ft.app(target=main)
