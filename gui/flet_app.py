import flet as ft
import asyncio
import queue
from typing import List

from core.models import GeneratorConfig, ColumnDefinition
from core.controller import GeneratorController
from gui.controls.column_card import ColumnControl
from gui.utils import Dialogs
from gui.handlers import ConfigHandlersMixin, GenerationHandlersMixin, DataHandlersMixin


class FletApp(ConfigHandlersMixin, GenerationHandlersMixin, DataHandlersMixin):
    """
    Main Flet application for Synthetic Data Generator.
    
    Uses mixin classes for handler organization:
    - ConfigHandlersMixin: save/load/reset config, provider changes
    - GenerationHandlersMixin: magic gen, start/stop, model refresh
    - DataHandlersMixin: import/export/analyze
    """
    
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
        self.imported_data: List[dict] = None
        self.current_export_format = None

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
        from core.models import AIProvider
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
                ft.PopupMenuItem(content=ft.Text("Export CSV"), on_click=lambda e: self._handle_export(e, "csv")),
                ft.PopupMenuItem(content=ft.Text("Export JSON"), on_click=lambda e: self._handle_export(e, "json")),
                ft.PopupMenuItem(content=ft.Text("Export SQL"), on_click=lambda e: self._handle_export(e, "sql")),
                ft.PopupMenuItem(content=ft.Text("Export PDF Report"), on_click=lambda e: self._handle_export(e, "pdf_report")),
                ft.PopupMenuItem(content=ft.Text("Export PDF Narrative"), on_click=lambda e: self._handle_export(e, "pdf_narrative")),
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
                        self.start_btn.disabled = False
                        self.start_btn.style = ft.ButtonStyle(bgcolor=ft.Colors.GREEN_700, color=ft.Colors.WHITE)
                        self.start_btn.update()
                        
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
