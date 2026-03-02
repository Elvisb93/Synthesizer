import flet as ft
import asyncio
import queue
from typing import List, Optional

from core.models import GeneratorConfig, ColumnDefinition
from core.controller import GeneratorController
from gui.controls.column_card import ColumnControl
from gui.utils import Dialogs
from gui.handlers import ConfigHandlersMixin, GenerationHandlersMixin, DataHandlersMixin, RagHandlersMixin


class FletApp(ConfigHandlersMixin, GenerationHandlersMixin, DataHandlersMixin, RagHandlersMixin):
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
        self.imported_data: Optional[List[dict]] = None
        self.current_export_format = None
        self.active_workspace_tab = "data"
        self.rag_files: List[dict] = []
        self.rag_task_presets: dict = {}
        self._runtime_config_signature = None
        self.file_assistant_mode = "Document Engine"

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
            label="Model",
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
        self.rows_field = ft.TextField(label="Rows to Generate", value="10", width=160, dense=True)
        self.sim_threshold_field = ft.TextField(label="Similarity Threshold", value="0.85", width=140, dense=True)
        self.max_retries_field = ft.TextField(label="Max Retries", value="50", width=120, dense=True)

        # === AI PROVIDER CONFIG ===
        from core.models import AIProvider
        self.provider_dropdown = ft.Dropdown(
            label="AI Provider",
            options=[ft.dropdown.Option(p.value) for p in AIProvider],
            value=AIProvider.LM_STUDIO.value,
            width=180,
            dense=True
        )
        self.provider_dropdown.on_change = self._on_provider_change

        self.api_key_field = ft.TextField(label="API Key (if required)", password=True, width=280, disabled=True, dense=True)
        self.test_connection_btn = ft.ElevatedButton("Test Connection", on_click=self._test_connection)

        # === RAG CONFIG ===
        self.rag_collection_field = ft.TextField(label="Collection", value="synthesizer_default", width=180, dense=True)
        self.rag_top_k_field = ft.TextField(label="Top K", value="5", width=80, dense=True)
        self.rag_min_score_field = ft.TextField(label="Min Score", value="0.25", width=100, dense=True)
        self.rag_max_context_chars_field = ft.TextField(label="Max Context Chars", value="3000", width=140, dense=True)
        self.rag_embedding_model_field = ft.TextField(label="Embedding Model", value="BAAI/bge-small-en-v1.5", width=280, dense=True)
        self.rag_source_filter_field = ft.TextField(label="Source Filter (optional)", value="", width=220, dense=True)
        self.rag_qdrant_url_field = ft.TextField(label="Qdrant URL", value=":memory:", width=220, dense=True)
        self.rag_qdrant_api_key_field = ft.TextField(label="Qdrant API Key", password=True, width=180, dense=True)
        self.rag_ocr_mode_dropdown = ft.Dropdown(
            label="OCR Mode",
            options=[ft.dropdown.Option("off"), ft.dropdown.Option("auto"), ft.dropdown.Option("on")],
            value="off",
            width=120,
            dense=True,
        )
        self.rag_ocr_dpi_field = ft.TextField(label="OCR DPI", value="150", width=90, dense=True)
        self.rag_ocr_max_pages_field = ft.TextField(label="OCR Max Pages", value="20", width=110, dense=True)
        self.rag_ocr_max_regions_field = ft.TextField(label="OCR Max Regions/Page", value="8", width=150, dense=True)
        self.rag_ocr_padding_field = ft.TextField(label="OCR Padding(px)", value="18", width=120, dense=True)
        self.rag_ocr_gap_multiplier_field = ft.TextField(label="OCR Gap Multiplier", value="2.5", width=140, dense=True)
        self.rag_ocr_min_chars_field = ft.TextField(label="OCR Min Chars", value="60", width=110, dense=True)
        self.rag_ocr_timeout_field = ft.TextField(label="OCR Timeout(ms)", value="4000", width=120, dense=True)
        self.rag_status_btn = ft.OutlinedButton("RAG Status", icon=ft.Icons.INFO_OUTLINE, on_click=self._on_rag_status)
        self.rag_clear_btn = ft.OutlinedButton("Clear Index", icon=ft.Icons.DELETE_OUTLINE, on_click=self._on_rag_clear)
        self.rag_config_block = ft.Column([
            ft.Row([
                self.rag_collection_field,
                self.rag_top_k_field,
                self.rag_min_score_field,
                self.rag_max_context_chars_field,
            ], spacing=10, wrap=True),
            ft.Row([
                self.rag_embedding_model_field,
                self.rag_source_filter_field,
                self.rag_qdrant_url_field,
                self.rag_qdrant_api_key_field,
            ], spacing=10, wrap=True),
            ft.Row([
                self.rag_ocr_mode_dropdown,
                self.rag_ocr_dpi_field,
                self.rag_ocr_max_pages_field,
                self.rag_ocr_max_regions_field,
            ], spacing=10, wrap=True),
            ft.Row([
                self.rag_ocr_padding_field,
                self.rag_ocr_gap_multiplier_field,
                self.rag_ocr_min_chars_field,
                self.rag_ocr_timeout_field,
            ], spacing=10, wrap=True),
            ft.Row([
                self.rag_status_btn,
                self.rag_clear_btn,
            ], spacing=8),
        ], visible=True)

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
            label="Describe what you want to generate",
            hint_text="e.g., Customer database with names, emails, and purchase history",
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
        self.magic_title = ft.Text("Step 3: Describe Output", size=18, weight=ft.FontWeight.BOLD)

        # === FILES WORKSPACE ===
        self.files_count_text = ft.Text("Indexed files: 0", color=ft.Colors.GREY_400, size=12)
        self.files_list_view = ft.ListView(spacing=6, auto_scroll=False, height=180)
        self.file_chat_view = ft.ListView(spacing=4, auto_scroll=True, height=180)
        self.preset_dropdown = ft.Dropdown(label="Task Preset", width=220, dense=True)
        self.preset_dropdown.on_change = self._on_file_preset_change
        self.preset_name_field = ft.TextField(label="Preset Name", width=180, dense=True)
        self.preset_save_btn = ft.ElevatedButton("Save Preset", icon=ft.Icons.SAVE, on_click=self._on_save_file_preset)
        self.preset_delete_btn = ft.OutlinedButton("Delete Preset", icon=ft.Icons.DELETE_OUTLINE, on_click=self._on_delete_file_preset)
        self.files_mode_dropdown = ft.Dropdown(
            label="Files Mode",
            options=[ft.dropdown.Option("Document Engine"), ft.dropdown.Option("Quick Q&A")],
            value="Document Engine",
            width=180,
            dense=True,
        )
        self.files_mode_dropdown.on_change = self._on_files_mode_change

        self.doc_mode_dropdown = ft.Dropdown(
            label="Doc Strategy",
            options=[
                ft.dropdown.Option("hybrid"),
                ft.dropdown.Option("factual by doc"),
                ft.dropdown.Option("creative"),
            ],
            value="hybrid",
            width=180,
            dense=True,
        )
        self.doc_pages_dropdown = ft.Dropdown(
            label="Pages",
            options=[
                ft.dropdown.Option("Let AI decide"),
                ft.dropdown.Option("1 page"),
                ft.dropdown.Option("2 pages"),
                ft.dropdown.Option("3 pages"),
                ft.dropdown.Option("5 pages"),
                ft.dropdown.Option("8 pages"),
                ft.dropdown.Option("10 pages"),
                ft.dropdown.Option("15 pages"),
                ft.dropdown.Option("20 pages"),
            ],
            value="Let AI decide",
            width=160,
            dense=True,
        )
        self.doc_quality_dropdown = ft.Dropdown(
            label="Quality",
            options=[ft.dropdown.Option("Fast"), ft.dropdown.Option("Thorough")],
            value="Fast",
            width=130,
            dense=True,
        )
        self.doc_audience_field = ft.TextField(label="Audience", value="General", width=170, dense=True)
        self.doc_tone_field = ft.TextField(label="Tone", value="professional", width=170, dense=True)
        self.doc_strategy_helper = ft.Text(
            "hybrid: grounded + synthesis | factual by doc: strictly grounded in files | creative: freer generation with minimal grounding",
            size=11,
            color=ft.Colors.GREY_400,
        )
        self.doc_bundle_label = ft.Text("Quick Presets:", size=11, color=ft.Colors.GREY_400, weight=ft.FontWeight.BOLD)
        self.doc_bundle_exec_btn = ft.OutlinedButton("Executive Brief", on_click=lambda e: self._apply_doc_bundle("Executive Brief"))
        self.doc_bundle_policy_btn = ft.OutlinedButton("Policy Draft", on_click=lambda e: self._apply_doc_bundle("Policy Draft"))
        self.doc_bundle_action_btn = ft.OutlinedButton("Action Plan", on_click=lambda e: self._apply_doc_bundle("Action Plan"))
        self.doc_bundle_meeting_btn = ft.OutlinedButton("Meeting Summary", on_click=lambda e: self._apply_doc_bundle("Meeting Summary"))
        self.doc_export_pdf_btn = ft.OutlinedButton("Export PDF", icon=ft.Icons.PICTURE_AS_PDF, on_click=self._on_export_document_pdf)
        self.doc_export_docx_btn = ft.OutlinedButton("Export DOCX", icon=ft.Icons.DESCRIPTION, on_click=self._on_export_document_docx)

        # === WORKSPACE TABS ===
        self.data_tab_btn = ft.ElevatedButton("Data Generation", on_click=lambda e: self._set_workspace_tab("data"))
        self.files_tab_btn = ft.OutlinedButton("Files", on_click=lambda e: self._set_workspace_tab("files"))

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
        self.debug_toggle = ft.Switch(
            label="Show logs and diagnostics",
            value=False,
            on_change=self._on_toggle_debug_view
        )

        self.advanced_settings_toggle = ft.Switch(
            label="Show advanced settings",
            value=False,
            on_change=self._on_toggle_advanced_settings
        )
        self.advanced_settings_container = ft.Container(
            content=ft.Column([
                ft.Text("Generation Tuning", size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_400),
                ft.Row([self.sim_threshold_field, self.max_retries_field], spacing=10, wrap=True),
                ft.Text("Cost Estimation", size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_400),
                self.cost_config_row,
                ft.Text("RAG Settings", size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_400),
                self.rag_config_block,
            ], spacing=8),
            visible=False,
            padding=10,
            border_radius=8,
            bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.WHITE),
            border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
        )

        self.debug_container = ft.Container(
            content=ft.Column([
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
            ], spacing=6),
            visible=False,
        )

        # === WORKSPACE CONTAINERS ===
        self.data_workspace_container = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Text("Step 4: Define Columns", size=18, weight=ft.FontWeight.BOLD),
                    ft.Container(expand=True),
                    self.add_col_btn,
                ]),
                ft.Text(
                    "Tip: keep instructions plain and specific, like 'US phone number' or 'purchase amount in USD'.",
                    size=12,
                    color=ft.Colors.GREY_400,
                ),
                ft.Container(
                    content=self.columns_list,
                    bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.WHITE),
                    border_radius=8,
                    padding=10,
                    border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                ),
                ft.Row([
                    self.start_btn,
                    self.export_btn,
                    self.analyze_btn,
                ], alignment=ft.MainAxisAlignment.START, wrap=True),
            ]),
            visible=True,
        )

        self.files_workspace_container = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Text("Step 4: Work With Files", size=18, weight=ft.FontWeight.BOLD),
                    ft.Container(expand=True),
                    self.files_count_text,
                ]),
                ft.Container(
                    content=self.files_list_view,
                    bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.WHITE),
                    border_radius=8,
                    padding=10,
                    border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                ),
                ft.Row([
                    self.files_mode_dropdown,
                    self.preset_dropdown,
                    self.preset_name_field,
                    self.preset_save_btn,
                    self.preset_delete_btn,
                ], spacing=8, wrap=True),
                ft.Row([
                    self.doc_mode_dropdown,
                    self.doc_pages_dropdown,
                    self.doc_quality_dropdown,
                    self.doc_audience_field,
                    self.doc_tone_field,
                    self.doc_export_pdf_btn,
                    self.doc_export_docx_btn,
                ], spacing=8, wrap=True),
                self.doc_strategy_helper,
                ft.Row([
                    self.doc_bundle_label,
                    self.doc_bundle_exec_btn,
                    self.doc_bundle_policy_btn,
                    self.doc_bundle_action_btn,
                    self.doc_bundle_meeting_btn,
                ], spacing=8, wrap=True),
                ft.Text("File Assistant Chat", size=16, weight=ft.FontWeight.BOLD),
                ft.Container(
                    content=self.file_chat_view,
                    bgcolor=ft.Colors.GREY_900,
                    border_radius=5,
                    padding=10,
                    border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                ),
            ]),
            visible=False,
        )

        # === BUILD LAYOUT ===
        self.page.add(
            # Header
            ft.Row([
                ft.Column([
                    ft.Text("Synthetic Data Generator", size=24, weight=ft.FontWeight.BOLD),
                    ft.Text(
                        "Simple flow: connect AI, describe output, then generate.",
                        size=12,
                        color=ft.Colors.GREY_400,
                    ),
                ], spacing=2),
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
                        ft.Text("Step 1: Basic Setup", size=14, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_300),
                        ft.Row([
                            self.model_dropdown,
                            self.refresh_models_btn,
                            ft.VerticalDivider(width=20),
                            self.rows_field,
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
                        ft.Text("Step 2: Connect AI", size=14, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_300),
                        ft.Row([
                            self.provider_dropdown,
                            self.api_key_field,
                            self.test_connection_btn,
                        ], spacing=10, wrap=True),
                        self.azure_row,
                        ft.Divider(height=6, color=ft.Colors.TRANSPARENT),
                        self.advanced_settings_toggle,
                        self.advanced_settings_container,
                    ]),
                    padding=15
                )
            ),

            # Magic Generator Section
            ft.Row([
                ft.Icon(ft.Icons.AUTO_AWESOME, color=ft.Colors.AMBER_400, size=20),
                self.magic_title,
            ], spacing=8),
            ft.Card(
                content=ft.Container(
                    content=ft.Row([self.magic_prompt, self.magic_btn], spacing=10),
                    padding=15
                )
            ),

            # Workspace Tabs
            ft.Row([self.data_tab_btn, self.files_tab_btn], spacing=8),
            self.data_workspace_container,
            self.files_workspace_container,

            # Logs Section
            ft.Container(
                content=ft.Row([
                    ft.Text("Logs & Status", size=16, weight=ft.FontWeight.BOLD),
                    ft.Container(width=10),
                    self.status_text,
                    ft.Container(width=10),
                    self.progress_bar,
                    ft.Container(expand=True),
                    self.clear_logs_btn,
                    self.debug_toggle
                ], alignment=ft.MainAxisAlignment.CENTER),
                padding=ft.padding.only(top=10, bottom=5)
            ),
            self.debug_container,
        )

        # Add one default column
        self._add_column()
        self._load_rag_presets()
        self._refresh_files_view()
        self._apply_workspace_mode()

    def _set_workspace_tab(self, tab_name: str):
        self.active_workspace_tab = tab_name
        self._apply_workspace_mode()
        self.page.update()

    def _on_toggle_advanced_settings(self, e):
        self.advanced_settings_container.visible = bool(self.advanced_settings_toggle.value)
        self.page.update()

    def _on_toggle_debug_view(self, e):
        self.debug_container.visible = bool(self.debug_toggle.value)
        self.page.update()

    def _resolve_document_mode(self) -> str:
        selected = (self.doc_mode_dropdown.value or "hybrid").strip().lower()
        mode_map = {
            "hybrid": "hybrid",
            "factual by doc": "strict_grounded",
            "creative": "pure",
            # Backward compatibility for old saved/runtime values.
            "strict_grounded": "strict_grounded",
            "pure": "pure",
        }
        return mode_map.get(selected, "hybrid")

    def _resolve_document_target_words(self) -> int:
        selection = (self.doc_pages_dropdown.value or "Let AI decide").strip().lower()
        if selection == "let ai decide":
            return 0

        pages = 0
        for token in selection.replace("-", " ").split():
            if token.isdigit():
                pages = int(token)
                break

        if pages <= 0:
            return 0

        words_per_page = 500
        return max(350, pages * words_per_page)

    def _apply_workspace_mode(self):
        is_files = self.active_workspace_tab == "files"
        self.data_workspace_container.visible = not is_files
        self.files_workspace_container.visible = is_files
        self.data_tab_btn.style = ft.ButtonStyle(bgcolor=ft.Colors.BLUE_700 if not is_files else None)
        self.files_tab_btn.style = ft.ButtonStyle(bgcolor=ft.Colors.BLUE_700 if is_files else None)

        if is_files:
            self.import_btn.content = ft.Row([ft.Icon(ft.Icons.UPLOAD_FILE), ft.Text("Import File")], spacing=6)
            self.import_btn.icon = ft.Icons.UPLOAD_FILE
            self.magic_title.value = "Step 3: File Task"
            self.magic_prompt.label = "Generate a long document or run file Q&A..."
            self.magic_prompt.hint_text = "e.g., 'Create a 3-part strategy memo with recommendations and implementation plan.'"
            label = "Generate Document" if (self.files_mode_dropdown.value or "Document Engine") == "Document Engine" else "Run File Task"
            self.magic_btn.content = ft.Row([ft.Icon(ft.Icons.SMART_TOY), ft.Text(label)], spacing=6)
        else:
            self.import_btn.content = ft.Row([ft.Icon(ft.Icons.TABLE_CHART), ft.Text("Import Data")], spacing=6)
            self.import_btn.icon = ft.Icons.TABLE_CHART
            self.magic_title.value = "Step 3: Describe Output"
            self.magic_prompt.label = "Describe what you want to generate"
            self.magic_prompt.hint_text = "e.g., 'Customer database with names, emails, phone numbers, and purchase history'"
            self.magic_btn.content = ft.Row([ft.Icon(ft.Icons.AUTO_AWESOME), ft.Text("Auto-Generate Schema")], spacing=6)

        if hasattr(self, "_on_files_mode_change"):
            self._on_files_mode_change(None)

    def _add_column(self, col_def: Optional[ColumnDefinition] = None):
        if col_def is None:
            col_ctrl = ColumnControl(self, index=len(self.columns), on_remove=self._remove_column)
        else:
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

                rag = s.get("rag", {})
                line5 = (
                    "RAG:     "
                    f"QUERIES: {fmt(rag.get('queries', 0))}  |  "
                    f"HIT RATE: {fmt_float(rag.get('hit_rate', 0.0))}%  |  "
                    f"AVG RETRIEVAL: {fmt_float(rag.get('avg_retrieval_ms', 0.0))}ms  |  "
                    f"AVG CONTEXT: {fmt_float(rag.get('avg_context_chars', 0.0))} chars  |  "
                    f"LAST HITS: {fmt(rag.get('last_hits', 0))}"
                )

                new_val = f"{line1}\n{line2}\n{line3}\n{line4}\n{line5}"
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
