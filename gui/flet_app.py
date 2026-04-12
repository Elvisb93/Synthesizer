import flet as ft
import asyncio
import queue
from typing import List, Optional

from core.models import ColumnDefinition, GeneratorConfig, RagBackend
from core.controller import GeneratorController
from gui.controls.column_card import ColumnControl
from gui.utils import Dialogs
from gui.handlers import ConfigHandlersMixin, GenerationHandlersMixin, DataHandlersMixin, RagHandlersMixin


class FletApp(ConfigHandlersMixin, GenerationHandlersMixin, DataHandlersMixin, RagHandlersMixin):
    """
    Main Flet application for Synthesizer Workspace.
    
    Uses mixin classes for handler organization:
    - ConfigHandlersMixin: save/load/reset config, provider changes
    - GenerationHandlersMixin: magic gen, start/stop, model refresh
    - DataHandlersMixin: import/export/analyze
    """
    
    def __init__(self, page: ft.Page, controller: GeneratorController):
        self.page = page
        self.controller = controller
        self.page.title = "Synthesizer Workspace"
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.bgcolor = ft.Colors.GREY_900
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
        self.log_queue.put("Workspace ready. Technical details will appear here when enabled.")

    def _setup_ui(self):
        # === TOOLBAR / ACTIONS ===
        self.save_btn = ft.ElevatedButton("Save Config", icon=ft.Icons.SAVE, on_click=self._on_save_config)
        self.load_btn = ft.ElevatedButton("Load Config", icon=ft.Icons.UPLOAD_FILE, on_click=self._on_load_config)
        self.import_btn = ft.ElevatedButton("Import Data", icon=ft.Icons.TABLE_CHART, on_click=self._on_import_data)
        self.reset_btn = ft.ElevatedButton("Reset Config", icon=ft.Icons.RESTART_ALT, on_click=self._on_reset_config, style=ft.ButtonStyle(color=ft.Colors.RED_400))
        self.help_btn = ft.IconButton(icon=ft.Icons.HELP_OUTLINE, tooltip="Help / Docs", on_click=self._show_help)

        # === MODEL CONFIGURATION ===
        self.model_dropdown = ft.Dropdown(
            label="AI Model",
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
        self.rows_field = ft.TextField(label="How Many Rows?", value="10", width=160, dense=True)
        self.sim_threshold_field = ft.TextField(label="Uniqueness Strictness", value="0.85", width=160, dense=True)
        self.max_retries_field = ft.TextField(label="Retry Limit", value="50", width=120, dense=True)

        # === AI PROVIDER CONFIG ===
        from core.models import AIProvider
        self.provider_dropdown = ft.Dropdown(
            label="AI Service",
            options=[ft.dropdown.Option(p.value) for p in AIProvider],
            value=AIProvider.LM_STUDIO.value,
            width=180,
            dense=True
        )
        self.provider_dropdown.on_change = self._on_provider_change

        self.api_key_field = ft.TextField(label="API Key (if required)", password=True, width=280, disabled=True, dense=True)
        self.test_connection_btn = ft.ElevatedButton("Check Connection", on_click=self._test_connection)

        # === RAG CONFIG ===
        self.rag_collection_field = ft.TextField(label="Search Collection", value="synthesizer_default", width=180, dense=True)
        self.rag_backend_dropdown = ft.Dropdown(
            label="RAG Backend",
            options=[
                ft.dropdown.Option(RagBackend.NATIVE.value),
                ft.dropdown.Option(RagBackend.LLAMA_INDEX.value),
            ],
            value=RagBackend.LLAMA_INDEX.value,
            width=150,
            dense=True,
        )
        self.rag_top_k_field = ft.TextField(label="Top Matches", value="5", width=90, dense=True)
        self.rag_min_score_field = ft.TextField(label="Minimum Match Score", value="0.25", width=140, dense=True)
        self.rag_max_context_chars_field = ft.TextField(label="Max Context Characters", value="3000", width=160, dense=True)
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
        self.rag_parser_mode_dropdown = ft.Dropdown(
            label="Parser Mode",
            options=[ft.dropdown.Option("auto"), ft.dropdown.Option("pdf_only"), ft.dropdown.Option("docling")],
            value="auto",
            width=130,
            dense=True,
        )
        self.rag_summary_top_k_field = ft.TextField(label="Summary Top K", value="3", width=110, dense=True)
        self.rag_dense_top_k_field = ft.TextField(label="Dense Top K", value="12", width=100, dense=True)
        self.rag_lexical_top_k_field = ft.TextField(label="Lexical Top K", value="12", width=110, dense=True)
        self.rag_parent_ctx_max_chars_field = ft.TextField(label="Parent Ctx Chars", value="1200", width=130, dense=True)
        self.rag_hybrid_switch = ft.Switch(label="Hybrid", value=True)
        self.rag_rerank_switch = ft.Switch(label="Rerank", value=True)
        self.rag_summary_switch = ft.Switch(label="Summary-First", value=True)
        self.rag_parent_ctx_switch = ft.Switch(label="Parent Context", value=True)
        self.rag_graph_switch = ft.Switch(label="GraphRAG", value=True)
        self.rag_late_interaction_switch = ft.Switch(label="Late Interaction", value=True)
        self.rag_graph_hops_field = ft.TextField(label="Graph Hops", value="1", width=95, dense=True)
        self.rag_graph_boost_field = ft.TextField(label="Graph Boost", value="0.08", width=100, dense=True)
        self.rag_late_interaction_weight_field = ft.TextField(label="Late Weight", value="0.2", width=95, dense=True)
        self.rag_status_btn = ft.OutlinedButton("Search Status", icon=ft.Icons.INFO_OUTLINE, on_click=self._on_rag_status)
        self.rag_clear_btn = ft.OutlinedButton("Clear Search Index", icon=ft.Icons.DELETE_OUTLINE, on_click=self._on_rag_clear)
        self.rag_config_block = ft.Column([
            ft.Row([
                self.rag_backend_dropdown,
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
                self.rag_parser_mode_dropdown,
                self.rag_summary_top_k_field,
                self.rag_dense_top_k_field,
                self.rag_lexical_top_k_field,
                self.rag_parent_ctx_max_chars_field,
            ], spacing=10, wrap=True),
            ft.Row([
                self.rag_hybrid_switch,
                self.rag_rerank_switch,
                self.rag_summary_switch,
                self.rag_parent_ctx_switch,
            ], spacing=10, wrap=True),
            ft.Row([
                self.rag_graph_switch,
                self.rag_late_interaction_switch,
                self.rag_graph_hops_field,
                self.rag_graph_boost_field,
                self.rag_late_interaction_weight_field,
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
            ft.Text("Optional estimate for model cost tracking", size=10, color=ft.Colors.GREY_500, italic=True)
        ], alignment=ft.MainAxisAlignment.START, visible=True)

        # === DATA WORKSPACE TASKS ===
        self.data_source_text = ft.Text(
            "Start from scratch or import a CSV/JSON file to use existing columns as a base.",
            size=12,
            color=ft.Colors.GREY_400,
        )
        self.data_prompt = ft.TextField(
            label="Describe the data you want",
            hint_text="e.g., Customer database with names, emails, phone numbers, and purchase history",
            expand=True,
            multiline=True,
            min_lines=1,
            max_lines=2,
            dense=True
        )
        self.data_magic_btn = ft.ElevatedButton(
            "Suggest Fields",
            icon=ft.Icons.AUTO_AWESOME,
            on_click=self._on_magic_generate
        )
        self.data_task_header = ft.Text("1. Describe Your Sample Data", size=18, weight=ft.FontWeight.BOLD)

        # === FILES WORKSPACE ===
        self.files_intro_text = ft.Text(
            "Add files first, then choose whether you want a document, grounded answers, or structured JSON.",
            color=ft.Colors.GREY_400,
            size=12,
        )
        self.files_count_text = ft.Text("Files ready: 0", color=ft.Colors.GREY_400, size=12)
        self.files_list_view = ft.ListView(spacing=6, auto_scroll=False, height=180)
        self.file_chat_view = ft.ListView(
            controls=[
                ft.Text(
                    "Results will appear here after you run a file task. Start by importing one or more files above.",
                    size=12,
                    color=ft.Colors.GREY_500,
                )
            ],
            spacing=4,
            auto_scroll=True,
            height=180,
        )
        self.rag_url_field = ft.TextField(label="Add Web Page", hint_text="https://example.com/page", expand=True, dense=True)
        self.rag_add_url_btn = ft.OutlinedButton("Add Web Page", icon=ft.Icons.LINK, on_click=self._on_add_rag_url)
        self.preset_dropdown = ft.Dropdown(label="Saved Prompt", width=220, dense=True)
        self.preset_dropdown.on_change = self._on_file_preset_change
        self.preset_name_field = ft.TextField(label="Save Prompt As", width=180, dense=True)
        self.preset_save_btn = ft.ElevatedButton("Save Prompt", icon=ft.Icons.SAVE, on_click=self._on_save_file_preset)
        self.preset_delete_btn = ft.OutlinedButton("Delete Prompt", icon=ft.Icons.DELETE_OUTLINE, on_click=self._on_delete_file_preset)
        self.files_mode_dropdown = ft.Dropdown(
            label="File Task",
            options=[
                ft.dropdown.Option("Document Engine"),
                ft.dropdown.Option("Quick Q&A"),
                ft.dropdown.Option("Structured JSON"),
            ],
            value="Document Engine",
            width=180,
            dense=True,
        )
        self.files_mode_dropdown.on_change = self._on_files_mode_change
        self.files_doc_mode_btn = ft.ElevatedButton("Draft a Document", icon=ft.Icons.ARTICLE_OUTLINED, on_click=lambda e: self._set_files_mode("Document Engine"))
        self.files_qa_mode_btn = ft.OutlinedButton("Ask Questions", icon=ft.Icons.HELP_OUTLINE, on_click=lambda e: self._set_files_mode("Quick Q&A"))
        self.files_json_mode_btn = ft.OutlinedButton("Build JSON", icon=ft.Icons.CODE, on_click=lambda e: self._set_files_mode("Structured JSON"))
        self.files_mode_helper = ft.Text(
            "Draft reports and summaries that stay grounded in the files you imported.",
            size=12,
            color=ft.Colors.GREY_400,
        )
        self.files_prompt = ft.TextField(
            label="What should the files help you produce?",
            hint_text="e.g., Create a 3-part strategy memo with recommendations and implementation plan.",
            expand=True,
            multiline=True,
            min_lines=1,
            max_lines=2,
            dense=True,
        )
        self.files_magic_btn = ft.ElevatedButton(
            "Generate Document",
            icon=ft.Icons.SMART_TOY,
            on_click=self._on_files_magic_task,
        )
        self.quick_qa_backend_dropdown = ft.Dropdown(
            label="Q&A Style",
            options=[
                ft.dropdown.Option("Broader Analysis"),
                ft.dropdown.Option("Pinpoint Quick"),
            ],
            value="Broader Analysis",
            width=170,
            dense=True,
        )
        self.quick_qa_backend_helper = ft.Text(
            "Broader Analysis uses the default LlamaIndex path. Pinpoint Quick switches Quick Q&A to the native retriever for tighter fact lookup.",
            size=11,
            color=ft.Colors.GREY_400,
        )
        self.json_template_path_field = ft.TextField(
            label="JSON Template File",
            hint_text="Select a .json template file",
            expand=True,
            dense=True,
        )
        self.json_template_browse_btn = ft.OutlinedButton(
            "Select Template",
            icon=ft.Icons.UPLOAD_FILE,
            on_click=self._on_pick_json_template,
        )
        self.json_target_key_field = ft.TextField(label="List Key To Fill", value="items", width=180, dense=True)
        self.json_mode_dropdown = ft.Dropdown(
            label="JSON Task",
            options=[
                ft.dropdown.Option("Standard Generation"),
                ft.dropdown.Option("Exhaustive Extraction"),
            ],
            value="Standard Generation",
            width=210,
            dense=True,
        )
        self.json_mode_dropdown.on_change = self._on_json_mode_change
        self.json_clear_existing_switch = ft.Switch(label="Replace Existing Items", value=True)
        self.json_export_btn = ft.OutlinedButton(
            "Export JSON",
            icon=ft.Icons.DOWNLOAD,
            on_click=self._on_export_json_template,
            visible=False,
        )
        self.json_template_helper = ft.Text(
            "Standard generation uses the row count above. Exhaustive extraction processes every imported chunk and ignores the row count.",
            size=11,
            color=ft.Colors.GREY_400,
        )
        self.json_template_container = ft.Container(
            content=ft.Column([
                ft.Row([
                    self.json_template_path_field,
                    self.json_template_browse_btn,
                ], spacing=8),
                ft.Row([
                    self.json_target_key_field,
                    self.json_mode_dropdown,
                    self.json_clear_existing_switch,
                    self.json_export_btn,
                ], spacing=8, wrap=True),
                self.json_template_helper,
            ], spacing=8),
            visible=False,
            padding=10,
            border_radius=8,
            bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.WHITE),
            border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
        )

        self.doc_mode_dropdown = ft.Dropdown(
            label="Grounding Style",
            options=[
                ft.dropdown.Option("Balanced"),
                ft.dropdown.Option("File-based"),
                ft.dropdown.Option("Creative"),
            ],
            value="Balanced",
            width=180,
            dense=True,
        )
        self.doc_pages_dropdown = ft.Dropdown(
            label="Length",
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
            label="Review Depth",
            options=[ft.dropdown.Option("Fast"), ft.dropdown.Option("Thorough")],
            value="Fast",
            width=130,
            dense=True,
        )
        self.doc_audience_field = ft.TextField(label="Audience", value="General", width=170, dense=True)
        self.doc_tone_field = ft.TextField(label="Tone", value="professional", width=170, dense=True)
        self.doc_chart_switch = ft.Switch(label="Include Charts", value=False)
        self.doc_flow_switch = ft.Switch(label="Include Flowchart", value=True)
        self.doc_max_charts_field = ft.TextField(label="Max Charts", value="3", width=110, dense=True)
        self.doc_strategy_helper = ft.Text(
            "Balanced blends file evidence with synthesis. File-based stays close to source material. Creative allows freer drafting.",
            size=11,
            color=ft.Colors.GREY_400,
        )
        self.doc_bundle_label = ft.Text("Quick Starting Points:", size=11, color=ft.Colors.GREY_400, weight=ft.FontWeight.BOLD)
        self.doc_bundle_exec_btn = ft.OutlinedButton("Executive Brief", on_click=lambda e: self._apply_doc_bundle("Executive Brief"))
        self.doc_bundle_policy_btn = ft.OutlinedButton("Policy Draft", on_click=lambda e: self._apply_doc_bundle("Policy Draft"))
        self.doc_bundle_action_btn = ft.OutlinedButton("Action Plan", on_click=lambda e: self._apply_doc_bundle("Action Plan"))
        self.doc_bundle_meeting_btn = ft.OutlinedButton("Meeting Summary", on_click=lambda e: self._apply_doc_bundle("Meeting Summary"))
        self.doc_export_pdf_btn = ft.OutlinedButton("Export PDF", icon=ft.Icons.PICTURE_AS_PDF, on_click=self._on_export_document_pdf)
        self.doc_export_docx_btn = ft.OutlinedButton("Export DOCX", icon=ft.Icons.DESCRIPTION, on_click=self._on_export_document_docx)

        # === WORKSPACE TABS ===
        self.data_tab_btn = ft.ElevatedButton("Generate Sample Data", icon=ft.Icons.TABLE_VIEW, on_click=lambda e: self._set_workspace_tab("data"))
        self.files_tab_btn = ft.OutlinedButton("Work With Files", icon=ft.Icons.FOLDER_OPEN, on_click=lambda e: self._set_workspace_tab("files"))

        # === COLUMNS ===
        self.columns_list = ft.Column(spacing=8)
        self.add_col_btn = ft.OutlinedButton("+ Add Field", on_click=lambda e: self._add_column())

        # === ACTION BAR ===
        self.start_btn = ft.ElevatedButton(
            content=ft.Row(
                [ft.Icon(ft.Icons.PLAY_ARROW, color=ft.Colors.WHITE), ft.Text("Generate Data", color=ft.Colors.WHITE)],
                alignment=ft.MainAxisAlignment.CENTER
            ),
            style=ft.ButtonStyle(bgcolor=ft.Colors.GREEN_700, color=ft.Colors.WHITE),
            on_click=self.toggle_generation
        )
        self.export_btn = ft.PopupMenuButton(
            content=ft.Row([ft.Icon(ft.Icons.DOWNLOAD, size=18), ft.Text("Export Results")], spacing=5),
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
        self.analyze_btn = ft.ElevatedButton("Review Quality", icon=ft.Icons.ANALYTICS, on_click=self._on_analyze, disabled=True)
        
        # Status & Progress
        self.status_text = ft.Text("Status: Ready to start", color=ft.Colors.GREY_400, size=12)
        self.progress_bar = ft.ProgressBar(value=0, width=200, visible=False, color=ft.Colors.BLUE_400)

        # === METRICS DISPLAY ===
        self.metrics_text = ft.Text("", size=11, color=ft.Colors.CYAN_200, font_family="monospace")

        # === LOGS ===
        self.clear_logs_btn = ft.TextButton("Clear Details", on_click=self._on_clear_logs, style=ft.ButtonStyle(color=ft.Colors.GREY_400))
        self.log_view = ft.ListView(spacing=2, auto_scroll=True, height=150)
        self.debug_toggle = ft.Switch(
            label="Show technical details",
            value=False,
            on_change=self._on_toggle_debug_view
        )

        self.advanced_settings_toggle = ft.Switch(
            label="Show technical settings",
            value=False,
            on_change=self._on_toggle_advanced_settings
        )
        self.advanced_settings_container = ft.Container(
            content=ft.Column([
                ft.Text("Most people can leave these settings closed unless they need finer control.", size=11, color=ft.Colors.GREY_500),
                ft.Text("Generation Tuning", size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_400),
                ft.Row([self.sim_threshold_field, self.max_retries_field], spacing=10, wrap=True),
                ft.Text("Cost Tracking", size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_400),
                self.cost_config_row,
                ft.Text("Retrieval And File Search", size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_400),
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

        self.workspace_badge = ft.Container(
            content=ft.Text("Current workflow: Generate Sample Data", color=ft.Colors.WHITE, size=11),
            padding=ft.padding.symmetric(horizontal=10, vertical=6),
            border_radius=999,
            bgcolor=ft.Colors.with_opacity(0.18, ft.Colors.AMBER_300),
        )
        self.workspace_hint_title = ft.Text(
            "Create realistic sample rows from scratch or from an imported file.",
            size=18,
            weight=ft.FontWeight.BOLD,
        )
        self.workspace_hint_text = ft.Text(
            "Best for demos, testing, seeded datasets, and enrichment. Use Suggest Fields if you want the app to draft a starting structure for you.",
            size=12,
            color=ft.Colors.GREY_300,
        )
        self.workspace_callout = ft.Container(
            content=ft.Column([
                self.workspace_badge,
                self.workspace_hint_title,
                self.workspace_hint_text,
            ], spacing=8),
            padding=18,
            border_radius=14,
            bgcolor=ft.Colors.with_opacity(0.08, ft.Colors.BLUE_300),
            border=ft.border.all(1, ft.Colors.with_opacity(0.25, ft.Colors.BLUE_100)),
        )

        self.quick_start_title = ft.Text("Quick start for sample data", size=18, weight=ft.FontWeight.BOLD)
        self.quick_start_hint = ft.Text(
            "Use the guided steps below or click a starter example to fill the prompt for you.",
            size=12,
            color=ft.Colors.GREY_300,
        )
        self.quick_start_step_1_title = ft.Text("1. Describe", size=14, weight=ft.FontWeight.BOLD)
        self.quick_start_step_1_body = ft.Text("Explain what you want to create in plain language.", size=12, color=ft.Colors.GREY_400)
        self.quick_start_step_2_title = ft.Text("2. Review", size=14, weight=ft.FontWeight.BOLD)
        self.quick_start_step_2_body = ft.Text("Check the suggested setup and make any edits you want.", size=12, color=ft.Colors.GREY_400)
        self.quick_start_step_3_title = ft.Text("3. Run", size=14, weight=ft.FontWeight.BOLD)
        self.quick_start_step_3_body = ft.Text("Generate results, review them, and export when you are happy.", size=12, color=ft.Colors.GREY_400)
        self.quick_start_examples_label = ft.Text("Starter examples", size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_300)
        self.quick_start_example_btn_1 = ft.OutlinedButton("", on_click=self._apply_quick_start_example)
        self.quick_start_example_btn_2 = ft.OutlinedButton("", on_click=self._apply_quick_start_example)
        self.quick_start_example_btn_3 = ft.OutlinedButton("", on_click=self._apply_quick_start_example)
        self.quick_start_container = ft.Container(
            content=ft.Column([
                self.quick_start_title,
                self.quick_start_hint,
                ft.Row([
                    ft.Container(
                        content=ft.Column([self.quick_start_step_1_title, self.quick_start_step_1_body], spacing=6),
                        padding=12,
                        border_radius=12,
                        bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.WHITE),
                        border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                        expand=True,
                    ),
                    ft.Container(
                        content=ft.Column([self.quick_start_step_2_title, self.quick_start_step_2_body], spacing=6),
                        padding=12,
                        border_radius=12,
                        bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.WHITE),
                        border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                        expand=True,
                    ),
                    ft.Container(
                        content=ft.Column([self.quick_start_step_3_title, self.quick_start_step_3_body], spacing=6),
                        padding=12,
                        border_radius=12,
                        bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.WHITE),
                        border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                        expand=True,
                    ),
                ], spacing=10, wrap=True),
                self.quick_start_examples_label,
                ft.Row([
                    self.quick_start_example_btn_1,
                    self.quick_start_example_btn_2,
                    self.quick_start_example_btn_3,
                ], spacing=8, wrap=True),
            ], spacing=12),
            padding=18,
            border_radius=14,
            bgcolor=ft.Colors.with_opacity(0.04, ft.Colors.WHITE),
            border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
        )

        # === WORKSPACE CONTAINERS ===
        self.document_settings_container = ft.Container(
            content=ft.Column([
                ft.Text("3. Choose How The Document Should Feel", size=18, weight=ft.FontWeight.BOLD),
                ft.Row([
                    self.doc_mode_dropdown,
                    self.doc_pages_dropdown,
                    self.doc_quality_dropdown,
                    self.doc_audience_field,
                    self.doc_tone_field,
                    self.doc_chart_switch,
                    self.doc_flow_switch,
                    self.doc_max_charts_field,
                ], spacing=8, wrap=True),
                self.doc_strategy_helper,
                ft.Row([
                    self.doc_bundle_label,
                    self.doc_bundle_exec_btn,
                    self.doc_bundle_policy_btn,
                    self.doc_bundle_action_btn,
                    self.doc_bundle_meeting_btn,
                ], spacing=8, wrap=True),
                ft.Row([
                    self.doc_export_pdf_btn,
                    self.doc_export_docx_btn,
                ], spacing=8, wrap=True),
            ], spacing=8),
            visible=True,
            padding=10,
            border_radius=8,
            bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.WHITE),
            border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
        )

        self.quick_qa_helper_container = ft.Container(
            content=ft.Column(
                [
                    ft.Text(
                        "3. Ask a question, request a summary, or draft a response using only the files you imported.",
                        size=12,
                        color=ft.Colors.GREY_400,
                    ),
                    ft.Row([self.quick_qa_backend_dropdown], spacing=8, wrap=True),
                    self.quick_qa_backend_helper,
                ],
                spacing=8,
            ),
            visible=False,
            padding=10,
            border_radius=8,
            bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.WHITE),
            border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
        )

        self.data_workspace_container = ft.Container(
            content=ft.Column([
                ft.Container(
                    content=ft.Column([
                        self.data_task_header,
                        self.data_source_text,
                        ft.Row([self.data_prompt, self.data_magic_btn], spacing=10, wrap=True),
                    ], spacing=8),
                    padding=15,
                    border_radius=10,
                    bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.WHITE),
                    border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                ),
                ft.Container(
                    content=ft.Column([
                        ft.Row([
                            ft.Text("2. Review Or Edit Fields", size=18, weight=ft.FontWeight.BOLD),
                            ft.Container(expand=True),
                            self.add_col_btn,
                        ]),
                        ft.Text(
                            "Keep instructions plain and specific, like 'US phone number' or 'purchase amount in USD'.",
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
                    ], spacing=8),
                    padding=15,
                    border_radius=10,
                    bgcolor=ft.Colors.with_opacity(0.02, ft.Colors.WHITE),
                    border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                ),
                ft.Container(
                    content=ft.Column([
                        ft.Text("3. Generate And Export", size=18, weight=ft.FontWeight.BOLD),
                        ft.Row([
                            self.start_btn,
                            self.export_btn,
                            self.analyze_btn,
                        ], alignment=ft.MainAxisAlignment.START, wrap=True),
                    ], spacing=8),
                    padding=15,
                    border_radius=10,
                    bgcolor=ft.Colors.with_opacity(0.02, ft.Colors.WHITE),
                    border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                ),
            ], spacing=12),
            visible=True,
        )

        self.files_workspace_container = ft.Container(
            content=ft.Column([
                ft.Container(
                    content=ft.Column([
                        ft.Row([
                            ft.Text("1. Add Files", size=18, weight=ft.FontWeight.BOLD),
                            ft.Container(expand=True),
                            self.files_count_text,
                        ]),
                        self.files_intro_text,
                        ft.Container(
                            content=self.files_list_view,
                            bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.WHITE),
                            border_radius=8,
                            padding=10,
                            border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                        ),
                        ft.Row([
                            self.rag_url_field,
                            self.rag_add_url_btn,
                        ], spacing=8),
                    ], spacing=8),
                    padding=15,
                    border_radius=10,
                    bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.WHITE),
                    border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                ),
                ft.Container(
                    content=ft.Column([
                        ft.Text("2. Choose What You Want From The Files", size=18, weight=ft.FontWeight.BOLD),
                        ft.Row([
                            self.files_doc_mode_btn,
                            self.files_qa_mode_btn,
                            self.files_json_mode_btn,
                        ], spacing=8, wrap=True),
                        self.files_mode_helper,
                        ft.Row([
                            self.preset_dropdown,
                            self.preset_name_field,
                            self.preset_save_btn,
                            self.preset_delete_btn,
                        ], spacing=8, wrap=True),
                    ], spacing=8),
                    padding=15,
                    border_radius=10,
                    bgcolor=ft.Colors.with_opacity(0.02, ft.Colors.WHITE),
                    border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                ),
                self.document_settings_container,
                self.quick_qa_helper_container,
                self.json_template_container,
                ft.Container(
                    content=ft.Column([
                        ft.Text("4. Describe The Result", size=18, weight=ft.FontWeight.BOLD),
                        ft.Row([self.files_prompt, self.files_magic_btn], spacing=10, wrap=True),
                    ], spacing=8),
                    padding=15,
                    border_radius=10,
                    bgcolor=ft.Colors.with_opacity(0.02, ft.Colors.WHITE),
                    border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                ),
                ft.Container(
                    content=ft.Column([
                        ft.Text("5. Review Output", size=18, weight=ft.FontWeight.BOLD),
                        ft.Container(
                            content=self.file_chat_view,
                            bgcolor=ft.Colors.GREY_900,
                            border_radius=5,
                            padding=10,
                            border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                        ),
                    ], spacing=8),
                    padding=15,
                    border_radius=10,
                    bgcolor=ft.Colors.with_opacity(0.02, ft.Colors.WHITE),
                    border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                ),
            ], spacing=12),
            visible=False,
        )

        # === BUILD LAYOUT ===
        self.page.add(
            # Header
            ft.Container(
                content=ft.Row([
                    ft.Row([
                        ft.Container(
                            content=ft.Icon(ft.Icons.AUTO_AWESOME_MOTION, color=ft.Colors.WHITE, size=22),
                            width=42,
                            height=42,
                            alignment=ft.Alignment(0, 0),
                            border_radius=12,
                            bgcolor=ft.Colors.with_opacity(0.18, ft.Colors.AMBER_300),
                        ),
                        ft.Column([
                            ft.Text("Synthesizer Workspace", size=26, weight=ft.FontWeight.BOLD),
                            ft.Text(
                                "Choose a task, connect your model, and follow one clear workflow at a time.",
                                size=12,
                                color=ft.Colors.GREY_300,
                            ),
                        ], spacing=2),
                    ], spacing=12),
                    ft.Container(expand=True),
                    self.help_btn
                ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
                padding=18,
                border_radius=16,
                bgcolor=ft.Colors.with_opacity(0.08, ft.Colors.BLUE_300),
                border=ft.border.all(1, ft.Colors.with_opacity(0.25, ft.Colors.BLUE_100)),
            ),
            
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
            ft.Text("Setup", size=18, weight=ft.FontWeight.BOLD),
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("1. Basic Setup", size=14, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_300),
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
            ft.Text("Connection", size=18, weight=ft.FontWeight.BOLD),
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("2. Connect Your AI", size=14, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_300),
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

            ft.Text("Choose A Task", size=18, weight=ft.FontWeight.BOLD),
            ft.Row([self.data_tab_btn, self.files_tab_btn], spacing=8),
            self.workspace_callout,
            self.quick_start_container,
            self.data_workspace_container,
            self.files_workspace_container,

            # Logs Section
            ft.Container(
                content=ft.Row([
                    ft.Text("Status", size=16, weight=ft.FontWeight.BOLD),
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

    def _set_files_mode(self, mode_name: str):
        self.files_mode_dropdown.value = mode_name
        self._on_files_mode_change(None)
        self.page.update()

    def _on_toggle_advanced_settings(self, e):
        self.advanced_settings_container.visible = bool(self.advanced_settings_toggle.value)
        self.page.update()

    def _on_toggle_debug_view(self, e):
        self.debug_container.visible = bool(self.debug_toggle.value)
        self.page.update()

    def _reset_file_chat_placeholder(self):
        self.file_chat_view.controls = [
            ft.Text(
                "Results will appear here after you run a file task. Start by importing one or more files above.",
                size=12,
                color=ft.Colors.GREY_500,
            )
        ]

    def _apply_quick_start_example(self, e):
        payload = getattr(e.control, "data", None) or {}
        workspace = payload.get("workspace")
        prompt = payload.get("prompt", "")
        mode = payload.get("mode")

        if workspace == "files":
            self._set_workspace_tab("files")
            if mode:
                self._set_files_mode(mode)
            if prompt and hasattr(self, "files_prompt") and not self.files_prompt.disabled:
                self.files_prompt.value = prompt
            self.page.update()
            return

        self._set_workspace_tab("data")
        if prompt and hasattr(self, "data_prompt"):
            self.data_prompt.value = prompt
        self.page.update()

    def _refresh_quick_start_content(self):
        if self.active_workspace_tab == "files":
            mode = (self.files_mode_dropdown.value or "Document Engine").strip()
            if mode == "Quick Q&A":
                self.quick_start_title.value = "Quick start for file questions"
                self.quick_start_hint.value = "Import files first, then ask grounded questions or request a concise summary."
                self.quick_start_step_1_body.value = "Add one or more files or a web page to the search workspace."
                self.quick_start_step_2_body.value = "Choose Ask Questions so the app stays focused on Q&A."
                self.quick_start_step_3_body.value = "Ask for a summary, action items, or a draft reply."
                examples = [
                    ("Summarize requests", "Quick Q&A", "Summarize the main requests and concerns from these files."),
                    ("List action items", "Quick Q&A", "Extract action items, owners, and deadlines from these files."),
                    ("Draft a reply", "Quick Q&A", "Draft a concise response based on the key points in these files."),
                ]
            elif mode == "Structured JSON":
                self.quick_start_title.value = "Quick start for structured JSON"
                self.quick_start_hint.value = "Choose a template and target list first, then run generation or extraction."
                self.quick_start_step_1_body.value = "Import files if you want grounded extraction from source material."
                self.quick_start_step_2_body.value = "Choose Build JSON and select the template file plus the list key."
                self.quick_start_step_3_body.value = "Run the task, review the preview, then export the filled JSON."
                examples = []
            else:
                self.quick_start_title.value = "Quick start for file-based documents"
                self.quick_start_hint.value = "Import files first, then choose the kind of document you want to draft."
                self.quick_start_step_1_body.value = "Add one or more files or a web page to the search workspace."
                self.quick_start_step_2_body.value = "Choose Draft a Document and adjust the document style if needed."
                self.quick_start_step_3_body.value = "Describe the report, brief, or summary you want to create."
                examples = [
                    ("Executive brief", "Document Engine", "Create an executive brief with key findings, risks, and recommended next steps."),
                    ("Action plan", "Document Engine", "Create an action plan with phases, owners, milestones, and measurable success criteria."),
                    ("Policy draft", "Document Engine", "Draft a policy document based on the imported files, including scope, requirements, and governance."),
                ]
        else:
            self.quick_start_title.value = "Quick start for sample data"
            self.quick_start_hint.value = "Describe the kind of rows you want, then review the suggested fields before you generate."
            self.quick_start_step_1_body.value = "Describe the dataset in plain language or import a CSV/JSON as your starting point."
            self.quick_start_step_2_body.value = "Use Suggest Fields if you want help drafting the field list."
            self.quick_start_step_3_body.value = "Generate rows, review quality, and export the final result."
            examples = [
                ("Customer contacts", None, "Create a customer contact dataset with name, email, phone number, company, and region."),
                ("Retail orders", None, "Create retail order data with order ID, customer name, product, quantity, order date, and total amount."),
                ("Support tickets", None, "Create support ticket data with ticket ID, issue type, customer priority, summary, status, and resolution note."),
            ]

        buttons = [
            self.quick_start_example_btn_1,
            self.quick_start_example_btn_2,
            self.quick_start_example_btn_3,
        ]
        for idx, btn in enumerate(buttons):
            if idx < len(examples):
                label, mode, prompt = examples[idx]
                btn.text = label
                btn.data = {
                    "workspace": "files" if self.active_workspace_tab == "files" else "data",
                    "mode": mode,
                    "prompt": prompt,
                }
                btn.visible = True
            else:
                btn.visible = False

    def _resolve_document_mode(self) -> str:
        selected = (self.doc_mode_dropdown.value or "Balanced").strip().lower()
        mode_map = {
            "balanced": "hybrid",
            "file-based": "strict_grounded",
            "creative": "pure",
            "hybrid": "hybrid",
            "factual by doc": "strict_grounded",
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
            self.workspace_badge.content.value = "Current workflow: Work With Files"
            self.workspace_badge.bgcolor = ft.Colors.with_opacity(0.18, ft.Colors.BLUE_300)
            self.workspace_hint_title.value = "Turn files into grounded answers, reports, or structured JSON."
            self.workspace_hint_text.value = (
                "Import files first, then choose the outcome you want. The app will keep the next steps focused on that file task only."
            )
        else:
            self.import_btn.content = ft.Row([ft.Icon(ft.Icons.TABLE_CHART), ft.Text("Import Data")], spacing=6)
            self.import_btn.icon = ft.Icons.TABLE_CHART
            self.data_task_header.value = "1. Describe Your Sample Data"
            self.workspace_badge.content.value = "Current workflow: Generate Sample Data"
            self.workspace_badge.bgcolor = ft.Colors.with_opacity(0.18, ft.Colors.AMBER_300)
            self.workspace_hint_title.value = "Create realistic sample rows from scratch or from an imported file."
            self.workspace_hint_text.value = (
                "Best for demos, testing, seeded datasets, and enrichment. Use Suggest Fields if you want the app to draft a starting structure for you."
            )

        if hasattr(self, "_on_files_mode_change"):
            self._on_files_mode_change(None)
        self._refresh_quick_start_content()

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
        **Synthesizer Workspace Help**

        **Generate Sample Data**
        - Describe the dataset you want.
        - Use **Suggest Fields** to draft a starter schema.
        - Review the fields, then run **Generate Data**.

        **Work With Files**
        - Import one or more files first.
        - Choose whether you want a document, grounded answers, or structured JSON.
        - Describe the result you want and run the task.

        **Technical Settings**
        - Hidden by default to keep the main flow simple.
        - Open them only if you need tuning, cost tracking, OCR, or retrieval settings.
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
                            [ft.Icon(ft.Icons.PLAY_ARROW, color=ft.Colors.WHITE), ft.Text("Generate Data", color=ft.Colors.WHITE)],
                            alignment=ft.MainAxisAlignment.CENTER
                        )
                        self.start_btn.disabled = False
                        self.start_btn.style = ft.ButtonStyle(bgcolor=ft.Colors.GREEN_700, color=ft.Colors.WHITE)
                        self.start_btn.update()
                        
                        self.export_btn.disabled = False
                        self.analyze_btn.disabled = False
                        self.progress_bar.visible = False
                        self.progress_bar.value = 0
                        self.status_text.value = "Status: Data is ready to review"
                        self.status_text.color = ft.Colors.GREEN_400
                        Dialogs.show_snackbar(self.page, "Data generation finished.")
                        progress_updated = True
                    else:
                        curr, total = data
                        if total > 0:
                            self.progress_bar.visible = True
                            self.progress_bar.value = curr / total
                            self.status_text.value = f"Status: Creating rows ({int((curr/total)*100)}%)"
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
