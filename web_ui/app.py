from __future__ import annotations

import gradio as gr

from core.app_config import PAGE_OPTIONS
from core.models import AIProvider, RagBackend
from web_ui.actions.config_actions import (
    DEFAULT_RAG_ADMIN_STATUS,
    HELP_MARKDOWN,
    clear_debug_details,
    clear_search_index,
    debug_details_markdown,
    get_search_status,
    load_config_file,
    refresh_debug_details,
    refresh_models,
    reset_config,
    save_config_file,
    test_connection,
)
from web_ui.actions.data_actions import (
    add_grid_row,
    apply_import_privacy_mode,
    apply_grid_row_edit,
    export_generated_data,
    export_power_bi_data,
    generate_data,
    import_data_file,
    load_grid_row_editor,
    request_stop_data_generation,
    remove_last_grid_row,
    review_generated_data_quality,
    save_grid_rows,
    suggest_fields,
    suggest_fields_and_sync_editor,
    sync_grid_row_editor,
    refresh_schema_overview,
    POWER_BI_PRIVACY_CHOICES,
    RESULT_QUALITY_CHOICES,
)
from web_ui.actions.files_actions import add_url_source, ask_quick_qa_chat, clear_files, files_mode_changed, files_mode_helper, register_uploaded_files, request_stop_files_task, run_files_task
from web_ui.actions.files_actions import (
    apply_doc_bundle,
    apply_file_preset,
    delete_file_preset,
    preset_choices,
    reindex_selected_source,
    remove_selected_source,
    save_file_preset,
)
from web_ui.adapters import (
    FIELD_TYPE_CHOICES,
    GRID_HEADERS,
    IMPORT_PRIVACY_CHOICES,
    field_records_to_grid_dataframe,
    field_rows_markup,
    imported_columns_markup,
)
from web_ui.runtime_cleanup import prepare_clean_workspace
from web_ui.state import activity_markdown, clear_runtime_session, new_session_state


FILE_SEARCH_PRESET_CHOICES = [
    "Best results (recommended)",
    "Faster answers",
    "Strict matching",
    "Wider search",
]

SCANNED_FILE_CHOICES = [
    "Normal documents",
    "Scanned PDFs or images",
]


def apply_file_search_preset(search_preset: str, scanned_file_mode: str):
    preset = search_preset if search_preset in FILE_SEARCH_PRESET_CHOICES else FILE_SEARCH_PRESET_CHOICES[0]
    scanned = scanned_file_mode if scanned_file_mode in SCANNED_FILE_CHOICES else SCANNED_FILE_CHOICES[0]

    settings = {
        "rag_backend": RagBackend.LLAMA_INDEX.value,
        "top_k": 5,
        "min_score": 0.25,
        "max_context_chars": 3000,
        "hybrid_search_enabled": True,
        "rerank_enabled": True,
        "summary_first_enabled": True,
        "parent_context_enabled": True,
        "graph_enabled": True,
        "late_interaction_enabled": True,
    }
    explanation = "Best balance of answer quality and speed."

    if preset == "Faster answers":
        settings.update(
            {
                "top_k": 3,
                "min_score": 0.30,
                "max_context_chars": 1800,
                "rerank_enabled": False,
                "graph_enabled": False,
                "late_interaction_enabled": False,
            }
        )
        explanation = "Searches less text so answers come back faster, but may miss details."
    elif preset == "Strict matching":
        settings.update(
            {
                "top_k": 4,
                "min_score": 0.45,
                "max_context_chars": 2200,
                "graph_enabled": False,
            }
        )
        explanation = "Only uses close matches. Best when you want fewer guesses."
    elif preset == "Wider search":
        settings.update(
            {
                "top_k": 8,
                "min_score": 0.10,
                "max_context_chars": 5000,
            }
        )
        explanation = "Looks through more text. Best when answers may be spread across files."

    ocr_mode = "auto" if scanned == "Scanned PDFs or images" else "off"
    parser_mode = "auto"
    scan_note = " OCR will be tried when files look scanned." if ocr_mode == "auto" else " OCR is off for faster normal document processing."

    return (
        settings["rag_backend"],
        settings["top_k"],
        settings["min_score"],
        settings["max_context_chars"],
        ocr_mode,
        parser_mode,
        settings["hybrid_search_enabled"],
        settings["rerank_enabled"],
        settings["summary_first_enabled"],
        settings["parent_context_enabled"],
        settings["graph_enabled"],
        settings["late_interaction_enabled"],
        f"**File search:** {explanation}{scan_note}",
    )


WEB_UI_CSS = """
.app-shell { width: calc(100% - 32px); max-width: 1440px; margin: 0 auto; }
.gradio-container, .contain, .app-shell, .app-shell * { box-sizing: border-box; }
.app-shell { overflow-x: hidden; }
.app-shell p, .app-shell li, .app-shell span { overflow-wrap: anywhere; }
.hero-copy h1 { margin-bottom: 4px; }
.hero-copy p { color: #475569; max-width: 920px; }
.workflow-step { border: 1px solid #d8dee8; border-radius: 8px; background: #ffffff !important; padding: 18px; margin-bottom: 16px; }
.workflow-step > .form, .workflow-step .form { background: #ffffff !important; border: 0 !important; }
.workflow-step textarea, .workflow-step input, .workflow-step select { background: #ffffff; }
.workflow-step h3 { margin-top: 0; }
.step-label { color: #0f172a; font-weight: 700; letter-spacing: 0; }
.schema-overview { margin: 12px 0 18px; }
.field-toolbar, .bottom-actions { margin-top: 14px; }
.schema-help { color: #475569; margin: 8px 0 12px; }
.status-panel { border-left: 4px solid #2563eb; padding: 4px 0 4px 12px; color: #334155; }
.primary-action-row button { min-height: 44px; }
.app-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; }
.app-header .hero-copy { flex: 1 1 auto; }
.reset-action { flex: 0 0 auto; max-width: 120px; }
@media (max-width: 700px) {
  body, .gradio-container, .contain { max-width: 100vw !important; overflow-x: hidden !important; }
  .app-shell { width: calc(100vw - 56px) !important; max-width: calc(100vw - 56px) !important; margin: 0 auto !important; }
  .app-shell * { min-width: 0 !important; max-width: 100% !important; }
  .app-header { align-items: stretch; flex-direction: column; }
  .reset-action { align-self: flex-end; width: 110px; }
  .workflow-step { padding: 12px; }
  .schema-grid { display: none !important; }
}
"""


def reset_workspace_session(session):
    if session is not None:
        clear_runtime_session(session)

    cleanup_report = prepare_clean_workspace()
    fresh_session = new_session_state(
        startup_collection_name=cleanup_report["startup_collection_name"],
        startup_message=cleanup_report["message"],
    )

    return (
        fresh_session,
        activity_markdown(fresh_session),
        "",
        10,
        RESULT_QUALITY_CHOICES[0],
        None,
        "Mask likely personal values",
        "Start from scratch or import a CSV/JSON file to use existing columns as a base. Imported files default to privacy masking so the preview and AI context avoid likely personal values.",
        imported_columns_markup([], "Mask likely personal values"),
        "No imported data yet.",
        gr.update(value=None, visible=False),
        "Review the schema at a glance, edit the grid, then save before generating.",
        field_rows_markup([]),
        field_records_to_grid_dataframe([]),
        gr.update(choices=["Row 1"], value="Row 1"),
        "",
        FIELD_TYPE_CHOICES[0],
        "",
        False,
        "Generation progress will appear here once you start a run.",
        gr.update(value=None, visible=False),
        "Quality review will appear here after generation.",
        None,
        None,
        "",
        [],
        gr.update(choices=[], value=None),
        "Document Engine",
        files_mode_helper("Document Engine"),
        gr.update(label="What document should the files help you create?", placeholder="e.g., Create an executive brief with findings, risks, and next steps.", interactive=True, visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        "",
        "",
        "Balanced",
        "Let AI decide",
        "Fast",
        "General",
        "professional",
        False,
        True,
        3,
        None,
        "items",
        "Standard Generation",
        True,
        "Files progress will appear here after you run a task.",
        None,
        None,
        None,
        "Results will appear here after you run a Files task.",
        fresh_session.startup_collection_name,
        DEFAULT_RAG_ADMIN_STATUS,
        debug_details_markdown(fresh_session),
    )


def create_app(
    *,
    startup_collection_name: str = "synthesizer_default",
    startup_cleanup_message: str | None = None,
) -> gr.Blocks:
    session_state = new_session_state(
        startup_collection_name=startup_collection_name,
        startup_message=startup_cleanup_message,
    )
    initial_preset_choices = preset_choices()
    initial_preset_value = initial_preset_choices[0] if initial_preset_choices else None
    with gr.Blocks(title="Synthesizer Workspace Web", css=WEB_UI_CSS) as app:
        session = gr.State(session_state)

        with gr.Column(elem_classes=["app-shell"]):
            with gr.Row(elem_classes=["app-header"]):
                gr.Markdown(
                    """
                    # Synthesizer Workspace
                    Create realistic sample data, review it, and package it for reporting.
                    """,
                    elem_classes=["hero-copy"],
                )
                reset_workspace_btn = gr.Button("Reset", variant="stop", elem_classes=["reset-action"])
            activity_log = gr.Markdown(activity_markdown(session_state))

            with gr.Accordion("AI Connection", open=True):
                gr.Markdown(
                    "Choose the AI service and model used for field suggestions, data generation, and review. Use **Check Connection** before a business user starts a run.",
                    elem_classes=["schema-help"],
                )
                provider = gr.Dropdown(
                    choices=[provider.value for provider in AIProvider],
                    value=AIProvider.LM_STUDIO.value,
                    label="AI Service",
                )
                model_id = gr.Dropdown(
                    choices=["local-model"],
                    value="local-model",
                    allow_custom_value=True,
                    label="Model",
                )
                api_key = gr.Textbox(label="API Key", type="password")
                with gr.Accordion("Azure Options", open=False):
                    azure_endpoint = gr.Textbox(label="Azure Endpoint")
                    azure_deployment = gr.Textbox(label="Azure Deployment")
                test_connection_btn = gr.Button("Check Connection", variant="primary")
                refresh_models_btn = gr.Button("Refresh Models")

            with gr.Accordion("Admin And Advanced Settings", open=False):
                gr.Markdown(
                    "These settings affect cost tracking, retry behavior, retrieval, saved configs, and troubleshooting. Leave defaults in place unless you are administering the workspace.",
                    elem_classes=["schema-help"],
                )
                with gr.Accordion("Generation Tuning", open=False):
                    with gr.Row():
                        similarity_threshold = gr.Number(label="Uniqueness Strictness", value=0.85)
                        max_retries = gr.Number(label="Retry Limit", value=50, precision=0)
                    gr.Markdown(
                        "Business users should normally use the **Result Quality** selector in the Generate tab. These values are low-level overrides.",
                        elem_classes=["schema-help"],
                    )
                with gr.Accordion("Cost Tracking", open=False):
                    with gr.Row():
                        input_price_per_1m = gr.Number(label="Input Price ($/1M)", value=0.15)
                        output_price_per_1m = gr.Number(label="Output Price ($/1M)", value=0.60)
                with gr.Accordion("Saved Workspace Config", open=False):
                    with gr.Row():
                        load_config_upload = gr.File(label="Load Config JSON", type="filepath")
                        save_config_btn = gr.Button("Save Config JSON")
                        reset_config_btn = gr.Button("Reset Config")
                    saved_config_file = gr.File(label="Saved Config Download", interactive=False)
                with gr.Accordion("Help & Docs", open=False):
                    gr.Markdown(HELP_MARKDOWN)
                with gr.Group(visible=False):
                    gr.Markdown(
                        "Use these only when working with uploaded files. The default is tuned for best results.",
                        elem_classes=["schema-help"],
                    )
                    file_search_preset = gr.Dropdown(
                        choices=FILE_SEARCH_PRESET_CHOICES,
                        value=FILE_SEARCH_PRESET_CHOICES[0],
                        label="How hard should Synthesizer search?",
                    )
                    with gr.Row():
                        quick_qa_mode = gr.Dropdown(
                            choices=["Broader Analysis", "Pinpoint Quick"],
                            value="Broader Analysis",
                            label="Question answering style",
                            info="Broader Analysis reads more context. Pinpoint Quick is better for short factual answers.",
                        )
                        scanned_file_mode = gr.Dropdown(
                            choices=SCANNED_FILE_CHOICES,
                            value=SCANNED_FILE_CHOICES[0],
                            label="Are the files scanned?",
                            info="Choose scanned only for image-based PDFs, screenshots, or photos of documents.",
                        )
                    file_search_status = gr.Markdown(
                        "**File search:** Best balance of answer quality and speed. OCR is off for faster normal document processing."
                    )
                    source_filter = gr.Textbox(
                        label="Only search files with this name (optional)",
                        placeholder="Example: renewal or policy.pdf",
                    )
                    with gr.Accordion("Technical Search Settings (admins only)", open=False):
                        with gr.Row():
                            rag_backend = gr.Dropdown(
                                choices=[RagBackend.NATIVE.value, RagBackend.LLAMA_INDEX.value],
                                value=RagBackend.LLAMA_INDEX.value,
                                label="Search Backend",
                            )
                            collection_name = gr.Textbox(label="Search Collection", value=session_state.startup_collection_name)
                        with gr.Row():
                            top_k = gr.Number(label="Top Matches", value=5, precision=0)
                            min_score = gr.Number(label="Minimum Match Score", value=0.25)
                            max_context_chars = gr.Number(label="Max Context Characters", value=3000, precision=0)
                        with gr.Row():
                            embedding_model = gr.Textbox(label="Embedding Model", value="BAAI/bge-small-en-v1.5")
                        with gr.Row():
                            qdrant_url = gr.Textbox(label="Qdrant URL", value=":memory:")
                            qdrant_api_key = gr.Textbox(label="Qdrant API Key", type="password")
                        with gr.Row():
                            ocr_mode = gr.Dropdown(choices=["off", "auto", "on"], value="off", label="OCR Mode")
                            parser_mode = gr.Dropdown(choices=["auto", "pdf_only", "docling"], value="auto", label="Parser Mode")
                        with gr.Accordion("OCR Details", open=False):
                            with gr.Row():
                                ocr_dpi = gr.Number(label="OCR DPI", value=150, precision=0)
                                ocr_max_pages = gr.Number(label="OCR Max Pages", value=20, precision=0)
                                ocr_max_regions_per_page = gr.Number(label="OCR Max Regions", value=8, precision=0)
                            with gr.Row():
                                ocr_region_padding_px = gr.Number(label="OCR Region Padding", value=18, precision=0)
                                ocr_gap_multiplier = gr.Number(label="OCR Gap Multiplier", value=2.5)
                                ocr_min_extracted_chars = gr.Number(label="OCR Min Chars", value=60, precision=0)
                                ocr_timeout_ms_per_page = gr.Number(label="OCR Timeout (ms)", value=4000, precision=0)
                        with gr.Accordion("Retrieval Details", open=False):
                            gr.Markdown(
                                "These controls change how evidence is found and ranked. Incorrect values can make answers slower or less accurate.",
                                elem_classes=["schema-help"],
                            )
                            with gr.Row():
                                hybrid_search_enabled = gr.Checkbox(label="Hybrid Search", value=True)
                                rerank_enabled = gr.Checkbox(label="Rerank", value=True)
                                summary_first_enabled = gr.Checkbox(label="Summary-First", value=True)
                                parent_context_enabled = gr.Checkbox(label="Parent Context", value=True)
                            with gr.Row():
                                graph_enabled = gr.Checkbox(label="Graph Retrieval", value=True)
                                late_interaction_enabled = gr.Checkbox(label="Late Interaction", value=True)
                            with gr.Row():
                                summary_top_k = gr.Number(label="Summary Top K", value=3, precision=0)
                                dense_top_k = gr.Number(label="Dense Top K", value=12, precision=0)
                                lexical_top_k = gr.Number(label="Lexical Top K", value=12, precision=0)
                            with gr.Row():
                                parent_context_max_chars = gr.Number(label="Parent Context Max Chars", value=1200, precision=0)
                                graph_hops = gr.Number(label="Graph Hops", value=1, precision=0)
                                graph_source_boost = gr.Number(label="Graph Boost", value=0.08)
                                late_interaction_weight = gr.Number(label="Late Weight", value=0.2)
                        with gr.Row():
                            rag_status_btn = gr.Button("Search Status")
                            rag_clear_btn = gr.Button("Clear Search Index")
                        rag_admin_status = gr.Markdown(DEFAULT_RAG_ADMIN_STATUS)

            with gr.Accordion("Diagnostics", open=False):
                with gr.Row():
                    refresh_debug_btn = gr.Button("Refresh Details")
                    clear_debug_btn = gr.Button("Clear Details")
                debug_details = gr.Markdown(debug_details_markdown(session_state))

            with gr.Tabs():
                with gr.Tab("Generate Sample Data"):
                    gr.Markdown(
                        """
                        ### Recommended Workflow
                        Follow the numbered sections below. The default settings are tuned for best-quality tabular output.
                        """,
                        elem_classes=["status-panel"],
                    )
                    with gr.Column(elem_classes=["workflow-step"]):
                        gr.Markdown("### 1. Describe Or Import\nTell Synthesizer what data you need, or import CSV/JSON rows to enrich.")
                        data_prompt = gr.Textbox(
                            label="Dataset Description",
                            placeholder="Example: Create realistic employee benefits renewal records with client, policy, premium, renewal date, and risk notes.",
                            lines=3,
                        )
                        suggest_fields_btn = gr.Button("Generate Fields", variant="primary")
                        num_rows = gr.Number(label="Rows To Generate", value=10, precision=0)
                        result_quality = gr.Dropdown(
                            choices=RESULT_QUALITY_CHOICES,
                            value=RESULT_QUALITY_CHOICES[0],
                            label="Result Quality",
                        )
                        with gr.Accordion("Prompt Starters", open=False):
                            with gr.Row():
                                example_btn_1 = gr.Button("Customer Contacts")
                                example_btn_2 = gr.Button("Support Tickets")
                                example_btn_3 = gr.Button("Insurance Inbox")

                    with gr.Accordion("Optional: Import Existing CSV/JSON", open=False):
                        gr.Markdown("Use existing rows as a starting point. Imported data is masked before model use by default.")
                        with gr.Row(elem_classes=["mini-bar"]):
                            with gr.Column(scale=2, min_width=240):
                                data_file = gr.File(label="Import CSV/JSON", type="filepath")
                            with gr.Column(scale=2, min_width=240):
                                import_privacy_mode = gr.Dropdown(
                                    choices=IMPORT_PRIVACY_CHOICES,
                                    value="Mask likely personal values",
                                    label="Import Privacy",
                                )
                            with gr.Column(scale=1, min_width=180):
                                apply_import_privacy_btn = gr.Button("Apply Privacy", variant="secondary")
                        data_status = gr.Markdown(
                            "Start from scratch or import a CSV/JSON file to use existing columns as a base. "
                            "Imported files default to privacy masking so the preview and AI context avoid likely personal values."
                        )
                        imported_columns = gr.HTML(imported_columns_markup([], "Mask likely personal values"))
                        with gr.Accordion("Imported Data Preview", open=False):
                            import_preview_text = gr.Markdown("No imported data yet.")
                            import_preview = gr.Dataframe(
                                label="Imported Data Preview",
                                interactive=False,
                                visible=False,
                                wrap=True,
                                max_height=240,
                            )

                    with gr.Column(elem_classes=["workflow-step"]):
                        gr.Markdown("### 2. Review Fields\nCheck the column names and describe what each generated field should contain.")
                        field_status = gr.Markdown("Review the schema at a glance, edit the grid, then save before generating.", elem_classes=["status-panel"])
                        gr.Markdown(
                            "Tip: use **Selected Row Editor** for friendlier labels, or edit the grid directly when you are comfortable with the column names.",
                            elem_classes=["schema-help"],
                        )
                        schema_overview = gr.HTML(field_rows_markup([]), elem_classes=["schema-overview"])
                        fields_grid = gr.Dataframe(
                            value=field_records_to_grid_dataframe([]),
                            headers=GRID_HEADERS,
                            datatype=["str", "str", "str", "str", "bool"],
                            interactive=True,
                            row_count=(8, "dynamic"),
                            wrap=True,
                            label="Editable Schema Grid",
                            max_height=460,
                            elem_classes=["schema-grid"],
                        )
                        with gr.Row(elem_classes=["field-toolbar"]):
                            new_field_btn = gr.Button("+ Add Row", variant="primary", scale=1)
                            remove_field_btn = gr.Button("Remove Last Row", scale=1)
                            save_field_btn = gr.Button("Save Rows", scale=1)
                        with gr.Accordion("Selected Row Editor", open=True):
                            gr.Markdown(
                                "Use this editor when you want a controlled type dropdown. It updates the visible grid only until you click **Save Rows**."
                            )
                            with gr.Row():
                                row_editor_choice = gr.Dropdown(
                                    choices=["Row 1"],
                                    value="Row 1",
                                    label="Row",
                                )
                                row_editor_apply_btn = gr.Button("Apply To Grid", variant="secondary")
                            with gr.Row():
                                row_editor_name = gr.Textbox(label="Name")
                                row_editor_type = gr.Dropdown(
                                    choices=FIELD_TYPE_CHOICES,
                                    value=FIELD_TYPE_CHOICES[0],
                                    label="Type",
                                )
                            row_editor_prompt = gr.Textbox(label="What Should This Field Contain?", lines=8)
                            row_editor_allow_duplicates = gr.Checkbox(label="Allow Duplicates", value=False)

                        with gr.Row(elem_classes=["bottom-actions primary-action-row"]):
                            generate_data_btn = gr.Button("Generate Data", variant="primary")
                            stop_data_btn = gr.Button("Stop", variant="stop")

                    with gr.Column(elem_classes=["workflow-step"]):
                        gr.Markdown("### 3. Generate And Review")
                        generation_progress = gr.Markdown("Generation progress will appear here once you start a run.")
                        generated_preview = gr.Dataframe(label="Generated Data Preview", interactive=False, visible=False, wrap=True)
                        with gr.Row(elem_classes=["primary-action-row"]):
                            review_quality_btn = gr.Button("Review Quality")
                        quality_report = gr.Markdown("Quality review will appear here after generation.")

                    with gr.Column(elem_classes=["workflow-step"]):
                        gr.Markdown("### 4. Export")
                        with gr.Accordion("Power BI Export", open=True):
                            power_bi_dataset_name = gr.Textbox(label="Dataset Name", value="Synthesizer Dataset")
                            power_bi_destination = gr.Textbox(
                                label="Destination Folder",
                                value=".web_ui_exports/power_bi",
                                placeholder="Use a local OneDrive/SharePoint synced folder for team refresh workflows.",
                            )
                            power_bi_privacy_mode = gr.Dropdown(
                                choices=POWER_BI_PRIVACY_CHOICES,
                                value=POWER_BI_PRIVACY_CHOICES[0],
                                label="Privacy Export Mode",
                            )
                            export_power_bi_btn = gr.Button("Export Power BI Run", variant="primary")
                        with gr.Accordion("Other Export Formats", open=False):
                            with gr.Row(elem_classes=["bottom-actions"]):
                                export_csv_btn = gr.Button("Use In Excel (CSV)")
                                export_narrative_pdf_btn = gr.Button("Narrative PDF")
                                export_json_btn = gr.Button("Developer JSON")
                                export_sql_btn = gr.Button("Developer SQL")
                        data_export_file = gr.File(label="Prepared Download", interactive=False)

                with gr.Tab("Work With Files"):
                    gr.Markdown("### 1. Add your sources\nUpload documents or add a web page. Files will be indexed when you run the task.")
                    with gr.Row():
                        files_upload = gr.File(label="Upload Files", type="filepath", file_count="multiple")
                        files_url = gr.Textbox(label="Add Web Page", placeholder="https://example.com/page")
                        add_url_btn = gr.Button("Add URL")
                        clear_files_btn = gr.Button("Clear Sources")
                    files_table = gr.Dataframe(headers=["name", "path"], datatype=["str", "str"], interactive=False, label="Current Sources")
                    with gr.Row():
                        selected_source = gr.Dropdown(choices=[], value=None, label="Selected Source")
                        reindex_source_btn = gr.Button("Re-index Selected")
                        remove_source_btn = gr.Button("Remove Selected")

                    gr.Markdown("### 2. Choose the result you want")
                    files_mode = gr.Dropdown(
                        choices=["Document Engine", "Quick Q&A", "Structured JSON"],
                        value="Document Engine",
                        label="File Task",
                    )
                    files_status = gr.Markdown(files_mode_helper("Document Engine"))
                    with gr.Group(visible=True) as preset_group:
                        with gr.Row():
                            preset_dropdown = gr.Dropdown(
                                choices=initial_preset_choices,
                                value=initial_preset_value,
                                label="Saved Prompt",
                            )
                            preset_name = gr.Textbox(label="Save Prompt As")
                            save_preset_btn = gr.Button("Save Prompt")
                            delete_preset_btn = gr.Button("Delete Prompt")
                    files_prompt = gr.Textbox(
                        label="What document should the files help you create?",
                        placeholder="e.g., Create an executive brief with findings, risks, and next steps.",
                        lines=3,
                    )

                    with gr.Group(visible=True) as document_group:
                        gr.Markdown("### Document Settings")
                        with gr.Row():
                            doc_bundle_exec_btn = gr.Button("Executive Brief")
                            doc_bundle_policy_btn = gr.Button("Policy Draft")
                            doc_bundle_action_btn = gr.Button("Action Plan")
                            doc_bundle_meeting_btn = gr.Button("Meeting Summary")
                        with gr.Row():
                            doc_mode = gr.Dropdown(choices=["Balanced", "File-based", "Creative"], value="Balanced", label="Grounding Style")
                            doc_pages = gr.Dropdown(
                                choices=PAGE_OPTIONS,
                                value="Let AI decide",
                                allow_custom_value=True,
                                label="Length",
                                info="Choose a preset or type a custom page count like 12 or 12 pages.",
                            )
                            doc_quality = gr.Dropdown(choices=["Fast", "Thorough"], value="Fast", label="Review Depth")
                        with gr.Row():
                            doc_audience = gr.Textbox(label="Audience", value="General")
                            doc_tone = gr.Textbox(label="Tone", value="professional")
                        with gr.Row():
                            doc_chart_enabled = gr.Checkbox(label="Include Charts", value=False)
                            doc_flow_enabled = gr.Checkbox(label="Include Flowchart", value=True)
                            doc_max_charts = gr.Number(label="Max Charts", value=3, precision=0)

                    with gr.Group(visible=False) as qa_group:
                        gr.Markdown("Quick Q&A keeps the answer grounded in the uploaded sources and supports follow-up questions.")
                        qa_question = gr.Textbox(
                            label="Ask The Uploaded Files",
                            placeholder="Ask a question, then ask follow-ups in the same chat.",
                            lines=2,
                        )
                        qa_ask_btn = gr.Button("Ask", variant="primary")

                    with gr.Group(visible=False) as json_group:
                        gr.Markdown("### Structured JSON Settings")
                        with gr.Row():
                            json_template_path = gr.File(label="JSON Template", type="filepath")
                            json_target_key = gr.Textbox(label="List Key To Fill", value="items")
                        with gr.Row():
                            json_mode = gr.Dropdown(choices=["Standard Generation", "Exhaustive Extraction"], value="Standard Generation", label="JSON Task")
                            json_clear_existing = gr.Checkbox(label="Replace Existing Items", value=True)

                    with gr.Row():
                        run_files_btn = gr.Button("Run Files Task", variant="primary")
                        stop_files_btn = gr.Button("Stop", variant="stop")
                    files_progress = gr.Markdown("Files progress will appear here after you run a task.")
                    files_download_pdf = gr.File(label="Document PDF", interactive=False)
                    files_download_docx = gr.File(label="Document DOCX", interactive=False)
                    files_download_json = gr.File(label="Structured JSON Download", interactive=False)
                    files_chat = gr.Markdown("Results will appear here after you run a Files task.")

        refresh_models_btn.click(
            fn=refresh_models,
            inputs=[session, model_id, provider, api_key, azure_endpoint, azure_deployment],
            outputs=[session, model_id, activity_log],
        )
        test_connection_btn.click(
            fn=test_connection,
            inputs=[session, model_id, provider, api_key, azure_endpoint, azure_deployment],
            outputs=[session, activity_log],
        )
        file_search_preset.change(
            fn=apply_file_search_preset,
            inputs=[file_search_preset, scanned_file_mode],
            outputs=[
                rag_backend,
                top_k,
                min_score,
                max_context_chars,
                ocr_mode,
                parser_mode,
                hybrid_search_enabled,
                rerank_enabled,
                summary_first_enabled,
                parent_context_enabled,
                graph_enabled,
                late_interaction_enabled,
                file_search_status,
            ],
            queue=False,
        )
        scanned_file_mode.change(
            fn=apply_file_search_preset,
            inputs=[file_search_preset, scanned_file_mode],
            outputs=[
                rag_backend,
                top_k,
                min_score,
                max_context_chars,
                ocr_mode,
                parser_mode,
                hybrid_search_enabled,
                rerank_enabled,
                summary_first_enabled,
                parent_context_enabled,
                graph_enabled,
                late_interaction_enabled,
                file_search_status,
            ],
            queue=False,
        )
        reset_workspace_btn.click(
            fn=reset_workspace_session,
            inputs=[session],
            outputs=[
                session,
                activity_log,
                data_prompt,
                num_rows,
                result_quality,
                data_file,
                import_privacy_mode,
                data_status,
                imported_columns,
                import_preview_text,
                import_preview,
                field_status,
                schema_overview,
                fields_grid,
                row_editor_choice,
                row_editor_name,
                row_editor_type,
                row_editor_prompt,
                row_editor_allow_duplicates,
                generation_progress,
                generated_preview,
                quality_report,
                data_export_file,
                files_upload,
                files_url,
                files_table,
                selected_source,
                files_mode,
                files_status,
                files_prompt,
                document_group,
                qa_group,
                json_group,
                preset_group,
                run_files_btn,
                stop_files_btn,
                preset_name,
                qa_question,
                doc_mode,
                doc_pages,
                doc_quality,
                doc_audience,
                doc_tone,
                doc_chart_enabled,
                doc_flow_enabled,
                doc_max_charts,
                json_template_path,
                json_target_key,
                json_mode,
                json_clear_existing,
                files_progress,
                files_download_pdf,
                files_download_docx,
                files_download_json,
                files_chat,
                collection_name,
                rag_admin_status,
                debug_details,
            ],
            queue=False,
        )
        save_config_btn.click(
            fn=save_config_file,
            inputs=[
                session,
                fields_grid,
                row_editor_choice,
                row_editor_name,
                row_editor_type,
                row_editor_prompt,
                row_editor_allow_duplicates,
                model_id,
                provider,
                api_key,
                azure_endpoint,
                azure_deployment,
                input_price_per_1m,
                output_price_per_1m,
                num_rows,
                similarity_threshold,
                max_retries,
                rag_backend,
                collection_name,
                top_k,
                min_score,
                max_context_chars,
                embedding_model,
                source_filter,
                qdrant_url,
                qdrant_api_key,
                ocr_mode,
                ocr_dpi,
                ocr_max_pages,
                ocr_max_regions_per_page,
                ocr_region_padding_px,
                ocr_gap_multiplier,
                ocr_min_extracted_chars,
                ocr_timeout_ms_per_page,
                parser_mode,
                hybrid_search_enabled,
                rerank_enabled,
                summary_first_enabled,
                summary_top_k,
                dense_top_k,
                lexical_top_k,
                parent_context_enabled,
                parent_context_max_chars,
                graph_enabled,
                graph_hops,
                graph_source_boost,
                late_interaction_enabled,
                late_interaction_weight,
                quick_qa_mode,
                doc_mode,
                doc_pages,
                doc_quality,
                doc_audience,
                doc_tone,
                doc_chart_enabled,
                doc_flow_enabled,
                doc_max_charts,
                import_privacy_mode,
            ],
            outputs=[session, saved_config_file, activity_log],
        )
        load_config_upload.change(
            fn=load_config_file,
            inputs=[session, load_config_upload],
            outputs=[
                session,
                model_id,
                provider,
                api_key,
                azure_endpoint,
                azure_deployment,
                input_price_per_1m,
                output_price_per_1m,
                num_rows,
                similarity_threshold,
                max_retries,
                rag_backend,
                collection_name,
                top_k,
                min_score,
                max_context_chars,
                embedding_model,
                source_filter,
                qdrant_url,
                qdrant_api_key,
                ocr_mode,
                ocr_dpi,
                ocr_max_pages,
                ocr_max_regions_per_page,
                ocr_region_padding_px,
                ocr_gap_multiplier,
                ocr_min_extracted_chars,
                ocr_timeout_ms_per_page,
                parser_mode,
                hybrid_search_enabled,
                rerank_enabled,
                summary_first_enabled,
                summary_top_k,
                dense_top_k,
                lexical_top_k,
                parent_context_enabled,
                parent_context_max_chars,
                graph_enabled,
                graph_hops,
                graph_source_boost,
                late_interaction_enabled,
                late_interaction_weight,
                quick_qa_mode,
                doc_mode,
                doc_pages,
                doc_quality,
                doc_audience,
                doc_tone,
                doc_chart_enabled,
                doc_flow_enabled,
                doc_max_charts,
                import_privacy_mode,
                fields_grid,
                field_status,
                activity_log,
            ],
        ).then(
            fn=sync_grid_row_editor,
            inputs=[fields_grid, row_editor_choice],
            outputs=[row_editor_choice, row_editor_name, row_editor_type, row_editor_prompt, row_editor_allow_duplicates],
            queue=False,
        ).then(
            fn=refresh_schema_overview,
            inputs=[fields_grid, row_editor_choice],
            outputs=[schema_overview],
            queue=False,
        )
        reset_config_btn.click(
            fn=reset_config,
            inputs=[session],
            outputs=[
                session,
                model_id,
                provider,
                api_key,
                azure_endpoint,
                azure_deployment,
                input_price_per_1m,
                output_price_per_1m,
                num_rows,
                similarity_threshold,
                max_retries,
                rag_backend,
                collection_name,
                top_k,
                min_score,
                max_context_chars,
                embedding_model,
                source_filter,
                qdrant_url,
                qdrant_api_key,
                ocr_mode,
                ocr_dpi,
                ocr_max_pages,
                ocr_max_regions_per_page,
                ocr_region_padding_px,
                ocr_gap_multiplier,
                ocr_min_extracted_chars,
                ocr_timeout_ms_per_page,
                parser_mode,
                hybrid_search_enabled,
                rerank_enabled,
                summary_first_enabled,
                summary_top_k,
                dense_top_k,
                lexical_top_k,
                parent_context_enabled,
                parent_context_max_chars,
                graph_enabled,
                graph_hops,
                graph_source_boost,
                late_interaction_enabled,
                late_interaction_weight,
                quick_qa_mode,
                doc_mode,
                doc_pages,
                doc_quality,
                doc_audience,
                doc_tone,
                doc_chart_enabled,
                doc_flow_enabled,
                doc_max_charts,
                import_privacy_mode,
                fields_grid,
                imported_columns,
                import_preview_text,
                import_preview,
                data_status,
                schema_overview,
                generated_preview,
                generation_progress,
                quality_report,
                data_export_file,
                files_table,
                selected_source,
                files_chat,
                files_status,
                files_progress,
                files_download_pdf,
                files_download_docx,
                files_download_json,
                rag_admin_status,
                activity_log,
                debug_details,
            ],
        )
        rag_status_btn.click(
            fn=get_search_status,
            inputs=[
                session,
                model_id,
                provider,
                api_key,
                azure_endpoint,
                azure_deployment,
                input_price_per_1m,
                output_price_per_1m,
                num_rows,
                similarity_threshold,
                max_retries,
                rag_backend,
                collection_name,
                top_k,
                min_score,
                max_context_chars,
                embedding_model,
                source_filter,
                qdrant_url,
                qdrant_api_key,
                ocr_mode,
                ocr_dpi,
                ocr_max_pages,
                ocr_max_regions_per_page,
                ocr_region_padding_px,
                ocr_gap_multiplier,
                ocr_min_extracted_chars,
                ocr_timeout_ms_per_page,
                parser_mode,
                hybrid_search_enabled,
                rerank_enabled,
                summary_first_enabled,
                summary_top_k,
                dense_top_k,
                lexical_top_k,
                parent_context_enabled,
                parent_context_max_chars,
                graph_enabled,
                graph_hops,
                graph_source_boost,
                late_interaction_enabled,
                late_interaction_weight,
                quick_qa_mode,
                doc_mode,
                doc_pages,
                doc_quality,
                doc_audience,
                doc_tone,
                doc_chart_enabled,
                doc_flow_enabled,
                doc_max_charts,
            ],
            outputs=[session, rag_admin_status, activity_log, debug_details],
        )
        rag_clear_btn.click(
            fn=clear_search_index,
            inputs=[
                session,
                model_id,
                provider,
                api_key,
                azure_endpoint,
                azure_deployment,
                input_price_per_1m,
                output_price_per_1m,
                num_rows,
                similarity_threshold,
                max_retries,
                rag_backend,
                collection_name,
                top_k,
                min_score,
                max_context_chars,
                embedding_model,
                source_filter,
                qdrant_url,
                qdrant_api_key,
                ocr_mode,
                ocr_dpi,
                ocr_max_pages,
                ocr_max_regions_per_page,
                ocr_region_padding_px,
                ocr_gap_multiplier,
                ocr_min_extracted_chars,
                ocr_timeout_ms_per_page,
                parser_mode,
                hybrid_search_enabled,
                rerank_enabled,
                summary_first_enabled,
                summary_top_k,
                dense_top_k,
                lexical_top_k,
                parent_context_enabled,
                parent_context_max_chars,
                graph_enabled,
                graph_hops,
                graph_source_boost,
                late_interaction_enabled,
                late_interaction_weight,
                quick_qa_mode,
                doc_mode,
                doc_pages,
                doc_quality,
                doc_audience,
                doc_tone,
                doc_chart_enabled,
                doc_flow_enabled,
                doc_max_charts,
            ],
            outputs=[session, files_table, selected_source, files_chat, files_status, rag_admin_status, activity_log, debug_details],
        )
        refresh_debug_btn.click(
            fn=refresh_debug_details,
            inputs=[session],
            outputs=[session, debug_details, activity_log],
            queue=False,
        )
        clear_debug_btn.click(
            fn=clear_debug_details,
            inputs=[session],
            outputs=[session, debug_details, activity_log],
            queue=False,
        )

        example_btn_1.click(lambda: "Create a customer contact dataset with name, email, phone number, company, and region.", outputs=[data_prompt])
        example_btn_2.click(lambda: "Create support ticket data with ticket ID, issue type, customer priority, summary, status, and resolution note.", outputs=[data_prompt])
        example_btn_3.click(lambda: "I need emails that you would find in a private medical insurance company inbox from clients, include minimum of 7 columns.", outputs=[data_prompt])

        data_file.change(
            fn=import_data_file,
            inputs=[session, data_file, import_privacy_mode],
            outputs=[
                session,
                num_rows,
                import_preview_text,
                imported_columns,
                import_preview,
                fields_grid,
                field_status,
                schema_overview,
                activity_log,
            ],
        ).then(
            fn=sync_grid_row_editor,
            inputs=[fields_grid, row_editor_choice],
            outputs=[row_editor_choice, row_editor_name, row_editor_type, row_editor_prompt, row_editor_allow_duplicates],
            queue=False,
        )
        apply_import_privacy_btn.click(
            fn=apply_import_privacy_mode,
            inputs=[session, import_privacy_mode],
            outputs=[session, import_privacy_mode, imported_columns, import_preview, import_preview_text, activity_log],
        )
        suggest_fields_btn.click(
            fn=suggest_fields_and_sync_editor,
            inputs=[session, data_prompt, model_id, provider, api_key, azure_endpoint, azure_deployment],
            outputs=[
                session,
                fields_grid,
                field_status,
                activity_log,
                row_editor_choice,
                row_editor_name,
                row_editor_type,
                row_editor_prompt,
                row_editor_allow_duplicates,
                schema_overview,
            ],
        )
        new_field_btn.click(
            fn=add_grid_row,
            inputs=[fields_grid],
            outputs=[fields_grid, field_status],
        ).then(
            fn=sync_grid_row_editor,
            inputs=[fields_grid, row_editor_choice],
            outputs=[row_editor_choice, row_editor_name, row_editor_type, row_editor_prompt, row_editor_allow_duplicates],
            queue=False,
        ).then(
            fn=refresh_schema_overview,
            inputs=[fields_grid, row_editor_choice],
            outputs=[schema_overview],
            queue=False,
        )
        save_field_btn.click(
            fn=save_grid_rows,
            inputs=[session, fields_grid],
            outputs=[session, fields_grid, field_status, activity_log],
        ).then(
            fn=sync_grid_row_editor,
            inputs=[fields_grid, row_editor_choice],
            outputs=[row_editor_choice, row_editor_name, row_editor_type, row_editor_prompt, row_editor_allow_duplicates],
            queue=False,
        ).then(
            fn=refresh_schema_overview,
            inputs=[fields_grid, row_editor_choice],
            outputs=[schema_overview],
            queue=False,
        )
        remove_field_btn.click(
            fn=remove_last_grid_row,
            inputs=[fields_grid],
            outputs=[fields_grid, field_status],
        ).then(
            fn=sync_grid_row_editor,
            inputs=[fields_grid, row_editor_choice],
            outputs=[row_editor_choice, row_editor_name, row_editor_type, row_editor_prompt, row_editor_allow_duplicates],
            queue=False,
        ).then(
            fn=refresh_schema_overview,
            inputs=[fields_grid, row_editor_choice],
            outputs=[schema_overview],
            queue=False,
        )
        fields_grid.input(
            fn=sync_grid_row_editor,
            inputs=[fields_grid, row_editor_choice],
            outputs=[row_editor_choice, row_editor_name, row_editor_type, row_editor_prompt, row_editor_allow_duplicates],
            queue=False,
        ).then(
            fn=refresh_schema_overview,
            inputs=[fields_grid, row_editor_choice],
            outputs=[schema_overview],
            queue=False,
        )
        row_editor_choice.change(
            fn=load_grid_row_editor,
            inputs=[fields_grid, row_editor_choice],
            outputs=[row_editor_name, row_editor_type, row_editor_prompt, row_editor_allow_duplicates],
            queue=False,
        ).then(
            fn=refresh_schema_overview,
            inputs=[fields_grid, row_editor_choice],
            outputs=[schema_overview],
            queue=False,
        )
        row_editor_apply_btn.click(
            fn=apply_grid_row_edit,
            inputs=[
                fields_grid,
                row_editor_choice,
                row_editor_name,
                row_editor_type,
                row_editor_prompt,
                row_editor_allow_duplicates,
            ],
            outputs=[
                fields_grid,
                row_editor_choice,
                row_editor_name,
                row_editor_type,
                row_editor_prompt,
                row_editor_allow_duplicates,
                field_status,
            ],
            queue=False,
        ).then(
            fn=refresh_schema_overview,
            inputs=[fields_grid, row_editor_choice],
            outputs=[schema_overview],
            queue=False,
        )
        generate_data_btn.click(
            fn=generate_data,
            inputs=[
                session,
                fields_grid,
                model_id,
                provider,
                api_key,
                azure_endpoint,
                azure_deployment,
                input_price_per_1m,
                output_price_per_1m,
                num_rows,
                similarity_threshold,
                max_retries,
                result_quality,
                rag_backend,
                collection_name,
                top_k,
                min_score,
                max_context_chars,
                embedding_model,
                source_filter,
                qdrant_url,
                qdrant_api_key,
                ocr_mode,
                ocr_dpi,
                ocr_max_pages,
                ocr_max_regions_per_page,
                ocr_region_padding_px,
                ocr_gap_multiplier,
                ocr_min_extracted_chars,
                ocr_timeout_ms_per_page,
                parser_mode,
                hybrid_search_enabled,
                rerank_enabled,
                summary_first_enabled,
                summary_top_k,
                dense_top_k,
                lexical_top_k,
                parent_context_enabled,
                parent_context_max_chars,
                graph_enabled,
                graph_hops,
                graph_source_boost,
                late_interaction_enabled,
                late_interaction_weight,
                quick_qa_mode,
                doc_mode,
                doc_pages,
                doc_quality,
                doc_audience,
                doc_tone,
                doc_chart_enabled,
                doc_flow_enabled,
                doc_max_charts,
            ],
            outputs=[session, generated_preview, generation_progress, field_status, activity_log],
        )
        stop_data_btn.click(
            fn=request_stop_data_generation,
            inputs=[session],
            outputs=[session, generation_progress, field_status, activity_log],
            queue=False,
        )
        export_csv_btn.click(fn=lambda s: export_generated_data(s, "csv"), inputs=[session], outputs=[session, data_export_file, field_status, activity_log])
        export_narrative_pdf_btn.click(fn=lambda s: export_generated_data(s, "pdf_narrative"), inputs=[session], outputs=[session, data_export_file, field_status, activity_log])
        export_json_btn.click(fn=lambda s: export_generated_data(s, "json"), inputs=[session], outputs=[session, data_export_file, field_status, activity_log])
        export_sql_btn.click(fn=lambda s: export_generated_data(s, "sql"), inputs=[session], outputs=[session, data_export_file, field_status, activity_log])
        export_power_bi_btn.click(
            fn=export_power_bi_data,
            inputs=[session, power_bi_dataset_name, power_bi_destination, power_bi_privacy_mode, model_id, provider],
            outputs=[session, data_export_file, field_status, activity_log],
        )
        review_quality_btn.click(fn=review_generated_data_quality, inputs=[session], outputs=[session, quality_report, activity_log])

        files_mode.change(
            fn=files_mode_changed,
            inputs=[files_mode],
            outputs=[files_status, files_prompt, document_group, qa_group, json_group, preset_group, run_files_btn, stop_files_btn],
        )
        files_upload.change(
            fn=register_uploaded_files,
            inputs=[session, files_upload, files_mode],
            outputs=[session, files_table, selected_source, files_status, activity_log],
        )
        add_url_btn.click(
            fn=add_url_source,
            inputs=[session, files_url, files_mode],
            outputs=[session, files_table, selected_source, files_status, activity_log],
        )
        clear_files_btn.click(
            fn=clear_files,
            inputs=[session, files_mode],
            outputs=[session, files_table, selected_source, files_chat, files_status, files_progress, activity_log],
        )
        reindex_source_btn.click(
            fn=reindex_selected_source,
            inputs=[
                session,
                selected_source,
                files_mode,
                model_id,
                provider,
                api_key,
                azure_endpoint,
                azure_deployment,
                input_price_per_1m,
                output_price_per_1m,
                num_rows,
                similarity_threshold,
                max_retries,
                rag_backend,
                collection_name,
                top_k,
                min_score,
                max_context_chars,
                embedding_model,
                source_filter,
                qdrant_url,
                qdrant_api_key,
                ocr_mode,
                ocr_dpi,
                ocr_max_pages,
                ocr_max_regions_per_page,
                ocr_region_padding_px,
                ocr_gap_multiplier,
                ocr_min_extracted_chars,
                ocr_timeout_ms_per_page,
                parser_mode,
                hybrid_search_enabled,
                rerank_enabled,
                summary_first_enabled,
                summary_top_k,
                dense_top_k,
                lexical_top_k,
                parent_context_enabled,
                parent_context_max_chars,
                graph_enabled,
                graph_hops,
                graph_source_boost,
                late_interaction_enabled,
                late_interaction_weight,
                quick_qa_mode,
                doc_mode,
                doc_pages,
                doc_quality,
                doc_audience,
                doc_tone,
                doc_chart_enabled,
                doc_flow_enabled,
                doc_max_charts,
            ],
            outputs=[session, files_table, selected_source, files_status, activity_log],
        )
        remove_source_btn.click(
            fn=remove_selected_source,
            inputs=[session, selected_source, files_mode],
            outputs=[session, files_table, selected_source, files_chat, files_status, activity_log],
        )
        preset_dropdown.change(
            fn=apply_file_preset,
            inputs=[session, preset_dropdown],
            outputs=[session, files_prompt, preset_name, activity_log],
            queue=False,
        )
        save_preset_btn.click(
            fn=save_file_preset,
            inputs=[session, preset_name, files_prompt, preset_dropdown],
            outputs=[session, preset_dropdown, preset_name, activity_log],
        )
        delete_preset_btn.click(
            fn=delete_file_preset,
            inputs=[session, preset_dropdown],
            outputs=[session, preset_dropdown, preset_name, activity_log],
        )
        doc_bundle_exec_btn.click(
            fn=lambda session: apply_doc_bundle(session, "Executive Brief"),
            inputs=[session],
            outputs=[session, files_prompt, doc_mode, doc_pages, doc_quality, doc_audience, doc_tone, activity_log],
            queue=False,
        )
        doc_bundle_policy_btn.click(
            fn=lambda session: apply_doc_bundle(session, "Policy Draft"),
            inputs=[session],
            outputs=[session, files_prompt, doc_mode, doc_pages, doc_quality, doc_audience, doc_tone, activity_log],
            queue=False,
        )
        doc_bundle_action_btn.click(
            fn=lambda session: apply_doc_bundle(session, "Action Plan"),
            inputs=[session],
            outputs=[session, files_prompt, doc_mode, doc_pages, doc_quality, doc_audience, doc_tone, activity_log],
            queue=False,
        )
        doc_bundle_meeting_btn.click(
            fn=lambda session: apply_doc_bundle(session, "Meeting Summary"),
            inputs=[session],
            outputs=[session, files_prompt, doc_mode, doc_pages, doc_quality, doc_audience, doc_tone, activity_log],
            queue=False,
        )
        run_files_btn.click(
            fn=run_files_task,
            inputs=[
                session,
                files_mode,
                files_prompt,
                model_id,
                provider,
                api_key,
                azure_endpoint,
                azure_deployment,
                input_price_per_1m,
                output_price_per_1m,
                num_rows,
                similarity_threshold,
                max_retries,
                rag_backend,
                collection_name,
                top_k,
                min_score,
                max_context_chars,
                embedding_model,
                source_filter,
                qdrant_url,
                qdrant_api_key,
                ocr_mode,
                ocr_dpi,
                ocr_max_pages,
                ocr_max_regions_per_page,
                ocr_region_padding_px,
                ocr_gap_multiplier,
                ocr_min_extracted_chars,
                ocr_timeout_ms_per_page,
                parser_mode,
                hybrid_search_enabled,
                rerank_enabled,
                summary_first_enabled,
                summary_top_k,
                dense_top_k,
                lexical_top_k,
                parent_context_enabled,
                parent_context_max_chars,
                graph_enabled,
                graph_hops,
                graph_source_boost,
                late_interaction_enabled,
                late_interaction_weight,
                quick_qa_mode,
                doc_mode,
                doc_pages,
                doc_quality,
                doc_audience,
                doc_tone,
                doc_chart_enabled,
                doc_flow_enabled,
                doc_max_charts,
                json_template_path,
                json_target_key,
                json_mode,
                json_clear_existing,
            ],
            outputs=[session, files_chat, files_status, files_progress, files_download_pdf, files_download_docx, files_download_json, activity_log],
        )
        qa_ask_btn.click(
            fn=ask_quick_qa_chat,
            inputs=[
                session,
                qa_question,
                model_id,
                provider,
                api_key,
                azure_endpoint,
                azure_deployment,
                input_price_per_1m,
                output_price_per_1m,
                num_rows,
                similarity_threshold,
                max_retries,
                rag_backend,
                collection_name,
                top_k,
                min_score,
                max_context_chars,
                embedding_model,
                source_filter,
                qdrant_url,
                qdrant_api_key,
                ocr_mode,
                ocr_dpi,
                ocr_max_pages,
                ocr_max_regions_per_page,
                ocr_region_padding_px,
                ocr_gap_multiplier,
                ocr_min_extracted_chars,
                ocr_timeout_ms_per_page,
                parser_mode,
                hybrid_search_enabled,
                rerank_enabled,
                summary_first_enabled,
                summary_top_k,
                dense_top_k,
                lexical_top_k,
                parent_context_enabled,
                parent_context_max_chars,
                graph_enabled,
                graph_hops,
                graph_source_boost,
                late_interaction_enabled,
                late_interaction_weight,
                quick_qa_mode,
                doc_mode,
                doc_pages,
                doc_quality,
                doc_audience,
                doc_tone,
                doc_chart_enabled,
                doc_flow_enabled,
                doc_max_charts,
            ],
            outputs=[session, files_chat, files_status, files_progress, activity_log],
        )
        qa_question.submit(
            fn=ask_quick_qa_chat,
            inputs=[
                session,
                qa_question,
                model_id,
                provider,
                api_key,
                azure_endpoint,
                azure_deployment,
                input_price_per_1m,
                output_price_per_1m,
                num_rows,
                similarity_threshold,
                max_retries,
                rag_backend,
                collection_name,
                top_k,
                min_score,
                max_context_chars,
                embedding_model,
                source_filter,
                qdrant_url,
                qdrant_api_key,
                ocr_mode,
                ocr_dpi,
                ocr_max_pages,
                ocr_max_regions_per_page,
                ocr_region_padding_px,
                ocr_gap_multiplier,
                ocr_min_extracted_chars,
                ocr_timeout_ms_per_page,
                parser_mode,
                hybrid_search_enabled,
                rerank_enabled,
                summary_first_enabled,
                summary_top_k,
                dense_top_k,
                lexical_top_k,
                parent_context_enabled,
                parent_context_max_chars,
                graph_enabled,
                graph_hops,
                graph_source_boost,
                late_interaction_enabled,
                late_interaction_weight,
                quick_qa_mode,
                doc_mode,
                doc_pages,
                doc_quality,
                doc_audience,
                doc_tone,
                doc_chart_enabled,
                doc_flow_enabled,
                doc_max_charts,
            ],
            outputs=[session, files_chat, files_status, files_progress, activity_log],
        )
        stop_files_btn.click(
            fn=request_stop_files_task,
            inputs=[session],
            outputs=[session, files_status, files_progress, activity_log],
            queue=False,
        )

    app.queue(default_concurrency_limit=2)
    return app


def launch_web_ui(*, server_name: str = "127.0.0.1", server_port: int = 7860, inbrowser: bool = True) -> None:
    cleanup_report = prepare_clean_workspace()
    app = create_app(
        startup_collection_name=cleanup_report["startup_collection_name"],
        startup_cleanup_message=cleanup_report["message"],
    )
    app.launch(server_name=server_name, server_port=server_port, inbrowser=inbrowser)


if __name__ == "__main__":
    launch_web_ui()
