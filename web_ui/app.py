from __future__ import annotations

import gradio as gr

from core.app_config import PAGE_OPTIONS
from core.models import AIProvider, RagBackend
from web_ui.actions.config_actions import load_config_file, refresh_models, save_config_file, test_connection
from web_ui.actions.data_actions import (
    add_grid_row,
    export_generated_data,
    generate_data,
    import_data_file,
    remove_last_grid_row,
    review_generated_data_quality,
    save_grid_rows,
    suggest_fields,
)
from web_ui.actions.files_actions import add_url_source, clear_files, files_mode_changed, files_mode_helper, register_uploaded_files, run_files_task
from web_ui.adapters import GRID_HEADERS, field_records_to_grid_dataframe
from web_ui.state import activity_markdown, new_session_state


WEB_UI_CSS = """
.app-shell { max-width: 1500px; margin: 0 auto; }
.prompt-shell, .schema-shell { border: 2px solid #111827; border-radius: 26px; background: #ffffff; }
.prompt-shell { padding: 16px 18px; margin-bottom: 22px; }
.mini-bar { margin: 6px 0 18px; }
.schema-shell { padding: 22px; margin-top: 10px; }
.field-toolbar { margin: 14px 0 0; }
.schema-help { color: #475569; margin: 8px 0 12px; }
.bottom-actions { margin-top: 18px; }
"""


def create_app() -> gr.Blocks:
    session_state = new_session_state()
    with gr.Blocks(title="Synthesizer Workspace Web") as app:
        session = gr.State(session_state)

        with gr.Column(elem_classes=["app-shell"]):
            gr.Markdown(
                """
                # Synthesizer Workspace
                Create realistic sample data or work with files through a browser-based workflow.
                """
            )
            activity_log = gr.Markdown(activity_markdown(session_state))

            with gr.Accordion("Connection And Technical Settings", open=False):
                with gr.Row():
                    provider = gr.Dropdown(
                        choices=[provider.value for provider in AIProvider],
                        value=AIProvider.LM_STUDIO.value,
                        label="AI Service",
                    )
                    model_id = gr.Dropdown(
                        choices=["local-model"],
                        value="local-model",
                        allow_custom_value=True,
                        label="AI Model",
                    )
                    api_key = gr.Textbox(label="API Key", type="password")
                with gr.Row():
                    azure_endpoint = gr.Textbox(label="Azure Endpoint")
                    azure_deployment = gr.Textbox(label="Azure Deployment")
                with gr.Row():
                    num_rows = gr.Number(label="How Many Rows?", value=10, precision=0)
                    similarity_threshold = gr.Number(label="Uniqueness Strictness", value=0.85)
                    max_retries = gr.Number(label="Retry Limit", value=50, precision=0)
                with gr.Row():
                    input_price_per_1m = gr.Number(label="Input Price ($/1M)", value=0.15)
                    output_price_per_1m = gr.Number(label="Output Price ($/1M)", value=0.60)
                with gr.Row():
                    refresh_models_btn = gr.Button("Refresh Models")
                    test_connection_btn = gr.Button("Check Connection")
                with gr.Row():
                    load_config_upload = gr.File(label="Load Config JSON", type="filepath")
                    save_config_btn = gr.Button("Save Config JSON")
                    saved_config_file = gr.File(label="Saved Config Download", interactive=False)
                with gr.Accordion("Retrieval Settings", open=False):
                    with gr.Row():
                        rag_backend = gr.Dropdown(
                            choices=[RagBackend.NATIVE.value, RagBackend.LLAMA_INDEX.value],
                            value=RagBackend.LLAMA_INDEX.value,
                            label="RAG Backend",
                        )
                        collection_name = gr.Textbox(label="Search Collection", value="synthesizer_default")
                        quick_qa_mode = gr.Dropdown(
                            choices=["Broader Analysis", "Pinpoint Quick"],
                            value="Broader Analysis",
                            label="Quick Q&A Style",
                        )
                    with gr.Row():
                        top_k = gr.Number(label="Top Matches", value=5, precision=0)
                        min_score = gr.Number(label="Minimum Match Score", value=0.25)
                        max_context_chars = gr.Number(label="Max Context Characters", value=3000, precision=0)
                    with gr.Row():
                        embedding_model = gr.Textbox(label="Embedding Model", value="BAAI/bge-small-en-v1.5")
                        source_filter = gr.Textbox(label="Source Filter")
                    with gr.Row():
                        qdrant_url = gr.Textbox(label="Qdrant URL", value=":memory:")
                        qdrant_api_key = gr.Textbox(label="Qdrant API Key", type="password")
                    with gr.Row():
                        ocr_mode = gr.Dropdown(choices=["off", "auto", "on"], value="off", label="OCR Mode")
                        parser_mode = gr.Dropdown(choices=["auto", "pdf_only", "docling"], value="auto", label="Parser Mode")

            with gr.Tabs():
                with gr.Tab("Generate Sample Data"):
                    with gr.Group(elem_classes=["prompt-shell"]):
                        with gr.Row():
                            data_prompt = gr.Textbox(
                                label="",
                                show_label=False,
                                placeholder="ai prompt",
                                lines=2,
                                scale=9,
                            )
                            suggest_fields_btn = gr.Button("Generate Fields", variant="primary", scale=2, min_width=180)
                        with gr.Row():
                            example_btn_1 = gr.Button("Customer Contacts")
                            example_btn_2 = gr.Button("Support Tickets")
                            example_btn_3 = gr.Button("Insurance Inbox")

                    with gr.Row(elem_classes=["mini-bar"]):
                        with gr.Column(scale=1, min_width=220):
                            data_file = gr.File(label="Import CSV/JSON", type="filepath")
                        with gr.Column(scale=4):
                            data_status = gr.Markdown("Start from scratch or import a CSV/JSON file to use existing columns as a base.")

                    import_preview_text = gr.Markdown("No imported data yet.")
                    import_preview = gr.Dataframe(label="Imported Data Preview", interactive=False, visible=False, wrap=True, max_height=260)

                    with gr.Group(elem_classes=["schema-shell"]):
                        field_status = gr.Markdown("Edit the visible rows directly, then save the list before generating.")
                        gr.Markdown(
                            "Allowed type values: `Short Text`, `Long Text`, `Numeric`, `Categorical`, `Boolean`, `Auto Increment (ID)`, `Faker / Deterministic`.",
                            elem_classes=["schema-help"],
                        )
                        fields_grid = gr.Dataframe(
                            value=field_records_to_grid_dataframe([]),
                            headers=GRID_HEADERS,
                            datatype=["str", "str", "str", "str", "bool"],
                            interactive=True,
                            wrap=True,
                            label="Editable Schema Rows",
                        )
                        with gr.Row(elem_classes=["field-toolbar"]):
                            new_field_btn = gr.Button("+ Add Row", variant="primary", scale=1)
                            remove_field_btn = gr.Button("Remove Last Row", scale=1)
                            save_field_btn = gr.Button("Save Rows", scale=1)

                        with gr.Row(elem_classes=["bottom-actions"]):
                            export_csv_btn = gr.Button("Export CSV")
                            export_json_btn = gr.Button("Export JSON")
                            export_sql_btn = gr.Button("Export SQL")
                            review_quality_btn = gr.Button("Review Quality")
                            generate_data_btn = gr.Button("Generate Data", variant="primary")

                    generation_progress = gr.Markdown("Generation progress will appear here once you start a run.")
                    data_export_file = gr.File(label="Prepared Download", interactive=False)
                    quality_report = gr.Markdown("Quality review will appear here after generation.")
                    generated_preview = gr.Dataframe(label="Generated Data Preview", interactive=False, visible=False, wrap=True)

                with gr.Tab("Work With Files"):
                    gr.Markdown("### 1. Add your sources\nUpload documents or add a web page. Files will be indexed when you run the task.")
                    with gr.Row():
                        files_upload = gr.File(label="Upload Files", type="filepath", file_count="multiple")
                        files_url = gr.Textbox(label="Add Web Page", placeholder="https://example.com/page")
                        add_url_btn = gr.Button("Add URL")
                        clear_files_btn = gr.Button("Clear Sources")
                    files_table = gr.Dataframe(headers=["name", "path"], datatype=["str", "str"], interactive=False, label="Current Sources")

                    gr.Markdown("### 2. Choose the result you want")
                    files_mode = gr.Dropdown(
                        choices=["Document Engine", "Quick Q&A", "Structured JSON"],
                        value="Document Engine",
                        label="File Task",
                    )
                    files_status = gr.Markdown(files_mode_helper("Document Engine"))
                    files_prompt = gr.Textbox(
                        label="What document should the files help you create?",
                        placeholder="e.g., Create an executive brief with findings, risks, and next steps.",
                        lines=3,
                    )

                    with gr.Group(visible=True) as document_group:
                        gr.Markdown("### Document Settings")
                        with gr.Row():
                            doc_mode = gr.Dropdown(choices=["Balanced", "File-based", "Creative"], value="Balanced", label="Grounding Style")
                            doc_pages = gr.Dropdown(choices=PAGE_OPTIONS, value="Let AI decide", label="Length")
                            doc_quality = gr.Dropdown(choices=["Fast", "Thorough"], value="Fast", label="Review Depth")
                        with gr.Row():
                            doc_audience = gr.Textbox(label="Audience", value="General")
                            doc_tone = gr.Textbox(label="Tone", value="professional")
                        with gr.Row():
                            doc_chart_enabled = gr.Checkbox(label="Include Charts", value=False)
                            doc_flow_enabled = gr.Checkbox(label="Include Flowchart", value=True)
                            doc_max_charts = gr.Number(label="Max Charts", value=3, precision=0)

                    with gr.Group(visible=False) as qa_group:
                        gr.Markdown("Quick Q&A keeps the answer grounded in the uploaded sources and returns citations when available.")

                    with gr.Group(visible=False) as json_group:
                        gr.Markdown("### Structured JSON Settings")
                        with gr.Row():
                            json_template_path = gr.File(label="JSON Template", type="filepath")
                            json_target_key = gr.Textbox(label="List Key To Fill", value="items")
                        with gr.Row():
                            json_mode = gr.Dropdown(choices=["Standard Generation", "Exhaustive Extraction"], value="Standard Generation", label="JSON Task")
                            json_clear_existing = gr.Checkbox(label="Replace Existing Items", value=True)

                    run_files_btn = gr.Button("Run Files Task", variant="primary")
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
        save_config_btn.click(
            fn=save_config_file,
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
                parser_mode,
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
                parser_mode,
                quick_qa_mode,
                doc_mode,
                doc_pages,
                doc_quality,
                doc_audience,
                doc_tone,
                doc_chart_enabled,
                doc_flow_enabled,
                doc_max_charts,
                fields_grid,
                activity_log,
            ],
        )

        example_btn_1.click(lambda: "Create a customer contact dataset with name, email, phone number, company, and region.", outputs=[data_prompt])
        example_btn_2.click(lambda: "Create support ticket data with ticket ID, issue type, customer priority, summary, status, and resolution note.", outputs=[data_prompt])
        example_btn_3.click(lambda: "I need emails that you would find in a private medical insurance company inbox from clients, include minimum of 7 columns.", outputs=[data_prompt])

        data_file.change(
            fn=import_data_file,
            inputs=[session, data_file],
            outputs=[
                session,
                num_rows,
                import_preview_text,
                import_preview,
                fields_grid,
                field_status,
                activity_log,
            ],
        )
        suggest_fields_btn.click(
            fn=suggest_fields,
            inputs=[session, data_prompt, model_id, provider, api_key, azure_endpoint, azure_deployment],
            outputs=[
                session,
                fields_grid,
                field_status,
                activity_log,
            ],
        )
        new_field_btn.click(
            fn=add_grid_row,
            inputs=[fields_grid],
            outputs=[fields_grid, field_status],
        )
        save_field_btn.click(
            fn=save_grid_rows,
            inputs=[session, fields_grid],
            outputs=[session, fields_grid, field_status, activity_log],
        )
        remove_field_btn.click(
            fn=remove_last_grid_row,
            inputs=[fields_grid],
            outputs=[fields_grid, field_status],
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
                parser_mode,
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
        export_csv_btn.click(fn=lambda s: export_generated_data(s, "csv"), inputs=[session], outputs=[session, data_export_file, field_status, activity_log])
        export_json_btn.click(fn=lambda s: export_generated_data(s, "json"), inputs=[session], outputs=[session, data_export_file, field_status, activity_log])
        export_sql_btn.click(fn=lambda s: export_generated_data(s, "sql"), inputs=[session], outputs=[session, data_export_file, field_status, activity_log])
        review_quality_btn.click(fn=review_generated_data_quality, inputs=[session], outputs=[session, quality_report, activity_log])

        files_mode.change(
            fn=files_mode_changed,
            inputs=[files_mode],
            outputs=[files_status, files_prompt, document_group, qa_group, json_group],
        )
        files_upload.change(
            fn=register_uploaded_files,
            inputs=[session, files_upload, files_mode],
            outputs=[session, files_table, files_status, activity_log],
        )
        add_url_btn.click(
            fn=add_url_source,
            inputs=[session, files_url, files_mode],
            outputs=[session, files_table, files_status, activity_log],
        )
        clear_files_btn.click(
            fn=clear_files,
            inputs=[session, files_mode],
            outputs=[session, files_table, files_chat, files_status, activity_log],
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
                parser_mode,
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
            outputs=[session, files_chat, files_status, files_download_pdf, files_download_docx, files_download_json, activity_log],
        )

    app.css = WEB_UI_CSS
    app.queue(default_concurrency_limit=2)
    return app


def launch_web_ui(*, server_name: str = "127.0.0.1", server_port: int = 7860, inbrowser: bool = True) -> None:
    app = create_app()
    app.launch(server_name=server_name, server_port=server_port, inbrowser=inbrowser)


if __name__ == "__main__":
    launch_web_ui()
