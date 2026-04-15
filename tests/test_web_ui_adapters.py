import os
import sys
import time as pytime
from types import SimpleNamespace

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.app_config import (
    build_generator_config,
    normalize_loaded_config,
    pages_label_from_target_words,
    resolve_document_target_words,
    serialize_ui_config,
)
from core.controller import GeneratorController
from core.document_engine.orchestrator import DocumentOrchestrator
from core.document_engine.models import DocumentGenerationOptions, DocumentMode
from core.llm_client import LLMClient
from core.models import AIProvider, ColumnDefinition, ColumnType
from web_ui.actions import files_actions
from web_ui.adapters import columns_to_field_records, field_records_to_columns, field_records_to_grid_dataframe, normalize_field_record
from web_ui.actions import data_actions
from web_ui.state import new_session_state
from web_ui import state as web_state


def test_build_generator_config_maps_web_values_to_runtime_models():
    config = build_generator_config(
        {
            "model_id": "local-model",
            "provider": AIProvider.LM_STUDIO.value,
            "doc_mode": "File-based",
            "doc_pages": "3 pages",
            "rag_backend": "Native",
            "num_rows": 12,
        },
        include_existing_data=False,
    )

    assert config.provider == AIProvider.LM_STUDIO
    assert config.num_rows == 12
    assert config.rag is not None
    assert config.rag.backend.value == "Native"
    assert config.document_engine is not None
    assert config.document_engine.mode == "strict_grounded"
    assert config.document_engine.target_words == 1500


def test_web_field_record_roundtrip_preserves_key_constraints():
    columns = [
        ColumnDefinition(
            name="email",
            type=ColumnType.SHORT_TEXT,
            prompt_instruction="Generate a business email address",
        ),
        ColumnDefinition(
            name="score",
            type=ColumnType.NUMERIC,
            prompt_instruction="A rating between 1 and 10",
        ),
    ]
    columns[0].constraints.allow_duplicates = False
    columns[0].constraints.regex_pattern = r".+@.+"
    columns[1].constraints.min_value = 1
    columns[1].constraints.max_value = 10

    records = columns_to_field_records(columns)
    restored = field_records_to_columns(records)

    assert [column.name for column in restored] == ["email", "score"]
    assert restored[0].constraints.regex_pattern == r".+@.+"
    assert restored[1].constraints.min_value == 1
    assert restored[1].constraints.max_value == 10


def test_web_config_roundtrip_keeps_browser_facing_labels():
    payload = serialize_ui_config(
        {
            "model_id": "demo-model",
            "provider": AIProvider.OPENAI.value,
            "api_key": "secret",
            "doc_mode": "Creative",
            "doc_pages": "2 pages",
            "quick_qa_mode": "Pinpoint Quick",
            "ocr_dpi": 200,
            "summary_top_k": 4,
            "graph_enabled": False,
        },
        columns=[],
    )

    normalized = normalize_loaded_config(payload)

    assert normalized["model_id"] == "demo-model"
    assert normalized["provider"] == AIProvider.OPENAI.value
    assert normalized["doc_mode"] == "Creative"
    assert normalized["doc_pages"] == "2 pages"
    assert normalized["quick_qa_mode"] == "Pinpoint Quick"
    assert normalized["ocr_dpi"] == 200
    assert normalized["summary_top_k"] == 4
    assert normalized["graph_enabled"] is False


def test_custom_document_page_inputs_are_supported():
    assert resolve_document_target_words("12") == 6000
    assert resolve_document_target_words("12 pages") == 6000
    assert resolve_document_target_words("7-page draft") == 3500
    assert pages_label_from_target_words(6000) == "12 pages"


def test_loaded_config_preserves_custom_document_page_count():
    normalized = normalize_loaded_config(
        {
            "document_engine": {
                "target_words": 6000,
            }
        }
    )

    assert normalized["doc_pages"] == "12 pages"


def test_build_generator_config_maps_advanced_rag_values_to_runtime_models():
    config = build_generator_config(
        {
            "rag_backend": "Native",
            "ocr_dpi": 175,
            "ocr_max_pages": 12,
            "hybrid_search_enabled": False,
            "rerank_enabled": False,
            "summary_first_enabled": False,
            "summary_top_k": 2,
            "dense_top_k": 9,
            "lexical_top_k": 7,
            "parent_context_enabled": False,
            "parent_context_max_chars": 900,
            "graph_enabled": False,
            "graph_hops": 2,
            "graph_source_boost": 0.15,
            "late_interaction_enabled": False,
            "late_interaction_weight": 0.35,
        },
        include_existing_data=False,
    )

    assert config.rag is not None
    assert config.rag.backend.value == "Native"
    assert config.rag.ocr_dpi == 175
    assert config.rag.ocr_max_pages == 12
    assert config.rag.hybrid_search_enabled is False
    assert config.rag.rerank_enabled is False
    assert config.rag.summary_first_enabled is False
    assert config.rag.summary_top_k == 2
    assert config.rag.dense_top_k == 9
    assert config.rag.lexical_top_k == 7
    assert config.rag.parent_context_enabled is False
    assert config.rag.parent_context_max_chars == 900
    assert config.rag.graph_enabled is False
    assert config.rag.graph_hops == 2
    assert config.rag.graph_source_boost == 0.15
    assert config.rag.late_interaction_enabled is False
    assert config.rag.late_interaction_weight == 0.35


def test_resolve_effective_rag_backend_uses_native_for_pinpoint_quick():
    assert files_actions.resolve_effective_rag_backend("Quick Q&A", "LlamaIndex", "Pinpoint Quick") == "Native"
    assert files_actions.resolve_effective_rag_backend("Quick Q&A", "LlamaIndex", "Broader Analysis") == "LlamaIndex"
    assert files_actions.resolve_effective_rag_backend("Document Engine", "LlamaIndex", "Pinpoint Quick") == "LlamaIndex"


def test_creative_polish_fallback_title_does_not_force_report_suffix():
    controller = GeneratorController()
    controller.llm_client = None

    result = controller._polish_document_for_publish(
        title="Write a story based on these positions",
        prompt="Write a story based on these positions of a couple in a club",
        text="Opening scene text.",
        audience="Adults",
        tone="erotic",
        target_words=900,
        mode=DocumentMode.PURE,
    )

    assert "Report" not in result["title"]
    assert result["text"] == "Opening scene text."


def test_creative_outline_fallback_uses_scene_structure():
    fake_llm = SimpleNamespace(generate_completion=lambda prompt, system_prompt=None: "not json")
    orchestrator = DocumentOrchestrator(fake_llm)

    outline = orchestrator._build_outline(
        DocumentGenerationOptions(
            prompt="Write an erotic story based on these positions",
            audience="Adults",
            tone="erotic",
            mode=DocumentMode.PURE,
            target_words=900,
        )
    )

    titles = [section.title for section in outline.sections]
    assert titles[0] == "Opening Scene"
    assert "Recommendations" not in titles


def test_request_stop_data_generation_marks_active_controller():
    class FakeController:
        def __init__(self):
            self.stop_requested = False

        def stop_generation(self):
            self.stop_requested = True

    session = new_session_state()
    controller = FakeController()
    web_state.register_runtime_controller(session, "data", controller)

    session, progress, status, activity = data_actions.request_stop_data_generation(session)

    assert controller.stop_requested is True
    assert "Stopping" in progress
    assert "Partial rows" in status
    assert "Stop requested" in activity
    web_state.clear_runtime_controller(session, "data")


def test_request_stop_files_task_marks_active_controller():
    class FakeController:
        def __init__(self):
            self.stop_requested = False

        def stop_document_generation(self):
            self.stop_requested = True

    session = new_session_state()
    controller = FakeController()
    web_state.register_runtime_controller(session, "files", controller)

    session, status, progress, activity = files_actions.request_stop_files_task(session)

    assert controller.stop_requested is True
    assert "Partial results" in status
    assert "Stopping" in progress
    assert "Stop requested" in activity
    web_state.clear_runtime_controller(session, "files")


def test_normalize_field_record_accepts_enum_types_for_dropdown_values():
    normalized = normalize_field_record(
        {
            "name": "email",
            "type": ColumnType.SHORT_TEXT,
            "prompt_instruction": "Business email",
        }
    )

    assert normalized["type"] == "Short Text"

    legacy = normalize_field_record({"type": "ColumnType.NUMERIC"})
    assert legacy["type"] == "Numeric"


def test_new_field_editor_returns_none_for_blank_number_inputs():
    session = new_session_state()

    result = data_actions.new_field_editor(session)

    assert result[8] is None
    assert result[9] is None
    assert result[10] is None
    assert result[11] is None


def test_add_field_appends_row_and_clears_editor():
    session = new_session_state()

    result = data_actions.add_field(
        session,
        "policy_number",
        ColumnType.SHORT_TEXT.value,
        "Unique insurance policy number",
        "",
        "",
        None,
        None,
        None,
        None,
        False,
        "",
    )

    updated_session = result[0]
    assert len(updated_session.fields) == 1
    assert updated_session.fields[0]["name"] == "policy_number"
    assert result[2] == "New"
    assert result[3] == ""
    assert "Added row **policy_number**" in result[-2]


def test_save_grid_rows_commits_visible_grid_to_session():
    session = new_session_state()
    grid = field_records_to_grid_dataframe([])
    grid.loc[0, "name"] = "claim_id"
    grid.loc[0, "type"] = "Short Text"
    grid.loc[0, "prompt_instruction"] = "Unique claim identifier"
    grid.loc[0, "allow_duplicates"] = False

    session, saved_grid, status, _ = data_actions.save_grid_rows(session, grid)

    assert len(session.fields) == 1
    assert session.fields[0]["name"] == "claim_id"
    assert "Saved **1** row" in status
    assert saved_grid["value"][0][0] == "Row 1"


def test_add_grid_row_expands_imported_grid():
    grid = field_records_to_grid_dataframe(
        [
            {"name": "client_id", "type": "Numeric", "prompt_instruction": "(Imported)", "allow_duplicates": False},
            {"name": "email_address", "type": "Short Text", "prompt_instruction": "(Imported)", "allow_duplicates": False},
        ]
    )

    updated_grid, status = data_actions.add_grid_row(grid)

    assert "blank row" in status.lower()
    assert len(updated_grid["value"]) == 3
    assert updated_grid["value"][-1][0] == "Row 3"


def test_sync_grid_row_editor_reads_selected_row_from_grid():
    grid = field_records_to_grid_dataframe(
        [
            {"name": "client_id", "type": "Numeric", "prompt_instruction": "(Imported)", "allow_duplicates": False},
            {"name": "email_address", "type": "Short Text", "prompt_instruction": "Work email", "allow_duplicates": True},
        ]
    )

    selector_update, name, field_type, prompt_instruction, allow_duplicates = data_actions.sync_grid_row_editor(grid, "Row 2")

    assert selector_update["choices"] == ["Row 1", "Row 2"]
    assert selector_update["value"] == "Row 2"
    assert name == "email_address"
    assert field_type == "Short Text"
    assert prompt_instruction == "Work email"
    assert allow_duplicates is True


def test_apply_grid_row_edit_updates_grid_with_dropdown_type_value():
    grid = field_records_to_grid_dataframe(
        [
            {"name": "client_id", "type": "Numeric", "prompt_instruction": "(Imported)", "allow_duplicates": False},
            {"name": "email_address", "type": "Short Text", "prompt_instruction": "Work email", "allow_duplicates": False},
        ]
    )

    updated_grid, selector_update, name, field_type, prompt_instruction, allow_duplicates, status = data_actions.apply_grid_row_edit(
        grid,
        "Row 2",
        "email_bucket",
        ColumnType.CATEGORICAL.value,
        "Email category bucket",
        True,
    )

    assert updated_grid["value"][1] == ["Row 2", "email_bucket", "Categorical", "Email category bucket", True]
    assert selector_update["value"] == "Row 2"
    assert name == "email_bucket"
    assert field_type == "Categorical"
    assert prompt_instruction == "Email category bucket"
    assert allow_duplicates is True
    assert "Save Rows" in status


def test_schema_fallback_can_extract_columns_from_raw_json():
    client = object.__new__(LLMClient)
    client.config = SimpleNamespace()
    client.on_log = None
    client.generate_completion = lambda prompt, system_prompt=None: """
    ```json
    {
      "columns": [
        {
          "name": "ticket_id",
          "type": "auto_increment",
          "prompt_instruction": "Unique identifier for each support ticket",
          "constraints": {"allow_duplicates": false}
        },
        {
          "name": "status",
          "type": "categorical",
          "prompt_instruction": "Current workflow state",
          "constraints": {"options": ["Open", "Pending", "Resolved"]}
        }
      ]
    }
    ```
    """

    columns = client._generate_schema_fallback("Support ticket dataset")

    assert len(columns) == 2
    assert columns[0]["type"] == "Auto Increment (ID)"
    assert columns[1]["type"] == "Categorical"
    assert columns[1]["constraints"]["options"] == ["Open", "Pending", "Resolved"]


def test_heuristic_schema_fallback_returns_email_rows():
    client = object.__new__(LLMClient)

    columns = client._generate_heuristic_schema("Insurance inbox emails from clients with at least 7 columns")

    assert len(columns) >= 7
    assert columns[0]["name"] == "message_id"
    assert any(column["name"] == "message_subject" for column in columns)
    assert any(column["name"] == "message_body" for column in columns)


def test_generate_data_uses_controller_and_returns_preview(monkeypatch):
    class FakeRow:
        def __init__(self, data):
            self.data = data

    class FakeController:
        def __init__(self):
            self.generated_rows = []
            self.on_log = None
            self.stop_requested = False

        def initialize(self, config, columns):
            self.config = config
            self.columns = columns

        def _run_generation_loop(self):
            if self.on_log:
                self.on_log("Generated row 1/2")
                self.on_log("Generation finished.")
            self.generated_rows = [
                FakeRow({"name": "Alice", "email": "alice@example.com"}),
                FakeRow({"name": "Bob", "email": "bob@example.com"}),
            ]

    monkeypatch.setattr(data_actions, "GeneratorController", FakeController)

    session = new_session_state()
    session.fields = columns_to_field_records(
        [
            ColumnDefinition(name="name", type=ColumnType.SHORT_TEXT, prompt_instruction="A full name"),
            ColumnDefinition(name="email", type=ColumnType.SHORT_TEXT, prompt_instruction="An email address"),
        ]
    )

    updates = list(
        data_actions.generate_data(
            session,
            field_records_to_grid_dataframe(session.fields),
            "local-model",
            AIProvider.LM_STUDIO.value,
            "",
            "",
            "",
            0.15,
            0.60,
            2,
            0.85,
            50,
            "LlamaIndex",
            "synthesizer_default",
            5,
            0.25,
            3000,
            "BAAI/bge-small-en-v1.5",
            "",
            ":memory:",
            "",
            "off",
            150,
            20,
            8,
            18,
            2.5,
            60,
            4000,
            "auto",
            True,
            True,
            True,
            3,
            12,
            12,
            True,
            1200,
            True,
            1,
            0.08,
            True,
            0.2,
            "Broader Analysis",
            "Balanced",
            "Let AI decide",
            "Fast",
            "General",
            "professional",
            False,
            True,
            3,
        )
    )

    session, preview, progress, status, activity = updates[-1]

    preview_df = preview["value"]
    assert isinstance(preview_df, pd.DataFrame)
    assert len(preview_df) == 2
    assert session.generated_rows[0]["name"] == "Alice"
    assert "Generated **2** row(s)" in status
    assert "Generation finished." in activity
    assert "Progress: **2/2** row(s) completed" in progress


def test_run_files_task_streams_progress_updates(monkeypatch):
    class FakeController:
        def __init__(self):
            self.on_log = None
            self.on_progress = None

        def ingest_documents(self, paths, force_reindex=False):
            if self.on_log:
                self.on_log("Starting ingest")
            pytime.sleep(0.02)
            if self.on_log:
                self.on_log("Finished ingest")
            return {"files_processed": len(paths), "chunks_created": 4, "vectors_upserted": 4}

        def ask_files(self, prompt):
            if self.on_log:
                self.on_log("Retrieving relevant chunks")
            if self.on_progress:
                self.on_progress(1, 2)
            pytime.sleep(0.02)
            if self.on_progress:
                self.on_progress(2, 2)
            if self.on_log:
                self.on_log("Answer ready")
            return {
                "answer": "Grounded answer",
                "citations": [{"source": "sample.pdf", "page": 1, "score": 0.91}],
            }

    monkeypatch.setattr(files_actions, "_build_files_controller", lambda session, **kwargs: (FakeController(), []))
    monkeypatch.setattr(files_actions.time, "sleep", lambda _: None)

    session = new_session_state()
    session.rag_files = ["C:\\temp\\sample.pdf"]

    updates = list(
        files_actions.run_files_task(
            session,
            "Quick Q&A",
            "What does the file say?",
            "local-model",
            AIProvider.LM_STUDIO.value,
            "",
            "",
            "",
            0.15,
            0.60,
            10,
            0.85,
            50,
            "LlamaIndex",
            "synthesizer_default",
            5,
            0.25,
            3000,
            "BAAI/bge-small-en-v1.5",
            "",
            ":memory:",
            "",
            "off",
            150,
            20,
            8,
            18,
            2.5,
            60,
            4000,
            "auto",
            True,
            True,
            True,
            3,
            12,
            12,
            True,
            1200,
            True,
            1,
            0.08,
            True,
            0.2,
            "Broader Analysis",
            "Balanced",
            "Let AI decide",
            "Fast",
            "General",
            "professional",
            False,
            True,
            3,
            "",
            "items",
            "Standard Generation",
            True,
        )
    )

    assert len(updates) >= 2
    final = updates[-1]
    assert "Grounded answer ready." in final[2]
    assert "### Files Progress" in final[3]
    assert "Status: **Completed**" in final[3]
    assert "Grounded answer" in final[1]
