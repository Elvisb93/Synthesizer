import os
import sys
from types import SimpleNamespace

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.app_config import build_generator_config, normalize_loaded_config, serialize_ui_config
from core.llm_client import LLMClient
from core.models import AIProvider, ColumnDefinition, ColumnType
from web_ui.adapters import columns_to_field_records, field_records_to_columns, field_records_to_grid_dataframe, normalize_field_record
from web_ui.actions import data_actions
from web_ui.state import new_session_state


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
        },
        columns=[],
    )

    normalized = normalize_loaded_config(payload)

    assert normalized["model_id"] == "demo-model"
    assert normalized["provider"] == AIProvider.OPENAI.value
    assert normalized["doc_mode"] == "Creative"
    assert normalized["doc_pages"] == "2 pages"
    assert normalized["quick_qa_mode"] == "Pinpoint Quick"


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
    assert saved_grid["value"].iloc[0]["row_id"] == "Row 1"


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
    assert updated_grid["value"].iloc[-1]["row_id"] == "Row 3"


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
            "auto",
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
