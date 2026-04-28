import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.models import ColumnDefinition, ColumnType
from web_ui.actions import data_actions
from web_ui.adapters import (
    columns_to_field_records,
    field_records_to_columns,
    field_records_to_grid_dataframe,
    field_rows_markup,
    imported_columns_markup,
    normalize_field_record,
)
from web_ui.state import new_session_state


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
    assert updated_grid["row_count"] == (6, "dynamic")


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


def test_imported_columns_markup_lists_all_headers():
    markup = imported_columns_markup(
        [{"email_id": 1, "sender_email": "alice@example.com", "email_text": "hello"}],
        "Keep original values",
    )

    assert "Imported columns (3)" in markup
    assert "email_id" in markup
    assert "sender_email" in markup
    assert "email_text" in markup


def test_field_rows_markup_groups_imported_and_generated_fields():
    markup = field_rows_markup(
        [
            {"name": "email_id", "type": "Numeric", "prompt_instruction": "(Imported)", "allow_duplicates": False},
            {"name": "priority_bucket", "type": "Categorical", "prompt_instruction": "Classify the message urgency", "allow_duplicates": True},
        ]
    )

    assert "Schema overview" in markup
    assert "Imported columns" in markup
    assert "New fields to generate" in markup
    assert "Imported column" in markup
    assert "New/generated column" in markup
