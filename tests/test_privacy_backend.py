from pathlib import Path
import re
from unittest.mock import MagicMock

import pandas as pd

from core.controller import GeneratorController
from core.models import AIProvider, ColumnDefinition, ColumnType, GeneratorConfig
from web_ui.adapters import detect_privacy_leaks, sanitize_imported_records
from web_ui.actions import data_actions
from web_ui.adapters import columns_to_field_records, field_records_to_grid_dataframe
from web_ui.state import new_session_state


EXAMPLE_CSV = Path(__file__).resolve().parents[1] / "examples" / "benefits_emails.csv"


def _example_columns() -> list[ColumnDefinition]:
    return [
        ColumnDefinition(name="email_id", type=ColumnType.NUMERIC, prompt_instruction="(Imported)"),
        ColumnDefinition(name="sender_email", type=ColumnType.SHORT_TEXT, prompt_instruction="(Imported)"),
        ColumnDefinition(name="sender_name", type=ColumnType.SHORT_TEXT, prompt_instruction="(Imported)"),
        ColumnDefinition(name="email_text", type=ColumnType.LONG_TEXT, prompt_instruction="(Imported)"),
        ColumnDefinition(name="Summary", type=ColumnType.LONG_TEXT, prompt_instruction="Provide a summary of @[email_text]"),
        ColumnDefinition(name="reply", type=ColumnType.LONG_TEXT, prompt_instruction="Provide a potential reply to the sender"),
    ]


def test_masked_benefits_email_rows_do_not_leak_known_pii_into_llm_prompts():
    df = pd.read_csv(EXAMPLE_CSV).iloc[[0, 1, 14]]
    raw_rows = df.to_dict(orient="records")
    masked_rows = sanitize_imported_records(raw_rows, "Mask likely personal values")
    assert detect_privacy_leaks(raw_rows, masked_rows) == []

    config = GeneratorConfig(model_id="test-model", num_rows=len(masked_rows), existing_data=masked_rows)
    controller = GeneratorController()
    controller.initialize(config, _example_columns())

    prompts: list[str] = []

    def fake_generate_completion(prompt: str, system_prompt: str = "") -> str:
        prompts.append(prompt)
        return "VALID" if "Review this data row" in prompt else "masked-result"

    controller.llm_client = MagicMock()
    controller.llm_client.generate_completion.side_effect = fake_generate_completion
    controller.validator = MagicMock()
    controller.validator.validate_regex.return_value = True
    controller.validator.is_unique.return_value = True

    controller._run_generation_loop()

    joined_prompts = "\n".join(prompts)
    assert "Angela Sanchez" not in joined_prompts
    assert "asanchez@example.com" not in joined_prompts
    assert "Sarah Patel" not in joined_prompts
    assert "HealthFirst Partners" not in joined_prompts
    assert "Benefits Coordinator" in joined_prompts
    assert "555-123-4567" not in joined_prompts
    assert "Treat only explicit angle-bracket mask tokens such as <NAME_1> or <EMAIL_2> as placeholders." in joined_prompts
    assert "<NAME" in joined_prompts
    assert "<EMAIL" in joined_prompts
    assert "<ORG" in joined_prompts
    assert "<PHONE" in joined_prompts


def test_real_file_masking_keeps_email_text_usable_and_avoids_obvious_false_positives():
    df = pd.read_csv(EXAMPLE_CSV)
    raw_rows = df.to_dict(orient="records")
    masked_rows = sanitize_imported_records(raw_rows, "Mask likely personal values")

    assert masked_rows[0]["email_id"] == raw_rows[0]["email_id"]
    assert masked_rows[1]["email_id"] == raw_rows[1]["email_id"]

    collapsed_placeholder = re.compile(r"^<[A-Z0-9_]+>$")
    for raw_row, masked_row in zip(raw_rows[:5], masked_rows[:5]):
        if len(str(raw_row["email_text"])) > 120:
            assert not collapsed_placeholder.match(str(masked_row["email_text"]))
            assert len(str(masked_row["email_text"])) > 120

    assert "Sarah Patel" not in str(masked_rows[1]["email_text"])
    assert "Benefits Team" not in str(masked_rows[4]["email_text"])
    assert "<ORG_" in str(masked_rows[4]["email_text"])
    assert "HealthFirst Partners" not in str(masked_rows[14]["email_text"])
    assert "Benefits Coordinator" in str(masked_rows[14]["email_text"])
    assert "555-123-4567" not in str(masked_rows[14]["email_text"])
    assert "<ORG_" in str(masked_rows[14]["email_text"])
    assert "<PHONE_" in str(masked_rows[14]["email_text"])
    assert "<ROLE_" not in str(masked_rows[14]["email_text"])


def test_detect_privacy_leaks_reports_unmasked_tokens():
    raw_rows = [
        {
            "sender_name": "Angela Sanchez",
            "sender_email": "asanchez@example.com",
            "email_text": "Angela Sanchez asked for help.",
        }
    ]
    masked_rows = [
        {
            "sender_name": "<NAME_1>",
            "sender_email": "<EMAIL_1>",
            "email_text": "Angela Sanchez asked for help.",
        }
    ]

    leaks = detect_privacy_leaks(raw_rows, masked_rows)

    assert leaks
    assert any("Angela Sanchez" in leak for leak in leaks)


def test_restored_export_rows_put_original_imported_values_back_after_masked_generation():
    df = pd.read_csv(EXAMPLE_CSV).head(2)

    session = new_session_state()
    session.raw_imported_data = df.to_dict(orient="records")
    session.imported_data = sanitize_imported_records(session.raw_imported_data, "Mask likely personal values")
    session.fields = [
        {"name": "email_id", "type": "Numeric", "prompt_instruction": "(Imported)", "allow_duplicates": False},
        {"name": "sender_email", "type": "Short Text", "prompt_instruction": "(Imported)", "allow_duplicates": False},
        {"name": "sender_name", "type": "Short Text", "prompt_instruction": "(Imported)", "allow_duplicates": False},
        {"name": "email_text", "type": "Long Text", "prompt_instruction": "(Imported)", "allow_duplicates": False},
        {"name": "Summary", "type": "Long Text", "prompt_instruction": "Provide a summary of @[email_text]", "allow_duplicates": True},
    ]
    session.generated_rows = [
        {
            "email_id": "<EMAIL_1>",
            "sender_email": "<EMAIL_1>",
            "sender_name": "<NAME_1>",
            "email_text": "<NAME_1> asked about coverage.",
            "Summary": "masked-result",
        },
        {
            "email_id": "<EMAIL_2>",
            "sender_email": "<EMAIL_2>",
            "sender_name": "<NAME_2>",
            "email_text": "<NAME_2> asked about benefits.",
            "Summary": "masked-result",
        },
    ]

    restored = data_actions.restore_original_imported_columns(session, session.generated_rows)

    assert restored[0]["email_id"] == df.iloc[0]["email_id"]
    assert restored[0]["sender_email"] == df.iloc[0]["sender_email"]
    assert restored[0]["sender_name"] == df.iloc[0]["sender_name"]
    assert restored[1]["sender_email"] == df.iloc[1]["sender_email"]
    assert restored[1]["Summary"] == "masked-result"


def test_generate_data_restores_original_imported_columns_after_masked_generation(monkeypatch):
    df = pd.read_csv(EXAMPLE_CSV).head(2)
    raw_rows = df.to_dict(orient="records")
    masked_rows = sanitize_imported_records(raw_rows, "Mask likely personal values")

    class FakeRow:
        def __init__(self, data):
            self.data = data

    class FakeController:
        def __init__(self):
            self.generated_rows = []
            self.on_log = None
            self.on_progress = None
            self.stop_requested = False

        def initialize(self, config, columns):
            self.config = config
            self.columns = columns

        def _run_generation_loop(self):
            if self.on_log:
                self.on_log("Generated row 1/2")
                self.on_log("Generated row 2/2")
                self.on_log("Generation finished.")
            if self.on_progress:
                self.on_progress(2, 2)
            self.generated_rows = [
                FakeRow(
                    {
                        "email_id": masked_rows[0]["email_id"],
                        "sender_email": masked_rows[0]["sender_email"],
                        "sender_name": masked_rows[0]["sender_name"],
                        "email_text": masked_rows[0]["email_text"],
                        "Summary": "summary one",
                        "reply": "reply one",
                    }
                ),
                FakeRow(
                    {
                        "email_id": masked_rows[1]["email_id"],
                        "sender_email": masked_rows[1]["sender_email"],
                        "sender_name": masked_rows[1]["sender_name"],
                        "email_text": masked_rows[1]["email_text"],
                        "Summary": "summary two",
                        "reply": "reply two",
                    }
                ),
            ]

    monkeypatch.setattr(data_actions, "GeneratorController", FakeController)

    session = new_session_state()
    session.import_privacy_mode = "Mask likely personal values"
    session.raw_imported_data = raw_rows
    session.imported_data = masked_rows
    session.fields = columns_to_field_records(_example_columns())

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

    session, preview, _, status, _ = updates[-1]
    preview_df = preview["value"]

    assert "Generated **2** row(s)" in status
    assert session.generated_rows[0]["email_id"] == raw_rows[0]["email_id"]
    assert session.generated_rows[0]["sender_email"] == raw_rows[0]["sender_email"]
    assert session.generated_rows[0]["sender_name"] == raw_rows[0]["sender_name"]
    assert session.generated_rows[0]["Summary"] == "summary one"
    assert preview_df.iloc[0]["sender_email"] == raw_rows[0]["sender_email"]
    assert preview_df.iloc[1]["sender_name"] == raw_rows[1]["sender_name"]


def test_generate_data_blocks_imported_rows_when_privacy_masking_is_not_active():
    df = pd.read_csv(EXAMPLE_CSV).head(1)
    raw_rows = df.to_dict(orient="records")

    session = new_session_state()
    session.import_privacy_mode = "Keep original values"
    session.raw_imported_data = raw_rows
    session.imported_data = raw_rows
    session.fields = columns_to_field_records(_example_columns())

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
            1,
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

    assert len(updates) == 1
    _, preview, progress, status, activity = updates[0]
    assert preview["visible"] is False
    assert "Generation blocked." in status
    assert "Mask likely personal values" in status
    assert "Generation progress will appear here once you start a run." == progress
    assert "Generation blocked: imported files must use privacy masking before AI generation." in activity


def test_generate_data_rebuilds_masked_existing_rows_from_raw_session_data(monkeypatch):
    df = pd.read_csv(EXAMPLE_CSV).head(1)
    raw_rows = df.to_dict(orient="records")
    expected_masked_rows = sanitize_imported_records(raw_rows, "Mask likely personal values")
    captured_existing_data = []

    class FakeController:
        def __init__(self):
            self.generated_rows = []
            self.on_log = None
            self.on_progress = None
            self.stop_requested = False

        def initialize(self, config, columns):
            captured_existing_data.append(config.existing_data)

        def _run_generation_loop(self):
            if self.on_progress:
                self.on_progress(0, 1)

    monkeypatch.setattr(data_actions, "GeneratorController", FakeController)

    session = new_session_state()
    session.import_privacy_mode = "Mask likely personal values"
    session.raw_imported_data = raw_rows
    session.imported_data = raw_rows
    session.fields = columns_to_field_records(_example_columns())

    list(
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
            1,
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

    assert captured_existing_data == [expected_masked_rows]
    assert session.imported_data == expected_masked_rows


def test_export_generated_data_csv_writes_original_imported_values_after_masked_generation(tmp_path):
    df = pd.read_csv(EXAMPLE_CSV).head(2)
    raw_rows = df.to_dict(orient="records")
    masked_rows = sanitize_imported_records(raw_rows, "Mask likely personal values")

    session = new_session_state()
    session.import_privacy_mode = "Mask likely personal values"
    session.raw_imported_data = raw_rows
    session.imported_data = masked_rows
    session.fields = columns_to_field_records(_example_columns())
    session.generated_rows = [
        {
            "email_id": masked_rows[0]["email_id"],
            "sender_email": masked_rows[0]["sender_email"],
            "sender_name": masked_rows[0]["sender_name"],
            "email_text": masked_rows[0]["email_text"],
            "Summary": "summary one",
            "reply": "reply one",
        },
        {
            "email_id": masked_rows[1]["email_id"],
            "sender_email": masked_rows[1]["sender_email"],
            "sender_name": masked_rows[1]["sender_name"],
            "email_text": masked_rows[1]["email_text"],
            "Summary": "summary two",
            "reply": "reply two",
        },
    ]

    data_actions.EXPORT_DIR = tmp_path
    try:
        session, download_path, status, _ = data_actions.export_generated_data(session, "csv")
    finally:
        data_actions.EXPORT_DIR = Path(".web_ui_exports")

    exported = pd.read_csv(download_path)

    assert "CSV" in status
    assert exported.iloc[0]["email_id"] == raw_rows[0]["email_id"]
    assert exported.iloc[0]["sender_email"] == raw_rows[0]["sender_email"]
    assert exported.iloc[1]["sender_name"] == raw_rows[1]["sender_name"]


def test_enrichment_mode_does_not_drop_rows_for_duplicate_generated_long_text():
    df = pd.read_csv(EXAMPLE_CSV).head(2)
    raw_rows = df.to_dict(orient="records")
    masked_rows = sanitize_imported_records(raw_rows, "Mask likely personal values")

    config = GeneratorConfig(model_id="test-model", num_rows=len(masked_rows), existing_data=masked_rows)
    controller = GeneratorController()
    controller.initialize(config, _example_columns())

    def fake_generate_completion(prompt: str, system_prompt: str = "") -> str:
        if "Review this data row" in prompt:
            return "VALID"
        if "column 'Summary'" in prompt:
            return "same-summary"
        if "column 'reply'" in prompt:
            return "same-reply"
        return "same-value"

    controller.llm_client = MagicMock()
    controller.llm_client.generate_completion.side_effect = fake_generate_completion
    controller._run_generation_loop()

    assert len(controller.generated_rows) == 2
    assert controller.generated_rows[0].data["Summary"] == "same-summary"
    assert controller.generated_rows[1].data["Summary"] == "same-summary"
