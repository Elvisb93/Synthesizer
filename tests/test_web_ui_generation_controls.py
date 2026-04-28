import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.models import AIProvider, ColumnDefinition, ColumnType
from web_ui.actions import data_actions
from web_ui.adapters import columns_to_field_records, field_records_to_grid_dataframe
from web_ui import state as web_state
from web_ui.state import new_session_state


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


def test_export_generated_data_prepares_narrative_pdf_download(monkeypatch, tmp_path):
    class FakeController:
        def export_narrative_pdf(self, filepath):
            Path(filepath).write_text("fake pdf bytes", encoding="utf-8")

    session = new_session_state()
    session.generated_rows = [{"title": "Claim Update", "body": "Policy details in narrative form."}]

    monkeypatch.setattr(data_actions, "_controller_from_session_rows", lambda session: (FakeController(), []))
    monkeypatch.setattr(data_actions, "EXPORT_DIR", tmp_path)

    session, download_path, status, activity = data_actions.export_generated_data(session, "pdf_narrative")

    assert download_path is not None
    assert download_path.endswith(".pdf")
    assert Path(download_path).exists()
    assert session.latest_downloads["data_pdf_narrative"] == download_path
    assert "Narrative PDF" in status
    assert "Narrative PDF" in activity
