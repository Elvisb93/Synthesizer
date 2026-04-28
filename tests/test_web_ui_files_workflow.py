import os
import sys
import time as pytime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.models import AIProvider
from web_ui.actions import files_actions
from web_ui import state as web_state
from web_ui.state import new_session_state


def test_resolve_effective_rag_backend_uses_native_for_pinpoint_quick():
    assert files_actions.resolve_effective_rag_backend("Quick Q&A", "LlamaIndex", "Pinpoint Quick") == "Native"
    assert files_actions.resolve_effective_rag_backend("Quick Q&A", "LlamaIndex", "Broader Analysis") == "LlamaIndex"
    assert files_actions.resolve_effective_rag_backend("Document Engine", "LlamaIndex", "Pinpoint Quick") == "LlamaIndex"


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


def test_apply_doc_bundle_updates_files_prompt_and_document_settings():
    session = new_session_state()

    session, prompt, doc_mode, doc_pages, doc_quality, doc_audience, doc_tone, activity = files_actions.apply_doc_bundle(
        session, "Policy Draft"
    )

    assert "policy document" in prompt.lower()
    assert doc_mode == "File-based"
    assert doc_pages == "5 pages"
    assert doc_quality == "Thorough"
    assert doc_audience == "Policy stakeholders"
    assert doc_tone == "formal"
    assert "Applied document bundle: Policy Draft" in activity


def test_file_prompt_presets_can_be_saved_loaded_and_deleted(monkeypatch, tmp_path):
    monkeypatch.setattr(files_actions, "PRESETS_FILE", tmp_path / ".rag_task_presets.json")

    session = new_session_state()

    session, dropdown_update, preset_name, activity = files_actions.save_file_preset(
        session,
        "Claims Follow Up",
        "Draft a follow-up note for the outstanding claims issues.",
        None,
    )

    assert "Claims Follow Up" in dropdown_update["choices"]
    assert dropdown_update["value"] == "Claims Follow Up"
    assert preset_name == "Claims Follow Up"
    assert "Saved prompt preset: Claims Follow Up" in activity

    session, prompt, loaded_name, activity = files_actions.apply_file_preset(session, "Claims Follow Up")

    assert "follow-up note" in prompt
    assert loaded_name == "Claims Follow Up"
    assert "Loaded prompt preset: Claims Follow Up" in activity

    session, dropdown_update, preset_name, activity = files_actions.delete_file_preset(session, "Claims Follow Up")

    assert "Claims Follow Up" not in dropdown_update["choices"]
    assert preset_name == ""
    assert "Deleted prompt preset: Claims Follow Up" in activity


def test_remove_selected_source_updates_files_list_and_selector():
    session = new_session_state()
    session.rag_files = ["C:\\temp\\alpha.pdf", "https://example.com/policy"]
    session.file_chat_history = [{"role": "assistant", "content": "Grounded answer"}]

    session, table, selector_update, files_chat, status, activity = files_actions.remove_selected_source(
        session,
        "C:\\temp\\alpha.pdf",
        "Document Engine",
    )

    table_df = table["value"] if isinstance(table, dict) else table
    assert session.rag_files == ["https://example.com/policy"]
    assert list(table_df["path"]) == ["https://example.com/policy"]
    assert selector_update["value"] == "https://example.com/policy"
    assert "Current files: **1**" in status
    assert "Removed source: alpha.pdf" in activity
    assert "Grounded answer" in files_chat


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
