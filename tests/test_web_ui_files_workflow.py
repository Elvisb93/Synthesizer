import os
import sys
import time as pytime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.models import AIProvider
from web_ui.app import apply_file_search_preset, reset_workspace_session
from web_ui.actions import files_actions
from web_ui import state as web_state
from web_ui.state import new_session_state


def test_resolve_effective_rag_backend_uses_native_for_pinpoint_quick():
    assert files_actions.resolve_effective_rag_backend("Quick Q&A", "LlamaIndex", "Pinpoint Quick") == "Native"
    assert files_actions.resolve_effective_rag_backend("Quick Q&A", "LlamaIndex", "Broader Analysis") == "LlamaIndex"
    assert files_actions.resolve_effective_rag_backend("Document Engine", "LlamaIndex", "Pinpoint Quick") == "LlamaIndex"


def test_plain_english_file_search_presets_map_to_retrieval_settings():
    result = apply_file_search_preset("Wider search", "Scanned PDFs or images")

    (
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
        status,
    ) = result

    assert rag_backend == "LlamaIndex"
    assert top_k == 8
    assert min_score == 0.10
    assert max_context_chars == 5000
    assert ocr_mode == "auto"
    assert parser_mode == "auto"
    assert hybrid_search_enabled is True
    assert rerank_enabled is True
    assert summary_first_enabled is True
    assert parent_context_enabled is True
    assert graph_enabled is True
    assert late_interaction_enabled is True
    assert "Looks through more text" in status
    assert "OCR will be tried" in status


def test_quick_qa_mode_hides_legacy_run_task_controls():
    updates = files_actions.files_mode_changed("Quick Q&A")

    assert len(updates) == 8
    assert updates[2]["visible"] is False
    assert updates[3]["visible"] is True
    assert updates[6]["visible"] is False
    assert updates[7]["visible"] is False


def test_reset_workspace_session_returns_fresh_state_and_clears_runtime(monkeypatch):
    class FakeController:
        pass

    session = new_session_state(startup_collection_name="old_collection")
    session.fields = [{"name": "old"}]
    session.rag_files = ["C:\\temp\\old.pdf"]
    session.generated_rows = [{"old": "row"}]
    web_state.register_runtime_controller(session, "files_chat", FakeController())

    monkeypatch.setattr(
        "web_ui.app.prepare_clean_workspace",
        lambda: {
            "startup_collection_name": "fresh_collection",
            "message": "Fresh workspace prepared for test.",
            "files_removed": 0,
            "directory_entries_removed": 0,
            "cleared_remote_collections": 0,
        },
    )

    result = reset_workspace_session(session)
    fresh_session = result[0]

    assert len(result) == 58
    assert fresh_session.runtime_id != session.runtime_id
    assert fresh_session.startup_collection_name == "fresh_collection"
    assert fresh_session.fields == []
    assert fresh_session.rag_files == []
    assert fresh_session.generated_rows == []
    assert web_state.get_runtime_controller(session, "files_chat") is None
    assert "Fresh workspace prepared for test." in result[1]
    assert result[27] == "Document Engine"
    assert result[55] == "fresh_collection"


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


def test_quick_qa_chat_reuses_indexed_controller_for_followups(monkeypatch):
    class FakeController:
        def __init__(self):
            self.ingest_count = 0
            self.questions = []

        def ingest_documents(self, paths, force_reindex=False):
            self.ingest_count += 1
            return {"files_processed": len(paths), "chunks_created": 2, "vectors_upserted": 2}

        def ask_files(self, prompt):
            self.questions.append(prompt)
            return {
                "answer": f"Answer {len(self.questions)}",
                "citations": [{"source": "sample.pdf", "page": 1, "score": 0.9}],
            }

    built = []

    def fake_build(session, **kwargs):
        controller = FakeController()
        built.append(controller)
        return controller, []

    monkeypatch.setattr(files_actions, "_build_files_controller", fake_build)

    session = new_session_state()
    session.rag_files = ["C:\\temp\\sample.pdf"]

    args = [
        session,
        "What is this file about?",
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
    ]

    first_updates = list(files_actions.ask_quick_qa_chat(*args))
    first_chat = first_updates[-1][1]

    args[1] = "What about the deadline?"
    second_updates = list(files_actions.ask_quick_qa_chat(*args))
    second_chat = second_updates[-1][1]

    assert len(built) == 1
    assert built[0].ingest_count == 1
    assert len(built[0].questions) == 2
    assert "Answer 1" in first_chat
    assert "Answer 2" in second_chat
    assert "Citations" in second_chat
    assert "What about the deadline?" in built[0].questions[-1]
