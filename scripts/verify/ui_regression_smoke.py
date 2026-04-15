"""
UI regression smoke test for the Flet app.

Covers:
- Data tab rendering basics
- Files tab mode switching
- Document generation start path
- Optional short app boot probe
"""

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.flet_app import FletApp  # noqa: E402


class FakePage:
    def __init__(self):
        self.title = ""
        self.theme_mode = None
        self.padding = 0
        self.scroll = None
        self.overlay = []
        self.controls = []

    def add(self, *controls):
        self.controls.extend(controls)

    def update(self):
        return None

    def run_task(self, task_or_coro):
        coro = None
        if asyncio.iscoroutine(task_or_coro):
            coro = task_or_coro
        elif callable(task_or_coro):
            maybe = task_or_coro()
            if asyncio.iscoroutine(maybe):
                coro = maybe
        if coro is None:
            return None

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(coro)
        finally:
            loop.close()
        return None


class FakeController:
    def __init__(self):
        self.on_log: Optional[Callable[[str], None]] = None
        self.on_progress: Optional[Callable[[int, int], None]] = None
        self.on_finished: Optional[Callable[[], None]] = None
        self.llm_client = None
        self.rag_service = None
        self.document_result = None
        self.generated_rows = []
        self.last_doc_request = {}

    def set_runtime_config(self, config):
        self.config = config
        self.llm_client = object()
        self.rag_service = object()

    def get_metrics(self):
        return {}

    def ingest_documents(self, paths, force_reindex=False):
        return {
            "files_processed": len(paths),
            "chunks_created": 0,
            "vectors_upserted": 0,
            "errors": [],
            "ocr_pages_total": 0,
            "ocr_regions_total": 0,
        }

    def clear_rag_collection(self):
        return None

    def get_rag_status(self):
        return {"enabled": True, "collection_name": "test", "collection_size": 0}

    def generate_document(self, prompt, **kwargs):
        self.last_doc_request = {"prompt": prompt, **kwargs}
        self.document_result = {
            "title": "Smoke Test",
            "outline": {},
            "text": "Smoke content.",
            "chunks": [],
            "citations": [],
            "mode": kwargs.get("mode", "hybrid"),
        }
        return self.document_result

    def ask_files(self, prompt):
        return {"answer": f"Echo: {prompt}", "citations": []}


def _assert(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def run_inprocess_ui_smoke():
    page = FakePage()
    controller = FakeController()
    app = FletApp(page, controller)

    # Flet controls assert if update() is called without a mounted real page.
    # For this in-process smoke harness, replace hot-path control updates with no-ops.
    for ctrl in [
        app.file_chat_view,
        app.files_list_view,
        app.log_view,
        app.progress_bar,
        app.status_text,
        app.start_btn,
        app.export_btn,
        app.analyze_btn,
        app.metrics_text,
    ]:
        if hasattr(ctrl, "update"):
            setattr(ctrl, "update", lambda: None)

    _assert(app.active_workspace_tab == "data", "Expected default workspace to be data.")
    _assert(app.data_workspace_container.visible is True, "Data container should be visible by default.")
    _assert(app.data_prompt.expand is not True, "Data prompt should stay in a bounded layout.")
    _assert(app.files_prompt.expand is not True, "Files prompt should stay in a bounded layout.")
    _assert(app.columns[0].prompt_field.expand is not True, "Column prompt should use an explicit width, not direct expansion.")

    # Data tab flow
    initial_cols = len(app.columns)
    app._add_column()
    _assert(len(app.columns) == initial_cols + 1, "Add Column did not append a new column.")

    # Files tab flow
    app._set_workspace_tab("files")
    _assert(app.files_workspace_container.visible is True, "Files container should be visible after tab switch.")
    _assert(app.data_workspace_container.visible is False, "Data container should be hidden in files mode.")

    # One-click preset bundle
    app._apply_doc_bundle("Executive Brief")
    _assert(app.doc_mode_dropdown.value == "Balanced", "Executive Brief should set doc mode to Balanced.")
    _assert(app.doc_pages_dropdown.value == "2 pages", "Executive Brief should set pages to 2 pages.")
    _assert(app.doc_quality_dropdown.value == "Fast", "Executive Brief should set quality mode to Fast.")

    # Document generation start
    app.files_mode_dropdown.value = "Document Engine"
    app._on_files_mode_change(None)
    app.files_prompt.value = "Generate a concise operational brief."
    app._on_files_magic_task(None)

    _assert(len(app.file_chat_view.controls) >= 2, "Expected user + assistant messages in file chat after generation.")
    _assert(bool(controller.last_doc_request), "Document generation request did not reach controller.")
    _assert(controller.last_doc_request.get("quality_mode") in {"Fast", "Thorough"}, "quality_mode not passed to controller.")


def run_short_boot_probe():
    cmd = [sys.executable, "main.py"]
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        out, err = proc.communicate(timeout=12)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()

    combined = f"{out}\n{err}".lower()
    if "traceback" in combined:
        raise RuntimeError("App boot probe emitted traceback.")


def main():
    print("Running in-process UI smoke...")
    run_inprocess_ui_smoke()
    print("In-process UI smoke passed.")

    print("Running short boot probe...")
    run_short_boot_probe()
    print("Boot probe passed.")

    print("UI regression smoke: PASS")


if __name__ == "__main__":
    main()
