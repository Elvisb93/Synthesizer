from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any
from uuid import uuid4


_RUNTIME_LOCK = Lock()
_RUNTIME_CONTROLLERS: dict[str, dict[str, Any]] = {}


@dataclass
class WebSessionState:
    runtime_id: str = field(default_factory=lambda: uuid4().hex)
    startup_collection_name: str = "synthesizer_default"
    active_tab: str = "data"
    files_mode: str = "Document Engine"
    import_privacy_mode: str = "Mask likely personal values"
    import_mask_mappings: list[dict[str, str]] = field(default_factory=list)
    raw_imported_data: list[dict] = field(default_factory=list)
    imported_data: list[dict] = field(default_factory=list)
    fields: list[dict] = field(default_factory=list)
    rag_files: list[str] = field(default_factory=list)
    file_chat_history: list[dict[str, str]] = field(default_factory=list)
    generated_rows: list[dict] = field(default_factory=list)
    latest_downloads: dict[str, str] = field(default_factory=dict)
    activity_log: list[str] = field(default_factory=lambda: ["Web UI preview ready."])


def new_session_state(
    *,
    startup_collection_name: str = "synthesizer_default",
    startup_message: str | None = None,
) -> WebSessionState:
    session = WebSessionState(startup_collection_name=startup_collection_name)
    if startup_message:
        session.activity_log.append(startup_message)
    return session


def append_activity(session: WebSessionState, message: str) -> WebSessionState:
    session.activity_log.append(message)
    if len(session.activity_log) > 50:
        session.activity_log = session.activity_log[-50:]
    return session


def register_runtime_controller(session: WebSessionState, task_name: str, controller: Any) -> None:
    with _RUNTIME_LOCK:
        task_map = _RUNTIME_CONTROLLERS.setdefault(session.runtime_id, {})
        task_map[task_name] = controller


def get_runtime_controller(session: WebSessionState, task_name: str) -> Any | None:
    with _RUNTIME_LOCK:
        return _RUNTIME_CONTROLLERS.get(session.runtime_id, {}).get(task_name)


def clear_runtime_controller(session: WebSessionState, task_name: str, controller: Any | None = None) -> None:
    with _RUNTIME_LOCK:
        task_map = _RUNTIME_CONTROLLERS.get(session.runtime_id)
        if not task_map:
            return
        existing = task_map.get(task_name)
        if controller is not None and existing is not controller:
            return
        task_map.pop(task_name, None)
        if not task_map:
            _RUNTIME_CONTROLLERS.pop(session.runtime_id, None)


def clear_runtime_session(session: WebSessionState) -> None:
    with _RUNTIME_LOCK:
        _RUNTIME_CONTROLLERS.pop(session.runtime_id, None)


def activity_markdown(session: WebSessionState) -> str:
    if not session.activity_log:
        return "No activity yet."
    return "\n".join(f"- {line}" for line in session.activity_log[-12:])
