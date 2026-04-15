from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WebSessionState:
    active_tab: str = "data"
    files_mode: str = "Document Engine"
    imported_data: list[dict] = field(default_factory=list)
    fields: list[dict] = field(default_factory=list)
    rag_files: list[str] = field(default_factory=list)
    file_chat_history: list[dict[str, str]] = field(default_factory=list)
    generated_rows: list[dict] = field(default_factory=list)
    latest_downloads: dict[str, str] = field(default_factory=dict)
    activity_log: list[str] = field(default_factory=lambda: ["Web UI preview ready."])


def new_session_state() -> WebSessionState:
    return WebSessionState()


def append_activity(session: WebSessionState, message: str) -> WebSessionState:
    session.activity_log.append(message)
    if len(session.activity_log) > 50:
        session.activity_log = session.activity_log[-50:]
    return session


def activity_markdown(session: WebSessionState) -> str:
    if not session.activity_log:
        return "No activity yet."
    return "\n".join(f"- {line}" for line in session.activity_log[-12:])
