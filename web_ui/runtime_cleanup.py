from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


RUNTIME_STATE_PATH = Path(".web_ui_runtime.json")
APP_TRANSIENT_DIRS = (
    Path(".web_ui_exports"),
    Path(".document_checkpoints"),
)
APP_TRANSIENT_FILE_PATTERNS = (
    ".rag_cache*.json",
    ".rag_manifest*.json",
    ".rag_ingestion_cache*.json",
)
SESSION_COLLECTION_PREFIX = "synthesizer_session_"


def fresh_session_collection_name(now: datetime | None = None) -> str:
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{SESSION_COLLECTION_PREFIX}{timestamp}"


def _is_memory_qdrant(qdrant_url: str | None) -> bool:
    lowered = str(qdrant_url or "").strip().lower()
    return lowered in {"", ":memory:", "memory://", "local-memory"}


def _load_runtime_registry() -> list[dict[str, str]]:
    if not RUNTIME_STATE_PATH.exists():
        return []
    try:
        payload = json.loads(RUNTIME_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []

    entries = payload.get("collections")
    if not isinstance(entries, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        collection_name = str(item.get("collection_name", "") or "").strip()
        qdrant_url = str(item.get("qdrant_url", "") or "").strip()
        if collection_name:
            normalized.append(
                {
                    "collection_name": collection_name,
                    "qdrant_url": qdrant_url,
                }
            )
    return normalized


def _save_runtime_registry(entries: list[dict[str, str]]) -> None:
    payload = {"collections": entries}
    RUNTIME_STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def record_runtime_collection(collection_name: str, qdrant_url: str) -> None:
    collection = str(collection_name or "").strip()
    qdrant = str(qdrant_url or "").strip()
    if not collection or _is_memory_qdrant(qdrant):
        return

    entries = _load_runtime_registry()
    entry = {
        "collection_name": collection,
        "qdrant_url": qdrant,
    }
    if entry not in entries:
        entries.append(entry)
        _save_runtime_registry(entries)


def _clear_registered_qdrant_collections(entries: list[dict[str, str]]) -> int:
    if not entries:
        return 0

    try:
        from qdrant_client import QdrantClient
    except Exception:
        return 0

    cleared = 0
    for entry in entries:
        qdrant_url = str(entry.get("qdrant_url", "") or "").strip()
        collection_name = str(entry.get("collection_name", "") or "").strip()
        if not collection_name or _is_memory_qdrant(qdrant_url):
            continue
        try:
            client = QdrantClient(url=qdrant_url)
            client.delete_collection(collection_name)
            cleared += 1
        except Exception:
            continue
    return cleared


def _clear_directory_contents(path: Path) -> int:
    if not path.exists():
        return 0

    removed = 0
    for child in list(path.iterdir()):
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except FileNotFoundError:
                pass
            except PermissionError:
                continue
        removed += 1
    return removed


def _cleanup_local_artifacts() -> dict[str, int]:
    file_count = 0
    dir_entry_count = 0

    for path in APP_TRANSIENT_DIRS:
        dir_entry_count += _clear_directory_contents(path)

    seen: set[Path] = set()
    for pattern in APP_TRANSIENT_FILE_PATTERNS:
        for path in Path(".").glob(pattern):
            if path in seen or not path.is_file():
                continue
            try:
                path.unlink()
                seen.add(path)
                file_count += 1
            except FileNotFoundError:
                continue
            except PermissionError:
                continue

    return {
        "files_removed": file_count,
        "directory_entries_removed": dir_entry_count,
    }


def prepare_clean_workspace() -> dict[str, Any]:
    entries = _load_runtime_registry()
    cleared_remote_collections = _clear_registered_qdrant_collections(entries)
    cleanup = _cleanup_local_artifacts()
    _save_runtime_registry([])

    startup_collection_name = fresh_session_collection_name()
    message = (
        "Fresh workspace prepared. "
        f"Cleared {cleanup['files_removed']} cache file(s), "
        f"{cleanup['directory_entries_removed']} export/checkpoint item(s), "
        f"and {cleared_remote_collections} persisted search collection(s)."
    )
    return {
        "startup_collection_name": startup_collection_name,
        "message": message,
        "files_removed": cleanup["files_removed"],
        "directory_entries_removed": cleanup["directory_entries_removed"],
        "cleared_remote_collections": cleared_remote_collections,
    }
