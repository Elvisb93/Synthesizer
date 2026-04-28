import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from web_ui import runtime_cleanup


def test_prepare_clean_workspace_removes_transient_artifacts(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    exports_dir = tmp_path / ".web_ui_exports"
    checkpoints_dir = tmp_path / ".document_checkpoints"
    exports_dir.mkdir()
    checkpoints_dir.mkdir()
    (exports_dir / "generated_rows_demo.csv").write_text("id\n1\n", encoding="utf-8")
    (checkpoints_dir / "job-1.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".rag_cache_demo.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".rag_manifest_demo.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".rag_ingestion_cache_demo.json").write_text("{}", encoding="utf-8")

    runtime_cleanup._save_runtime_registry(  # noqa: SLF001 - intentional test coverage of persisted startup state
        [{"collection_name": "demo_collection", "qdrant_url": "http://localhost:6333"}]
    )
    monkeypatch.setattr(runtime_cleanup, "_clear_registered_qdrant_collections", lambda entries: len(entries))

    report = runtime_cleanup.prepare_clean_workspace()

    assert report["files_removed"] == 3
    assert report["directory_entries_removed"] == 2
    assert report["cleared_remote_collections"] == 1
    assert report["startup_collection_name"].startswith(runtime_cleanup.SESSION_COLLECTION_PREFIX)
    assert list(exports_dir.iterdir()) == []
    assert list(checkpoints_dir.iterdir()) == []
    assert not list(tmp_path.glob(".rag_cache*.json"))
    assert not list(tmp_path.glob(".rag_manifest*.json"))
    assert not list(tmp_path.glob(".rag_ingestion_cache*.json"))
    registry = runtime_cleanup._load_runtime_registry()  # noqa: SLF001 - intentional test coverage of persisted startup state
    assert registry == []


def test_record_runtime_collection_skips_memory_backed_targets(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    runtime_cleanup.record_runtime_collection("demo_a", ":memory:")
    runtime_cleanup.record_runtime_collection("demo_b", "memory://")
    runtime_cleanup.record_runtime_collection("demo_c", "http://localhost:6333")

    registry = runtime_cleanup._load_runtime_registry()  # noqa: SLF001 - intentional test coverage of persisted startup state
    assert registry == [{"collection_name": "demo_c", "qdrant_url": "http://localhost:6333"}]
