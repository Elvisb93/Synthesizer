"""Versioned Power BI export runs for generated tabular data."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List

from ..models import AIProvider, ColumnDefinition, RowData


POWER_BI_INDEX_COLUMNS = [
    "run_id",
    "dataset_name",
    "dataset_slug",
    "created_at",
    "row_count",
    "column_count",
    "schema_hash",
    "privacy_export_mode",
    "source_mode",
    "provider",
    "model",
    "data_path",
    "schema_path",
    "metadata_path",
]


@dataclass(frozen=True)
class PowerBiExportResult:
    run_id: str
    run_dir: str
    data_path: str
    schema_path: str
    metadata_path: str
    index_path: str
    schema_changed: bool
    previous_schema_hash: str | None = None


def slugify_dataset_name(name: str) -> str:
    """Return a folder-safe slug while preserving readable words."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", (name or "").strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "dataset"


def schema_hash(columns: Iterable[ColumnDefinition]) -> str:
    payload = [
        {
            "name": col.name,
            "type": col.type.value if hasattr(col.type, "value") else str(col.type),
        }
        for col in columns
    ]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _power_bi_type_hint(column: ColumnDefinition) -> str:
    value = column.type.value if hasattr(column.type, "value") else str(column.type)
    lowered = value.lower()
    if "numeric" in lowered or "increment" in lowered:
        return "number"
    if "boolean" in lowered:
        return "boolean"
    return "text"


def _column_schema(column: ColumnDefinition) -> dict[str, Any]:
    constraints = column.constraints.model_dump() if column.constraints else {}
    return {
        "name": column.name,
        "type": column.type.value if hasattr(column.type, "value") else str(column.type),
        "power_bi_type_hint": _power_bi_type_hint(column),
        "prompt_instruction": column.prompt_instruction,
        "allow_duplicates": bool(constraints.get("allow_duplicates", False)),
        "constraints": constraints,
    }


def _write_data_csv(path: Path, rows: List[RowData], columns: List[ColumnDefinition]) -> None:
    fieldnames = [col.name for col in columns]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.data.get(name, "") for name in fieldnames})


def _relative_path(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def _latest_schema_hash(index_path: Path, dataset_slug: str) -> str | None:
    if not index_path.exists():
        return None
    with index_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        latest: str | None = None
        for row in reader:
            if row.get("dataset_slug") == dataset_slug:
                latest = row.get("schema_hash") or None
        return latest


def _append_index(index_path: Path, row: dict[str, Any]) -> None:
    index_exists = index_path.exists()
    with index_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=POWER_BI_INDEX_COLUMNS)
        if not index_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in POWER_BI_INDEX_COLUMNS})


def export_power_bi_run(
    destination_dir: str | Path,
    generated_rows: List[RowData],
    columns: List[ColumnDefinition],
    *,
    dataset_name: str,
    privacy_export_mode: str = "Restored imported values",
    source_mode: str = "fresh_generation",
    provider: AIProvider | str = AIProvider.LM_STUDIO,
    model: str = "local-model",
    app_version: str | None = None,
    log_fn=None,
) -> PowerBiExportResult:
    """Export rows to a timestamped Power BI run folder and append index.csv."""
    if not generated_rows:
        raise ValueError("No generated rows available for Power BI export.")
    if not columns:
        raise ValueError("No schema columns available for Power BI export.")

    base_dir = Path(destination_dir or ".web_ui_exports/power_bi").expanduser()
    base_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = base_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now().isoformat(timespec="seconds")
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    clean_name = (dataset_name or "").strip() or "Dataset"
    dataset_slug = slugify_dataset_name(clean_name)
    run_id = f"{stamp}_{dataset_slug}"
    run_dir = runs_dir / run_id

    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_id = f"{stamp}_{dataset_slug}_{suffix}"
        run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)

    data_path = run_dir / "data.csv"
    schema_path = run_dir / "schema.json"
    metadata_path = run_dir / "metadata.json"
    index_path = base_dir / "index.csv"

    current_schema_hash = schema_hash(columns)
    previous_schema_hash = _latest_schema_hash(index_path, dataset_slug)
    schema_changed = bool(previous_schema_hash and previous_schema_hash != current_schema_hash)

    schema_payload = {
        "schema_hash": current_schema_hash,
        "columns": [_column_schema(column) for column in columns],
    }
    metadata = {
        "run_id": run_id,
        "dataset_name": clean_name,
        "dataset_slug": dataset_slug,
        "created_at": created_at,
        "row_count": len(generated_rows),
        "column_count": len(columns),
        "privacy_export_mode": privacy_export_mode,
        "source_mode": source_mode,
        "provider": provider.value if hasattr(provider, "value") else str(provider),
        "model": model,
        "schema_hash": current_schema_hash,
        "schema_changed_from_previous_run": schema_changed,
        "previous_schema_hash": previous_schema_hash,
        "files": {
            "data": "data.csv",
            "schema": "schema.json",
            "metadata": "metadata.json",
        },
        "app_version": app_version or "",
    }

    _write_data_csv(data_path, generated_rows, columns)
    schema_path.write_text(json.dumps(schema_payload, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    _append_index(
        index_path,
        {
            **metadata,
            "data_path": _relative_path(data_path, base_dir),
            "schema_path": _relative_path(schema_path, base_dir),
            "metadata_path": _relative_path(metadata_path, base_dir),
        },
    )

    if log_fn:
        log_fn(f"Power BI export run created at {run_dir}")

    return PowerBiExportResult(
        run_id=run_id,
        run_dir=str(run_dir),
        data_path=str(data_path),
        schema_path=str(schema_path),
        metadata_path=str(metadata_path),
        index_path=str(index_path),
        schema_changed=schema_changed,
        previous_schema_hash=previous_schema_hash,
    )
