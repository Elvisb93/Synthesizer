from __future__ import annotations

import csv
import json

import pytest

from core.exporters import export_power_bi_run
from core.exporters import power_bi_exporter
from core.models import ColumnDefinition, ColumnType, RowData


def _columns(extra: bool = False) -> list[ColumnDefinition]:
    columns = [
        ColumnDefinition(name="id", type=ColumnType.AUTO_INCREMENT),
        ColumnDefinition(name="customer_name", type=ColumnType.SHORT_TEXT, prompt_instruction="Customer full name"),
        ColumnDefinition(name="premium", type=ColumnType.NUMERIC, prompt_instruction="Annual premium"),
    ]
    if extra:
        columns.append(ColumnDefinition(name="active", type=ColumnType.BOOLEAN))
    return columns


def _rows() -> list[RowData]:
    return [
        RowData(data={"premium": 1200, "customer_name": "Ada Lovelace", "id": 1}),
        RowData(data={"premium": 950, "customer_name": "Grace Hopper", "id": 2}),
    ]


def test_power_bi_export_creates_versioned_run_contract(tmp_path):
    result = export_power_bi_run(
        tmp_path,
        _rows(),
        _columns(),
        dataset_name="Customer Contacts!",
        privacy_export_mode="Restored imported values",
        source_mode="fresh_generation",
        provider="OpenAI",
        model="gpt-test",
    )

    data_path = tmp_path / "runs" / result.run_id / "data.csv"
    schema_path = tmp_path / "runs" / result.run_id / "schema.json"
    metadata_path = tmp_path / "runs" / result.run_id / "metadata.json"
    index_path = tmp_path / "index.csv"

    assert data_path.exists()
    assert schema_path.exists()
    assert metadata_path.exists()
    assert index_path.exists()
    assert result.run_id.endswith("_customer_contacts")

    with data_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["id", "customer_name", "premium"]
        rows = list(reader)
    assert rows[0] == {"id": "1", "customer_name": "Ada Lovelace", "premium": "1200"}

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["schema_hash"]
    assert [column["name"] for column in schema["columns"]] == ["id", "customer_name", "premium"]
    assert schema["columns"][2]["power_bi_type_hint"] == "number"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["dataset_name"] == "Customer Contacts!"
    assert metadata["dataset_slug"] == "customer_contacts"
    assert metadata["row_count"] == 2
    assert metadata["column_count"] == 3
    assert metadata["privacy_export_mode"] == "Restored imported values"
    assert metadata["source_mode"] == "fresh_generation"
    assert metadata["provider"] == "OpenAI"
    assert metadata["model"] == "gpt-test"

    with index_path.open(newline="", encoding="utf-8") as handle:
        index_rows = list(csv.DictReader(handle))
    assert len(index_rows) == 1
    assert index_rows[0]["data_path"] == f"runs/{result.run_id}/data.csv"
    assert index_rows[0]["schema_path"] == f"runs/{result.run_id}/schema.json"
    assert index_rows[0]["metadata_path"] == f"runs/{result.run_id}/metadata.json"


def test_power_bi_export_does_not_overwrite_previous_runs(tmp_path):
    first = export_power_bi_run(tmp_path, _rows(), _columns(), dataset_name="Demo")
    second = export_power_bi_run(tmp_path, _rows(), _columns(), dataset_name="Demo")

    assert first.run_id != second.run_id
    assert (tmp_path / "runs" / first.run_id / "data.csv").exists()
    assert (tmp_path / "runs" / second.run_id / "data.csv").exists()

    with (tmp_path / "index.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 2


def test_power_bi_export_warns_when_schema_changes_for_same_dataset(tmp_path):
    first = export_power_bi_run(tmp_path, _rows(), _columns(), dataset_name="Demo")
    second = export_power_bi_run(tmp_path, _rows(), _columns(extra=True), dataset_name="Demo")

    first_metadata = json.loads((tmp_path / "runs" / first.run_id / "metadata.json").read_text(encoding="utf-8"))
    metadata = json.loads((tmp_path / "runs" / second.run_id / "metadata.json").read_text(encoding="utf-8"))
    assert second.schema_changed is True
    assert second.previous_schema_hash == first_metadata["schema_hash"]
    assert metadata["schema_changed_from_previous_run"] is True
    assert metadata["previous_schema_hash"] == first_metadata["schema_hash"]


def test_power_bi_export_does_not_append_index_when_run_write_fails(tmp_path, monkeypatch):
    def fail_write(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(power_bi_exporter, "_write_data_csv", fail_write)

    with pytest.raises(RuntimeError, match="disk full"):
        export_power_bi_run(tmp_path, _rows(), _columns(), dataset_name="Broken")

    assert not (tmp_path / "index.csv").exists()
