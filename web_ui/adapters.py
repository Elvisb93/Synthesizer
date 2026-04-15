from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd

from core.models import ColumnConstraints, ColumnDefinition, ColumnType


FIELD_TYPE_CHOICES = [column_type.value for column_type in ColumnType]
SUMMARY_HEADERS = ["name", "type", "prompt_instruction", "rules"]
GRID_HEADERS = ["row_id", "name", "type", "prompt_instruction", "allow_duplicates"]


def blank_field_record() -> dict[str, Any]:
    return {
        "name": "",
        "type": ColumnType.SHORT_TEXT.value,
        "prompt_instruction": "",
        "options": "",
        "regex_pattern": "",
        "min_value": "",
        "max_value": "",
        "min_length": "",
        "max_length": "",
        "allow_duplicates": False,
        "faker_provider": "",
    }


def normalize_field_type_value(raw_value: Any) -> str:
    if isinstance(raw_value, ColumnType):
        return raw_value.value

    text = str(raw_value or "").strip()
    if not text:
        return ColumnType.SHORT_TEXT.value

    if text.startswith("ColumnType."):
        text = text.split(".", 1)[1].strip()

    for column_type in ColumnType:
        if text in {column_type.value, column_type.name}:
            return column_type.value

    return ColumnType.SHORT_TEXT.value


def normalize_field_record(record: dict[str, Any] | None) -> dict[str, Any]:
    if record and "constraints" in record:
        try:
            return definition_to_field_record(ColumnDefinition(**record))
        except Exception:
            pass
    merged = blank_field_record()
    if record:
        merged.update(record)
    merged["name"] = str(merged.get("name", "") or "").strip()
    merged["type"] = normalize_field_type_value(merged.get("type", ColumnType.SHORT_TEXT.value))
    merged["prompt_instruction"] = str(merged.get("prompt_instruction", "") or "").strip()
    merged["options"] = str(merged.get("options", "") or "").strip()
    merged["regex_pattern"] = str(merged.get("regex_pattern", "") or "").strip()
    merged["faker_provider"] = str(merged.get("faker_provider", "") or "").strip()
    merged["allow_duplicates"] = bool(merged.get("allow_duplicates", False))
    for key in ("min_value", "max_value", "min_length", "max_length"):
        merged[key] = "" if merged.get(key, "") in (None, "") else merged.get(key)
    return merged


def definition_to_field_record(column: ColumnDefinition) -> dict[str, Any]:
    constraints = column.constraints or ColumnConstraints()
    return normalize_field_record(
        {
            "name": column.name,
            "type": column.type.value,
            "prompt_instruction": column.prompt_instruction,
            "options": ",".join(constraints.options or []),
            "regex_pattern": constraints.regex_pattern or "",
            "min_value": "" if constraints.min_value is None else constraints.min_value,
            "max_value": "" if constraints.max_value is None else constraints.max_value,
            "min_length": "" if constraints.min_length is None else constraints.min_length,
            "max_length": "" if constraints.max_length is None else constraints.max_length,
            "allow_duplicates": bool(constraints.allow_duplicates),
            "faker_provider": constraints.faker_provider or "",
        }
    )


def field_record_to_definition(record: dict[str, Any]) -> ColumnDefinition:
    record = normalize_field_record(record)
    try:
        column_type = ColumnType(record["type"])
    except ValueError:
        column_type = ColumnType.SHORT_TEXT

    constraints_kwargs: dict[str, Any] = {
        "options": [item.strip() for item in record["options"].split(",") if item.strip()],
        "regex_pattern": record["regex_pattern"] or None,
        "faker_provider": record["faker_provider"] or None,
        "allow_duplicates": bool(record["allow_duplicates"]),
    }

    for key in ("min_value", "max_value"):
        raw = record.get(key, "")
        if str(raw).strip():
            constraints_kwargs[key] = float(raw)

    for key in ("min_length", "max_length"):
        raw = record.get(key, "")
        if str(raw).strip():
            constraints_kwargs[key] = int(float(raw))

    return ColumnDefinition(
        name=record["name"],
        type=column_type,
        prompt_instruction=record["prompt_instruction"],
        constraints=ColumnConstraints(**constraints_kwargs),
    )


def columns_to_field_records(columns: list[ColumnDefinition]) -> list[dict[str, Any]]:
    return [definition_to_field_record(column) for column in columns]


def field_records_to_columns(records: list[dict[str, Any]]) -> list[ColumnDefinition]:
    columns: list[ColumnDefinition] = []
    for record in records:
        normalized = normalize_field_record(record)
        if not normalized["name"]:
            continue
        columns.append(field_record_to_definition(normalized))
    return columns


def field_choice_labels(records: list[dict[str, Any]]) -> list[str]:
    labels = []
    for index, record in enumerate(records, start=1):
        normalized = normalize_field_record(record)
        name = normalized["name"] or f"Field {index}"
        labels.append(f"{index}. {name}")
    return labels


def field_record_from_choice(records: list[dict[str, Any]], choice: str | None) -> dict[str, Any]:
    if not choice:
        return blank_field_record()
    try:
        index = int(str(choice).split(".", 1)[0]) - 1
    except Exception:
        return blank_field_record()
    if index < 0 or index >= len(records):
        return blank_field_record()
    return normalize_field_record(records[index])


def summary_rules_text(record: dict[str, Any]) -> str:
    normalized = normalize_field_record(record)
    bits: list[str] = []
    if normalized["options"]:
        bits.append("options")
    if normalized["regex_pattern"]:
        bits.append("regex")
    if str(normalized["min_value"]).strip() or str(normalized["max_value"]).strip():
        bits.append("numeric range")
    if str(normalized["min_length"]).strip() or str(normalized["max_length"]).strip():
        bits.append("length")
    if normalized["faker_provider"]:
        bits.append(f"faker={normalized['faker_provider']}")
    if normalized["allow_duplicates"]:
        bits.append("duplicates allowed")
    return ", ".join(bits) if bits else "basic"


def field_summary_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        normalized = normalize_field_record(record)
        rows.append(
            {
                "name": normalized["name"],
                "type": normalized["type"],
                "prompt_instruction": normalized["prompt_instruction"],
                "rules": summary_rules_text(normalized),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_HEADERS) if rows else pd.DataFrame(columns=SUMMARY_HEADERS)


def field_rows_markup(records: list[dict[str, Any]], selected_choice: str | None = None) -> str:
    if not records:
        return (
            "<div style=\"padding:16px;border:1px dashed #cbd5e1;border-radius:14px;background:#f8fafc;color:#475569;\">"
            "<strong>No rows yet.</strong><br>"
            "Use <em>Generate Fields</em> or <em>Add Row</em> to build your schema."
            "</div>"
        )

    selected_index = None
    if selected_choice:
        try:
            selected_index = int(str(selected_choice).split(".", 1)[0]) - 1
        except Exception:
            selected_index = None

    rows: list[str] = [
        (
            "<div style=\"display:grid;grid-template-columns:110px 220px 180px minmax(320px,1fr);gap:12px;"
            "padding:10px 14px;border-bottom:1px solid #d8dee8;font-weight:700;color:#0f172a;background:#f8fafc;\">"
            "<div>Row ID</div>"
            "<div>Name</div>"
            "<div>Type</div>"
            "<div>Prompt instruction</div>"
            "</div>"
        )
    ]
    for index, record in enumerate(records, start=1):
        normalized = normalize_field_record(record)
        prompt = escape(normalized["prompt_instruction"] or "No instruction yet.")
        rules = escape(summary_rules_text(normalized))
        background = "#eff6ff" if selected_index == index - 1 else "#ffffff"
        border = "#2563eb" if selected_index == index - 1 else "#d8dee8"
        rows.append(
            "".join(
                [
                    (
                        f"<div style=\"display:grid;grid-template-columns:110px 220px 180px minmax(320px,1fr);gap:12px;"
                        f"padding:12px 14px;border-top:1px solid {border};background:{background};align-items:start;\">"
                    ),
                    (
                        "<div style=\"display:inline-flex;align-items:center;justify-content:center;width:88px;"
                        "min-height:40px;border:1px solid #111827;border-radius:14px;background:#ffffff;font-weight:700;color:#111827;\">"
                        f"Row {index}</div>"
                    ),
                    (
                        "<div style=\"min-height:40px;padding:10px 14px;border:1px solid #d8dee8;border-radius:14px;"
                        "background:#ffffff;color:#111827;font-weight:600;\">"
                        f"{escape(normalized['name'] or f'field_{index}')}"
                        "</div>"
                    ),
                    (
                        "<div style=\"min-height:40px;padding:10px 14px;border:1px solid #d8dee8;border-radius:14px;"
                        "background:#ffffff;color:#111827;\">"
                        f"{escape(normalized['type'])}"
                        "</div>"
                    ),
                    (
                        "<div style=\"min-height:40px;padding:10px 14px;border:1px solid #d8dee8;border-radius:14px;"
                        "background:#ffffff;color:#111827;\">"
                        f"{prompt}"
                        f"<div style=\"margin-top:6px;color:#6b7280;font-size:0.92rem;\">Rules: {rules}</div>"
                        "</div>"
                    ),
                    "</div>",
                ]
            )
        )

    return (
        "<div style=\"border:1px solid #d8dee8;border-radius:18px;overflow:hidden;background:#ffffff;\">"
        + "".join(rows)
        + "</div>"
    )


def field_records_to_grid_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        normalized = normalize_field_record(record)
        rows.append(
            {
                "row_id": f"Row {index}",
                "name": normalized["name"],
                "type": normalized["type"],
                "prompt_instruction": normalized["prompt_instruction"],
                "allow_duplicates": bool(normalized["allow_duplicates"]),
            }
        )
    if not rows:
        rows.append(
            {
                "row_id": "Row 1",
                "name": "",
                "type": ColumnType.SHORT_TEXT.value,
                "prompt_instruction": "",
                "allow_duplicates": False,
            }
        )
    return pd.DataFrame(rows, columns=GRID_HEADERS)


def import_preview_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).head(25)


def infer_field_records_from_dataframe(df: pd.DataFrame) -> list[dict[str, Any]]:
    inferred: list[dict[str, Any]] = []
    for column_name in df.columns:
        series = df[column_name]
        if pd.api.types.is_integer_dtype(series):
            column_type = ColumnType.NUMERIC.value
        elif pd.api.types.is_float_dtype(series):
            column_type = ColumnType.NUMERIC.value
        elif pd.api.types.is_bool_dtype(series):
            column_type = ColumnType.BOOLEAN.value
        else:
            column_type = ColumnType.SHORT_TEXT.value
        inferred.append(
            normalize_field_record(
                {
                    "name": str(column_name),
                    "type": column_type,
                    "prompt_instruction": "(Imported)",
                }
            )
        )
    return inferred


def build_schema_context(records: list[dict[str, Any]]) -> str | None:
    if not records:
        return None
    sample_row = records[0]
    context_lines = []
    for header, value in sample_row.items():
        type_hint = "String"
        if isinstance(value, bool):
            type_hint = "Boolean"
        elif isinstance(value, int):
            type_hint = "Integer"
        elif isinstance(value, float):
            type_hint = "Float"
        context_lines.append(f"Column: {header} ({type_hint}) | Sample: {value}")
    return "\n".join(context_lines)


def visibility_for_field_type(field_type: str) -> dict[str, bool]:
    selected = (field_type or ColumnType.SHORT_TEXT.value).strip()
    return {
        "show_options": selected == ColumnType.CATEGORICAL.value,
        "show_numeric": selected == ColumnType.NUMERIC.value,
        "show_text": selected in {ColumnType.SHORT_TEXT.value, ColumnType.LONG_TEXT.value},
        "show_faker": selected == ColumnType.DETERMINISTIC.value,
    }
