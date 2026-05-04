from __future__ import annotations

import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from core.app_config import build_generator_config
from core.controller import GeneratorController
from core.llm_client import LLMClient
from core.models import AIProvider, ColumnDefinition, ColumnType, RowData
from web_ui.adapters import (
    detect_privacy_leaks,
    IMPORT_PRIVACY_CHOICES,
    blank_field_record,
    build_schema_context,
    field_choice_labels,
    field_records_to_grid_dataframe,
    field_record_from_choice,
    field_records_to_columns,
    field_rows_markup,
    GRID_HEADERS,
    imported_columns_markup,
    import_preview_dataframe,
    infer_field_records_from_dataframe,
    mask_imported_records,
    normalize_field_record,
    normalize_field_type_value,
    sanitize_imported_records,
    visibility_for_field_type,
)
from web_ui.state import (
    WebSessionState,
    activity_markdown,
    append_activity,
    clear_runtime_controller,
    get_runtime_controller,
    register_runtime_controller,
)
from web_ui.runtime_cleanup import record_runtime_collection


EXPORT_DIR = Path(".web_ui_exports")
POWER_BI_EXPORT_DIR = EXPORT_DIR / "power_bi"
POWER_BI_PRIVACY_CHOICES = [
    "Restored imported values",
    "Masked imported values where available",
]
RESULT_QUALITY_CHOICES = ["Best Quality", "Balanced", "Fast Draft"]


def _row_id_value(selected_choice: str | None) -> str:
    if not selected_choice:
        return "New"
    try:
        index = int(str(selected_choice).split(".", 1)[0])
        return f"Row {index}"
    except Exception:
        return "New"


def _number_field_value(value: Any) -> Any:
    return None if value in ("", None) else value


def _combined_activity_markdown(session: WebSessionState, live_logs: list[str] | None = None) -> str:
    lines = list(session.activity_log[-8:])
    if live_logs:
        lines.extend(live_logs[-8:])
    if not lines:
        return "No activity yet."
    return "\n".join(f"- {line}" for line in lines[-12:])


def _generation_progress_markdown(
    *,
    done: int,
    target: int,
    retries: int,
    current_row: int,
    last_event: str,
    started_at: float,
    live_logs: list[str],
    is_running: bool,
    metrics: dict[str, Any] | None = None,
) -> str:
    safe_target = max(target, 1)
    percent = min(100, round((done / safe_target) * 100))
    elapsed = max(0, int(time.time() - started_at))
    state_label = "Running" if is_running else "Completed"
    recent_lines = "\n".join(f"- {line}" for line in live_logs[-6:]) if live_logs else "- Waiting for the model to return output..."
    cost_lines = _estimated_cost_markdown(metrics)
    return (
        f"### Generation Progress\n"
        f"- Status: **{state_label}**\n"
        f"- Progress: **{done}/{target}** row(s) completed ({percent}%)\n"
        f"- Current row: **{min(max(current_row, 1), safe_target)}**\n"
        f"- Retries so far: **{retries}**\n"
        f"- Last event: {last_event}\n"
        f"- Elapsed: **{elapsed}s**\n\n"
        f"{cost_lines}\n\n"
        f"**Recent events**\n{recent_lines}"
    )


def _estimated_cost_markdown(metrics: dict[str, Any] | None) -> str:
    if not metrics:
        return "**Estimated usage**\n- Token usage unavailable until the model reports usage."

    total = metrics.get("total") or {}
    avg_row = metrics.get("avg_row") or {}
    input_tokens = int(total.get("in", 0) or 0)
    output_tokens = int(total.get("out", 0) or 0)
    total_tokens = int(total.get("used", 0) or 0)
    if total_tokens <= 0:
        return (
            "**Estimated usage**\n"
            "- Token usage unavailable from this provider/runtime, so estimated cost cannot be calculated."
        )

    total_cost = float(total.get("cost", 0.0) or 0.0)
    avg_cost = float(avg_row.get("cost", 0.0) or 0.0)
    return (
        "**Estimated usage**\n"
        f"- Tokens: **{total_tokens:,}** total ({input_tokens:,} in / {output_tokens:,} out)\n"
        f"- Estimated cost: **${total_cost:.6f}** total"
        + (f" / **${avg_cost:.6f}** per completed row" if avg_cost > 0 else "")
    )


def _quality_settings(
    quality_preset: str,
    similarity_threshold: float,
    max_retries: int,
) -> tuple[float, int, str]:
    preset = quality_preset if quality_preset in RESULT_QUALITY_CHOICES else "Best Quality"
    if preset == "Fast Draft":
        return min(float(similarity_threshold or 0.85), 0.82), max(20, int(max_retries or 20)), "Fast Draft"
    if preset == "Balanced":
        return max(0.84, float(similarity_threshold or 0.85)), max(50, int(max_retries or 50)), "Balanced"
    return max(0.88, float(similarity_threshold or 0.88)), max(80, int(max_retries or 80)), "Best Quality"


def _schema_guidance(records: list[dict[str, Any]]) -> list[str]:
    guidance: list[str] = []
    normalized = [normalize_field_record(record) for record in records]
    generated = [record for record in normalized if record["prompt_instruction"] != "(Imported)"]
    if not normalized:
        return ["Add or generate fields before running data generation."]
    vague = [
        record["name"]
        for record in generated
        if len(str(record.get("prompt_instruction", "")).strip()) < 12
    ]
    if vague:
        guidance.append(f"Add clearer instructions for: {', '.join(vague[:4])}.")
    categorical_without_options = [
        record["name"]
        for record in generated
        if record["type"] == ColumnType.CATEGORICAL.value and not str(record.get("options", "")).strip()
    ]
    if categorical_without_options:
        guidance.append(f"Add allowed options for categorical fields: {', '.join(categorical_without_options[:4])}.")
    if not generated:
        guidance.append("All fields are imported. Add at least one new field if you want enrichment.")
    return guidance


def _schema_status(records: list[dict[str, Any]], prefix: str) -> str:
    guidance = _schema_guidance(records)
    if not guidance:
        return f"{prefix} The schema looks ready for generation."
    lines = [prefix, "", "**Before generating:**"]
    lines.extend(f"- {item}" for item in guidance)
    return "\n".join(lines)


def _controller_metrics(controller: Any) -> dict[str, Any]:
    try:
        return controller.get_metrics() or {}
    except Exception:
        return {}


def request_stop_data_generation(session: WebSessionState):
    controller = get_runtime_controller(session, "data")
    if controller is None:
        append_activity(session, "No data generation is currently running.")
        return session, "Generation progress will appear here once you start a run.", "No active generation to stop.", activity_markdown(session)

    if hasattr(controller, "stop_generation"):
        controller.stop_generation()
    else:
        controller.stop_requested = True
    append_activity(session, "Stop requested for sample data generation.")
    return (
        session,
        "### Generation Progress\n- Status: **Stopping**\n- Finishing the current unit of work before stopping.",
        "Stop requested. Partial rows will remain available for export.",
        activity_markdown(session),
    )


def _grid_rows(grid_value: Any) -> list[dict[str, Any]]:
    if isinstance(grid_value, pd.DataFrame):
        return grid_value.fillna("").to_dict(orient="records")
    if isinstance(grid_value, list):
        if not grid_value:
            return []
        if isinstance(grid_value[0], dict):
            return [{key: row.get(key, "") for key in GRID_HEADERS} for row in grid_value]
        return [
            {
                "row_id": row[0] if len(row) > 0 else "",
                "name": row[1] if len(row) > 1 else "",
                "type": row[2] if len(row) > 2 else "",
                "prompt_instruction": row[3] if len(row) > 3 else "",
                "allow_duplicates": row[4] if len(row) > 4 else False,
            }
            for row in grid_value
        ]
    return []


def _bool_value(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"true", "1", "yes", "y"}


def _grid_row_choices(rows: list[dict[str, Any]]) -> list[str]:
    return [f"Row {index}" for index in range(1, len(rows) + 1)]


def _selected_grid_row_index(rows: list[dict[str, Any]], selected_choice: str | None) -> int:
    if not rows:
        return 0
    if selected_choice:
        try:
            selected_index = int(str(selected_choice).split(" ", 1)[1]) - 1
            if 0 <= selected_index < len(rows):
                return selected_index
        except Exception:
            pass
    return 0


def _grid_update(df: pd.DataFrame):
    visible_rows = max(6, min(len(df.index) + 1, 16))
    return gr.update(value=df.values.tolist(), row_count=(visible_rows, "dynamic"))


def _schema_overview_markdown(grid_value: Any, selected_choice: str | None = None) -> str:
    rows = _grid_rows(grid_value)
    records = [
        normalize_field_record(
            {
                "name": row.get("name", ""),
                "type": row.get("type", ColumnType.SHORT_TEXT.value),
                "prompt_instruction": row.get("prompt_instruction", ""),
                "allow_duplicates": _bool_value(row.get("allow_duplicates", False)),
            }
        )
        for row in rows
        if any(str(row.get(key, "") or "").strip() for key in ("name", "prompt_instruction"))
    ]
    return field_rows_markup(records, selected_choice=selected_choice)


def refresh_schema_overview(grid_value: Any, selected_choice: str | None = None):
    return _schema_overview_markdown(grid_value, selected_choice=selected_choice)


def _coerce_privacy_mode(raw_mode: str | None) -> str:
    preferred_default = "Mask likely personal values"
    fallback = preferred_default if preferred_default in IMPORT_PRIVACY_CHOICES else IMPORT_PRIVACY_CHOICES[0]
    mode = (raw_mode or fallback).strip()
    return mode if mode in IMPORT_PRIVACY_CHOICES else fallback


def _imported_column_names(session: WebSessionState) -> list[str]:
    names: list[str] = []
    for record in session.fields:
        normalized = normalize_field_record(record)
        if normalized["prompt_instruction"] == "(Imported)" and normalized["name"]:
            names.append(normalized["name"])
    return names


_UNRESOLVED_PLACEHOLDER_RE = re.compile(r"<([A-Z]+)(?:_[0-9A-Z]+)+>")


def _replace_unresolved_placeholder_tokens(text: str) -> str:
    replacements = {
        "NAME": "the sender",
        "EMAIL": "the listed email",
        "PHONE": "the listed phone number",
        "ORG": "the organization",
        "ROLE": "the role",
        "IDENTIFIER": "the reference number",
        "DATE": "the listed date",
        "TIME": "the listed time",
        "ADDRESS": "the listed address",
        "URL": "the linked resource",
    }

    def replace(match: re.Match[str]) -> str:
        return replacements.get(match.group(1), "the referenced value")

    return _UNRESOLVED_PLACEHOLDER_RE.sub(replace, text)


def restore_original_imported_columns(session: WebSessionState, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows or not session.raw_imported_data:
        return list(rows)

    imported_columns = _imported_column_names(session)
    if not imported_columns:
        return list(rows)

    restored_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        restored = dict(row)
        row_mapping = session.import_mask_mappings[index] if index < len(session.import_mask_mappings) else {}
        for key, value in list(restored.items()):
            if isinstance(value, str) and row_mapping:
                for placeholder, original in sorted(row_mapping.items(), key=lambda item: len(item[0]), reverse=True):
                    restored[key] = restored[key].replace(placeholder, original)
            if isinstance(restored.get(key), str):
                restored[key] = _replace_unresolved_placeholder_tokens(restored[key])
        if index < len(session.raw_imported_data):
            original = session.raw_imported_data[index]
            for column_name in imported_columns:
                if column_name in original:
                    restored[column_name] = original[column_name]
        restored_rows.append(restored)
    return restored_rows


def _apply_import_records(
    session: WebSessionState,
    records: list[dict[str, Any]],
    *,
    file_label: str,
    privacy_mode: str,
    reusing_existing_fields: bool = False,
):
    session.import_privacy_mode = _coerce_privacy_mode(privacy_mode)
    session.raw_imported_data = list(records)
    session.imported_data, session.import_mask_mappings = mask_imported_records(session.raw_imported_data, session.import_privacy_mode)
    privacy_leaks = detect_privacy_leaks(session.raw_imported_data, session.imported_data)
    if not reusing_existing_fields:
        session.fields = infer_field_records_from_dataframe(pd.DataFrame(session.raw_imported_data))

    status = (
        f"Imported **{len(session.raw_imported_data)}** row(s) from `{file_label}`. "
        f"Review the inferred fields, adjust their types, then generate."
    )
    if session.import_privacy_mode == "Mask likely personal values":
        status += " Privacy masking is active for the preview and AI context."
        if privacy_leaks:
            status += f" Privacy audit warning: {len(privacy_leaks)} token(s) still appear to leak."

    return (
        session,
        len(session.raw_imported_data),
        gr.update(value=status),
        gr.update(value=imported_columns_markup(session.raw_imported_data, session.import_privacy_mode)),
        gr.update(value=import_preview_dataframe(session.imported_data), visible=bool(session.imported_data)),
        _grid_update(field_records_to_grid_dataframe(session.fields)),
        status,
        refresh_schema_overview(field_records_to_grid_dataframe(session.fields)),
        activity_markdown(session),
    )


def _requested_minimum_rows(prompt: str) -> int:
    text = (prompt or "").lower()
    match = re.search(r"(?:minimum of|at least|minimum)\s+(\d+)\s+(?:columns|fields|rows)", text)
    if match:
        return max(1, int(match.group(1)))
    return 0


def _records_from_grid(grid_value: Any, existing_records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
    rows = _grid_rows(grid_value)
    built: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, row in enumerate(rows):
        existing = normalize_field_record(existing_records[index]) if index < len(existing_records) else blank_field_record()
        name = str(row.get("name", "") or "").strip()
        field_type = str(row.get("type", "") or "").strip() or existing["type"]
        prompt_instruction = str(row.get("prompt_instruction", "") or "").strip()
        if not any([name, prompt_instruction]):
            continue
        record = normalize_field_record(
            {
                **existing,
                "name": name,
                "type": field_type,
                "prompt_instruction": prompt_instruction,
                "allow_duplicates": _bool_value(row.get("allow_duplicates", existing["allow_duplicates"])),
            }
        )
        if not record["name"]:
            return existing_records, f"Row {index + 1} needs a name before it can be saved."
        lowered = record["name"].lower()
        if lowered in seen_names:
            return existing_records, f"Row {index + 1} duplicates the name **{record['name']}**."
        seen_names.add(lowered)
        built.append(record)
    return built, None


def add_grid_row(grid_value: Any):
    rows = _grid_rows(grid_value)
    cleaned = [
        {
            "row_id": str(row.get("row_id", "") or f"Row {index + 1}"),
            "name": str(row.get("name", "") or ""),
            "type": str(row.get("type", "") or ColumnType.SHORT_TEXT.value),
            "prompt_instruction": str(row.get("prompt_instruction", "") or ""),
            "allow_duplicates": _bool_value(row.get("allow_duplicates", False)),
        }
        for index, row in enumerate(rows)
    ]
    cleaned.append(
        {
            "row_id": f"Row {len(cleaned) + 1}",
            "name": "",
            "type": ColumnType.SHORT_TEXT.value,
            "prompt_instruction": "",
            "allow_duplicates": False,
        }
    )
    df = pd.DataFrame(cleaned, columns=GRID_HEADERS)
    return _grid_update(df), "Added a new blank row at the bottom."


def remove_last_grid_row(grid_value: Any):
    rows = _grid_rows(grid_value)
    if len(rows) <= 1:
        df = field_records_to_grid_dataframe([])
        return _grid_update(df), "At least one editable row stays available."
    trimmed = rows[:-1]
    for index, row in enumerate(trimmed, start=1):
        row["row_id"] = f"Row {index}"
    df = pd.DataFrame(trimmed, columns=GRID_HEADERS)
    return _grid_update(df), f"Removed Row {len(rows)}."


def save_grid_rows(session: WebSessionState, grid_value: Any):
    records, error = _records_from_grid(grid_value, session.fields)
    if error:
        df = pd.DataFrame(_grid_rows(grid_value), columns=GRID_HEADERS)
        return session, _grid_update(df), error, activity_markdown(session)
    session.fields = records
    append_activity(session, f"Saved {len(records)} schema row(s).")
    df = field_records_to_grid_dataframe(records)
    return session, _grid_update(df), _schema_status(records, f"Saved **{len(records)}** row(s)."), activity_markdown(session)


def sync_grid_row_editor(grid_value: Any, selected_choice: str | None):
    rows = _grid_rows(grid_value)
    if not rows:
        rows = _grid_rows(field_records_to_grid_dataframe([]))
    choices = _grid_row_choices(rows)
    selected_index = _selected_grid_row_index(rows, selected_choice)
    selected_value = choices[selected_index]
    row = rows[selected_index]
    return (
        gr.update(choices=choices, value=selected_value),
        str(row.get("name", "") or ""),
        normalize_field_type_value(row.get("type", ColumnType.SHORT_TEXT.value)),
        str(row.get("prompt_instruction", "") or ""),
        _bool_value(row.get("allow_duplicates", False)),
    )


def load_grid_row_editor(grid_value: Any, selected_choice: str | None):
    _, name, field_type, prompt_instruction, allow_duplicates = sync_grid_row_editor(grid_value, selected_choice)
    return name, field_type, prompt_instruction, allow_duplicates


def apply_grid_row_edit(
    grid_value: Any,
    selected_choice: str | None,
    name: str,
    field_type: str,
    prompt_instruction: str,
    allow_duplicates: bool,
):
    rows = _grid_rows(grid_value)
    if not rows:
        rows = _grid_rows(field_records_to_grid_dataframe([]))

    selected_index = _selected_grid_row_index(rows, selected_choice)
    for index, row in enumerate(rows, start=1):
        row["row_id"] = f"Row {index}"

    rows[selected_index] = {
        "row_id": f"Row {selected_index + 1}",
        "name": str(name or "").strip(),
        "type": normalize_field_type_value(field_type or ColumnType.SHORT_TEXT.value),
        "prompt_instruction": str(prompt_instruction or "").strip(),
        "allow_duplicates": _bool_value(allow_duplicates),
    }

    df = pd.DataFrame(rows, columns=GRID_HEADERS)
    selector_update, editor_name, editor_type, editor_prompt, editor_allow_duplicates = sync_grid_row_editor(
        df, f"Row {selected_index + 1}"
    )
    status = f"Updated **Row {selected_index + 1}** in the editor. Click **Save Rows** to persist it."
    return (
        _grid_update(df),
        selector_update,
        editor_name,
        editor_type,
        editor_prompt,
        editor_allow_duplicates,
        status,
    )


def _field_editor_outputs(record: dict[str, Any], choices: list[str], selected_choice: str | None):
    visibility = visibility_for_field_type(record["type"])
    return (
        gr.update(choices=choices, value=selected_choice),
        _row_id_value(selected_choice),
        record["name"],
        record["type"],
        record["prompt_instruction"],
        record["options"],
        record["regex_pattern"],
        _number_field_value(record["min_value"]),
        _number_field_value(record["max_value"]),
        _number_field_value(record["min_length"]),
        _number_field_value(record["max_length"]),
        record["allow_duplicates"],
        record["faker_provider"],
        gr.update(visible=visibility["show_options"]),
        gr.update(visible=visibility["show_text"]),
        gr.update(visible=visibility["show_numeric"]),
        gr.update(visible=visibility["show_faker"]),
    )


def field_type_changed(field_type: str):
    visibility = visibility_for_field_type(field_type)
    return (
        gr.update(visible=visibility["show_options"]),
        gr.update(visible=visibility["show_text"]),
        gr.update(visible=visibility["show_numeric"]),
        gr.update(visible=visibility["show_faker"]),
    )


def new_field_editor(session: WebSessionState):
    record = blank_field_record()
    return (
        session,
        *_field_editor_outputs(record, field_choice_labels(session.fields), None),
        field_rows_markup(session.fields, None),
        "Ready to add a new field.",
        activity_markdown(session),
    )


def load_selected_field(session: WebSessionState, selected_choice: str | None):
    record = field_record_from_choice(session.fields, selected_choice)
    if not selected_choice:
        return (
            session,
            *_field_editor_outputs(record, field_choice_labels(session.fields), None),
            field_rows_markup(session.fields, None),
            "Select a field to edit, or start a new one.",
            activity_markdown(session),
        )
    return (
        session,
        *_field_editor_outputs(record, field_choice_labels(session.fields), selected_choice),
        field_rows_markup(session.fields, selected_choice),
        f"Editing **{record['name']}**.",
        activity_markdown(session),
    )


def save_field(
    session: WebSessionState,
    selected_choice: str | None,
    name: str,
    field_type: str,
    prompt_instruction: str,
    options: str,
    regex_pattern: str,
    min_value: Any,
    max_value: Any,
    min_length: Any,
    max_length: Any,
    allow_duplicates: bool,
    faker_provider: str,
):
    record = normalize_field_record(
        {
            "name": name,
            "type": field_type,
            "prompt_instruction": prompt_instruction,
            "options": options,
            "regex_pattern": regex_pattern,
            "min_value": min_value,
            "max_value": max_value,
            "min_length": min_length,
            "max_length": max_length,
            "allow_duplicates": allow_duplicates,
            "faker_provider": faker_provider,
        }
    )
    if not record["name"]:
        return (
            session,
            *_field_editor_outputs(record, field_choice_labels(session.fields), selected_choice),
            field_rows_markup(session.fields, selected_choice),
            "Field name is required.",
            activity_markdown(session),
        )

    fields = list(session.fields)
    selected_index = None
    if selected_choice:
        try:
            selected_index = int(str(selected_choice).split(".", 1)[0]) - 1
        except Exception:
            selected_index = None

    duplicate_name = next(
        (idx for idx, existing in enumerate(fields) if normalize_field_record(existing)["name"].lower() == record["name"].lower()),
        None,
    )
    if duplicate_name is not None and duplicate_name != selected_index:
        return (
            session,
            *_field_editor_outputs(record, field_choice_labels(fields), selected_choice),
            field_rows_markup(fields, selected_choice),
            f"A field named **{record['name']}** already exists.",
            activity_markdown(session),
        )

    if selected_index is None or selected_index < 0 or selected_index >= len(fields):
        fields.append(record)
        append_activity(session, f"Added field: {record['name']}")
    else:
        fields[selected_index] = record
        append_activity(session, f"Updated field: {record['name']}")

    session.fields = fields
    choices = field_choice_labels(fields)
    new_index = next(idx for idx, existing in enumerate(fields) if normalize_field_record(existing)["name"] == record["name"])
    selected_value = choices[new_index]
    return (
        session,
        *_field_editor_outputs(record, choices, selected_value),
        field_rows_markup(fields, selected_value),
        f"Saved field **{record['name']}**.",
        activity_markdown(session),
    )


def add_field(
    session: WebSessionState,
    name: str,
    field_type: str,
    prompt_instruction: str,
    options: str,
    regex_pattern: str,
    min_value: Any,
    max_value: Any,
    min_length: Any,
    max_length: Any,
    allow_duplicates: bool,
    faker_provider: str,
):
    record = normalize_field_record(
        {
            "name": name,
            "type": field_type,
            "prompt_instruction": prompt_instruction,
            "options": options,
            "regex_pattern": regex_pattern,
            "min_value": min_value,
            "max_value": max_value,
            "min_length": min_length,
            "max_length": max_length,
            "allow_duplicates": allow_duplicates,
            "faker_provider": faker_provider,
        }
    )
    if not record["name"]:
        return (
            session,
            *_field_editor_outputs(record, field_choice_labels(session.fields), None),
            field_rows_markup(session.fields, None),
            "Enter a row name before adding it.",
            activity_markdown(session),
        )

    fields = list(session.fields)
    duplicate_name = next(
        (idx for idx, existing in enumerate(fields) if normalize_field_record(existing)["name"].lower() == record["name"].lower()),
        None,
    )
    if duplicate_name is not None:
        choice = field_choice_labels(fields)[duplicate_name]
        return (
            session,
            *_field_editor_outputs(record, field_choice_labels(fields), choice),
            field_rows_markup(fields, choice),
            f"A row named **{record['name']}** already exists. Edit the existing row or choose a different name.",
            activity_markdown(session),
        )

    fields.append(record)
    session.fields = fields
    append_activity(session, f"Added field: {record['name']}")
    blank_record = blank_field_record()
    return (
        session,
        *_field_editor_outputs(blank_record, field_choice_labels(fields), None),
        field_rows_markup(fields, None),
        f"Added row **{record['name']}**. You can enter the next one now.",
        activity_markdown(session),
    )


def remove_field(session: WebSessionState, selected_choice: str | None):
    if not selected_choice:
        return (
            session,
            *_field_editor_outputs(blank_field_record(), field_choice_labels(session.fields), None),
            field_rows_markup(session.fields, None),
            "Select a field first.",
            activity_markdown(session),
        )

    try:
        selected_index = int(str(selected_choice).split(".", 1)[0]) - 1
    except Exception:
        selected_index = -1

    fields = list(session.fields)
    if selected_index < 0 or selected_index >= len(fields):
        return (
            session,
            *_field_editor_outputs(blank_field_record(), field_choice_labels(fields), None),
            field_rows_markup(fields, None),
            "Could not find that field.",
            activity_markdown(session),
        )

    removed = normalize_field_record(fields.pop(selected_index))
    session.fields = fields
    append_activity(session, f"Removed field: {removed['name']}")

    choices = field_choice_labels(fields)
    selected_value = choices[min(selected_index, len(choices) - 1)] if choices else None
    record = field_record_from_choice(fields, selected_value)
    status = f"Removed field **{removed['name']}**."
    return (
        session,
        *_field_editor_outputs(record, choices, selected_value),
        field_rows_markup(fields, selected_value),
        status,
        activity_markdown(session),
    )


def import_data_file(session: WebSessionState, file_path: str, privacy_mode: str):
    if not file_path:
        append_activity(session, "No data file selected.")
        return (
            session,
            10,
            gr.update(value="Select a CSV or JSON file to begin."),
            gr.update(value="No imported columns yet."),
            gr.update(value=pd.DataFrame(), visible=False),
            _grid_update(field_records_to_grid_dataframe(session.fields)),
            "Select a CSV or JSON file to begin.",
            refresh_schema_overview(field_records_to_grid_dataframe(session.fields)),
            activity_markdown(session),
        )

    if str(file_path).lower().endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_json(file_path)

    append_activity(session, f"Imported {len(df)} row(s) from {os.path.basename(file_path)}")
    return _apply_import_records(
        session,
        df.to_dict(orient="records"),
        file_label=os.path.basename(file_path),
        privacy_mode=privacy_mode,
    )


def apply_import_privacy_mode(session: WebSessionState, privacy_mode: str):
    privacy_mode = _coerce_privacy_mode(privacy_mode)
    session.import_privacy_mode = privacy_mode
    if not session.raw_imported_data:
        append_activity(session, f"Privacy mode set to {privacy_mode.lower()}.")
        session.import_mask_mappings = []
        return (
            session,
            privacy_mode,
            gr.update(value="No imported columns yet."),
            gr.update(value=pd.DataFrame(), visible=False),
            "Import a CSV or JSON file to preview and clean personal values.",
            activity_markdown(session),
        )

    session.imported_data, session.import_mask_mappings = mask_imported_records(session.raw_imported_data, privacy_mode)
    privacy_leaks = detect_privacy_leaks(session.raw_imported_data, session.imported_data)
    append_activity(session, f"Applied import privacy mode: {privacy_mode}.")
    status = f"Updated imported data preview. Privacy mode: **{privacy_mode}**."
    if privacy_mode == "Mask likely personal values":
        status += " The model will use the masked import context."
        if privacy_leaks:
            status += f" Privacy audit warning: **{len(privacy_leaks)}** token(s) may still leak."
    else:
        status += " The model will use the original import values."
    return (
        session,
        privacy_mode,
        gr.update(value=imported_columns_markup(session.raw_imported_data, session.import_privacy_mode)),
        gr.update(value=import_preview_dataframe(session.imported_data), visible=True),
        status,
        activity_markdown(session),
    )


def suggest_fields(
    session: WebSessionState,
    prompt: str,
    current_model: str,
    provider: str,
    api_key: str,
    azure_endpoint: str,
    azure_deployment: str,
):
    prompt = (prompt or "").strip()
    if not prompt:
        append_activity(session, "Field suggestion skipped: no prompt provided.")
        return (
            session,
            _grid_update(field_records_to_grid_dataframe(session.fields)),
            "Add a plain-language data description first.",
            activity_markdown(session),
        )

    try:
        if session.import_privacy_mode == "Mask likely personal values" and session.raw_imported_data:
            session.imported_data, session.import_mask_mappings = mask_imported_records(
                session.raw_imported_data,
                session.import_privacy_mode,
            )
            privacy_leaks = detect_privacy_leaks(session.raw_imported_data, session.imported_data)
            if privacy_leaks:
                append_activity(session, f"Generation blocked: privacy audit found {len(privacy_leaks)} remaining leak(s).")
                leak_preview = "; ".join(privacy_leaks[:3])
                return (
                    session,
                    gr.update(value=pd.DataFrame(), visible=False),
                    "Field suggestion failed because privacy masking left unresolved personal values.",
                    f"Privacy audit failed before generation. Example leak(s): {leak_preview}",
                    activity_markdown(session),
                )

        config = build_generator_config(
            {
                "model_id": current_model or "local-model",
                "provider": provider,
                "api_key": api_key,
                "azure_endpoint": azure_endpoint,
                "azure_deployment": azure_deployment,
            },
            include_existing_data=False,
        )
        client = LLMClient(config)
        context = build_schema_context(session.imported_data)
        schema_list = client.generate_schema(prompt, context=context)
        minimum_rows = _requested_minimum_rows(prompt)
        if minimum_rows and len(schema_list) < minimum_rows:
            heuristic_rows = client._generate_heuristic_schema(prompt)
            existing_names = {str(item.get("name", "")).strip().lower() for item in schema_list}
            for item in heuristic_rows:
                key = str(item.get("name", "")).strip().lower()
                if key and key not in existing_names:
                    schema_list.append(item)
                    existing_names.add(key)
                if len(schema_list) >= minimum_rows:
                    break
        if not schema_list:
            append_activity(session, "Field suggestion returned no usable columns.")
            return (
                session,
                _grid_update(field_records_to_grid_dataframe(session.fields)),
                "No fields could be generated from that prompt. Try a simpler description or check the local model output.",
                activity_markdown(session),
            )

        existing_by_name = {
            normalize_field_record(record)["name"].lower(): normalize_field_record(record)
            for record in session.fields
            if normalize_field_record(record)["name"]
        }
        last_added_name = None
        for item in schema_list:
            constraints = item.get("constraints") or {}
            record = normalize_field_record(
                {
                    "name": item.get("name", ""),
                    "type": item.get("type", ColumnType.SHORT_TEXT.value),
                    "prompt_instruction": item.get("prompt_instruction", ""),
                    "options": ",".join(constraints.get("options", []) or []),
                    "regex_pattern": constraints.get("regex_pattern", "") or "",
                    "min_value": constraints.get("min_value", ""),
                    "max_value": constraints.get("max_value", ""),
                    "min_length": constraints.get("min_length", ""),
                    "max_length": constraints.get("max_length", ""),
                    "allow_duplicates": bool(constraints.get("allow_duplicates", False)),
                    "faker_provider": constraints.get("faker_provider", "") or "",
                }
            )
            if not record["name"]:
                continue
            key = record["name"].lower()
            if key not in existing_by_name:
                existing_by_name[key] = record
                last_added_name = record["name"]

        session.fields = list(existing_by_name.values())
        choices = field_choice_labels(session.fields)
        selected_choice = None
        if last_added_name:
            for choice in choices:
                if choice.endswith(last_added_name):
                    selected_choice = choice
                    break
        selected_record = field_record_from_choice(session.fields, selected_choice)
        append_activity(session, f"Field suggestion complete: {len(schema_list)} suggestion(s) returned.")
        status = _schema_status(session.fields, f"Suggested fields are ready. Total editable fields: **{len(session.fields)}**.")
        return (
            session,
            _grid_update(field_records_to_grid_dataframe(session.fields)),
            status,
            activity_markdown(session),
        )
    except Exception as exc:
        append_activity(session, f"Field suggestion failed: {exc}")
        return (
            session,
            _grid_update(field_records_to_grid_dataframe(session.fields)),
            "Field suggestion failed. Check provider settings and try again.",
            activity_markdown(session),
        )


def suggest_fields_and_sync_editor(
    session: WebSessionState,
    prompt: str,
    current_model: str,
    provider: str,
    api_key: str,
    azure_endpoint: str,
    azure_deployment: str,
):
    session, _, status, activity = suggest_fields(
        session,
        prompt,
        current_model,
        provider,
        api_key,
        azure_endpoint,
        azure_deployment,
    )
    df = field_records_to_grid_dataframe(session.fields)
    row_selector, row_name, row_type, row_prompt, row_allow_duplicates = sync_grid_row_editor(df, None)
    selected_choice = row_selector.get("value") if isinstance(row_selector, dict) else None
    schema_overview = refresh_schema_overview(df, selected_choice)
    return (
        session,
        _grid_update(df),
        status,
        activity,
        row_selector,
        row_name,
        row_type,
        row_prompt,
        row_allow_duplicates,
        schema_overview,
    )


def _controller_from_session_rows(session: WebSessionState) -> tuple[GeneratorController, list[ColumnDefinition]]:
    columns = field_records_to_columns(session.fields)
    controller = GeneratorController()
    controller.columns = columns
    controller.generated_rows = [RowData(data=row) for row in session.generated_rows]
    return controller, columns


def _power_bi_rows_for_privacy_mode(session: WebSessionState, privacy_export_mode: str) -> list[dict[str, Any]]:
    rows = restore_original_imported_columns(session, session.generated_rows)
    if privacy_export_mode != "Masked imported values where available":
        return rows

    imported_columns = _imported_column_names(session)
    if not imported_columns or not session.imported_data:
        return rows

    masked_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        masked = dict(row)
        if index < len(session.imported_data):
            imported = session.imported_data[index]
            for column_name in imported_columns:
                if column_name in imported:
                    masked[column_name] = imported[column_name]
        masked_rows.append(masked)
    return masked_rows


def generate_data(
    session: WebSessionState,
    grid_value: Any,
    current_model: str,
    provider: str,
    api_key: str,
    azure_endpoint: str,
    azure_deployment: str,
    input_price_per_1m: float,
    output_price_per_1m: float,
    num_rows: int,
    similarity_threshold: float,
    max_retries: int,
    result_quality: str,
    rag_backend: str,
    collection_name: str,
    top_k: int,
    min_score: float,
    max_context_chars: int,
    embedding_model: str,
    source_filter: str,
    qdrant_url: str,
    qdrant_api_key: str,
    ocr_mode: str,
    ocr_dpi: int,
    ocr_max_pages: int,
    ocr_max_regions_per_page: int,
    ocr_region_padding_px: int,
    ocr_gap_multiplier: float,
    ocr_min_extracted_chars: int,
    ocr_timeout_ms_per_page: int,
    parser_mode: str,
    hybrid_search_enabled: bool,
    rerank_enabled: bool,
    summary_first_enabled: bool,
    summary_top_k: int,
    dense_top_k: int,
    lexical_top_k: int,
    parent_context_enabled: bool,
    parent_context_max_chars: int,
    graph_enabled: bool,
    graph_hops: int,
    graph_source_boost: float,
    late_interaction_enabled: bool,
    late_interaction_weight: float,
    quick_qa_mode: str,
    doc_mode: str,
    doc_pages: str,
    doc_quality: str,
    doc_audience: str,
    doc_tone: str,
    doc_chart_enabled: bool,
    doc_flow_enabled: bool,
    doc_max_charts: int,
):
    grid_records, grid_error = _records_from_grid(grid_value, session.fields)
    if grid_error:
        append_activity(session, grid_error)
        yield (
            session,
            gr.update(value=pd.DataFrame(), visible=False),
            "Generation progress will appear here once you start a run.",
            grid_error,
            activity_markdown(session),
        )
        return

    session.fields = grid_records
    columns = field_records_to_columns(session.fields)
    if not columns:
        append_activity(session, "Generation skipped: no fields defined.")
        yield (
            session,
            gr.update(value=pd.DataFrame(), visible=False),
            "Generation progress will appear here once you start a run.",
            "Add at least one field before generating.",
            activity_markdown(session),
        )
        return

    if session.raw_imported_data and session.import_privacy_mode != "Mask likely personal values":
        append_activity(session, "Generation blocked: imported files must use privacy masking before AI generation.")
        yield (
            session,
            gr.update(value=pd.DataFrame(), visible=False),
            "Generation progress will appear here once you start a run.",
            "Generation blocked. Imported files must use `Mask likely personal values` before the model can process them.",
            activity_markdown(session),
        )
        return

    try:
        if session.import_privacy_mode == "Mask likely personal values" and session.raw_imported_data:
            session.imported_data, session.import_mask_mappings = mask_imported_records(
                session.raw_imported_data,
                session.import_privacy_mode,
            )
            privacy_leaks = detect_privacy_leaks(session.raw_imported_data, session.imported_data)
            if privacy_leaks:
                append_activity(session, f"Generation blocked: privacy audit found {len(privacy_leaks)} remaining leak(s).")
                leak_preview = "; ".join(privacy_leaks[:3])
                yield (
                    session,
                    gr.update(value=pd.DataFrame(), visible=False),
                    "Generation progress will appear here once you start a run.",
                    f"Privacy audit failed before generation. Example leak(s): {leak_preview}",
                    activity_markdown(session),
                )
                return

        effective_similarity_threshold, effective_max_retries, effective_quality = _quality_settings(
            result_quality,
            similarity_threshold,
            max_retries,
        )
        config = build_generator_config(
            {
                "model_id": current_model or "local-model",
                "provider": provider,
                "api_key": api_key,
                "azure_endpoint": azure_endpoint,
                "azure_deployment": azure_deployment,
                "input_price_per_1m": input_price_per_1m,
                "output_price_per_1m": output_price_per_1m,
                "num_rows": int(num_rows or 10),
                "similarity_threshold": effective_similarity_threshold,
                "max_retries": effective_max_retries,
                "rag_backend": rag_backend,
                "collection_name": collection_name,
                "top_k": int(top_k or 5),
                "min_score": min_score,
                "max_context_chars": int(max_context_chars or 3000),
                "embedding_model": embedding_model,
                "source_filter": source_filter,
                "qdrant_url": qdrant_url,
                "qdrant_api_key": qdrant_api_key,
                "ocr_mode": ocr_mode,
                "ocr_dpi": int(ocr_dpi or 150),
                "ocr_max_pages": int(ocr_max_pages or 20),
                "ocr_max_regions_per_page": int(ocr_max_regions_per_page or 8),
                "ocr_region_padding_px": int(ocr_region_padding_px or 18),
                "ocr_gap_multiplier": float(ocr_gap_multiplier or 2.5),
                "ocr_min_extracted_chars": int(ocr_min_extracted_chars or 60),
                "ocr_timeout_ms_per_page": int(ocr_timeout_ms_per_page or 4000),
                "parser_mode": parser_mode,
                "hybrid_search_enabled": bool(hybrid_search_enabled),
                "rerank_enabled": bool(rerank_enabled),
                "summary_first_enabled": bool(summary_first_enabled),
                "summary_top_k": int(summary_top_k or 3),
                "dense_top_k": int(dense_top_k or 12),
                "lexical_top_k": int(lexical_top_k or 12),
                "parent_context_enabled": bool(parent_context_enabled),
                "parent_context_max_chars": int(parent_context_max_chars or 1200),
                "graph_enabled": bool(graph_enabled),
                "graph_hops": int(graph_hops or 1),
                "graph_source_boost": float(graph_source_boost or 0.08),
                "late_interaction_enabled": bool(late_interaction_enabled),
                "late_interaction_weight": float(late_interaction_weight or 0.2),
                "quick_qa_mode": quick_qa_mode,
                "doc_mode": doc_mode,
                "doc_pages": doc_pages,
                "doc_quality": doc_quality,
                "doc_audience": doc_audience,
                "doc_tone": doc_tone,
                "doc_chart_enabled": doc_chart_enabled,
                "doc_flow_enabled": doc_flow_enabled,
                "doc_max_charts": int(doc_max_charts or 3),
            },
            include_existing_data=True,
            existing_data=session.imported_data or None,
        )
        record_runtime_collection(collection_name, qdrant_url)

        controller = GeneratorController()
        collected_logs: list[str] = []
        target_count = config.num_rows
        progress_state = {
            "done": 0,
            "target": target_count,
            "retries": 0,
            "current_row": 1,
            "last_event": "Preparing generation run...",
        }

        def handle_log(message: str):
            text = str(message)
            collected_logs.append(text)
            progress_state["last_event"] = text
            generated_match = re.search(r"Generated row (\d+)/(\d+)", text)
            if generated_match:
                progress_state["done"] = int(generated_match.group(1))
                progress_state["target"] = int(generated_match.group(2))
                progress_state["current_row"] = min(progress_state["done"] + 1, progress_state["target"])
            if "Retrying" in text or "failed validation" in text or "Duplicate value" in text or "Regex failed" in text:
                progress_state["retries"] += 1

        def handle_progress(done: int, target: int):
            progress_state["done"] = done
            progress_state["target"] = target
            progress_state["current_row"] = min(done + 1, max(target, 1))

        controller.on_log = handle_log
        controller.on_progress = handle_progress
        controller.initialize(config, columns)
        register_runtime_controller(session, "data", controller)
        started_at = time.time()

        yield (
            session,
            gr.update(value=pd.DataFrame(), visible=False),
            _generation_progress_markdown(
                done=0,
                target=target_count,
                retries=0,
                current_row=1,
                last_event="Starting generation...",
                started_at=started_at,
                live_logs=collected_logs,
                is_running=True,
                metrics=_controller_metrics(controller),
            ),
            f"Starting generation for **{target_count}** row(s) using **{effective_quality}**.",
            _combined_activity_markdown(session, collected_logs),
        )

        worker = threading.Thread(target=controller._run_generation_loop, daemon=True)
        worker.start()

        last_snapshot: tuple[int, int, int, str] | None = None
        while worker.is_alive():
            session.generated_rows = [row.data for row in controller.generated_rows]
            snapshot = (
                progress_state["done"],
                progress_state["retries"],
                progress_state["current_row"],
                progress_state["last_event"],
                str(_controller_metrics(controller).get("total", {})),
            )
            if snapshot != last_snapshot:
                last_snapshot = snapshot
                yield (
                    session,
                    gr.update(value=pd.DataFrame(), visible=False),
                    _generation_progress_markdown(
                        done=progress_state["done"],
                        target=progress_state["target"],
                        retries=progress_state["retries"],
                        current_row=progress_state["current_row"],
                        last_event=progress_state["last_event"],
                        started_at=started_at,
                        live_logs=collected_logs,
                        is_running=True,
                        metrics=_controller_metrics(controller),
                    ),
                    f"Generating row **{min(max(progress_state['current_row'], 1), max(progress_state['target'], 1))}** of **{progress_state['target']}** using **{effective_quality}**.",
                    _combined_activity_markdown(session, collected_logs),
                )
            time.sleep(0.35)

        worker.join()

        generated = restore_original_imported_columns(session, [row.data for row in controller.generated_rows])
        session.generated_rows = generated
        clear_runtime_controller(session, "data", controller)
        for line in collected_logs[-10:]:
            append_activity(session, line)

        preview = pd.DataFrame(generated).head(50) if generated else pd.DataFrame()
        generated_count = len(generated)
        if generated_count == 0:
            if controller.stop_requested:
                status = "Generation stopped before any rows were completed."
            else:
                status = "Generation finished, but no rows were produced. Check the activity log for validation or provider errors."
        else:
            if controller.stop_requested:
                status = f"Stopped early with **{generated_count}** row(s) out of requested **{target_count}**. Partial export is ready."
                append_activity(session, f"Web generation stopped with {generated_count} row(s).")
            else:
                status = f"Generated **{generated_count}** row(s) out of requested **{target_count}**."
                append_activity(session, f"Web generation finished with {generated_count} row(s).")

        yield (
            session,
            gr.update(value=preview, visible=not preview.empty),
            _generation_progress_markdown(
                done=generated_count,
                target=target_count,
                retries=progress_state["retries"],
                current_row=target_count,
                last_event="Generation stopped by user." if controller.stop_requested else progress_state["last_event"],
                started_at=started_at,
                live_logs=collected_logs,
                is_running=False,
                metrics=_controller_metrics(controller),
            ),
            status,
            activity_markdown(session),
        )
    except Exception as exc:
        clear_runtime_controller(session, "data")
        append_activity(session, f"Generation failed: {exc}")
        yield (
            session,
            gr.update(value=pd.DataFrame(), visible=False),
            "### Generation Progress\n- Status: **Failed**\n- Check the activity log for the error details.",
            "Generation failed. Check the activity log and provider settings.",
            activity_markdown(session),
        )


def export_generated_data(session: WebSessionState, export_format: str):
    if not session.generated_rows:
        return session, None, "Generate data first.", activity_markdown(session)

    controller, _ = _controller_from_session_rows(session)
    controller.generated_rows = [RowData(data=row) for row in restore_original_imported_columns(session, session.generated_rows)]
    EXPORT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        export_specs = {
            "csv": {
                "path": EXPORT_DIR / f"generated_rows_{timestamp}.csv",
                "label": "CSV",
                "handler": lambda output_path: controller.export_csv(output_path),
            },
            "json": {
                "path": EXPORT_DIR / f"generated_rows_{timestamp}.json",
                "label": "JSON",
                "handler": lambda output_path: controller.export_json(output_path),
            },
            "sql": {
                "path": EXPORT_DIR / f"generated_rows_{timestamp}.sql",
                "label": "SQL",
                "handler": lambda output_path: controller.export_sql(output_path),
            },
            "pdf_narrative": {
                "path": EXPORT_DIR / f"generated_rows_narrative_{timestamp}.pdf",
                "label": "Narrative PDF",
                "handler": lambda output_path: controller.export_narrative_pdf(output_path),
            },
        }
        spec = export_specs.get(export_format)
        if spec is None:
            append_activity(session, f"Unsupported export format requested: {export_format}")
            return session, None, "Unsupported export format.", activity_markdown(session)

        path = spec["path"]
        spec["handler"](str(path))
        session.latest_downloads[f"data_{export_format}"] = str(path)
        append_activity(session, f"Prepared {spec['label']} download.")
        return session, str(path), f"Prepared **{spec['label']}** export.", activity_markdown(session)
    except Exception as exc:
        append_activity(session, f"Export failed: {exc}")
        return session, None, "Export failed.", activity_markdown(session)


def export_power_bi_data(
    session: WebSessionState,
    dataset_name: str,
    destination_dir: str,
    privacy_export_mode: str,
    current_model: str,
    provider: str,
):
    if not session.generated_rows:
        return session, None, "Generate data first.", activity_markdown(session)

    dataset_name = str(dataset_name or "").strip() or "Synthesizer Dataset"
    destination = Path(str(destination_dir or "").strip() or POWER_BI_EXPORT_DIR)
    privacy_mode = privacy_export_mode if privacy_export_mode in POWER_BI_PRIVACY_CHOICES else POWER_BI_PRIVACY_CHOICES[0]
    rows = _power_bi_rows_for_privacy_mode(session, privacy_mode)
    source_mode = "imported_enrichment" if session.raw_imported_data else "fresh_generation"

    try:
        controller, _ = _controller_from_session_rows(session)
        controller.generated_rows = [RowData(data=row) for row in rows]
        try:
            controller.config.model_id = current_model or controller.config.model_id
            controller.config.provider = AIProvider(provider) if provider else controller.config.provider
        except Exception:
            pass

        result = controller.export_power_bi_run(
            str(destination),
            dataset_name=dataset_name,
            privacy_export_mode=privacy_mode,
            source_mode=source_mode,
        )
        session.latest_downloads["power_bi_data"] = result.data_path
        session.latest_downloads["power_bi_schema"] = result.schema_path
        session.latest_downloads["power_bi_metadata"] = result.metadata_path
        session.latest_downloads["power_bi_index"] = result.index_path

        warning = ""
        if result.schema_changed:
            warning = " Schema differs from the previous run for this dataset; existing Power BI reports may need query/model updates."
        append_activity(session, f"Prepared Power BI run {result.run_id}.")
        return (
            session,
            result.data_path,
            f"Prepared **Power BI** export run `{result.run_id}` in `{result.run_dir}`.{warning}",
            activity_markdown(session),
        )
    except Exception as exc:
        append_activity(session, f"Power BI export failed: {exc}")
        return session, None, "Power BI export failed.", activity_markdown(session)


def review_generated_data_quality(session: WebSessionState):
    if not session.generated_rows:
        return session, "Generate data first, then review quality.", activity_markdown(session)

    controller, _ = _controller_from_session_rows(session)
    report = controller.analyze_quality()
    if not report:
        append_activity(session, "Quality review returned no metrics.")
        return session, "No quality metrics were produced.", activity_markdown(session)

    lines = ["### Data Quality Review"]
    recommendations: list[str] = []
    for column, data in report.items():
        diversity = float(data.get("diversity_score", 0) or 0)
        null_count = int(data.get("null_count", 0) or 0)
        lines.append(f"**{column}**")
        lines.append(f"- Diversity score: {diversity:.1%}")
        lines.append(f"- Null count: {null_count}")
        top = data.get("top_frequent", {})
        if top:
            top_text = ", ".join(f"{key} ({value})" for key, value in list(top.items())[:5])
            lines.append(f"- Frequent values: {top_text}")
        if null_count:
            recommendations.append(f"`{column}` has blank values. Tighten its field instruction or use Best Quality.")
        if diversity < 0.35 and len(session.generated_rows) > 3:
            recommendations.append(f"`{column}` has low variety. Add examples, allowed options, or allow duplicates only when repeated values are expected.")
    if recommendations:
        lines.append("\n### Recommended Fixes")
        lines.extend(f"- {item}" for item in recommendations[:5])
    else:
        lines.append("\nNo obvious quality issues found in the generated rows.")
    append_activity(session, "Quality review ready.")
    return session, "\n".join(lines), activity_markdown(session)
