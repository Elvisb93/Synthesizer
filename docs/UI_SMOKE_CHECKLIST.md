# UI Smoke Checklist

Use this checklist after every web UI-related change.

## Automated Smoke

Run:

```bash
python -m pytest tests/test_web_ui_runtime_config.py tests/test_web_ui_schema_editor.py tests/test_web_ui_privacy_import_export.py tests/test_web_ui_generation_controls.py tests/test_web_ui_files_workflow.py tests/test_web_ui_startup_cleanup.py -q
```

These suites check core browser flows, schema editing, privacy import/export, generation controls, files actions, reset behavior, startup cleanup, and search admin callbacks.

## Launch

1. Run `python main.py` from the project root.
2. Confirm the Gradio app opens without traceback/errors.

## Global Layout

1. Header and top-level controls render with no overlapping elements.
2. `Connection And Technical Settings` opens/closes cleanly.
3. `Debug Details` opens/closes cleanly.

## Data Generation Tab

1. `Generate Sample Data` tab is selectable and visible.
2. The editable schema grid renders with `row_id`, `name`, `type`, `prompt_instruction`, and `allow_duplicates`.
3. `+ Add Row` adds another editable row without layout break.
4. The selected row editor syncs correctly when a different row is selected.

## Files Tab

1. Switch to `Work With Files` successfully.
2. Files table renders the empty state or current indexed sources.
3. Files mode dropdown changes mode without broken layout.
4. Prompt preset controls appear for `Document Engine` and `Quick Q&A`, and hide for `Structured JSON`.
5. `Selected Source`, `Re-index Selected`, and `Remove Selected` controls stay usable.

## Basic Actions

1. `Reset Config` works and UI remains stable.
2. `Help & Docs` accordion opens correctly.
3. `Search Status` and `Clear Search Index` return sensible messages.
4. No red traceback/errors appear in terminal while performing checks.

## Pass Criteria

- All checks above pass.
- No runtime exceptions are emitted during interaction.
