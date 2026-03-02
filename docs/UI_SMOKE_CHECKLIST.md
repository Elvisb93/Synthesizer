# UI Smoke Checklist

Use this checklist after every UI-related change.

## Automated Smoke

Run:

```bash
py scripts/verify/ui_regression_smoke.py
```

This script checks core tab flows, document-generation start path, and a short app boot probe.

## Launch

1. Run `py main.py` from the project root.
2. Confirm the app opens without traceback/errors.

## Global Layout

1. Header and action buttons render with no overlapping controls.
2. `Show advanced settings` toggle opens/closes the advanced panel cleanly.
3. `Show logs and diagnostics` toggle opens/closes diagnostics cleanly.

## Data Generation Tab

1. `Data Generation` tab is selectable and visible.
2. In `Step 4: Define Columns`, the first column card renders correctly:
   - `Column Name`, `Data Type`, and prompt field are all visible in one row.
   - No oversized gray/blank stretched input area appears.
3. `+ Add Column` adds another card without layout break.
4. `Show Advanced Options` on a column toggles fields without visual corruption.

## Files Tab

1. Switch to `Files` tab successfully.
2. Files list area renders (empty-state message or indexed files).
3. Files mode dropdown changes mode without broken layout.
4. Magic action button text updates correctly for selected file mode.

## Basic Actions

1. `Reset` works and UI remains stable.
2. `Help` dialog opens and closes cleanly.
3. No red traceback/errors in terminal while performing checks.

## Pass Criteria

- All checks above pass.
- No runtime exceptions are emitted during interaction.
