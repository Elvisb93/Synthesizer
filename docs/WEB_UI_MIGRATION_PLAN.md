# Web UI Migration Record

This document is now an archive of the completed Flet-to-web migration rather than a forward plan.

## Status

The migration is complete as of `April 23, 2026`.

Current state:

- `main.py` launches the Gradio web app.
- The legacy `gui/` Flet package has been removed.
- `flet` is no longer a runtime dependency.
- The browser UI is the only supported application surface.

## Final Outcome

The app now runs through a browser-first architecture built on `web_ui/`:

```text
Synthesizer/
├── core/                    # Source of truth for business logic
├── web_ui/                  # Gradio UI layer
│   ├── app.py
│   ├── state.py
│   ├── adapters.py
│   └── actions/
│       ├── config_actions.py
│       ├── data_actions.py
│       └── files_actions.py
├── tests/
└── main.py                  # Web entrypoint
```

## Delivered Parity

The web UI now covers the key workflows that originally drove the migration:

- sample-data schema editing and generation
- CSV/JSON/SQL exports
- narrative PDF export for generated sample data
- quality review for generated rows
- Files workspace upload + URL ingest
- `Document Engine`, `Quick Q&A`, and `Structured JSON` modes
- document PDF/DOCX and structured JSON downloads
- per-source reindex and remove actions
- saved file-task prompt presets
- one-click document bundles
- config save/load/reset
- help/docs panel
- search status and search-index clearing
- debug details panel for active runtime controllers

## Architecture Decisions That Survived The Cutover

### 1. `core/` Stayed UI-Independent

This was the most important migration constraint, and it held. The web app calls into `GeneratorController` and related `core/` modules without pushing Gradio concerns down into business logic.

### 2. Explicit Session State Replaced UI-Object State

The web app uses `gr.State` plus `WebSessionState` and a runtime-controller registry. That replaced the older desktop-style pattern of storing mutable UI state directly on the app object.

### 3. Browser-Native Download Flow Replaced File Pickers

Exports now return generated files back to Gradio components for download instead of relying on OS save dialogs.

### 4. The Files Workspace Became A First-Class Web Flow

Rather than trying to recreate desktop widgets exactly, the web version leans on Gradio primitives: uploads, chat-style output, dataframes, accordions, and explicit download targets.

## Cutover Changes

The final cutover included:

- switching `main.py` to `web_ui.launch_web_ui()`
- removing `web_main.py`
- removing the `gui/` package
- removing Flet-only tests and smoke scripts
- updating docs to make the browser UI the default path

## What Remains Future Work

These are no longer migration blockers; they are normal product enhancements:

- richer live metrics visualization
- broader end-to-end smoke coverage beyond the focused `tests/test_web_ui_*.py` regression modules
- any future deployment/auth/multi-user concerns if the app moves beyond local usage

## Lessons Learned

- Keeping `core/` clean made the UI cutover practical.
- Rebuilding interactions around browser-native primitives was faster than chasing desktop parity.
- Action modules (`web_ui/actions/`) are a better fit than embedding behavior directly inside the layout file.

## Recommendation Going Forward

Treat this migration as finished.

Future UI work should:

1. build on `web_ui/`
2. extend `core/` only when behavior truly belongs there
3. update browser regression tests alongside user-facing changes
