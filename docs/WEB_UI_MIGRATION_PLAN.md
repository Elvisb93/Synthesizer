# Web UI Migration Plan

## Goal

Move the UI from Flet to a browser-first web app to improve day-to-day development ergonomics while keeping the existing `core/` business logic intact.

## Recommendation

Use **Gradio** for the first web migration.

Why it fits this repo well:

- The app is already Python-first and controller-driven.
- The current UX is mostly forms, file upload, tabbed workflows, chat-like output, tables, and downloads.
- Gradio already covers the main primitives we need:
  - `Blocks` for custom layout and event wiring
  - `Tab` for `Generate Sample Data` and `Work With Files`
  - `Chatbot` for the Files assistant output
  - `File` for uploads
  - `Dataframe` for preview/edit surfaces
  - `DownloadButton` for exports
  - `State` and `queue()` for session state and long-running tasks

## Why This Is Not A Full Rewrite

The repo is in a good position for this change:

- `main.py` is only a Flet bootstrap.
- `core/controller.py` already owns most orchestration.
- `core/` remains UI-independent.

The main rewrite cost is in `gui/`, especially:

- `gui/flet_app.py`
- `gui/controls/column_card.py`
- `gui/utils.py`
- `gui/handlers/*.py`

Those files currently mix:

- widget creation
- session state
- async/task orchestration
- file picker behavior
- user feedback formatting

That coupling is the main migration target.

## Proposed Strategy

Do a **parallel UI migration**, not an in-place rewrite.

Keep Flet running until the web UI reaches parity for the core workflows, then switch the default entrypoint.

## Target Architecture

```text
Synthesizer/
├── core/                    # Keep as-is; remains source of truth
├── gui/                     # Existing Flet UI (temporary during migration)
├── web_ui/                  # New Gradio UI layer
│   ├── app.py               # Gradio Blocks app
│   ├── state.py             # Session state helpers / view models
│   ├── adapters.py          # UI-to-controller conversion helpers
│   ├── actions/
│   │   ├── data_actions.py
│   │   ├── files_actions.py
│   │   └── config_actions.py
│   └── components/
│       ├── data_tab.py
│       ├── files_tab.py
│       └── shared.py
└── main.py / web_main.py    # Temporary dual entrypoints during rollout
```

## Migration Principles

1. Keep `core/` untouched unless a change clearly improves UI-independence.
2. Extract plain Python helpers before rebuilding UI interactions.
3. Preserve the two main workflows:
   - `Generate Sample Data`
   - `Work With Files`
4. Ship in slices so we can verify behavior after each phase.
5. Do not remove Flet until exports, file ingestion, and long-running tasks are all proven in the web UI.

## Phase Plan

### Phase 0: Create A UI-Neutral Surface

Goal: reduce Flet-specific assumptions before building Gradio.

Tasks:

- Identify logic in `gui/handlers/` that should become plain functions.
- Extract config-building helpers from Flet widget access into plain dict/model adapters.
- Extract file/result formatting helpers away from Flet controls.
- Introduce a lightweight session/view-state shape for:
  - active tab
  - files mode
  - imported data preview
  - rag file list
  - file chat history
  - last exportable outputs

Deliverable:

- New reusable helpers that both Flet and Gradio could call.

### Phase 1: Stand Up A Minimal Gradio Shell

Goal: boot a web app with the same top-level navigation.

Tasks:

- Add `web_ui/app.py`.
- Build a `gr.Blocks()` shell with:
  - title/header
  - setup/config area
  - `Generate Sample Data` tab
  - `Work With Files` tab
  - status/log panel
- Add `gr.State` session containers.
- Enable `queue()` for long-running operations.

Deliverable:

- App launches locally in browser with static controls and session state.

### Phase 2: Migrate Data Generation Workflow

Goal: reach parity for the sample-data workflow first.

Tasks:

- Rebuild config controls:
  - model
  - provider
  - pricing
  - row count
  - retry/similarity settings
- Replace `ColumnControl` with a web-friendly schema editor.
- Support:
  - manual field add/remove
  - import CSV/JSON
  - magic schema generation
  - generation start/stop
  - quality review
  - CSV/JSON/SQL/PDF exports

Recommended implementation detail:

- Use a `Dataframe` or JSON-backed editor for field definitions instead of recreating Flet card widgets one-for-one.

Deliverable:

- Full Data tab usable in Gradio without opening Flet.

### Phase 3: Migrate Files Workspace

Goal: bring over the highest-value workflow after the Data tab is stable.

Tasks:

- Rebuild file ingest flow using Gradio file upload.
- Rebuild imported file list and reindex/remove actions.
- Rebuild task modes:
  - `Document Engine`
  - `Quick Q&A`
  - `Structured JSON`
- Render output in a `Chatbot` plus structured preview blocks.
- Add document settings, quick presets, and JSON template controls.
- Expose PDF/DOCX/JSON downloads through web buttons.

Deliverable:

- Browser-based Files workspace with chat-style results and downloads.

### Phase 4: Logs, Progress, And Export Polish

Goal: replace the Flet queue/update loop cleanly.

Tasks:

- Map controller logs/progress to Gradio output components.
- Decide whether to:
  - poll controller state, or
  - append logs during action completion only
- Make export behavior browser-friendly:
  - return generated files for download
  - store recent outputs in session state
- Make long-running tasks clearly cancellable where possible.

Deliverable:

- Stable feedback loop for generation, ingest, and document tasks.

### Phase 5: Verification And Cutover

Goal: make the web UI the default with low migration risk.

Tasks:

- Add web smoke tests for critical flows.
- Update docs and screenshots.
- Add a temporary toggle:
  - `python main.py` -> web UI by default
  - optional legacy Flet launcher for fallback
- Remove Flet from default requirements only after parity is confirmed.

Deliverable:

- Web UI becomes the primary development path.

## Recommended Implementation Order

1. Extract shared helpers from `gui/handlers/`.
2. Build the Gradio shell.
3. Finish the Data tab end-to-end.
4. Finish Files workspace end-to-end.
5. Swap the default entrypoint.
6. Retire Flet later, not immediately.

## High-Risk Areas

### 1. Column Editing UX

The Flet `ColumnControl` card model does not map neatly to web primitives. Recreating it exactly in Gradio will slow the migration down.

Recommendation:

- simplify the schema editor for the web version
- use tabular editing or an accordion-based form
- preserve capability first, exact visual parity second

### 2. File Picker Assumptions

Current helpers in `gui/utils.py` rely on desktop-style pick/save dialogs. Browser flow is different.

Recommendation:

- uploads should be browser-native
- exports should return files directly for download
- avoid any assumption that the UI can open OS save dialogs

### 3. Long-Running Task Feedback

Flet currently relies on queue-driven control updates. Gradio will need a simpler response model.

Recommendation:

- start with coarse progress/status updates
- only add streaming/polling if users truly need it

### 4. Session State

Flet stores a lot of mutable state on `self`. In Gradio, session state must be made explicit.

Recommendation:

- define one session object early
- keep it serializable where practical

## What We Should Not Do

- Do not rewrite `GeneratorController` just to match a new UI framework.
- Do not migrate both tabs at once.
- Do not remove Flet before browser export and file workflows are proven.
- Do not chase pixel parity with the Flet layout.

## Definition Of Done

The migration is complete when:

- the app runs locally in a browser without Flet
- both main workflows are available
- file ingestion and exports work end-to-end
- generation/document tasks remain responsive during development
- core tests still pass
- at least one web smoke test covers each main workflow

## Final Call

**Gradio is the right first move** for this repo.

It gives us the fastest path from a desktop-style Python UI to a browser-based development workflow without forcing an immediate FastAPI + React rewrite.

If the product later needs a more custom multi-user web app, we can still evolve toward FastAPI plus a dedicated frontend. But for the next step, Gradio is the lowest-risk migration path.
