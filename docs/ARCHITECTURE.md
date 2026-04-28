# System Architecture

This document describes the current browser-first architecture of Synthesizer after the Flet retirement.

## Architecture Diagram

```text
┌────────────────────────────────────────────────────────────────────┐
│                         Web UI Layer (`web_ui/`)                  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │               Gradio Blocks App (`web_ui/app.py`)           │  │
│  │  • Generate Sample Data tab                                 │  │
│  │  • Files workspace tab                                      │  │
│  │  • Connection / RAG settings                                │  │
│  │  • Help, admin, and debug panels                            │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                 │                       │                          │
│                 ▼                       ▼                          │
│      `actions/data_actions.py`   `actions/files_actions.py`       │
│      `actions/config_actions.py` `adapters.py` / `state.py`       │
└────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│                         Core Layer (`core/`)                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                 GeneratorController                          │  │
│  │  • sample-data generation                                   │  │
│  │  • Files workspace orchestration                            │  │
│  │  • export entrypoints                                       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│      │             │                │                │            │
│      ▼             ▼                ▼                ▼            │
│  PromptBuilder   LLMClient      RAG stack      Exporters          │
│  Metrics         Schema agent   Document engine Validators        │
└────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│                    External Services / File Outputs               │
│  • OpenAI / Azure / LM Studio / Gemini-compatible endpoints      │
│  • Qdrant (`:memory:` or server)                                 │
│  • CSV / JSON / SQL / PDF / DOCX outputs                         │
└────────────────────────────────────────────────────────────────────┘
```

## Layer Responsibilities

### Web UI Layer (`web_ui/`)

**Purpose:** Present the browser experience and translate user interactions into controller calls.

**Key modules:**

- `app.py`
  - Builds the Gradio `Blocks` layout.
  - Declares tabs, forms, buttons, downloads, admin controls, and debug panels.
  - Wires component events to action functions.
- `actions/config_actions.py`
  - Connection testing, model refresh, config save/load/reset.
  - Search status and search-index clearing.
  - Debug details formatting.
- `actions/data_actions.py`
  - Sample-data schema editing, import, generation, stop requests, quality review, and exports.
- `actions/files_actions.py`
  - File upload and URL ingest, source management, prompt presets, document bundles, file tasks, and Files-mode exports.
- `adapters.py`
  - Converts between UI grid rows and domain-friendly field structures.
- `state.py`
  - Holds per-session state and a runtime-controller registry for active tasks.
- `runtime_cleanup.py`
  - Clears transient runtime artifacts before launch.
  - Assigns a fresh session collection name at startup.

**Important boundary:** `web_ui/` may import from `core/`, but `core/` must never import from `web_ui/`.

### Core Layer (`core/`)

**Purpose:** Hold all business logic independently of Gradio or any other UI framework.

**Key components:**

- `controller.py`
  - Main orchestration layer for both app workflows.
  - Owns runtime config, generation lifecycle, exports, and Files workspace operations.
- `prompt_builder.py`
  - Dependency resolution and prompt construction for sample-data generation.
- `llm_client.py`
  - Provider abstraction, model listing, connection testing, schema generation, and retrieval integration.
- `schema_agent.py`
  - Structured schema generation with retry logic.
- `metrics.py`
  - Token, timing, and RAG telemetry aggregation.
- `validator.py`
  - Exact and semantic uniqueness checks.
- `analytics.py`
  - Quality review for generated rows.
- `rag/`
  - Retrieval stack, backends, parsers, chunking, embeddings, and stores.
- `document_engine/`
  - Long-form document planning, synthesis, and validation.
- `exporters/`
  - CSV, JSON, SQL, narrative/quality PDF, and document PDF/DOCX export implementations.

### Persistence And Outputs

Synthesizer is primarily local-file oriented:

- Configs persist as JSON.
- Sample data imports use CSV/JSON.
- Sample data exports support CSV, JSON, SQL, and narrative PDF.
- Files workspace exports support PDF, DOCX, and populated JSON templates.
- RAG can use either in-memory or external Qdrant storage.
- `.web_ui_exports/`, `.document_checkpoints/`, and local `.rag_*` cache/manifests are treated as transient workspace artifacts rather than long-term source files.

## Runtime Flow

### 1. App Startup

Before the Gradio app is created, startup cleanup prepares a fresh local workspace:

- transient `.rag_*` cache/manifests are cleared
- previous export/checkpoint artifacts are cleared
- a fresh per-launch collection name is assigned for the session

```text
main.py
  → web_ui.launch_web_ui()
  → web_ui.create_app()
  → Gradio Blocks app
  → app.queue(default_concurrency_limit=2)
```

### 2. Session State Flow

```text
Browser session
  → gr.State(WebSessionState)
  → stores fields / imported data / file history / downloads
  → runtime controllers tracked by session runtime_id
```

The runtime-controller registry lets the app:

- stop active data generation
- stop active Files tasks
- inspect debug state for long-running work

### 3. Sample-Data Flow

```text
User edits schema or asks for field suggestions
  → web_ui.actions.data_actions
  → GeneratorController / LLMClient / schema agent
  → session.fields updated

User starts generation
  → GeneratorController.initialize(config, columns)
  → controller callbacks append logs/progress
  → background worker thread runs generation loop
  → session.generated_rows updated
  → Dataframe preview + export buttons enabled
```

### 4. Sample-Data Export Flow

```text
User clicks export button
  → data_actions.export_generated_data(...)
  → temporary output path created
  → GeneratorController.export_<format>(path)
  → file path returned to Gradio download component
```

### 5. Files Workspace Flow

```text
User uploads files or adds a URL
  → files_actions register sources in session
  → files_actions builds runtime controller
  → controller.ingest_documents(...)

User runs Files task
  → mode branch:
      • Document Engine
      • Quick Q&A
      • Structured JSON
  → controller delegates to RAG / document_engine / JSON generation
  → chat output, status text, and downloadable artifacts returned to UI
```

### 6. Admin / Debug Flow

```text
User opens admin panels
  → config_actions.get_search_status(...)
  → config_actions.clear_search_index(...)
  → config_actions.refresh_debug_details(...)
```

These actions are intentionally controller-backed so the web UI can inspect real runtime state without moving business logic into the view layer.

## Concurrency Model

The app uses two complementary mechanisms:

### Gradio Queue

- `app.queue(default_concurrency_limit=2)` serializes and manages browser-triggered work.
- Keeps long-running requests from blocking the whole interface.

### Controller-Backed Background Work

- Sample-data generation still runs in a worker thread started from `data_actions.py`.
- Files tasks use runtime controllers plus callback logging/progress collection.
- Stop buttons signal the active controller rather than killing threads directly.

## Dependency Graph

```text
main.py
  └─> web_ui.launch_web_ui()
        └─> web_ui.create_app()
              ├─> web_ui.actions.config_actions
              ├─> web_ui.actions.data_actions
              ├─> web_ui.actions.files_actions
              ├─> web_ui.adapters
              └─> web_ui.state
                    └─> runtime controller registry

web_ui.actions.*
  └─> core.controller.GeneratorController
        ├─> core.llm_client.LLMClient
        │     └─> core.schema_agent.SchemaGeneratorAgent
        ├─> core.rag.*
        ├─> core.document_engine.*
        ├─> core.analytics.QualityAnalyzer
        ├─> core.validator.UniquenessValidator
        ├─> core.prompt_builder
        ├─> core.metrics
        └─> core.exporters.*
```

## Design Principles

### 1. UI Independence In Core

- `core/` owns behavior.
- `web_ui/` owns presentation and session orchestration.
- The controller never depends on Gradio component types.

### 2. Thin UI Actions

- Action modules should build config/state from inputs, call the controller, and format outputs.
- Heavy logic belongs in `core/` unless it is purely view-specific.

### 3. Explicit Session State

- Mutable browser state lives in `WebSessionState`.
- Runtime controller references are keyed by `runtime_id` so simultaneous sessions stay isolated.

### 4. Local-First File Workflows

- Browser upload/download replaces desktop file-picker assumptions.
- Qdrant defaults to `:memory:` so the Files workspace works locally without extra setup.

### 5. Incremental Extensibility

- New export formats belong in `core/exporters/` plus minimal UI wiring.
- New Files features should extend `files_actions.py` and `app.py` without changing core abstractions unless needed.

## Security Notes

- API keys are user-supplied and only sent to the selected model provider.
- Generated content is intended to be synthetic; uniqueness and validation settings help reduce accidental duplication.
- File ingest and export remain local-machine operations unless the selected model or vector store points to a remote service.

## Performance Notes

- Uniqueness validation caches embeddings when semantic checks are enabled.
- Retrieval context is capped to control prompt growth.
- OCR defaults to `off` to avoid unnecessary CPU cost.
- Large Files tasks favor coarse progress reporting over highly granular UI streaming.

## Extensibility Points

### Adding A New Export Format

1. Create `core/exporters/<format>_exporter.py`.
2. Export it in `core/exporters/__init__.py`.
3. Add a controller wrapper in `core/controller.py`.
4. Add a button and download target in `web_ui/app.py`.
5. Handle the new format in `web_ui/actions/data_actions.py` or `web_ui/actions/files_actions.py`.

### Adding A New Column Type

1. Extend `ColumnType` in `core/models.py`.
2. Update prompt logic in `core/prompt_builder.py`.
3. Update UI choices in `web_ui/adapters.py` and, if needed, labels in `web_ui/app.py`.
4. Add tests.

### Extending Files / RAG Behavior

1. Add or update logic under `core/rag/` or `core/document_engine/`.
2. Expose the capability through `GeneratorController`.
3. Add browser wiring in `web_ui/actions/files_actions.py`.
4. Update the relevant docs and regression tests.
