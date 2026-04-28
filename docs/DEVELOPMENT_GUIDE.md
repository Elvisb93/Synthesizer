# Development Guide

This guide captures the current development patterns for the web-only Synthesizer codebase.

## Table of Contents

- [Development Principles](#development-principles)
- [Code Organization Rules](#code-organization-rules)
- [Adding New Features](#adding-new-features)
- [Long-Running Work Patterns](#long-running-work-patterns)
- [RAG Development Notes](#rag-development-notes)
- [Testing Guidelines](#testing-guidelines)
- [UI Regression Checks](#ui-regression-checks)
- [Common Patterns](#common-patterns)
- [Code Style](#code-style)

## Development Principles

### 1. Core Must Stay UI-Independent

**Rule:** `core/` must never import from `web_ui/`.

**Why:** This keeps the business logic testable and reusable regardless of frontend.

```python
# ❌ WRONG - core depending on web UI state
from web_ui.state import WebSessionState

# ✅ CORRECT - use callbacks / plain values
def __init__(self, on_log=None):
    self.on_log = on_log

def log(self, message: str) -> None:
    if self.on_log:
        self.on_log(message)
```

### 2. Keep `web_ui/app.py` Declarative

**Rule:** Put layout and event wiring in `web_ui/app.py`, and move non-trivial behavior into `web_ui/actions/`.

**Why:** The app file should stay readable and testable.

### 3. Prefer Action Modules Over View Logic Creep

**Rule:** Add or extend focused action modules instead of embedding logic directly inside component callbacks.

**Why:** This keeps browser behavior easy to test with adapter/action tests.

### 4. Preserve Public APIs

**Rule:** When refactoring shared code, keep stable imports through `__init__.py` exports where practical.

**Why:** This reduces accidental breakage for scripts and tests.

### 5. RAG Must Remain Additive

**Rule:** Files workflows should degrade gracefully when retrieval is unavailable.

**Why:** Retrieval failures should not crash unrelated generation flows.

## Code Organization Rules

### File Placement

#### Core Business Logic → `core/`

- Domain models
- Controllers and orchestration
- Exporters
- RAG / document engine logic
- Validation and analytics

#### Web UI → `web_ui/`

- `app.py` for layout and wiring
- `actions/` for browser callbacks
- `adapters.py` for UI/data conversion
- `state.py` for per-session state and runtime-controller tracking

#### Tests → `tests/`

- Unit tests for `core/`
- Adapter/action regression tests for `web_ui/`
- Integration tests for exports and retrieval behavior

#### Utilities → `scripts/`

- Verification helpers
- Evaluation scripts
- Debug tooling

### Module Size Guidelines

- **Target:** roughly 200-400 lines when possible
- **Review point:** once a module becomes difficult to scan, split it by responsibility
- **Avoid:** tiny “one helper only” modules unless they clarify a strong boundary

### Naming Conventions

- **Modules:** `snake_case.py`
- **Classes:** `PascalCase`
- **Functions:** `snake_case()`
- **Constants:** `UPPER_SNAKE_CASE`
- **Private helpers:** `_leading_underscore`

## Adding New Features

### 1. Adding A New Column Type

1. Add the enum value in `core/models.py`.
2. Update prompt handling in `core/prompt_builder.py`.
3. Update grid/choice handling in `web_ui/adapters.py` and any related labels in `web_ui/app.py`.
4. Add validation if needed.
5. Add tests.

Example:

```python
# core/models.py
class ColumnType(Enum):
    PHONE_NUMBER = "Phone Number"

# core/prompt_builder.py
if column.type == ColumnType.PHONE_NUMBER:
    prompt += "Generate a valid phone number in format: (XXX) XXX-XXXX\n"
```

### 2. Adding A New Export Format

1. Create `core/exporters/<format>_exporter.py`.
2. Export it from `core/exporters/__init__.py`.
3. Add a wrapper method to `core/controller.py`.
4. Add a browser button/download target in `web_ui/app.py`.
5. Extend `export_generated_data()` or the relevant Files action.
6. Add tests.

Example:

```python
# core/controller.py
def export_excel(self, path: str) -> None:
    from core.exporters import export_excel
    export_excel(self.generated_rows, self.columns, path)
```

### 3. Adding A New AI Provider

1. Add the enum value in `core/models.py`.
2. Extend provider handling in `core/llm_client.py`.
3. Update provider choices in `web_ui/app.py`.
4. Add any provider-specific fields only if truly required.
5. Test model listing and connection.

### 4. Adding A New Web UI Action Area

1. Create or extend a focused file under `web_ui/actions/`.
2. Keep inputs/outputs plain and testable.
3. Wire it from `web_ui/app.py`.
4. Add regression coverage in the focused `tests/test_web_ui_*.py` module that matches the behavior you changed.

## Long-Running Work Patterns

### Pattern 1: Controller Callbacks For Progress

Use controller callbacks to collect logs and progress without pushing UI code into `core/`.

```python
logs: list[str] = []

def handle_log(message: str) -> None:
    logs.append(str(message))

controller.on_log = handle_log
```

### Pattern 2: Runtime Controller Registration

When an action launches long-running work, register the controller in `web_ui/state.py`.

Why this matters:

- stop buttons can find the active controller
- debug panels can inspect runtime state
- multiple browser sessions remain isolated by `runtime_id`

### Pattern 3: Background Work Behind Gradio Events

Gradio already handles request orchestration through `app.queue(...)`, but some controller work still runs in background threads for responsiveness and stop support.

Use this pattern when:

- generation loops run for many rows
- the user may press Stop
- you need incremental log/progress capture

Avoid this pattern when:

- a simple synchronous action can finish quickly
- the result is just a small config or formatting transformation

## RAG Development Notes

### Current Stack

- Parser routing: `RouterParser`
- PDF parsing: `HybridPdfParser`
- OCR: `RapidOcrEngine` (optional)
- Native chunking: `SemanticDoubleBufferChunker`
- Native embeddings: `FastEmbedEmbedder`
- Store: `QdrantVectorStore`
- Alternate backend: `LlamaIndex` ingestion + retrieval path

### UI Integration Points

- `web_ui/actions/files_actions.py`
  - source registration and ingest
  - reindex/remove actions
  - prompt presets and document bundles
  - Files task execution and exports
- `web_ui/actions/config_actions.py`
  - search status
  - clear search index
  - config save/load/reset
- `web_ui/app.py`
  - Files controls, retrieval settings, and debug/admin panels

### Files Workspace Modes

- `Document Engine`
- `Quick Q&A`
- `Structured JSON`

When changing Files behavior, update:

1. the relevant `core/` orchestration
2. the `GeneratorController` surface if needed
3. the Files actions
4. the UI wiring/tests/docs

### Common RAG Runtime Issue

- `WinError 10061` usually means the configured Qdrant server is not running.
- For local-first use, set `qdrant_url` to `:memory:`.

### OCR Notes

- Keep `ocr_mode=off` as the default for lighter machines.
- `auto` should stay selective rather than forcing full-page OCR.
- Missing OCR dependencies should degrade gracefully.

## Testing Guidelines

### Unit Tests

- Test `core/` logic without the UI.
- Mock external providers and file/network boundaries when possible.
- Use fixtures for repeated config/setup.

Example:

```python
from core.prompt_builder import get_execution_order
from core.models import ColumnDefinition, ColumnType

def test_topological_sort():
    columns = [
        ColumnDefinition(name="A", type=ColumnType.SHORT_TEXT, prompt_instruction=""),
        ColumnDefinition(name="B", type=ColumnType.SHORT_TEXT, prompt_instruction="Use {A}"),
    ]
    assert get_execution_order(columns) == ["A", "B"]
```

### Web UI Regression Tests

- Use the focused `tests/test_web_ui_*.py` modules as the fast regression layer for app wiring and action behavior.
- Add targeted assertions for new exports, reset flows, startup cleanup, Files actions, and download preparation.

### Integration Tests

- Use real temporary files for exporters when reasonable.
- Prefer narrow end-to-end tests over large brittle UI tests.

## UI Regression Checks

Run after UI-facing changes:

```bash
python -m pytest tests/test_web_ui_runtime_config.py tests/test_web_ui_schema_editor.py tests/test_web_ui_privacy_import_export.py tests/test_web_ui_generation_controls.py tests/test_web_ui_files_workflow.py tests/test_web_ui_startup_cleanup.py -q
```

Minimum manual smoke check:

```bash
python main.py
```

## Common Patterns

### Pattern: Callback-Based Communication

```python
class GeneratorController:
    def __init__(self, on_log=None, on_progress=None):
        self.on_log = on_log
        self.on_progress = on_progress
```

This lets the controller report progress without knowing anything about Gradio components.

### Pattern: Action Functions Return UI-Ready Values

Keep action functions focused on:

1. building config/state from inputs
2. calling `core/`
3. returning plain strings, lists, file paths, or component updates

### Pattern: Lazy Imports

Use lazy imports inside controller wrappers or expensive optional paths to avoid circular imports and unnecessary startup cost.

```python
def export_csv(self, path: str) -> None:
    from core.exporters import export_csv
    export_csv(self.generated_rows, self.columns, path)
```

### Pattern: Validate Early

Validate configuration and user input before launching long-running work.

```python
def validate_config(config: GeneratorConfig) -> list[str]:
    errors: list[str] = []
    if config.num_rows < 1:
        errors.append("Number of rows must be positive")
    return errors
```

## Code Style

### Imports

```python
# Standard library
from pathlib import Path
from typing import Any

# Third-party
import gradio as gr

# Local
from core.models import ColumnDefinition
from web_ui.state import WebSessionState
```

### Docstrings

Use docstrings on public helpers and non-obvious behavior, especially in `core/`.

### Type Hints

- Use type hints for public functions and controller surfaces.
- Keep internal helpers typed when it improves readability.

### Error Handling

- Catch specific exceptions where you can recover meaningfully.
- Prefer user-facing status messages in action functions.
- Keep raw tracebacks out of normal success paths.

## Checklist For New Features

Before wrapping up a feature:

- [ ] Core logic stays UI-independent
- [ ] New browser behavior lives in `web_ui/actions/` or `web_ui/app.py`
- [ ] Tests were added or updated
- [ ] Documentation was updated
- [ ] Manual smoke run completed for UI/code changes
- [ ] Public APIs remain intentional
- [ ] Error handling is in place

## Common Pitfalls

### ❌ Pitfall 1: Putting Web UI Logic In `core/`

```python
# WRONG
from web_ui.state import WebSessionState
```

**Fix:** pass plain values, callbacks, or config objects instead.

### ❌ Pitfall 2: Bloated `app.py`

If a callback starts doing real work inline, move it to `web_ui/actions/`.

### ❌ Pitfall 3: Forgetting Stop/Debug Plumbing

If a task can run for a while, register its controller so stop buttons and debug details keep working.

### ❌ Pitfall 4: Export Feature Without Download Wiring

Adding a new exporter in `core/` is not enough; the web UI also needs a returned file path for Gradio download components.

### ❌ Pitfall 5: Unbounded Growing Inputs

Avoid unconstrained expanding multiline inputs in schema/forms; keep them inside clearly bounded rows, groups, or accordions.

## Getting Help

- **Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)
- **Structure:** [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- **RAG:** [RAG_GUIDE.md](RAG_GUIDE.md)
