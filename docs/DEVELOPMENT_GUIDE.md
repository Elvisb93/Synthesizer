# Development Guide

This guide provides rules, patterns, and best practices for developing features in the Synthetic Data Generator project.

## Table of Contents

- [Development Principles](#development-principles)
- [Code Organization Rules](#code-organization-rules)
- [Adding New Features](#adding-new-features)
- [Async Patterns (Critical)](#async-patterns-critical)
- [Testing Guidelines](#testing-guidelines)
- [RAG Development Notes](#rag-development-notes)
- [UI Regression Checks](#ui-regression-checks)
- [Common Patterns](#common-patterns)
- [Code Style](#code-style)

## Development Principles

### 1. Core Must Be UI-Independent

**Rule:** The `core/` package must NEVER import from `gui/`.

**Why:** Enables testing business logic without UI, supports future UI frameworks.

**Example:**

```python
# ❌ WRONG - core importing from gui
from gui.utils import Dialogs

# ✅ CORRECT - use callbacks
def __init__(self, on_log=None):
    self.on_log = on_log
    
def log(self, message):
    if self.on_log:
        self.on_log(message)
```

### 2. Async-First in GUI

**Rule:** All UI operations that may block MUST be async and use `page.run_task()`.

**Why:** Prevents UI freezing, maintains responsiveness.

**Example:**

```python
# ❌ WRONG - calling async from sync handler
def on_export_click(self, e):
    path = await save_file(...)  # SyntaxError!

# ✅ CORRECT - wrap in async task
async def export_data(self, format_type):
    path = await save_file(...)
    # ... export logic ...

def _handle_export(self, e, format_type):
    self.page.run_task(lambda: self.export_data(format_type))
```

### 3. Mixin Pattern for Handlers

**Rule:** Group related event handlers into focused mixin classes.

**Why:** Keeps `flet_app.py` manageable, improves testability.

**Example:**

```python
# gui/handlers/new_feature_handlers.py
class NewFeatureHandlersMixin:
    def _on_feature_action(self, e):
        # Implementation
        pass

# gui/flet_app.py
class FletApp(
    ConfigHandlersMixin,
    GenerationHandlersMixin,
    DataHandlersMixin,
    NewFeatureHandlersMixin  # Add new mixin
):
    pass
```

### 4. Backward Compatibility

**Rule:** Maintain public APIs via `__init__.py` exports.

**Why:** Prevents breaking existing code during refactoring.

**Example:**

```python
# core/__init__.py
from .models import ColumnDefinition
from .new_module import NewClass

__all__ = ["ColumnDefinition", "NewClass"]
```

### 5. Local-First RAG Is Optional

**Rule:** RAG must be additive and fail-safe. Generation should continue even when RAG is disabled or unavailable.

**Why:** Prevents retrieval outages from blocking core generation workflows.

**Implementation Notes:**

- Keep RAG under `core/rag/` with interface-based adapters.
- Never couple core logic to one vector provider implementation.
- Keep retrieval bounded (`max_context_chars`) to avoid uncontrolled token growth.

## Code Organization Rules

### File Placement

#### Core Business Logic → `core/`

- Domain models
- Business logic
- AI integration
- Data validation
- Export/import logic

#### UI Code → `gui/`

- Flet components
- Event handlers
- UI utilities
- Layout code

#### Tests → `tests/`

- Unit tests
- Integration tests
- Test fixtures

#### Utilities → `scripts/`

- Debug scripts → `scripts/debug/`
- Verification scripts → `scripts/verify/`
- Demo scripts → `scripts/demo/`

### Module Size Guidelines

- **Target:** 200-300 lines per module
- **Maximum:** 500 lines before considering split
- **Minimum:** 50 lines (avoid over-fragmentation)

### Naming Conventions

- **Modules:** `snake_case.py`
- **Classes:** `PascalCase`
- **Functions:** `snake_case()`
- **Constants:** `UPPER_SNAKE_CASE`
- **Private:** `_leading_underscore`

## Adding New Features

### 1. Adding a New Column Type

**Steps:**

1. Add enum value to `ColumnType` in `core/models.py`
2. Update prompt templates in `core/prompt_builder.py`
3. Add UI dropdown option in `gui/controls/column_card.py`
4. Add validation logic if needed
5. Write tests

**Example:**

```python
# 1. core/models.py
class ColumnType(Enum):
    # ... existing types ...
    PHONE_NUMBER = "Phone Number"

# 2. core/prompt_builder.py
def construct_prompt(row, column, context):
    if column.type == ColumnType.PHONE_NUMBER:
        prompt += "Generate a valid phone number in format: (XXX) XXX-XXXX\n"
    # ...

# 3. gui/controls/column_card.py
self.type_dropdown = ft.Dropdown(
    options=[
        # ... existing options ...
        ft.dropdown.Option("Phone Number"),
    ]
)

# 4. tests/test_phone_number.py
def test_phone_number_generation():
    # Test implementation
    pass
```

### 2. Adding a New Export Format

**Steps:**

1. Create `core/exporters/new_format_exporter.py`
2. Implement export function
3. Export in `core/exporters/__init__.py`
4. Add controller method in `core/controller.py`
5. Add menu item in `gui/flet_app.py`
6. Add handler in `gui/handlers/data_handlers.py`
7. Write tests

**Example:**

```python
# 1. core/exporters/excel_exporter.py
def export_excel(rows, columns, path):
    """Export data to Excel format."""
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    # ... implementation ...
    wb.save(path)

# 2. core/exporters/__init__.py
from .excel_exporter import export_excel
__all__ = [..., "export_excel"]

# 3. core/controller.py
def export_excel(self, path):
    from core.exporters import export_excel
    export_excel(self.generated_rows, self.columns, path)

# 4. gui/handlers/data_handlers.py
async def export_data(self, format_type):
    # ... existing code ...
    elif format_type == "excel":
        self.controller.export_excel(path)

# 5. gui/flet_app.py
self.export_btn = ft.PopupMenuButton(
    items=[
        # ... existing items ...
        ft.PopupMenuItem(
            content=ft.Text("Export Excel"),
            on_click=lambda e: self._handle_export(e, "excel")
        ),
    ]
)
```

### 3. Adding a New AI Provider

**Steps:**

1. Add enum value to `AIProvider` in `core/models.py`
2. Add base URL to `PROVIDER_BASE_URLS` in `core/llm_client.py`
3. Update provider dropdown in `gui/flet_app.py`
4. Add provider-specific config fields if needed
5. Test connection and model listing

**Example:**

```python
# 1. core/models.py
class AIProvider(Enum):
    # ... existing providers ...
    ANTHROPIC = "Anthropic Claude"

# 2. core/llm_client.py
PROVIDER_BASE_URLS = {
    # ... existing URLs ...
    AIProvider.ANTHROPIC: "https://api.anthropic.com/v1",
}

# 3. gui/flet_app.py
self.provider_dropdown = ft.Dropdown(
    options=[
        # ... existing options ...
        ft.dropdown.Option("Anthropic Claude", AIProvider.ANTHROPIC.value),
    ]
)
```

### 4. Adding a New UI Handler Category

**Steps:**

1. Create `gui/handlers/new_category_handlers.py`
2. Define mixin class with handler methods
3. Export in `gui/handlers/__init__.py`
4. Add mixin to `FletApp` inheritance
5. Call handlers from UI components

**Example:**

```python
# 1. gui/handlers/template_handlers.py
class TemplateHandlersMixin:
    """Mixin for template management handlers."""
    
    async def _save_template(self, e):
        # Implementation
        pass
    
    async def _load_template(self, e):
        # Implementation
        pass

# 2. gui/handlers/__init__.py
from .template_handlers import TemplateHandlersMixin
__all__ = [..., "TemplateHandlersMixin"]

# 3. gui/flet_app.py
class FletApp(
    ConfigHandlersMixin,
    GenerationHandlersMixin,
    DataHandlersMixin,
    TemplateHandlersMixin  # Add new mixin
):
    pass
```

## Async Patterns (Critical)

### Pattern 1: Async File Operations

**Problem:** File pickers are async, button handlers are sync.

**Solution:** Create async method + sync wrapper.

```python
# Async implementation
async def save_config(self):
    path = await save_file(
        title="Save Configuration",
        default_name="config.json",
        filter_pairs=("JSON files", "*.json")
    )
    if path:
        # Save logic
        pass

# Sync wrapper for button
def _on_save_config(self, e):
    self.page.run_task(self.save_config)

# Usage in UI
ft.ElevatedButton(
    "Save Config",
    on_click=self._on_save_config
)
```

### Pattern 2: Long-Running Tasks

**Problem:** Generation takes time, UI must remain responsive.

**Solution:** Background thread + queue-based communication.

```python
# In controller
def start_generation_thread(self):
    self.stop_event.clear()
    thread = threading.Thread(target=self._generation_loop)
    thread.daemon = True
    thread.start()

def _generation_loop(self):
    while not self.stop_event.is_set():
        # Generate data
        self.log_queue.put(f"Generated row {i}")
        self.progress_queue.put({"current": i, "total": total})

# In GUI async loop
async def start_async_loop(self):
    while True:
        # Process log queue
        while not self.controller.log_queue.empty():
            msg = self.controller.log_queue.get()
            self.log_view.controls.append(ft.Text(msg))
        
        # Process progress queue
        while not self.controller.progress_queue.empty():
            data = self.controller.progress_queue.get()
            self.progress_bar.value = data["current"] / data["total"]
        
        self.page.update()
        await asyncio.sleep(0.1)
```

### Pattern 3: Async Task Scheduling

**Problem:** Need to run async task from sync context.

**Solution:** Use `page.run_task()`.

```python
# ❌ WRONG
def on_click(self, e):
    asyncio.create_task(self.async_method())  # RuntimeError!

# ✅ CORRECT
def on_click(self, e):
    self.page.run_task(self.async_method)
```

## RAG Development Notes

### Current Stack

- Parser: `HybridPdfParser` (`pypdfium2` text + OCR policy)
- OCR: `RapidOcrEngine` (`rapidocr-onnxruntime`, optional)
- Chunking: `SemanticDoubleBufferChunker`
- Embeddings: `fastembed` (`FastEmbedEmbedder`)
- Store: `QdrantVectorStore` (`:memory:` by default, or `http://localhost:6333`)
- Orchestrator: `RagService`

### Config Surface

`RagConfig` in `core/models.py` includes:

- `collection_name`
- `top_k`
- `min_score`
- `max_context_chars`
- `embedding_model`
- `source_filter`
- `qdrant_url`
- `qdrant_api_key`
- `ocr_mode` (`off`/`auto`/`on`)
- `ocr_dpi`
- `ocr_max_pages`
- `ocr_max_regions_per_page`
- `ocr_region_padding_px`
- `ocr_gap_multiplier`
- `ocr_min_extracted_chars`
- `ocr_timeout_ms_per_page`

### UI Integration Points

- `gui/handlers/rag_handlers.py`
  - `_import_file_for_rag`
  - `_on_files_magic_task`
  - `_on_file_preset_change`
  - `_on_save_file_preset`
  - `_on_delete_file_preset`
  - `_on_rag_status`
  - `_on_rag_clear`
- `gui/handlers/data_handlers.py`
  - `_on_import_data` routes by active workspace tab
- `gui/handlers/config_handlers.py`
  - Save/load/reset of `rag` block

### Workspace Behavior

- `Data Generation` tab: import CSV/JSON + schema/generation flow
- `Files` tab: import PDFs + `Document Engine` / `Quick Q&A` task modes
- Shared toolbar import button is context-aware by tab

### Files UX Controls (Document Engine)

- Strategy labels in UI:
  - `hybrid` -> grounded + synthesis
  - `factual by doc` -> strictly grounded in files (`strict_grounded` internally)
  - `creative` -> freer generation (`pure` internally)
- Length control:
  - UI uses `Pages` dropdown and maps to words internally (`~500 words/page`)
  - `Let AI decide` sends auto-target (`target_words <= 0`) and controller resolves size
- Quality mode:
  - `Fast` -> fewer checks/retries, faster output
  - `Thorough` -> stricter consistency/retry behavior
- One-click doc bundles:
  - `Executive Brief`, `Policy Draft`, `Action Plan`, `Meeting Summary`

### Metrics Integration Points

- `LLMClient.get_rag_stats()` tracks retrieval behavior.
- `core/metrics.py` merges RAG metrics into `stats.rag`.
- `gui/flet_app.py` renders RAG telemetry in the metrics panel.

### Tests

- Unit tests:
  - `tests/test_rag_chunking.py`
  - `tests/test_rag_config.py`
  - `tests/test_rag_retriever.py`
  - `tests/test_rag_generation_integration.py`
  - `tests/test_metrics_rag.py`
  - `tests/test_rag_ocr.py`
- Live integration test:
  - `tests/test_rag_lmstudio_live.py`

Run live test:

```bash
RUN_LIVE_LMSTUDIO_RAG=1 py -m pytest tests/test_rag_lmstudio_live.py -q -s
```

The live test targets `examples/benefits_email_narative.pdf`, queries an email-related question, and asserts retrieval/hit metrics so RAG usage is verified.

### Common RAG Runtime Issue

- If logs show `WinError 10061` on retrieval, Qdrant URL points to a server that is not running.
- For local-first usage, set `qdrant_url` to `:memory:`.

### OCR Runtime Notes

- Keep `ocr_mode=off` as default for low-resource machines.
- `auto` mode should only OCR sparse pages or large-gap regions; avoid full-page OCR unless needed.
- If OCR backend is missing, ingestion still proceeds with text extraction.

## Testing Guidelines

### Unit Tests

- Test core logic independently of UI
- Mock external dependencies (LLM APIs, file I/O)
- Use pytest fixtures for common setup

```python
# tests/test_prompt_builder.py
import pytest
from core.prompt_builder import get_execution_order
from core.models import ColumnDefinition, ColumnType

def test_topological_sort():
    columns = [
        ColumnDefinition(name="A", type=ColumnType.SHORT_TEXT, prompt_instruction=""),
        ColumnDefinition(name="B", type=ColumnType.SHORT_TEXT, prompt_instruction="Use {A}"),
    ]
    order = get_execution_order(columns)
    assert order == ["A", "B"]
```

### Integration Tests

- Test component interactions
- Use real (but small) data
- Test error handling

```python
# tests/test_export_integration.py
def test_csv_export_roundtrip(tmp_path):
    # Generate data
    controller = GeneratorController()
    # ... setup ...
    
    # Export
    csv_path = tmp_path / "test.csv"
    controller.export_csv(str(csv_path))
    
    # Verify
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) == expected_rows
```

### Test Organization

```
tests/
├── test_controller.py       # Controller tests
├── test_prompt_builder.py   # Prompt building tests
├── test_validator.py         # Validation tests
├── test_exporters.py         # Export tests
└── fixtures/                 # Test data
    └── sample_config.json
```

## UI Regression Checks

Run after UI-facing changes:

```bash
py scripts/verify/ui_regression_smoke.py
```

Minimum manual run:

```bash
py main.py
```

## Common Patterns

### Pattern: Callback-Based Communication

**Use Case:** Core needs to notify GUI without depending on it.

```python
# Core
class GeneratorController:
    def __init__(self, on_log=None, on_progress=None):
        self.on_log = on_log
        self.on_progress = on_progress
    
    def log(self, message):
        if self.on_log:
            self.on_log(message)

# GUI
controller = GeneratorController(
    on_log=lambda msg: self.log_queue.put(msg),
    on_progress=lambda data: self.progress_queue.put(data)
)
```

### Pattern: Lazy Imports

**Use Case:** Avoid circular imports, reduce startup time.

```python
# ❌ WRONG - import at module level
from core.llm_client import LLMClient

def some_method(self):
    client = LLMClient(...)

# ✅ CORRECT - import when needed
def some_method(self):
    from core.llm_client import LLMClient
    client = LLMClient(...)
```

### Pattern: Configuration Validation

**Use Case:** Validate config before use.

```python
def validate_config(config: GeneratorConfig) -> list[str]:
    """Return list of validation errors."""
    errors = []
    
    if config.num_rows < 1:
        errors.append("Number of rows must be positive")
    
    if config.provider == AIProvider.AZURE_OPENAI:
        if not config.azure_endpoint:
            errors.append("Azure endpoint required")
    
    return errors
```

## Code Style

### Imports

```python
# Standard library
import asyncio
import json
from typing import List, Optional

# Third-party
import flet as ft
from langchain_openai import ChatOpenAI

# Local
from core.models import ColumnDefinition
from gui.utils import Dialogs
```

### Docstrings

```python
def construct_prompt(row: dict, column: ColumnDefinition, context: str) -> str:
    """
    Construct LLM prompt for generating a column value.
    
    Args:
        row: Partially generated row data
        column: Column definition with type and constraints
        context: Additional context from existing columns
    
    Returns:
        Formatted prompt string for LLM
    """
    pass
```

### Type Hints

```python
# Use type hints for public APIs
def export_csv(rows: List[dict], columns: List[ColumnDefinition], path: str) -> None:
    pass

# Optional for internal helpers
def _format_value(val):
    pass
```

### Error Handling

```python
# Be specific with exceptions
try:
    df = pd.read_csv(path)
except FileNotFoundError:
    self.log(f"File not found: {path}")
except pd.errors.ParserError:
    self.log(f"Invalid CSV format: {path}")
except Exception as e:
    self.log(f"Unexpected error: {e}")
```

## Checklist for New Features

Before submitting a new feature:

- [ ] Core logic is UI-independent
- [ ] Async operations use `page.run_task()`
- [ ] Public API exported in `__init__.py`
- [ ] Tests written and passing
- [ ] Documentation updated
- [ ] No breaking changes to existing APIs
- [ ] Error handling implemented
- [ ] Type hints added for public APIs
- [ ] Code follows project style
- [ ] Tested manually in UI

## Common Pitfalls

### ❌ Pitfall 1: Calling Async from Sync

```python
# WRONG
def on_click(self, e):
    result = await self.async_method()  # SyntaxError
```

**Fix:** Use `page.run_task()` or make handler async.

### ❌ Pitfall 2: Blocking UI Thread

```python
# WRONG
def on_generate(self, e):
    for i in range(1000):
        self.generate_row()  # UI freezes!
```

**Fix:** Use background thread or async task.

### ❌ Pitfall 3: Core Importing GUI

```python
# WRONG in core/controller.py
from gui.utils import Dialogs
Dialogs.show_snackbar(...)
```

**Fix:** Use callbacks.

### ❌ Pitfall 4: Forgetting to Update `__init__.py`

```python
# Created core/new_module.py
# Forgot to export in core/__init__.py
# Result: ImportError for users
```

**Fix:** Always update `__init__.py` when adding public APIs.

### ❌ Pitfall 5: Unbounded `TextField(expand=True)` in Column/Form Layouts

```python
# WRONG (can create oversized stretched gray inputs in unconstrained containers)
ft.TextField(expand=True, multiline=True)
```

**Fix:** Keep fields in constrained rows/containers and only use `expand=True` where layout bounds are explicit.

## Getting Help

- **Architecture Questions:** See [ARCHITECTURE.md](ARCHITECTURE.md)
- **Structure Questions:** See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- **Bug Reports:** Create GitHub issue
- **Feature Requests:** Create GitHub issue with "enhancement" label
