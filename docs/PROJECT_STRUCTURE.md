# Project Structure

This document provides a detailed explanation of the Synthesizer project structure after the 2026-02 refactoring.

## Directory Overview

```
Synthesizer/
├── core/                   # Core business logic (UI-independent)
│   ├── exporters/          # Data export modules
│   │   ├── __init__.py     # Public API exports
│   │   ├── csv_exporter.py
│   │   ├── json_exporter.py
│   │   ├── sql_exporter.py
│   │   └── pdf_exporter.py
│   ├── __init__.py         # Core package exports
│   ├── models.py           # Domain models (ColumnDefinition, GeneratorConfig, etc.)
│   ├── schemas.py          # LangChain output parsing schemas
│   ├── controller.py       # Generation orchestration
│   ├── llm_client.py       # LLM provider abstraction
│   ├── prompt_builder.py   # Prompt construction & dependency resolution
│   ├── metrics.py          # Token usage & cost calculation
│   ├── validator.py        # Uniqueness validation
│   ├── analytics.py        # Data quality analysis
│   └── schema_agent.py     # LangGraph schema generation agent
│
├── gui/                    # Flet UI layer
│   ├── handlers/           # Event handler mixins
│   │   ├── __init__.py
│   │   ├── config_handlers.py      # Save/load/reset config
│   │   ├── generation_handlers.py  # Magic gen, start/stop, model refresh
│   │   └── data_handlers.py        # Import/export/analyze
│   ├── controls/           # Reusable UI components
│   │   └── column_card.py
│   ├── __init__.py         # GUI package exports
│   ├── flet_app.py         # Main application (uses handler mixins)
│   └── utils.py            # UI utilities (dialogs, file pickers)
│
├── tests/                  # Test suite
│   ├── test_*.py           # Unit tests
│   └── ...
│
├── scripts/                # Utility scripts
│   ├── debug/              # Debug scripts
│   ├── verify/             # Verification scripts
│   └── demo/               # Demo scripts
│
├── docs/                   # Documentation
│   ├── README.md
│   ├── ARCHITECTURE.md
│   ├── DEVELOPMENT_GUIDE.md
│   └── PROJECT_STRUCTURE.md (this file)
│
├── assets/                 # Static assets
├── main.py                 # Application entry point
└── README.md               # Project README
```

## Module Responsibilities

### Core Layer (`core/`)

**Purpose:** Contains all business logic, completely independent of the UI framework.

#### `models.py` - Domain Models

- `ColumnType` - Enum of supported column types
- `AIProvider` - Enum of supported AI providers
- `ColumnConstraints` - Validation constraints for columns
- `ColumnDefinition` - Column schema definition
- `GeneratorConfig` - Generation configuration
- `RowData` - Generated row data structure
- `FAKER_PROVIDERS` - Available Faker providers

#### `schemas.py` - LangChain Schemas

- `ColumnConstraintsSchema` - Pydantic schema for constraints
- `ColumnSchema` - Pydantic schema for columns
- `Schema` - Pydantic schema for full dataset

**Why separate?** Domain models are stable, while LangChain schemas may change with AI provider updates.

#### `controller.py` - Generation Orchestration

- Coordinates generation workflow
- Manages threading for async generation
- Delegates to specialized modules (prompt_builder, metrics, exporters)
- Provides thin wrapper methods for backward compatibility

#### `prompt_builder.py` - Prompt Construction

- `get_dependencies()` - Extract column dependencies
- `get_execution_order()` - Topological sort for generation order
- `construct_prompt()` - Build LLM prompts with context

#### `metrics.py` - Metrics Calculation

- `calculate_metrics()` - Token usage, cost estimation, performance stats

#### `exporters/` - Data Export

- `csv_exporter.py` - CSV export
- `json_exporter.py` - JSON export
- `sql_exporter.py` - SQL INSERT statements
- `pdf_exporter.py` - PDF quality reports & narratives

**Why a sub-package?** Exporters are a cohesive unit that may grow (Excel, Parquet, etc.)

#### `llm_client.py` - LLM Abstraction

- Supports multiple providers (LM Studio, OpenAI, Azure, Gemini, etc.)
- Handles model listing, connection testing
- Delegates schema generation to `schema_agent.py`

#### `validator.py` - Uniqueness Validation

- Hash-based exact duplicate detection
- Semantic similarity checking with sentence-transformers
- Embedding caching for performance

#### `analytics.py` - Quality Analysis

- Diversity scoring
- Null detection
- Frequency analysis

### GUI Layer (`gui/`)

**Purpose:** Flet-based user interface, organized using the mixin pattern.

#### `flet_app.py` - Main Application

- **Inherits from:** `ConfigHandlersMixin`, `GenerationHandlersMixin`, `DataHandlersMixin`
- **Responsibilities:**
  - UI layout construction (`_setup_ui`)
  - Async event loop (`start_async_loop`)
  - Column management (`_add_column`, `_remove_column`)
  - Controller callback initialization

**Why mixins?** Breaks down a 891-line God class into focused, testable units.

#### `handlers/` - Event Handler Mixins

**`config_handlers.py`** - Configuration I/O

- `_save_config()` - Save config to JSON
- `_load_config()` - Load config from JSON
- `_reset_config()` - Reset to defaults
- `_on_provider_change()` - Update UI based on provider

**`generation_handlers.py`** - Generation Operations

- `_refresh_models()` - Fetch available models from provider
- `_test_connection()` - Test LLM connection
- `_on_magic_generate()` - AI-powered schema generation
- `toggle_generation()` - Start/stop data generation

**`data_handlers.py`** - Data Operations

- `_on_import_data()` - Import CSV/JSON
- `export_data()` - Export generated data
- `_handle_export()` - Sync wrapper for async export
- `_on_analyze()` - Quality analysis dialog

**Critical Pattern:** All async operations use `page.run_task()` to schedule tasks in Flet's event loop.

#### `controls/` - Reusable Components

- `column_card.py` - Column definition UI card

#### `utils.py` - UI Utilities

- `Dialogs` - Snackbar and dialog helpers
- `pick_file()` - Async file picker
- `save_file()` - Async save dialog

## Package Exports (`__init__.py`)

### `core/__init__.py`

Exports the public API for core functionality:

```python
from .models import (
    ColumnType, AIProvider, ColumnConstraints,
    ColumnDefinition, GeneratorConfig, RowData, FAKER_PROVIDERS
)
from .controller import GeneratorController
from .llm_client import LLMClient
from .validator import UniquenessValidator
from .analytics import QualityAnalyzer
```

**Why?** Maintains backward compatibility. Existing code can still do:

```python
from core.models import ColumnDefinition  # Still works
from core import ColumnDefinition         # Also works
```

### `gui/__init__.py`

Exports the main entry point:

```python
from .flet_app import main
```

### `core/exporters/__init__.py`

Exports all export functions:

```python
from .csv_exporter import export_csv
from .json_exporter import export_json
from .sql_exporter import export_sql
from .pdf_exporter import PDFReportGenerator
```

## Design Patterns

### 1. Mixin Pattern (GUI Handlers)

**Problem:** `flet_app.py` was 891 lines with mixed responsibilities.

**Solution:** Split handlers into focused mixins by domain:

- Config I/O → `ConfigHandlersMixin`
- Generation → `GenerationHandlersMixin`
- Data operations → `DataHandlersMixin`

**Benefits:**

- Each mixin is independently testable
- Clear separation of concerns
- Easy to add new handler categories

### 2. Async Wrapper Pattern

**Problem:** Flet button handlers are sync, but file pickers and export operations are async.

**Solution:** Create sync wrapper methods that use `page.run_task()`:

```python
async def export_data(self, format_type):
    # Async implementation
    path = await save_file(...)
    # ...

def _handle_export(self, e, format_type):
    """Sync wrapper for button click."""
    self.page.run_task(lambda: self.export_data(format_type))
```

### 3. Dependency Injection (Controller)

**Problem:** Controller needed to delegate to specialized modules without tight coupling.

**Solution:** Import functions/classes as needed, maintain thin wrapper methods:

```python
def export_csv(self, path):
    from core.exporters import export_csv
    export_csv(self.generated_rows, self.columns, path)
```

## File Size Metrics

### Before Refactoring

- `core/controller.py`: 551 lines
- `gui/flet_app.py`: 891 lines
- `gui/main_window.py`: 724 lines (dead code)

### After Refactoring

- `core/controller.py`: ~250 lines (-55%)
- `core/prompt_builder.py`: ~80 lines (new)
- `core/metrics.py`: ~60 lines (new)
- `core/exporters/*.py`: ~200 lines total (new)
- `gui/flet_app.py`: ~450 lines (-50%)
- `gui/handlers/*.py`: ~600 lines total (new)
- Dead code removed: 724 lines

**Net Result:** Better organization, improved testability, no loss of functionality.

## Import Paths

### Recommended Patterns

**Within `core/`:**

```python
from .models import ColumnDefinition
from .schemas import Schema
from .exporters import export_csv
```

**From `gui/` to `core/`:**

```python
from core.models import GeneratorConfig
from core.controller import GeneratorController
```

**External imports:**

```python
from core import GeneratorController, ColumnDefinition
from gui import main
```

## Testing Structure

Tests mirror the source structure:

```
tests/
├── test_controller.py      # Controller tests
├── test_validator.py        # Validator tests
├── test_analytics.py        # Analytics tests
├── test_prompt_builder.py   # Prompt building tests
└── ...
```

## Migration Notes

### Breaking Changes

**None.** All public APIs preserved via `__init__.py` exports.

### Deprecated Patterns

- ❌ `from core.models import Schema` → ✅ `from core.schemas import Schema`
- ❌ Calling async methods directly from sync handlers → ✅ Use `page.run_task()`

## Future Expansion

### Adding New Export Formats

1. Create `core/exporters/new_format_exporter.py`
2. Add export function
3. Export in `core/exporters/__init__.py`
4. Add menu item in `gui/flet_app.py`
5. Add handler in `gui/handlers/data_handlers.py`

### Adding New Column Types

1. Add enum value to `ColumnType` in `core/models.py`
2. Update prompt templates in `core/prompt_builder.py`
3. Update UI dropdown in `gui/controls/column_card.py`
4. Add validation logic if needed

### Adding New AI Providers

1. Add enum value to `AIProvider` in `core/models.py`
2. Add base URL to `PROVIDER_BASE_URLS` in `core/llm_client.py`
3. Update provider dropdown in `gui/flet_app.py`
4. Add provider-specific config fields if needed
