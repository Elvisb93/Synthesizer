# Project Structure

This document provides a detailed explanation of the Synthesizer project structure after the web-first UI cutover.

## Directory Overview

```
Synthesizer/
├── core/                   # Core business logic (UI-independent)
│   ├── exporters/          # Data export modules
│   │   ├── __init__.py     # Public API exports
│   │   ├── csv_exporter.py
│   │   ├── document_docx_exporter.py
│   │   ├── document_pdf_exporter.py
│   │   ├── json_exporter.py
│   │   ├── sql_exporter.py
│   │   └── pdf_exporter.py
│   ├── document_engine/    # Long-form document generation pipeline
│   │   ├── orchestrator.py
│   │   ├── models.py
│   │   └── validators.py
│   ├── rag/                # Retrieval-augmented generation modules
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
├── web_ui/                 # Gradio web UI layer
│   ├── actions/            # Config, data, and files callback logic
│   ├── __init__.py         # Web UI exports
│   ├── adapters.py         # UI/data adapters
│   ├── app.py              # Main application
│   ├── runtime_cleanup.py  # Startup cleanup + fresh-session collection handling
│   └── state.py            # Session state helpers
│
├── tests/                  # Test suite
│   ├── test_*.py           # Unit tests
│   └── ...
│
├── scripts/                # Utility scripts (verification/evaluation/debug)
│   ├── verify/             # Automated smoke/verification scripts
│   └── *.py                # Additional root utility scripts
│
├── docs/                   # Documentation
│   ├── README.md
│   ├── ARCHITECTURE.md
│   ├── DEVELOPMENT_GUIDE.md
│   └── PROJECT_STRUCTURE.md (this file)
│
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
- Coordinates Files workspace document generation and file Q&A
- Coordinates JSON template generation and exhaustive extraction flows
- Manages threading for async generation
- Delegates to specialized modules (prompt_builder, metrics, RAG, document_engine, exporters)
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
- `document_pdf_exporter.py` - long-form document PDF export
- `document_docx_exporter.py` - long-form document DOCX export

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

### Web UI Layer (`web_ui/`)

**Purpose:** Browser-first user interface built with Gradio.

#### `app.py` - Main Application

- Builds the `Blocks` layout
- Wires buttons/inputs to action functions
- Exposes browser downloads and admin panels

#### `actions/` - UI Action Modules

- `config_actions.py` - config save/load/reset, help, search admin, debug details
- `data_actions.py` - sample-data import, schema editing, generation, exports
- `files_actions.py` - files upload/indexing, presets, bundles, file tasks, source actions

#### `adapters.py` - UI/Data Conversions

- Grid/schema conversion helpers
- Browser-friendly field normalization

#### `state.py` - Session State

- Per-session web state
- Runtime controller registry for active tasks

#### `runtime_cleanup.py` - Startup Cleanup

- Clears transient local RAG caches/manifests
- Clears prior export/checkpoint artifacts
- Tracks non-memory collections used at runtime
- Creates a fresh collection name for each app launch

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

### `core/exporters/__init__.py`

Exports all export functions:

```python
from .csv_exporter import export_csv
from .json_exporter import export_json
from .sql_exporter import export_sql
from .pdf_exporter import PDFReportGenerator
from .document_pdf_exporter import DocumentPDFExporter
from .document_docx_exporter import DocumentDocxExporter
```

## Design Patterns

### 1. Action Module Pattern

UI callbacks are grouped by domain in `web_ui/actions/`, which keeps the app layout declarative and the behavior testable.

### 2. Dependency Injection (Controller)

**Problem:** Controller needed to delegate to specialized modules without tight coupling.

**Solution:** Import functions/classes as needed, maintain thin wrapper methods:

```python
def export_csv(self, path):
    from core.exporters import export_csv
    export_csv(self.generated_rows, self.columns, path)
```

## Code Organization Notes

- Handler mixins split UI concerns by domain (config, generation, data, files/RAG).
- `core/document_engine/` and `core/rag/` keep long-form and retrieval logic UI-independent.
- `core/exporters/` now contains both tabular and long-form document exporters.

## Import Paths

### Recommended Patterns

**Within `core/`:**

```python
from .models import ColumnDefinition
from .schemas import Schema
from .exporters import export_csv
```

**From `web_ui/` to `core/`:**

```python
from core.models import GeneratorConfig
from core.controller import GeneratorController
```

**External imports:**

```python
from core import GeneratorController, ColumnDefinition
from web_ui import launch_web_ui
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
- ❌ Embedding UI state or component logic inside `core/` → ✅ Keep UI orchestration in `web_ui/actions/`

## Future Expansion

### Adding New Export Formats

1. Create `core/exporters/new_format_exporter.py`
2. Add export function
3. Export in `core/exporters/__init__.py`
4. Add UI wiring in `web_ui/app.py`
5. Add callback handling in `web_ui/actions/data_actions.py`

### Adding New Column Types

1. Add enum value to `ColumnType` in `core/models.py`
2. Update prompt templates in `core/prompt_builder.py`
3. Update field type choices in `web_ui/adapters.py` / `web_ui/app.py`
4. Add validation logic if needed

### Adding New AI Providers

1. Add enum value to `AIProvider` in `core/models.py`
2. Add base URL to `PROVIDER_BASE_URLS` in `core/llm_client.py`
3. Update provider dropdown in `web_ui/app.py`
4. Add provider-specific config fields if needed
