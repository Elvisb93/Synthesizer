# System Architecture

This document provides a high-level overview of the Synthetic Data Generator architecture.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         GUI Layer                            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              FletApp (Main Application)                 │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │ │
│  │  │   Config     │  │ Generation   │  │     Data     │ │ │
│  │  │   Handlers   │  │   Handlers   │  │   Handlers   │ │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘ │ │
│  │                     ┌──────────────┐                   │ │
│  │                     │     RAG      │                   │ │
│  │                     │   Handlers   │                   │ │
│  │                     └──────────────┘                   │ │
│  └────────────────────────────────────────────────────────┘ │
│                            │                                 │
│                            │ Uses                            │
│                            ▼                                 │
└─────────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────────────────────────────────────┐
│                        Core Layer                            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │           GeneratorController (Orchestrator)            │ │
│  │                                                          │ │
│  │  Delegates to:                                          │ │
│  │  • PromptBuilder  - Dependency resolution & prompts    │ │
│  │  • Metrics        - Token usage & cost tracking        │ │
│  │  • Exporters      - Data export (CSV/JSON/SQL/PDF/DOCX)│ │
│  └────────────────────────────────────────────────────────┘ │
│                            │                                 │
│                            │ Uses                            │
│                            ▼                                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                     LLMClient                           │ │
│  │  • Provider abstraction (OpenAI, Azure, LM Studio...)  │ │
│  │  • Model listing & connection testing                  │ │
│  │  • Optional RAG retrieval + retrieval metrics          │ │
│  │  • Delegates schema gen to SchemaGeneratorAgent        │ │
│  └────────────────────────────────────────────────────────┘ │
│                            │                                 │
│                            │ Uses (optional)                 │
│                            ▼                                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                    RAG Subsystem                        │ │
│  │  • RagService orchestration                             │ │
│  │  • HybridPdfParser (text + OCR policy)                  │ │
│  │  • RapidOcrEngine (optional OCR backend)                │ │
│  │  • SemanticDoubleBufferChunker                          │ │
│  │  • FastEmbedEmbedder (local ONNX embeddings)            │ │
│  │  • QdrantVectorStore (server or :memory:)               │ │
│  └────────────────────────────────────────────────────────┘ │
│                            │                                 │
│                            │ Uses                            │
│                            ▼                                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              SchemaGeneratorAgent (LangGraph)           │ │
│  │  • Structured output parsing with Pydantic             │ │
│  │  • Retry logic for malformed responses                 │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                 UniquenessValidator                     │ │
│  │  • Hash-based exact duplicate detection                │ │
│  │  • Semantic similarity with sentence-transformers      │ │
│  │  • Embedding caching for performance                   │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                   QualityAnalyzer                       │ │
│  │  • Diversity scoring                                    │ │
│  │  • Null detection & frequency analysis                 │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                             │
                             │ Persists to
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                              │
│  • CSV files (export/import)                                │
│  • JSON files (export/import/config)                        │
│  • SQL files (export)                                       │
│  • PDF/DOCX files (export - reports, narratives, documents) │
└─────────────────────────────────────────────────────────────┘
```

## Layer Responsibilities

### GUI Layer (`gui/`)

**Purpose:** User interface and event handling

**Key Components:**

- `FletApp` - Main application window, inherits from handler mixins
- `ConfigHandlersMixin` - Config save/load/reset
- `GenerationHandlersMixin` - Magic generation, start/stop, model refresh
- `DataHandlersMixin` - Import/export/analyze
- `RagHandlersMixin` - File import/index, file chat/tasks, presets, status, clear index
- `ColumnControl` - Reusable column definition UI component

**Dependencies:**

- Depends on `core` layer (imports models, controller)
- No dependencies on specific UI framework details in core

**Communication:**

- Calls `GeneratorController` methods
- Receives updates via callbacks (log_queue, progress_queue)
- Uses async patterns with `page.run_task()` for Flet integration

### Core Layer (`core/`)

**Purpose:** Business logic, completely UI-independent

**Key Components:**

#### Orchestration

- `GeneratorController` - Coordinates generation workflow, manages threading
- `PromptBuilder` - Dependency resolution, topological sorting, prompt construction
- `Metrics` - Token usage tracking, cost estimation

#### AI Integration

- `LLMClient` - Provider abstraction (OpenAI, Azure, LM Studio, Gemini, etc.)
- `SchemaGeneratorAgent` - LangGraph-based schema generation with retry logic

#### RAG (Optional)

- `RagService` - Coordinates parse -> chunk -> embed -> upsert/search
- `HybridPdfParser` - Text extraction with OCR policy (`off`/`auto`/`on`)
- `RapidOcrEngine` - OCR adapter used in auto/on modes
- `SemanticDoubleBufferChunker` - Overlap + context-preserving chunking
- `FastEmbedEmbedder` - Local embedding model inference
- `QdrantVectorStore` - Vector storage/retrieval (local server or in-memory)

#### Validation & Analysis

- `UniquenessValidator` - Duplicate detection (hash + semantic similarity)
- `QualityAnalyzer` - Data quality metrics (diversity, nulls, frequency)

#### Data Export

- `exporters/` - Modular export system (CSV, JSON, SQL, PDF, DOCX)

**Dependencies:**

- LangChain for LLM integration
- sentence-transformers for semantic similarity
- pandas for data manipulation (optional)
- reportlab for PDF generation

**Communication:**

- Exposes public API via `__init__.py`
- Uses callbacks for async updates (log_callback, progress_callback)
- Thread-safe queue-based communication

### Data Layer

**Purpose:** Persistence

**Formats:**

- **CSV** - Primary data export/import format
- **JSON** - Configuration persistence, data export
- **SQL** - INSERT statements for database import
- **PDF** - Quality reports and narrative summaries
- **DOCX** - Long-form document export from Files workspace

## Data Flow

### 1. Configuration Flow

```
User Input (GUI)
  → ConfigHandlersMixin._save_config()
  → JSON file (config.json)
  
JSON file (config.json)
  → ConfigHandlersMixin._load_config()
  → Update GUI fields
```

### 2. Schema Generation Flow (Magic Generator)

```
User Prompt (GUI)
  → GenerationHandlersMixin._on_magic_generate()
  → LLMClient.generate_schema()
  → SchemaGeneratorAgent (LangGraph)
  → LLM API (OpenAI/Azure/LM Studio/etc.)
  → Structured Output (Pydantic Schema)
  → Parse to ColumnDefinitions
  → Update GUI (add column cards)
```

### 3. Data Generation Flow

```
User clicks "Start Generation"
  → GenerationHandlersMixin.toggle_generation()
  → GeneratorController.initialize(config, columns)
  → GeneratorController.start_generation_thread()
  
Background Thread:
  → PromptBuilder.get_execution_order() (topological sort)
  → For each row:
      → PromptBuilder.construct_prompt(row, column, context)
      → (Optional) LLMClient.retrieve_context() via RagService
      → Inject Retrieved Context block into prompt
      → LLMClient.generate_completion()
      → UniquenessValidator.is_unique() (check duplicates)
      → If unique: commit row
      → If duplicate: retry with modified prompt
      → Update progress_queue
  → Metrics.calculate_metrics()
  → Update log_queue
  
Main Thread (async loop):
  → Read from log_queue → Update log view
  → Read from progress_queue → Update progress bar
  → Fetch metrics → Update metrics display
```

### 4. Export Flow

```
User selects export format
  → DataHandlersMixin._handle_export(format_type)
  → page.run_task(export_data)
  → FilePicker (save_file)
  → GeneratorController.export_X(path)
  → Exporters module (csv/json/sql/pdf/docx)
  → Write to file
```

### 5. Import Flow

```
User clicks "Import Data"
  → DataHandlersMixin._on_import_data()
  → FilePicker (pick_file)
  → pandas.read_csv/read_json
  → Parse to ColumnDefinitions
  → Clear existing columns
  → Add imported columns (marked as "Imported")
  → Update row count
```

### 6. RAG Ingestion + Retrieval Flow

```
User switches to "Files" tab and clicks "Import File"
  → DataHandlersMixin routes to RagHandlersMixin._import_file_for_rag()
  → GeneratorController.ingest_documents(paths)
  → create_rag_backend(...)
      → default: LlamaIndexRagService.ingest_documents()
          → HybridPdfParser/RouterParser parse()
          → OCR fallback (optional; off/auto/on)
          → LlamaIndex IngestionPipeline
          → Hugging Face embeddings
          → Qdrant vector upsert
      → alternate: RagService.ingest_documents()
          → HybridPdfParser.parse()
          → SemanticDoubleBufferChunker.chunk()
          → FastEmbedEmbedder.embed_documents()
          → QdrantVectorStore.upsert_chunks()

During generation:
  → Row agent asks LLMClient.retrieve_context(query)
  → RagService.search()/format_hits()
  → Prompt includes "Retrieved Context" section
  → LLM response generated with grounding hints

During file tasks/chat:
  → User prompt (Magic input in Files tab)
  → Files mode branch:
      → Document Engine mode:
          → GeneratorController.generate_document(...)
          → DocumentOrchestrator.run(...) with backend-prepared RAG context when available
          → Export available as PDF/DOCX
      → Quick Q&A mode:
          → GeneratorController.ask_files(prompt)
          → Default: backend answer_query() / synthesized response when available
          → Optional override: "Pinpoint Quick" switches Quick Q&A to Native
          → Fallback: LLMClient.retrieve_context() + generate_completion()
          → Citations returned to chat
      → Structured JSON mode:
          → User selects JSON template + target key
          → Standard Generation:
              → GeneratorController.generate_json_batch(...)
              → json_agent LangGraph loop + validator + template injection
          → Exhaustive Extraction:
              → GeneratorController.generate_exhaustive_extraction(...)
              → RagService.get_all_chunks(...)
              → chunk_agent extraction + critique + template injection
```

## Threading Model

### Main Thread (Flet UI)

- Runs Flet event loop
- Handles all UI updates
- Processes log_queue and progress_queue
- **Never blocks** - all long-running operations delegated to background threads

### Generation Thread

- Started by `GeneratorController.start_generation_thread()`
- Runs generation loop
- Writes to thread-safe queues (log_queue, progress_queue)
- Can be stopped via `stop_event` flag

### Thread Communication

```
Generation Thread                Main Thread (UI)
      │                               │
      │  log_queue.put(msg)           │
      ├──────────────────────────────>│
      │                               │ async loop reads queue
      │                               │ updates log view
      │                               │
      │  progress_queue.put(data)     │
      ├──────────────────────────────>│
      │                               │ async loop reads queue
      │                               │ updates progress bar
      │                               │
      │  <stop_event.is_set()>        │
      │<──────────────────────────────┤
      │                               │ user clicks stop
      │  graceful shutdown            │
```

## Async Patterns (Flet)

### Problem

Flet button handlers are synchronous, but file operations and long-running tasks need to be async.

### Solution: Wrapper Pattern

```python
# Async implementation
async def export_data(self, format_type):
    path = await save_file(...)  # Async file picker
    self.controller.export_csv(path)

# Sync wrapper for button click
def _handle_export(self, e, format_type):
    self.page.run_task(lambda: self.export_data(format_type))

# Usage in UI setup
ft.PopupMenuItem(
    content=ft.Text("Export CSV"),
    on_click=lambda e: self._handle_export(e, "csv")
)
```

**Key Rule:** Always use `page.run_task()` to schedule async operations from sync handlers.

## Dependency Graph

```
main.py
  └─> gui.flet_app.main()
        └─> FletApp(page, controller)
              ├─> GeneratorController
              │     ├─> LLMClient
              │     │     └─> SchemaGeneratorAgent
              │     ├─> RagService (optional)
              │     │     ├─> HybridPdfParser
              │     │     ├─> RapidOcrEngine (optional)
              │     │     ├─> SemanticDoubleBufferChunker
              │     │     ├─> FastEmbedEmbedder
              │     │     └─> QdrantVectorStore
              │     ├─> UniquenessValidator
              │     ├─> QualityAnalyzer
              │     ├─> PromptBuilder
              │     ├─> Metrics
              │     └─> Exporters
              └─> Handler Mixins
                    ├─> ConfigHandlersMixin
                    ├─> GenerationHandlersMixin
                    ├─> DataHandlersMixin
                    └─> RagHandlersMixin
```

## Design Principles

### 1. Separation of Concerns

- **Core** = Business logic (no UI dependencies)
- **GUI** = Presentation layer (depends on core)
- **Tests** = Verify both layers independently

### 2. Dependency Inversion

- Core defines interfaces (callbacks, abstract methods)
- GUI implements concrete handlers
- Controller doesn't know about Flet

### 3. Single Responsibility

- Each module has one clear purpose
- Mixins group related handlers
- Exporters are separate modules

### 4. Open/Closed Principle

- Easy to add new column types (extend enum)
- Easy to add new export formats (new exporter module)
- Easy to add new AI providers (extend enum + config)
- Easy to add new RAG store/embedding/parser adapters behind interfaces

### 5. Async-First (GUI)

- All UI operations use Flet's async patterns
- No blocking operations in main thread
- Background threads for long-running tasks

## Security Considerations

### API Keys

- Stored in config.json (user's local machine)
- Never logged or transmitted except to AI provider
- User responsible for key security

### File Operations

- All file paths validated before use
- File pickers prevent directory traversal
- Export operations check write permissions

### LLM Integration

- No user data sent to AI except prompts
- Generated data is synthetic (no PII)
- Configurable similarity thresholds prevent leakage

## Performance Optimizations

### 1. Embedding Caching (UniquenessValidator)

- Embeddings cached to avoid re-computation
- Reduces semantic similarity check time by ~90%

### 2. Topological Sorting (PromptBuilder)

- Columns generated in dependency order
- Enables context-aware generation
- Prevents circular dependencies

### 3. Async UI Updates

- Log and progress updates batched
- UI updates at 10Hz (100ms intervals)
- Prevents UI freezing during generation

### 4. Thread-Safe Queues

- Lock-free communication between threads
- No blocking on UI thread
- Graceful shutdown on stop

### 5. Local-First RAG

- Embeddings run locally (no embedding API round-trip)
- Qdrant supports both server mode and `:memory:` mode (default)
- Retrieval context is capped by `max_context_chars` to control token growth
- OCR defaults to `off` to minimize CPU/RAM overhead
- `auto` OCR performs targeted region scans for sparse/gapped layouts

## Extensibility Points

### Adding New Column Types

1. Add to `ColumnType` enum in `core/models.py`
2. Update prompt templates in `core/prompt_builder.py`
3. Add UI dropdown option in `gui/controls/column_card.py`

### Adding New Export Formats

1. Create `core/exporters/new_format_exporter.py`
2. Export function in `core/exporters/__init__.py`
3. Add menu item in `gui/flet_app.py`
4. Add handler in `gui/handlers/data_handlers.py`

### Adding New AI Providers

1. Add to `AIProvider` enum in `core/models.py`
2. Add base URL to `PROVIDER_BASE_URLS` in `core/llm_client.py`
3. Update provider dropdown in `gui/flet_app.py`
4. Add provider-specific config if needed

### Extending RAG

1. Add parser/chunker/embedder/store implementation under `core/rag/`
2. Keep compatibility with `core/rag/interfaces.py`
3. Wire into `RagService` composition
4. Add tests under `tests/test_rag_*.py`
5. If UI behavior changes, update `RagHandlersMixin` and docs

### Adding New Validation Rules

1. Extend `UniquenessValidator` or create new validator
2. Call from `GeneratorController` generation loop
3. Add UI controls for threshold configuration
