# Synthetic Data Generator (Local LLM)

A modern desktop application for generating synthetic tabular data using local Large Language Models (via LM Studio) or cloud providers. Designed for privacy, modularity, and strict semantic uniqueness.

## 🚀 Key Features

* **Modern UI**: Built with **Flet** (Flutter for Python) featuring a responsive layout, dark mode, and intuitive controls.
* **Agentic Generation**: Uses **LangGraph** agents to generate semantic, context-aware data rows with intelligent retry logic.
* **Magic Schema Generator**: Describe your dataset in plain English, and the AI will auto-generate the schema. **Now Context-Aware**: If you import a file, the AI sees your existing columns and suggests relevant new columns without overwriting your data.
* **Strict Uniqueness**:
  * **Short Text**: Enforces exact match uniqueness (SHA256).
  * **Long Text**: Uses `sentence-transformers` to reject semantically similar outputs (e.g., "Good product" vs "Great product").
* **Flexible Constraints**:
  * **Regex**: Enforce patterns (e.g., email, phone).
  * **Numeric/Text**: Set min/max values and lengths.
  * **Cross-Column Logic**: Define rules like `End Date` > `Start Date`.
* **Data Enrichment**: Import existing CSV/JSON files and use AI to generate new columns based on existing data.
* **Local RAG (New)**: Optional retrieval-augmented generation for grounded outputs using `pypdfium2` + `fastembed` + `qdrant-client`.
* **Multiple Exports**: CSV, JSON, SQL inserts, PDF Reports, and "Narrative" PDF documents.

## 🗺️ Code Structure

```
synthetic_data_gen/
├── core/                 # Business logic and backend systems
│   ├── controller.py     # ORCHESTRATOR: Manages the generation lifecycle
│   ├── llm_client.py     # INTEGRATION: Wrapper for LLM APIs (LangChain)
│   ├── schema_agent.py   # AGENT: "Magic" schema generator
│   ├── row_agent.py      # AGENT: Intelligent row generator
│   ├── rag/              # RAG: parser/chunker/embedder/store/retriever
│   ├── validator.py      # LOGIC: Uniqueness and constraint validation
│   └── exporters.py      # LOGIC: PDF/file export handling
├── gui/                  # User Interface
│   ├── flet_app.py       # UI: Main Application & Async Loop
│   ├── utils.py          # UI: Helper functions (Dialogs, Snackbars)
│   └── controls/         # UI: Reusable Widgets
│       └── column_card.py # COMPONENT: Column definition card
├── scripts/              # Tools: Verification, debugging, and demo scripts
├── examples/             # Artifacts: Sample outputs and test data
├── tests/                # Unit Tests
├── main.py               # BOOTSTRAP: App entry point
└── requirements.txt      # DEPS: Python dependencies
```

## 🛠️ Development & UI Architecture

This project uses a modular Flet architecture with an asynchronous event loop to ensure UI responsiveness during heavy AI generation tasks.

### 1. Application Entry Point

* **`main.py`**: Initializes the Flet page and the `FletApp` instance.
* **`gui/flet_app.py`**: Contains the core `FletApp` class.
  * **`start_async_loop()`**: The heartbeat of the application. It runs indefinitely, consuming queues (`log_queue`, `progress_queue`) from the `GeneratorController` and updating the UI. This replaces standard threading to prevent UI freezes.

### 2. Component Structure

* **`gui/controls/`**: Reusable UI widgets.
  * **`column_card.py`**: Defines the `ColumnControl` class, managing individual column settings (Type, Regex, Constraints).
* **`gui/utils.py`**: Static helper methods for `Dialogs` (Snackbars, File Pickers) to keep the main logic clean.

### 3. State Management

* **`FletApp`**: Holds high-level UI state (`columns` list, `imported_data`).
* **`GeneratorController` (`core/controller.py`)**: Manages the business logic, background threads for generation, and data state. It communicates back to the UI via callbacks (`on_log`, `on_progress`) which feed into the `FletApp` queues.

### 4. Adding New Features

* **New Widgets**: Create a new file in `gui/controls/`, inherit from `ft.UserControl` or `ft.Card`, and import it in `flet_app.py`.
* **New Logic**: Add methods to `GeneratorController`, then expose them in the UI via `FletApp` methods.

## ⚡ Quick Start

### Prerequisites

1. **Python 3.10+** installed.
2. **LM Studio** (optional, recommended for free local AI): Running server at `http://localhost:1234/v1`.

### Installation

```bash
pip install -r requirements.txt
```

### Running the App

```bash
python main.py
```

## 🧠 Configuration & Usage

### 1. Connection

Go to the **AI Configuration** section.

* **Provider**: Choose LM Studio (Local), OpenAI, Gemini, or Azure.
* **API Key**: Required for cloud providers.
* **Test**: Click "Test Connection" to verify.

### 2. Defining Schema

You have two options:

* **Manual**: Click "+ Add Column". Choose type (Short Text, Numeric, etc.). Open "Advanced" for regex, min/max, and logic constraints.
* **Magic Generation**:
  1. **Fresh Start**: Type a description (e.g., *"Customer database"*) and click **"Auto-Generate Schema"**.
  2. **Context-Aware Mode**: Import a CSV first. The generator will read your headers and first row to understand your data context (e.g., inferring types from "2023-10-25" or "100.50") and suggest new, relevant columns to append. It strictly protects your existing data from being overwritten.

### 3. Generation

* **Start**: Click "Start Generation".
* **Stop**: The button changes to **"STOP"** (Red) during operation. Click it to cancel gracefully; existing data is preserved.
* **Progress**: Watch the live logs and progress bar.

### 4. Metrics & Cost Estimation (New!)

The application tracks token usage and provides real-time cost estimates:

* **Token Counting**: Tracks `prompt_tokens` (input) and `completion_tokens` (output).
* **Cost Calculation**: Input/Output costs are calculated separately based on your configured pricing.
  * *Default*: ~$0.15/1M Input, ~$0.60/1M Output (GPT-4o-mini rates).
  * *Configurable*: Update these rates in the **AI Configuration** settings.
* **Savings Estimation**:
  * Deterministic columns (Faker, Regex, Auto-Increment) do *not* use the LLM.
  * The app estimates how much you saved by calculating what those columns *would* have cost if generated by the LLM (based on the average cost of your actual LLM columns).
* **Real-time Display**: View Totals and Averages per row in the log window.

### 5. Local RAG + File Assistant (Optional)

RAG is integrated into a dedicated **Files** workspace and works local-first.

1. Open **Files** tab.
2. Click **Import File** (same toolbar button, context-aware by tab) and select one or more PDFs.
3. Use the Magic input as a **file task/chat prompt** (summarize, extract actions, draft reply, etc.).
4. Review answers with source citations (file + page) shown in chat.

Default first-run settings:

* Collection: `synthesizer_default`
* Qdrant URL: `:memory:` (no Qdrant server required)
* Embedding model: `BAAI/bge-small-en-v1.5`

In **Data Generation** mode, generation still works as before, and retrieved context is injected when available.

RAG metrics are displayed in the metrics panel:

* Queries
* Hit rate
* Average retrieval latency (ms)
* Average context size (chars)
* Last retrieval hit count

Files tab includes editable task presets that persist across restarts in `.rag_task_presets.json`.

### 6. Live RAG Verification Test (LM Studio + gpt-oss-20b)

The integration test `tests/test_rag_lmstudio_live.py` validates end-to-end RAG behavior against:

* `C:\Users\longs\Documents\GitHub\Synthesizer\examples\benefits_email_narative.pdf`
* LM Studio model: `gpt-oss-20b`

Run it with:

```bash
RUN_LIVE_LMSTUDIO_RAG=1 py -m pytest tests/test_rag_lmstudio_live.py -q -s
```

By default, it uses in-memory Qdrant (`:memory:`) so no local Qdrant server is required.

If you see `WinError 10061`, your Qdrant URL is likely set to `http://localhost:6333` without a running Qdrant instance. Switch it back to `:memory:`.

## 📅 Completed Development Phases

* **Phase 1-2**: Validator Engine & Flet UI Rewrite ✅
* **Phase 3**: Inter-Column Dependencies (`@[col]`) ✅
* **Phase 4**: Advanced Constraints (Regex, Logic, Min/Max) ✅
* **Phase 5**: Agentic Row Generation (LangGraph) ✅
* **Phase 6**: Magic Schema Generator ✅
* **Phase 7**: Data Import & Enrichment ✅
* **Phase 8**: Quality Reporting & PDF Export ✅
* **Phase 9**: UX Polish & Documentation ✅

---
*Built with ❤️ using Flet, LangChain, and LangGraph.*
