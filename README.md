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
* **Local RAG + OCR Fallback (New)**: Retrieval-augmented generation for grounded outputs using `pypdfium2` + `fastembed` + `qdrant-client`, with optional OCR modes (`off`/`auto`/`on`) for scanned PDFs.
* **Files Workspace (Document Engine + Q&A + Structured JSON)**:
  * `Doc Strategy`: `hybrid`, `factual by doc`, `creative` (with inline helper text).
  * `Pages`: choose fixed page count or `Let AI decide` for model-selected length.
  * `Quality`: `Fast` (quicker, fewer checks) or `Thorough` (stricter consistency checks).
  * One-click bundles: `Executive Brief`, `Policy Draft`, `Action Plan`, `Meeting Summary`.
  * `Structured JSON`: select a JSON template, set a target key, then run either standard item generation or exhaustive chunk-by-chunk extraction into the target array.
* **Multiple Exports**: CSV, JSON, SQL inserts, PDF reports, PDF documents, and DOCX documents.
* **Regression Safety**: Includes `scripts/verify/ui_regression_smoke.py` for automated UI smoke checks.

## 🗺️ Code Structure

```text
Synthesizer/
├── core/                     # Business logic and backend systems
│   ├── controller.py         # Generation + document orchestration
│   ├── llm_client.py         # LLM provider abstraction
│   ├── schema_agent.py       # Magic schema generator agent
│   ├── row_agent.py          # Row generation agent
│   ├── rag/                  # RAG parser/chunker/embedder/store/retriever
│   ├── document_engine/      # Long-form document generation pipeline
│   ├── exporters/            # CSV/JSON/SQL/PDF/DOCX exporters
│   └── models.py             # Core domain/config models
├── gui/                      # Flet UI layer
│   ├── flet_app.py           # Main app + layout
│   ├── handlers/             # Config, generation, data, RAG handlers
│   ├── controls/             # Reusable widgets (column card)
│   └── utils.py              # Dialogs and file-picker helpers
├── scripts/verify/           # Automated verification scripts
├── tests/                    # Unit/integration tests
├── docs/                     # Architecture and developer docs
└── main.py                   # App entrypoint
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
* **Test**: Click "Test Connection" to run a live provider/model handshake using the currently selected settings.

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
3. Choose **Files Mode**:
   - **Document Engine**: generate long-form docs from prompt + retrieved context.
   - **Quick Q&A**: run grounded Q&A with citations.
   - **Structured JSON**: populate a JSON template target array from the selected model, or exhaustively extract grounded instruction/response pairs from ingested files.
4. Use document controls (Document Engine mode):
   - **Doc Strategy**:
     - `hybrid`: grounded + synthesis
     - `factual by doc`: strictly grounded in imported files
     - `creative`: freer generation with minimal grounding
   - **Pages**: fixed page count or **Let AI decide**.
   - **Quality**:
     - `Fast`: fewer retries/checks, faster output
     - `Thorough`: stricter validation and consistency checks
5. Optionally apply one-click bundles: `Executive Brief`, `Policy Draft`, `Action Plan`, `Meeting Summary`.
6. Review output in File Assistant chat and export with **Export PDF** / **Export DOCX**.

Structured JSON mode adds:

* **JSON Template**: select a `.json` template file.
* **Target Key**: dot-path to the target array (for example `items` or `data.messages`).
* **Template Mode**:
  * `Standard Generation`: generate `Rows` number of items into the target array.
  * `Exhaustive Extraction`: process every ingested chunk and inject grounded pairs into the target array.
* **Export JSON**: save the populated template after generation.

OCR options are available in **AI Configuration -> RAG Settings**:

* `off` (default): no OCR (lowest CPU/RAM impact)
* `auto`: uses text extraction first, then OCR fallback for sparse pages and large-gap regions
* `on`: full-page OCR path (heavier)

Default first-run settings:

* Collection: `synthesizer_default`
* Qdrant URL: `:memory:` (no Qdrant server required)
* Embedding model: `BAAI/bge-small-en-v1.5`
* OCR mode: `off`

In **Data Generation** mode, generation still works as before, and retrieved context is injected when available.
In **Document Engine** mode, if retrieval is unavailable or empty, generation continues with non-RAG context instead of hard failing.

RAG metrics are displayed in the metrics panel:

* Queries
* Hit rate
* Average retrieval latency (ms)
* Average context size (chars)
* Last retrieval hit count

Files tab includes editable task presets that persist across restarts in `.rag_task_presets.json`.

### 6. UI Regression Smoke Check

Run this after UI changes:

```bash
py scripts/verify/ui_regression_smoke.py
```

The script validates Data tab basics, Files tab mode flow, document-generation start path, and a short boot probe.

### 7. Live RAG Verification Test (LM Studio)

The integration test `tests/test_rag_lmstudio_live.py` validates end-to-end RAG behavior against:

* `C:\Users\longs\Documents\GitHub\Synthesizer\examples\benefits_email_narative.pdf`
* LM Studio model: configurable. Recent live verification in this repo was run successfully with `qwen/qwen3.5-9b`.

Run it with:

```bash
RUN_LIVE_LMSTUDIO_RAG=1 py -m pytest tests/test_rag_lmstudio_live.py -q -s
```

By default, it uses in-memory Qdrant (`:memory:`) so no local Qdrant server is required.

If you see `WinError 10061`, your Qdrant URL is likely set to `http://localhost:6333` without a running Qdrant instance. Switch it back to `:memory:`.

If RAG initialization fails locally, make sure the environment has:

* `fastembed`
* `qdrant-client`
* optional OCR/doc parsing extras as needed

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
