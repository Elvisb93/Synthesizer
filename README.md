# Synthesizer

Synthesizer is a browser-first Gradio application for:

- generating synthetic tabular data
- enriching imported CSV/JSON rows with new AI-generated fields
- working with files through local-first RAG, grounded Q&A, document generation, and structured JSON extraction

It supports LM Studio for local models and also works with cloud providers through the shared `core/` runtime.

## Current App Model

- Entry point: [main.py](/C:/Users/longs/Documents/GitHub/Synthesizer/main.py:1)
- Web UI: [web_ui/app.py](/C:/Users/longs/Documents/GitHub/Synthesizer/web_ui/app.py:1)
- Business logic: [core/controller.py](/C:/Users/longs/Documents/GitHub/Synthesizer/core/controller.py:1)

The legacy Flet UI has been removed. The active app is the Gradio web UI only.

## Key Capabilities

- Schema drafting from plain-English prompts via `Generate Fields`
- CSV/JSON import with privacy masking before model use
- Decode-on-export so imported source fields can be restored after masked generation
- Synthetic row generation with uniqueness checks and validation
- Versioned Power BI-ready tabular exports for local, OneDrive, or SharePoint-synced folders
- Files workspace with:
  - `Document Engine`
  - `Quick Q&A`
  - `Structured JSON`
- Local-first RAG with `LlamaIndex` and `Native` backends
- Exports for CSV, JSON, SQL, PDF, and DOCX

## Workspace Behavior

Each app launch now starts with a fresh workspace:

- previous local RAG cache/manifests are cleared
- previous export/checkpoint artifacts are cleared
- prior session collections are not reused by default
- a fresh session collection name is assigned on startup

This behavior is implemented in [web_ui/runtime_cleanup.py](/C:/Users/longs/Documents/GitHub/Synthesizer/web_ui/runtime_cleanup.py:1).

## Project Structure

```text
Synthesizer/
├── core/                      # Business logic, RAG, document engine, exporters
├── web_ui/                    # Gradio app, actions, adapters, session/runtime cleanup
├── tests/                     # Unit and integration tests
├── scripts/                   # Developer verification and evaluation utilities
├── docs/                      # Architecture and developer documentation
├── examples/                  # Example input files
└── main.py                    # App launch entrypoint
```

## Setup

### Requirements

- Python 3.10+
- `uv` for environment and command management
- LM Studio running at `http://localhost:1234/v1` if you want local model usage

### Install

```bash
uv pip install -r requirements.txt
```

This installs the full app stack, including the default `LlamaIndex` RAG backend,
Docling parser support, OCR, and local vector search dependencies.

## Run

From the repo root:

```bash
uv run main.py
```

If you need a direct interpreter command instead of `uv run`:

```bat
.venv\Scripts\python.exe main.py
```

## Using The App

### Generate Sample Data

1. Open `Generate Sample Data`.
2. Describe the dataset, or import a CSV/JSON file.
3. Use `Generate Fields` to draft the schema.
4. Review the editable schema grid and row editor.
5. Click `Generate Data`.

Notes:

- The schema editor uses rows, not the old column-card UI.
- Imported data can be privacy-masked before the model sees it.
- Export restores original imported fields while preserving generated fields.

### Work With Files

1. Open `Work With Files`.
2. Upload files or add a URL source.
3. Choose one mode:
   - `Document Engine`
   - `Quick Q&A`
   - `Structured JSON`
4. Run the task and export the result if needed.

RAG defaults:

- backend: `LlamaIndex`
- collection: a fresh per-launch session collection
- Qdrant URL: `:memory:`
- embedding model: `BAAI/bge-small-en-v1.5`

## Verification

Focused web UI regression:

```bash
uv run python -m pytest tests/test_web_ui_runtime_config.py tests/test_web_ui_schema_editor.py tests/test_web_ui_privacy_import_export.py tests/test_web_ui_generation_controls.py tests/test_web_ui_files_workflow.py tests/test_web_ui_startup_cleanup.py -q
```

Privacy backend regression:

```bash
uv run python -m pytest tests/test_privacy_backend.py -q
```

Live RAG verification against LM Studio:

```bash
RUN_LIVE_LMSTUDIO_RAG=1 uv run python -m pytest tests/test_rag_lmstudio_live.py -q -s
```

Backend comparison utility:

```bash
uv run python scripts/evaluate_rag_backends.py --spec examples/rag_eval_spec.sample.json --model "your-lm-studio-model"
```

## Notes

- `:memory:` means no external Qdrant server is required.
- If you point `qdrant_url` at `http://localhost:6333`, you must have Qdrant running.
- Files in `.web_ui_exports/`, `.document_checkpoints/`, and `.rag_*` are treated as transient runtime artifacts.

## Related Docs

- [docs/ARCHITECTURE.md](/C:/Users/longs/Documents/GitHub/Synthesizer/docs/ARCHITECTURE.md:1)
- [docs/DEVELOPMENT_GUIDE.md](/C:/Users/longs/Documents/GitHub/Synthesizer/docs/DEVELOPMENT_GUIDE.md:1)
- [docs/POWER_BI_EXPORT_GUIDE.md](/C:/Users/longs/Documents/GitHub/Synthesizer/docs/POWER_BI_EXPORT_GUIDE.md:1)
- [docs/RAG_GUIDE.md](/C:/Users/longs/Documents/GitHub/Synthesizer/docs/RAG_GUIDE.md:1)
- [docs/PROJECT_STRUCTURE.md](/C:/Users/longs/Documents/GitHub/Synthesizer/docs/PROJECT_STRUCTURE.md:1)
