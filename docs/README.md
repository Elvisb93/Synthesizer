# Architecture Documentation

This directory contains architecture, workflow, and development documentation for the web-first Synthesizer project.

## Documents

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Complete system architecture overview
- **[DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md)** - Development guidelines and best practices
- **[INSURANCE_BROKER_GUIDE.md](INSURANCE_BROKER_GUIDE.md)** - Non-technical guide for insurance brokers and employee benefits teams
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Detailed project structure explanation
- **[PRESIDIO_PRIVACY_MASKING.md](PRESIDIO_PRIVACY_MASKING.md)** - How imported-row masking works, where Presidio is used in the app, and the current privacy guardrails/limitations
- **[RAG_GUIDE.md](RAG_GUIDE.md)** - Local-first RAG + OCR fallback architecture, configuration, and testing, including Files workspace `Structured JSON` mode, query-seeded graph retrieval, and late-interaction reranking notes
- **[UI_SMOKE_CHECKLIST.md](UI_SMOKE_CHECKLIST.md)** - Manual + automated web UI verification workflow
- **[agent_rules.md](agent_rules.md)** - User-defined development rules captured across sessions

## Quick Links

- [Project Structure](#project-structure)
- [Core Principles](#core-principles)
- [Adding New Features](#adding-new-features)

## Verification Commands

Run these after UI changes:

```bash
python -m pytest tests/test_web_ui_runtime_config.py tests/test_web_ui_schema_editor.py tests/test_web_ui_privacy_import_export.py tests/test_web_ui_generation_controls.py tests/test_web_ui_files_workflow.py tests/test_web_ui_startup_cleanup.py -q
python main.py
```

## Project Structure

```
Synthesizer/
├── core/              # Business logic (no UI dependencies)
│   ├── exporters/     # Data export modules
│   ├── rag/           # Retrieval-augmented generation subsystem
│   ├── document_engine/ # Long-form document orchestration
│   ├── models.py      # Domain models
│   ├── schemas.py     # LangChain output schemas
│   ├── controller.py  # Generation orchestration
│   └── ...
├── web_ui/            # Gradio web UI layer
│   ├── app.py         # Main application
│   ├── actions/       # UI callback logic
│   ├── adapters.py    # UI/data adapters
│   └── state.py       # Session state helpers
├── tests/             # Test suite
└── docs/              # Documentation
```

## Core Principles

1. **Separation of Concerns** - Core logic is independent of UI
2. **Action Separation** - Web UI callbacks are organized by responsibility
3. **Async-Friendly** - Long-running tasks stream progress through the Gradio workflow
4. **Core Independence** - Business logic remains outside the UI layer

## Adding New Features

See [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) for detailed instructions on:

- Adding new column types
- Creating new export formats
- Extending the web UI
- Writing tests
