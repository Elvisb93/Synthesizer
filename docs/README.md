# Architecture Documentation

This directory contains comprehensive documentation for the Synthetic Data Generator project architecture, design patterns, and development guidelines.

## Documents

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Complete system architecture overview
- **[DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md)** - Development guidelines and best practices
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Detailed project structure explanation

## Quick Links

- [Project Structure](#project-structure)
- [Core Principles](#core-principles)
- [Adding New Features](#adding-new-features)

## Project Structure

```
Synthesizer/
├── core/              # Business logic (no UI dependencies)
│   ├── exporters/     # Data export modules
│   ├── models.py      # Domain models
│   ├── schemas.py     # LangChain output schemas
│   ├── controller.py  # Generation orchestration
│   └── ...
├── gui/               # Flet UI layer
│   ├── handlers/      # Event handler mixins
│   ├── controls/      # Reusable UI components
│   └── flet_app.py    # Main application
├── tests/             # Test suite
└── docs/              # Documentation
```

## Core Principles

1. **Separation of Concerns** - Core logic is independent of UI
2. **Mixin Pattern** - UI handlers organized by responsibility
3. **Async-First** - All UI operations use Flet's async patterns
4. **Backward Compatibility** - Public APIs preserved via `__init__.py` exports

## Adding New Features

See [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) for detailed instructions on:

- Adding new column types
- Creating new export formats
- Extending the UI
- Writing tests
