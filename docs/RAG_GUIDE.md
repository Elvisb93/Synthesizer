# RAG Guide

This document describes how Retrieval-Augmented Generation (RAG) is implemented in this project and how to use it safely.

## Overview

RAG is local-first and integrated into the **Files** workspace. Users import PDFs, then ask file-grounded questions/tasks through the existing Magic input.

Current pipeline:

1. Parse PDF text with `pypdfium2`
2. Chunk text with semantic overlap (`SemanticDoubleBufferChunker`)
3. Create embeddings with `fastembed`
4. Store/search vectors in Qdrant (`qdrant-client`)
5. Inject top hits into prompt as a bounded "Retrieved Context" block
6. Return answer + citations (source, page, score)

## Key Files

- `core/rag/service.py` - end-to-end orchestration
- `core/rag/parsers/pdfium_parser.py` - PDF parser
- `core/rag/chunking/semantic_double_buffer.py` - chunking strategy
- `core/rag/embeddings/fastembed_embedder.py` - embedding adapter
- `core/rag/stores/qdrant_store.py` - vector store adapter
- `core/llm_client.py` - retrieval + RAG telemetry
- `core/models.py` - `RagConfig`
- `gui/handlers/rag_handlers.py` - file import/index/chat/presets/status/clear
- `gui/flet_app.py` - Data vs Files workspace tabs

## Configuration

RAG settings are part of `GeneratorConfig.rag` (`RagConfig`).

Important fields:

- `collection_name`: Qdrant collection
- `top_k`: number of retrieved hits
- `min_score`: retrieval score cutoff
- `max_context_chars`: hard cap for injected context
- `embedding_model`: FastEmbed model ID
- `source_filter`: optional source path filter for retrieval
- `qdrant_url`: server URL or `:memory:`
- `qdrant_api_key`: optional key for managed/private Qdrant

Notes:

- Default `qdrant_url` is `:memory:` for zero-setup local use.
- RAG behavior is driven by imported files (no UI toggle required).

## UI Workflow

1. Open **Files** tab.
2. Click **Import File** to ingest one or more PDFs.
3. Ask file questions/tasks in Magic input.
4. Read answer + citations in File Assistant chat.
5. Use per-file actions to re-index/remove as needed.
6. Monitor RAG metrics in the metrics panel.

The toolbar import button is mode-aware:

- **Data Generation tab** -> CSV/JSON import for enrichment
- **Files tab** -> PDF import for RAG

## Task Presets

Files mode supports editable task presets:

- Select preset -> prompt is loaded into Magic input
- Save preset -> persists to `.rag_task_presets.json`
- Delete preset -> removes from local preset store

## Metrics

RAG telemetry is exposed in `stats.rag`:

- `queries`
- `hit_rate`
- `avg_retrieval_ms`
- `avg_context_chars`
- `last_hits`

## Testing

### Fast tests

```bash
py -m pytest tests/test_rag_chunking.py tests/test_rag_config.py tests/test_rag_retriever.py tests/test_rag_generation_integration.py tests/test_metrics_rag.py -q
```

### Live LM Studio test

```bash
RUN_LIVE_LMSTUDIO_RAG=1 py -m pytest tests/test_rag_lmstudio_live.py -q -s
```

What the live test verifies:

- Ingest from `examples/benefits_email_narative.pdf`
- Retrieval returns non-empty context for an email-focused question
- LM Studio (`gpt-oss-20b`) returns grounded output
- RAG hit metrics are non-zero

## Operational Notes

- Use `:memory:` for default local use and tests when no Qdrant server is running.
- For persistent indexing, run Qdrant and set `qdrant_url` to the server endpoint.
- Keep `max_context_chars` conservative to avoid token inflation.
- If RAG fails, generation should continue with non-RAG prompts.

### Troubleshooting

- **`WinError 10061` during retrieval:** Qdrant server URL is configured but not running. Set `qdrant_url` to `:memory:`.
- **"Could not find relevant context":** try lowering `min_score` (e.g. `0.10`) or clearing index and re-importing files.
- **LM Studio appears idle during file query:** retrieval returned no context, so no LLM call was sent. Check RAG status, file list, and retrieval settings.
