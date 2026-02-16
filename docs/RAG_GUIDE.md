# RAG Guide

This document describes how Retrieval-Augmented Generation (RAG) is implemented in this project and how to use it safely.

## Overview

RAG is optional and local-first. When enabled, the generator retrieves relevant context from ingested PDFs and injects that context into generation prompts.

Current pipeline:

1. Parse PDF text with `pypdfium2`
2. Chunk text with semantic overlap (`SemanticDoubleBufferChunker`)
3. Create embeddings with `fastembed`
4. Store/search vectors in Qdrant (`qdrant-client`)
5. Inject top hits into prompt as a bounded "Retrieved Context" block

## Key Files

- `core/rag/service.py` - end-to-end orchestration
- `core/rag/parsers/pdfium_parser.py` - PDF parser
- `core/rag/chunking/semantic_double_buffer.py` - chunking strategy
- `core/rag/embeddings/fastembed_embedder.py` - embedding adapter
- `core/rag/stores/qdrant_store.py` - vector store adapter
- `core/llm_client.py` - retrieval + RAG telemetry
- `core/models.py` - `RagConfig`
- `gui/handlers/rag_handlers.py` - ingest/status/clear actions

## Configuration

RAG settings are part of `GeneratorConfig.rag` (`RagConfig`).

Important fields:

- `enabled`: turn retrieval on/off
- `collection_name`: Qdrant collection
- `top_k`: number of retrieved hits
- `min_score`: retrieval score cutoff
- `max_context_chars`: hard cap for injected context
- `embedding_model`: FastEmbed model ID
- `source_filter`: optional source path filter for retrieval
- `qdrant_url`: server URL or `:memory:`
- `qdrant_api_key`: optional key for managed/private Qdrant

## UI Workflow

1. Enable RAG in **AI Configuration**.
2. Configure collection/model/Qdrant URL.
3. Click **Ingest PDF** and select a document.
4. Optionally set **Source Filter** to constrain retrieval.
5. Start generation.
6. Monitor RAG metrics in the metrics panel.

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

- Use `:memory:` for tests when no Qdrant server is running.
- For persistent indexing, run Qdrant and set `qdrant_url` to the server endpoint.
- Keep `max_context_chars` conservative to avoid token inflation.
- If RAG fails, generation should continue with non-RAG prompts.
