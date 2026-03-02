# RAG Guide

This document describes how Retrieval-Augmented Generation (RAG) works in this project and how it is used by the Files workspace.

## Overview

RAG is local-first and integrated into the **Files** tab.

Current pipeline:

1. Parse PDF text with `pypdfium2` (`HybridPdfParser`)
2. Optionally run OCR fallback (`off`/`auto`/`on`)
3. Chunk text (`SemanticDoubleBufferChunker`)
4. Embed chunks (`fastembed`)
5. Upsert/search vectors in Qdrant (`qdrant-client`)
6. Inject bounded retrieved context into prompts
7. Return grounded output (+ citations when available)

Files workspace supports two runtime modes:

- **Document Engine**: long-form document generation from prompt + available context.
- **Quick Q&A**: grounded file question answering with citations.

## Supported File Types

- Current ingestion path is **PDF-focused** in the Files tab.
- Import dialog filters for `*.pdf` (with optional `*.*` fallback selection).

## Key Files

- `core/rag/service.py` - end-to-end parse/chunk/embed/store/search orchestration
- `core/rag/parsers/hybrid_pdf_parser.py` - PDF native text + OCR fallback policy
- `core/rag/ocr/rapidocr_engine.py` - OCR adapter (RapidOCR)
- `core/rag/chunking/semantic_double_buffer.py` - chunking strategy
- `core/rag/embeddings/fastembed_embedder.py` - embedding adapter
- `core/rag/stores/qdrant_store.py` - vector store adapter
- `core/llm_client.py` - retrieval integration + RAG telemetry
- `core/controller.py` - document and file-task orchestration with RAG fallback behavior
- `core/models.py` - `RagConfig`
- `gui/handlers/rag_handlers.py` - file import/index, mode switching, presets, doc bundles
- `gui/flet_app.py` - Files tab controls and strategy helper text

## Configuration

RAG settings live in `GeneratorConfig.rag` (`RagConfig`).

Important fields:

- `collection_name`
- `top_k`
- `min_score`
- `max_context_chars`
- `embedding_model`
- `source_filter`
- `qdrant_url`
- `qdrant_api_key`
- `ocr_mode` (`off` | `auto` | `on`)
- `ocr_dpi`
- `ocr_max_pages`
- `ocr_max_regions_per_page`
- `ocr_region_padding_px`
- `ocr_gap_multiplier`
- `ocr_min_extracted_chars`
- `ocr_timeout_ms_per_page`

Notes:

- Default `qdrant_url` is `:memory:` for zero-setup local use.
- Retrieval settings are read from UI at runtime before file operations.

## Files Workspace UX

1. Open **Files** tab.
2. Click **Import File** to ingest one or more PDFs.
3. Select **Files Mode**:
   - `Document Engine`
   - `Quick Q&A`
4. In Document Engine mode, set document controls:
   - **Doc Strategy**:
     - `hybrid`: grounded + synthesis
     - `factual by doc`: strictly grounded in files
     - `creative`: freer generation with minimal grounding
   - **Pages**: fixed page count or `Let AI decide`
   - **Quality**: `Fast` or `Thorough`
   - **Audience** and **Tone**
5. Optional one-click bundles:
   - `Executive Brief`
   - `Policy Draft`
   - `Action Plan`
   - `Meeting Summary`
6. Run task from Magic input and review chat output.

Toolbar import behavior is mode-aware:

- **Data Generation tab** -> CSV/JSON import for enrichment
- **Files tab** -> PDF import for RAG

## Presets

Files mode supports editable task presets:

- Select preset -> prompt loaded into Magic input
- Save preset -> persists to `.rag_task_presets.json`
- Delete preset -> removes from local preset store

Document Engine also supports built-in one-click bundles (above), which apply strategy/pages/quality/tone/audience defaults.

## Fallback Behavior

- If RAG initialization fails, Files features degrade gracefully instead of crashing the app.
- In Document Engine mode, document generation can proceed with non-RAG context when retrieval is empty/unavailable.
- In Quick Q&A mode, empty retrieval returns a user-facing "insufficient context" response.

## Metrics

RAG telemetry is exposed in `stats.rag`:

- `queries`
- `hit_rate`
- `avg_retrieval_ms`
- `avg_context_chars`
- `last_hits`

Ingest report includes OCR counters:

- `ocr_pages_total`
- `ocr_pages_full`
- `ocr_regions_total`
- `ocr_failures`

## Testing

Fast tests:

```bash
py -m pytest tests/test_rag_chunking.py tests/test_rag_config.py tests/test_rag_retriever.py tests/test_rag_generation_integration.py tests/test_metrics_rag.py -q
py -m pytest tests/test_rag_ocr.py -q
```

Live LM Studio test:

```bash
RUN_LIVE_LMSTUDIO_RAG=1 py -m pytest tests/test_rag_lmstudio_live.py -q -s
```

UI regression smoke:

```bash
py scripts/verify/ui_regression_smoke.py
```

## Operational Notes

- Use `:memory:` when no Qdrant server is running.
- For persistent indexing, run Qdrant and set `qdrant_url` to your server endpoint.
- Keep `max_context_chars` conservative to avoid token inflation.

### Troubleshooting

- **`WinError 10061` during retrieval:** Qdrant URL points to a server that is not running. Set `qdrant_url` to `:memory:`.
- **No relevant context found:** try lowering `min_score` (for example `0.10`) or re-indexing files.
- **LM Studio appears idle in Quick Q&A:** retrieval returned no context, so no generation call was sent.
- **OCR in `auto`/`on` not activating:** ensure `rapidocr-onnxruntime` is installed and inspect ingest `ocr_*` counters.
