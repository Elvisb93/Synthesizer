# RAG Guide

This document describes how Retrieval-Augmented Generation (RAG) works in this project and how it is used by the Files workspace.

## Overview

RAG is local-first and integrated into the **Files** tab.

Current pipeline:

1. Parse source files with routed parsers (`RouterParser`)
   - PDF: `HybridPdfParser` (native + OCR fallback)
   - Text/Markdown/JSON/CSV/DOCX: text parser
   - Excel (`.xlsx/.xls`): multi-sheet parser
   - HTML + URLs: HTML/text extraction
   - Images: OCR parser (when dependencies available)
2. Optionally run OCR fallback (`off`/`auto`/`on`)
3. Chunk text (`SemanticDoubleBufferChunker`)
4. Create both chunk vectors and document-summary vectors
5. Embed records (`fastembed`)
6. Upsert/search vectors in Qdrant (`qdrant-client`)
7. Retrieve via summary-first + hybrid dense/lexical + rerank
8. Expand context with parent page snippets (configurable)
9. Expand candidate sources with local Shadow Graph links and query-seeded graph discovery
10. Optionally apply late-interaction token scoring using a local weighted token/order/proximity scorer
11. Inject bounded retrieved context into prompts
12. Return grounded output (+ citations when available)

Files workspace supports three runtime modes:

- **Document Engine**: long-form document generation from prompt + available context.
- **Quick Q&A**: grounded file question answering with citations.
- **Structured JSON**: JSON template population and exhaustive grounded extraction into a target array.

## Supported File Types

- PDF (`.pdf`)
- Text and markup (`.txt`, `.md`, `.json`, `.csv`, `.docx`, `.html`, `.htm`)
- Excel (`.xlsx`, `.xls`) with per-sheet extraction
- Images (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif`, `.tiff`) via OCR parser
- HTTP/HTTPS URLs via routed parser (controller/API path)

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
- `parser_mode` (`auto` | `pdf_only` | `docling`)
- `hybrid_search_enabled`
- `rerank_enabled`
- `summary_first_enabled`
- `summary_top_k`
- `dense_top_k`
- `lexical_top_k`
- `parent_context_enabled`
- `parent_context_max_chars`
- `graph_enabled`
- `graph_hops`
- `graph_source_boost`
- `late_interaction_enabled`
- `late_interaction_weight`
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
2. Click **Import File** to ingest one or more supported files.
3. Optional: paste a **URL** in the Files workspace and click **Index URL**.
4. Select **Files Mode**:
   - `Document Engine`
   - `Quick Q&A`
   - `Structured JSON`
5. In Document Engine mode, set document controls:
   - **Doc Strategy**:
     - `hybrid`: grounded + synthesis
     - `factual by doc`: strictly grounded in files
     - `creative`: freer generation with minimal grounding
   - **Pages**: fixed page count or `Let AI decide`
     - Page/word targets are treated as a minimum target for content planning, not a hard ceiling on each chunk
   - **Quality**: `Fast` or `Thorough`
   - **Audience** and **Tone**
6. Optional one-click bundles:
   - `Executive Brief`
   - `Policy Draft`
   - `Action Plan`
   - `Meeting Summary`
7. In Structured JSON mode, set:
   - **JSON Template**
   - **Target Key**
   - **Template Mode**: `Standard Generation` or `Exhaustive Extraction`
8. Run task from Magic input and review chat output.

Toolbar import behavior is mode-aware:

- **Data Generation tab** -> CSV/JSON import for enrichment
- **Files tab** -> multi-format import for RAG (PDF, Excel, images, text/markup, URLs)

## Export Formatting

- Document and narrative PDF exporters now apply markdown-aware rendering.
- The renderer preserves heading hierarchy, bullet/numbered lists, fenced code blocks, and basic markdown table rows with cleaner spacing.
- This is used for model-generated outputs so professional reports are more readable without manual cleanup.

## Presets

Files mode supports editable task presets:

- Select preset -> prompt loaded into Magic input
- Save preset -> persists to `.rag_task_presets.json`
- Delete preset -> removes from local preset store

Document Engine also supports built-in one-click bundles (above), which apply strategy/pages/quality/tone/audience defaults.
Structured JSON mode supports exporting the populated template back to disk as JSON.

## Fallback Behavior

- If RAG initialization fails, Files features degrade gracefully instead of crashing the app.
- In Document Engine mode, document generation can proceed with non-RAG context when retrieval is empty/unavailable.
- In Quick Q&A mode, empty retrieval returns a user-facing "insufficient context" response.
- In Structured JSON exhaustive mode, extraction requires RAG to be initialized and at least one file to be ingested.

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

Recent live verification was also run successfully against LM Studio model `qwen/qwen3.5-9b`, covering:

- connection test
- structured JSON generation/export
- local RAG ingest/search
- graph boost metadata
- late interaction metadata
- grounded Q&A
- exhaustive extraction

UI regression smoke:

```bash
py scripts/verify/ui_regression_smoke.py
```

## Operational Notes

- Use `:memory:` when no Qdrant server is running.
- For persistent indexing, run Qdrant and set `qdrant_url` to your server endpoint.
- Keep `max_context_chars` conservative to avoid token inflation.
- `parser_mode=docling` requires optional Docling dependency; if unavailable, runtime degrades to `auto`.
- Graph expansion is local and entity/theme based. It now supports query-seeded source discovery as well as source-to-source expansion.
- Late interaction remains a local approximation, but now weighs token importance, coverage, token order, and compact match windows instead of relying only on simple n-gram overlap.
- Optional advanced deps live in `requirements-rag-optional.txt` (currently Docling).

### Troubleshooting

- **`WinError 10061` during retrieval:** Qdrant URL points to a server that is not running. Set `qdrant_url` to `:memory:`.
- **No relevant context found:** try lowering `min_score` (for example `0.10`) or re-indexing files.
- **LM Studio appears idle in Quick Q&A:** retrieval returned no context, so no generation call was sent.
- **RAG is not configured:** install `fastembed` and `qdrant-client` in the active environment, then restart the app.
- **OCR in `auto`/`on` not activating:** ensure `rapidocr-onnxruntime` is installed and inspect ingest `ocr_*` counters.
