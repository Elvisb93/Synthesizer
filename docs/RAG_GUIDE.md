# RAG Guide

This document describes how Retrieval-Augmented Generation (RAG) works in this project and how it is used by the `Work With Files` workspace.

## Overview

RAG is local-first and integrated into the **Work With Files** tab.

Current architecture:

1. Parse source files with routed parsers (`RouterParser`)
   - PDF: `HybridPdfParser` (native + OCR fallback)
   - Text/Markdown/JSON/CSV/DOCX: text parser
   - Excel (`.xlsx/.xls`): multi-sheet parser
   - HTML + URLs: HTML/text extraction
   - Images: OCR parser (when dependencies available)
2. Choose a backend through `RagConfig.backend`
   - `LlamaIndex` is the default
   - `Native` remains available as a local fallback/alternative
3. Index and retrieve through the selected backend
4. Inject bounded retrieved context into prompts
5. Return grounded output (+ citations when available)

Files workspace supports three runtime modes:

- **Document Engine**: long-form document generation from prompt + available context.
- **Quick Q&A**: grounded file question answering with citations.
- **Structured JSON**: JSON template population and exhaustive grounded extraction into a target array.

Quick Q&A also supports a mode-level override:

- **Broader Analysis**: uses the default `LlamaIndex` path
- **Pinpoint Quick**: switches only `Quick Q&A` to the native backend for narrower fact lookup

## Backend Behavior

### Default backend: `LlamaIndex`

The current default Files backend uses:

1. Routed parsing via the existing local parser stack
2. `LlamaIndex` `IngestionPipeline`
3. Hugging Face embeddings
4. Qdrant vector storage
5. Local LM Studio-compatible synthesis through an OpenAI-compatible bridge
6. Response synthesis for `Quick Q&A`
7. Prepared grounding summaries for `Document Engine`

Important details:

- No LlamaCloud or paid Llama API is required.
- Metadata extraction is enabled only for smaller ingests.
- Larger ingests automatically skip metadata extraction to avoid long indexing times.
- `Document Engine` uses the selected backend's retrieval/context-preparation path automatically.

### Alternate backend: `Native`

The original native backend remains available and still uses:

1. `SemanticDoubleBufferChunker`
2. `fastembed`
3. Qdrant storage
4. Summary-first retrieval
5. Hybrid dense + lexical search
6. Parent context expansion
7. Local graph expansion
8. Late-interaction reranking

## Supported File Types

- PDF (`.pdf`)
- Text and markup (`.txt`, `.md`, `.json`, `.csv`, `.docx`, `.html`, `.htm`)
- Excel (`.xlsx`, `.xls`) with per-sheet extraction
- Images (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif`, `.tiff`) via OCR parser
- HTTP/HTTPS URLs via routed parser (controller/API path)

## Key Files

- `core/rag/factory.py` - backend selection and construction
- `core/rag/service.py` - native end-to-end parse/chunk/embed/store/search orchestration
- `core/rag/backends/base.py` - shared backend protocol
- `core/rag/backends/llamaindex_backend.py` - default `LlamaIndex` backend
- `core/rag/backends/local_openai_llm.py` - LM Studio/OpenAI-compatible local LLM bridge for `LlamaIndex`
- `core/rag/parsers/hybrid_pdf_parser.py` - PDF native text + OCR fallback policy
- `core/rag/ocr/rapidocr_engine.py` - OCR adapter (RapidOCR)
- `core/rag/chunking/semantic_double_buffer.py` - chunking strategy
- `core/rag/embeddings/fastembed_embedder.py` - embedding adapter
- `core/rag/stores/qdrant_store.py` - vector store adapter
- `core/llm_client.py` - retrieval integration + RAG telemetry
- `core/controller.py` - document and file-task orchestration with RAG fallback behavior
- `core/models.py` - `RagConfig`
- `web_ui/actions/files_actions.py` - file import/index, mode switching, presets, source actions, and document bundles
- `web_ui/app.py` - `Work With Files` tab controls, search admin, and browser download wiring
- `web_ui/runtime_cleanup.py` - startup cleanup and fresh per-launch collection setup

## Configuration

RAG settings live in `GeneratorConfig.rag` (`RagConfig`).

Important fields:

- `backend` (`LlamaIndex` | `Native`)
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

- Default backend is `LlamaIndex`.
- Default `qdrant_url` is `:memory:` for zero-setup local use.
- The app now starts each launch with a fresh session collection name instead of reusing the old shared default collection by default.
- Retrieval settings are read from UI at runtime before file operations.
- `LlamaIndex` uses local LM Studio-compatible generation only when a local model/base URL is configured.

## Files Workspace UX

1. Open **Work With Files**.
2. Upload one or more files to ingest supported sources.
3. Optional: paste a **URL** in the Files workspace and click **Add URL**.
4. Select **Files Mode**:
   - `Document Engine`
   - `Quick Q&A`
   - `Structured JSON`
5. Optional: select **Q&A Style** when in `Quick Q&A` mode:
   - `Broader Analysis`
   - `Pinpoint Quick`
6. In Document Engine mode, set document controls:
   - **Doc Strategy**:
     - `hybrid`: grounded + synthesis
     - `factual by doc`: strictly grounded in files
     - `creative`: freer generation with minimal grounding
   - **Pages**: fixed page count or `Let AI decide`
     - Page/word targets are treated as a minimum target for content planning, not a hard ceiling on each chunk
   - **Quality**: `Fast` or `Thorough`
   - **Audience** and **Tone**
7. Optional one-click bundles:
   - `Executive Brief`
   - `Policy Draft`
   - `Action Plan`
   - `Meeting Summary`
8. In Structured JSON mode, set:
   - **JSON Template**
   - **Target Key**
   - **Template Mode**: `Standard Generation` or `Exhaustive Extraction`
9. Run the task and review the chat-style output plus downloads.

Browser workflow is mode-aware:

- **Generate Sample Data** -> CSV/JSON import for enrichment
- **Work With Files** -> multi-format upload/add-URL flow for RAG

## Export Formatting

- Document and narrative PDF exporters now apply markdown-aware rendering.
- The renderer preserves heading hierarchy, bullet/numbered lists, fenced code blocks, and basic markdown table rows with cleaner spacing.
- This is used for model-generated outputs so professional reports are more readable without manual cleanup.

## Presets

Files mode supports editable task presets:

- Select preset -> prompt loaded into the Files prompt box
- Save preset -> persists to `.rag_task_presets.json`
- Delete preset -> removes from local preset store

Document Engine also supports built-in one-click bundles (above), which apply strategy/pages/quality/tone/audience defaults.
Structured JSON mode supports exporting the populated template back to disk as JSON.

## Fallback Behavior

- If RAG initialization fails, Files features degrade gracefully instead of crashing the app.
- In Document Engine mode, document generation can proceed with non-RAG context when retrieval is empty/unavailable.
- In Quick Q&A mode, empty retrieval returns a user-facing "insufficient context" response.
- In Structured JSON exhaustive mode, extraction requires RAG to be initialized and at least one file to be ingested.
- If `LlamaIndex` synthesis fails for a request, controller-level fallback behavior still allows the Files flow to continue.

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
uv run python -m pytest tests/test_rag_chunking.py tests/test_rag_config.py tests/test_rag_retriever.py tests/test_rag_generation_integration.py tests/test_metrics_rag.py -q
uv run python -m pytest tests/test_rag_ocr.py -q
uv run python -m pytest tests/test_llamaindex_backend_pipeline.py -q
```

Live LM Studio test:

```bash
RUN_LIVE_LMSTUDIO_RAG=1 uv run python -m pytest tests/test_rag_lmstudio_live.py -q -s
```

Backend comparison:

```bash
uv run python scripts/evaluate_rag_backends.py --spec examples/rag_eval_spec.sample.json --model "your-lm-studio-model"
```

This writes a JSON report comparing `Native` and `LlamaIndex` on the same local documents for:

- ingest time
- `Quick Q&A`
- `Document Engine`
- backend status metadata

Recent local backend verification on `April 12, 2026` also confirmed that the current `LlamaIndex` path uses size-aware metadata extraction gating, which reduced a previously slow large-ingest path down to a practical local runtime.

Recent live verification was also run successfully against LM Studio model `qwen/qwen3.5-9b`, covering:

- connection test
- structured JSON generation/export
- local RAG ingest/search
- graph boost metadata
- late interaction metadata
- grounded Q&A
- exhaustive extraction

Web UI regression checks:

```bash
uv run python -m pytest tests/test_web_ui_runtime_config.py tests/test_web_ui_files_workflow.py tests/test_web_ui_startup_cleanup.py -q
```

## Operational Notes

- Use `:memory:` when no Qdrant server is running.
- For persistent indexing, run Qdrant and set `qdrant_url` to your server endpoint.
- On app startup, local transient RAG caches/manifests and prior workspace exports/checkpoints are cleared automatically.
- Keep `max_context_chars` conservative to avoid token inflation.
- `parser_mode=docling` uses the Docling dependency included in `requirements.txt`; if unavailable, runtime degrades to `auto`.
- The default `LlamaIndex` backend still uses the existing local parser/OCR stack; this is not a separate hosted ingestion service.
- Graph expansion is local and entity/theme based. It now supports query-seeded source discovery as well as source-to-source expansion.
- Late interaction remains a local approximation, but now weighs token importance, coverage, token order, and compact match windows instead of relying only on simple n-gram overlap.
- Advanced RAG dependencies are included in `requirements.txt` (`Docling`, OCR extras, and `LlamaIndex` packages).

### Troubleshooting

- **`WinError 10061` during retrieval:** Qdrant URL points to a server that is not running. Set `qdrant_url` to `:memory:`.
- **No relevant context found:** try lowering `min_score` (for example `0.10`) or re-indexing files.
- **LM Studio appears idle in Quick Q&A:** retrieval returned no context, so no generation call was sent.
- **RAG is not configured:** install `fastembed` and `qdrant-client` in the active environment, then restart the app.
- **OCR in `auto`/`on` not activating:** ensure `rapidocr-onnxruntime` is installed and inspect ingest `ocr_*` counters.
- **`LlamaIndex` indexing feels too slow:** larger ingests should now skip metadata extraction automatically. Check backend status for `metadata_extraction_enabled` and `metadata_extraction_last_reason`.
