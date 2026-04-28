# Work With Files Pipeline

This document explains how the `Work With Files` workflow currently works in the Gradio app, step by step, and what each setting changes in the final output.

Primary code paths:

- UI: `web_ui/app.py`
- Gradio action wiring: `web_ui/actions/files_actions.py`
- Config building: `core/app_config.py`
- Runtime controller: `core/controller.py`
- Document engine orchestration: `core/document_engine/orchestrator.py`
- Startup cleanup + fresh-session collection setup: `web_ui/runtime_cleanup.py`

## 1. What the feature does

`Work With Files` is a file-grounded generation workflow. You give the app one or more files or URLs, the system indexes them into RAG, and then one of 3 task modes runs on top of that indexed knowledge:

- `Document Engine`
- `Quick Q&A`
- `Structured JSON`

The UI entry point is in `web_ui/app.py`.

Important startup behavior:

- each app launch starts with a fresh local workspace
- transient local RAG caches/manifests from prior runs are cleared
- a fresh per-launch collection name is assigned by default

## 2. End-to-end pipeline

### Step 1: Source registration

When you upload files or add a URL, Gradio stores those paths in the session state as the current file workspace.

What happens:

- Uploaded file paths are stored in `session.rag_files`
- URLs are appended to the same list
- The `Current Sources` table is refreshed

Relevant code:

- `register_uploaded_files()` in `web_ui/actions/files_actions.py`
- `add_url_source()` in `web_ui/actions/files_actions.py`

### Step 2: Mode selection

The `File Task` dropdown decides which downstream runtime path is used:

- `Document Engine`: generate a grounded PDF/DOCX document
- `Quick Q&A`: answer a question using the indexed files
- `Structured JSON`: generate JSON from a template, or exhaustively extract from all chunks

Relevant code:

- `files_mode_changed()` in `web_ui/actions/files_actions.py`

### Step 3: Runtime config is built

When you click `Run Files Task`, the app converts all current UI settings into a `GeneratorConfig`.

This is where:

- document settings are converted into `DocumentEngineConfig`
- retrieval settings are converted into `RagConfig`
- provider/model/API settings are attached

Relevant code:

- `_build_files_controller()` in `web_ui/actions/files_actions.py`
- `build_generator_config()` in `core/app_config.py`

### Step 4: Controller is created

A fresh `GeneratorController` is created for the run and initialized with:

- LLM client
- chosen RAG backend
- document orchestrator
- chart generator

Relevant code:

- `GeneratorController.set_runtime_config()` in `core/controller.py`
- `GeneratorController.initialize_rag()` in `core/controller.py`

### Step 5: Files are indexed into RAG

Before the task itself runs, the sources are ingested into the selected RAG backend.

What this usually means:

- files are parsed
- OCR may run if needed
- chunks are created
- embeddings are created
- vectors are stored
- graph/chunk relations may also be built

Relevant code:

- `run_files_task()` in `web_ui/actions/files_actions.py`
- `GeneratorController.ingest_documents()` in `core/controller.py`

### Step 6: The selected task runs

After indexing, the pipeline branches by mode:

#### A. Document Engine

The system:

1. Resolves document mode and target words
2. Builds an outline
3. Generates the document section by section
4. Retrieves context per section
5. Writes chunks with retries and validation
6. Runs consistency checks during long documents
7. Polishes the final document
8. Optionally generates charts/flowcharts
9. Exports PDF and DOCX

Relevant code:

- `GeneratorController.generate_document()` in `core/controller.py`
- `DocumentOrchestrator.run()` in `core/document_engine/orchestrator.py`

#### B. Quick Q&A

The system:

1. Searches the indexed sources
2. Builds grounded context
3. Either uses backend-native answer synthesis if available
4. Or falls back to normal RAG retrieval + final answer generation
5. Returns citations

Relevant code:

- `GeneratorController.ask_files()` in `core/controller.py`

#### C. Structured JSON

There are 2 paths:

- `Standard Generation`
  - uses the JSON template and generates a chosen number of items
- `Exhaustive Extraction`
  - processes every chunk in RAG and extracts verified items into the target array

Relevant code:

- `GeneratorController.generate_json_batch()` in `core/controller.py`
- `GeneratorController.generate_exhaustive_extraction()` in `core/controller.py`

### Step 7: Progress updates are streamed to Gradio

The app runs the work in a background thread and updates:

- current stage
- done/target progress
- last event
- recent logs

Relevant code:

- `_files_progress_markdown()` in `web_ui/actions/files_actions.py`
- `run_files_task()` in `web_ui/actions/files_actions.py`

### Step 8: Export artifacts are prepared

Depending on mode, the workflow prepares:

- PDF and DOCX for `Document Engine`
- JSON file for `Structured JSON`
- chat/citation output for `Quick Q&A`

Relevant code:

- `run_files_task()` in `web_ui/actions/files_actions.py`
- `export_document_pdf()` and `export_document_docx()` in `core/controller.py`

## 3. How Document Engine works internally

### 3.1 Outline planning

The document engine first decides what shape the document should have.

It chooses a different outline style depending on the request:

- narrative-style requests use story/scene beats
- comparison/recommendation requests use comparison sections
- everything else uses standard report sections

Relevant code:

- `_build_outline()` in `core/document_engine/orchestrator.py`

### 3.2 Section-by-section retrieval

For each section, the system builds a retrieval query using:

- the original prompt
- the current section title and purpose
- the rolling summary of what has already been written
- the next section boundary
- comparison-specific instructions when relevant

Relevant code:

- `_build_retrieval_query()` in `core/document_engine/orchestrator.py`

### 3.3 Source-aware guidance

When multiple sources are present in citations, the writer prompt explicitly tells the model not to silently let one source dominate.

For comparison tasks it also adds extra rules to:

- compare sources directly
- surface tradeoffs and exclusions
- justify recommendations against alternatives

Relevant code:

- `_build_source_guidance()` in `core/document_engine/orchestrator.py`

### 3.4 Chunk writing and validation

Each section is written in chunks. For each chunk the system:

1. retrieves context
2. builds a structured generation prompt
3. asks the model for JSON with a `chunk`
4. sanitizes the output
5. validates it
6. retries if needed

Relevant code:

- `_generate_chunk_with_retries()` in `core/document_engine/orchestrator.py`
- `_build_generation_prompt()` in `core/document_engine/orchestrator.py`

### 3.5 Running state and consistency

After each chunk, the engine updates:

- position in the document
- rolling summary
- tail content
- named entities / fact registry
- style signals

At configured intervals, it also runs a consistency audit and stores patch instructions for future sections.

Relevant code:

- `_update_state()` in `core/document_engine/orchestrator.py`
- `_consistency_check()` in `core/document_engine/orchestrator.py`

### 3.6 Final assembly

At the end, all chunks are assembled by section into final text, and citations are aggregated.

Relevant code:

- `_build_result()` in `core/document_engine/orchestrator.py`

### 3.7 Publish polish

After orchestration finishes, the controller runs a final polish step that:

- cleans titles
- removes prompt residue and meta artifacts
- rewrites into cleaner publish-ready output
- uses different polish behavior for narrative vs report-style output

Relevant code:

- `_polish_document_for_publish()` in `core/controller.py`

### 3.8 Visual generation

If enabled, chart generation runs after the document text is complete.

The current implementation:

- retrieves context again for the full prompt
- tries to generate grounded chart specs from real retrieved evidence
- falls back to spreadsheet-derived charts, relevance charts, or flowcharts
- embeds visuals into PDF and DOCX

Important nuance:

- charts are disabled in `Creative` mode because `Creative` maps to `pure`, and `_build_document_charts()` exits early for `PURE`
- flowcharts now work as a standalone fallback even when no normal chart can be grounded

Relevant code:

- `_build_document_charts()` in `core/controller.py`
- `DocumentChartGenerator.generate()` in `core/charts/generator.py`

## 4. What each setting changes

This section is grouped by where the setting acts in the pipeline.

### 4.1 File Task

UI setting:

- `Document Engine`
- `Quick Q&A`
- `Structured JSON`

Effect:

- Chooses the entire downstream execution path
- Changes what outputs are produced

Relevant code:

- `web_ui/app.py`
- `run_files_task()` in `web_ui/actions/files_actions.py`

### 4.2 Prompt / request text

UI setting:

- main prompt textbox

Effect:

- drives retrieval query intent
- drives document outline shape
- influences whether the request is treated as narrative or comparison-oriented
- influences Q&A answer generation
- is ignored for `Structured JSON` in template-driven mode

Relevant code:

- `files_prompt` in `web_ui/app.py`
- `_build_outline()` in `core/document_engine/orchestrator.py`

### 4.3 Grounding Style

UI values:

- `Balanced`
- `File-based`
- `Creative`

Mapping:

- `Balanced` -> `hybrid`
- `File-based` -> `strict_grounded`
- `Creative` -> `pure`

Effect:

- `Balanced`
  - uses retrieved context when relevant
  - still allows some synthesis
  - best general-purpose option
- `File-based`
  - applies the strictest grounding rules
  - best when you want the output tightly tied to uploaded evidence
- `Creative`
  - does not rely on file grounding during document writing
  - best for freeform story-like output
  - chart generation is skipped in this mode

Relevant code:

- `resolve_document_mode()` in `core/app_config.py`
- `_build_generation_prompt()` in `core/document_engine/orchestrator.py`
- `_retrieve_context()` in `core/document_engine/orchestrator.py`
- `_build_document_charts()` in `core/controller.py`

### 4.4 Length

UI setting:

- preset pages or manual page text
- `Let AI decide`

Effect:

- converted to `target_words`
- page count uses about `500 words per page`
- `Let AI decide` becomes `0`, which triggers auto-length selection inside the controller

Examples:

- `3 pages` -> about `1500 words`
- `12 pages` -> about `6000 words`
- `Let AI decide` -> model/fallback heuristic decides

Relevant code:

- `resolve_document_target_words()` in `core/app_config.py`
- `generate_document()` in `core/controller.py`

### 4.5 Review Depth

UI values:

- `Fast`
- `Thorough`

Effect:

- changes runtime tuning for the document engine
- affects chunk sizing, retries, consistency checking cadence, and validation strictness indirectly

General behavior:

- `Fast`
  - fewer retries
  - less aggressive consistency checking
  - faster output
  - more likely to be shorter and looser
- `Thorough`
  - more checking
  - more careful generation
  - slower, but generally better for difficult grounded documents

Relevant code:

- `generate_document()` in `core/controller.py`
- `_build_document_runtime_tuning()` in `core/controller.py`

### 4.6 Audience

UI setting:

- free text

Effect:

- passed into outline generation
- passed into chunk-writing prompts
- passed into final polish
- influences framing, assumptions, and style

Examples:

- `General` -> broader explanation
- `Executives` -> concise, high-level framing
- `Children` -> simplified style

Relevant code:

- `DocumentGenerationOptions` setup in `core/controller.py`
- `_build_outline()` in `core/document_engine/orchestrator.py`
- `_build_generation_prompt()` in `core/document_engine/orchestrator.py`

### 4.7 Tone

UI setting:

- free text

Effect:

- used during document writing and final polish
- also helps classify requests as narrative or comparison-like in some cases

Examples:

- `professional`
- `casual`
- `formal`
- `erotic`
- `technical`

Relevant code:

- `DocumentGenerationOptions` setup in `core/controller.py`
- `_is_narrative_request_from_parts()` in `core/document_engine/orchestrator.py`
- `_build_generation_prompt()` in `core/document_engine/orchestrator.py`

### 4.8 Include Charts

UI setting:

- checkbox

Effect:

- if enabled, the system tries to generate chart artifacts after the document text is complete
- these visuals are then embedded into both PDF and DOCX exports

Important nuance:

- no charts are attempted in `Creative` mode because that mode is `PURE`
- charts depend on usable retrieved context and grounded chart specs or fallbacks

Relevant code:

- `chart_enabled` in `core/controller.py`
- `DocumentPDFExporter` and `DocumentDocxExporter`

### 4.9 Include Flowchart

UI setting:

- checkbox

Effect:

- allows the system to append a process flow visual
- now also works as a standalone fallback when no numeric chart is available

Good use cases:

- procedural analysis
- decision workflows
- method explanations

Relevant code:

- `DocumentChartGenerator.generate()` in `core/charts/generator.py`

### 4.10 Max Charts

UI setting:

- integer

Effect:

- caps the number of visuals produced
- currently clamped to a safe range during generation

Relevant code:

- `doc_max_charts` in `web_ui/app.py`
- `generate_document()` in `core/controller.py`

### 4.11 RAG Backend

UI values:

- `LlamaIndex`
- `Native`

Effect:

- chooses the retrieval backend implementation
- changes how search, context prep, answer synthesis, and graph behavior are performed

Current practical difference:

- `LlamaIndex` is the default general backend
- `Native` can be forced for `Quick Q&A` when `Pinpoint Quick` is used

Relevant code:

- `resolve_effective_rag_backend()` in `web_ui/actions/files_actions.py`
- `initialize_rag()` in `core/controller.py`

### 4.12 Collection Name

UI setting:

- text

Effect:

- defines the RAG collection that receives the indexed files
- by default, the app assigns a fresh session collection on launch
- changing it manually isolates one workspace from another or intentionally points back to a persistent collection

Use this when:

- you want separate projects
- you do not want retrieval mixing across unrelated source sets

Relevant code:

- `RagConfig.collection_name` in `core/models.py`

### 4.13 Top K

UI setting:

- integer

Effect:

- controls how many top retrieval results are considered during search

Typical tradeoff:

- lower `top_k`
  - faster
  - tighter context
  - higher risk of missing relevant supporting evidence
- higher `top_k`
  - broader coverage
  - more context and more potential source coverage
  - higher risk of noise

Relevant code:

- `RagConfig.top_k` in `core/models.py`
- `initialize_rag()` in `core/controller.py`

### 4.14 Min Score

UI setting:

- float

Effect:

- sets the minimum retrieval score required for hits to be included

Tradeoff:

- higher score
  - stricter filtering
  - less noise
  - easier to miss weak-but-important clauses
- lower score
  - broader evidence
  - more noise

Relevant code:

- `RagConfig.min_score` in `core/models.py`

### 4.15 Max Context Chars

UI setting:

- integer

Effect:

- caps how much retrieved text is formatted into the prompt context

Tradeoff:

- lower value
  - cheaper/faster
  - may truncate supporting detail
- higher value
  - richer grounding
  - larger prompts and slower generation

Relevant code:

- `RagConfig.max_context_chars` in `core/models.py`

### 4.16 Embedding Model

UI setting:

- model name

Effect:

- changes how chunks are embedded for vector retrieval
- directly affects semantic search quality

Relevant code:

- `RagConfig.embedding_model` in `core/models.py`

### 4.17 Source Filter

UI setting:

- text

Effect:

- narrows retrieval or extraction to matching sources
- useful when multiple unrelated files are indexed in the same collection

Relevant code:

- `RagConfig.source_filter` in `core/models.py`
- `ask_files()` and `generate_exhaustive_extraction()` in `core/controller.py`

### 4.18 OCR settings

Settings:

- `ocr_mode`
- `ocr_dpi`
- `ocr_max_pages`
- `ocr_max_regions_per_page`
- `ocr_region_padding_px`
- `ocr_gap_multiplier`
- `ocr_min_extracted_chars`
- `ocr_timeout_ms_per_page`

Effect:

- these affect how image-heavy or scanned documents are read during ingestion

General interpretation:

- higher DPI
  - can improve OCR accuracy
  - slower and heavier
- higher max pages/regions
  - allows deeper extraction
  - slower on big scans
- padding/gap settings
  - influence region grouping for OCR
- min extracted chars
  - helps reject low-value OCR output
- timeout
  - controls per-page OCR patience

Relevant code:

- `RagConfig` in `core/models.py`
- `build_generator_config()` in `core/app_config.py`
- `initialize_rag()` in `core/controller.py`

### 4.19 Parser Mode

UI setting:

- parser mode string

Effect:

- changes how documents are parsed before chunking
- can affect chunk quality, table extraction, and text cleanliness depending on backend support

Relevant code:

- `RagConfig.parser_mode` in `core/models.py`

### 4.20 Hybrid Search

UI setting:

- checkbox

Effect:

- enables mixed retrieval strategies instead of relying on just one
- usually improves recall for harder queries

Relevant code:

- `RagConfig.hybrid_search_enabled` in `core/models.py`

### 4.21 Rerank

UI setting:

- checkbox

Effect:

- adds a reranking pass after initial retrieval
- can improve final ordering of evidence

Relevant code:

- `RagConfig.rerank_enabled` in `core/models.py`

### 4.22 Summary First

UI setting:

- checkbox

Effect:

- lets the system use summary-oriented retrieval before dense chunk retrieval
- useful when large files need faster top-level narrowing

Related settings:

- `summary_top_k`
- `dense_top_k`
- `lexical_top_k`

Relevant code:

- `RagConfig.summary_first_enabled` in `core/models.py`

### 4.23 Parent Context

UI settings:

- `parent_context_enabled`
- `parent_context_max_chars`

Effect:

- expands retrieved chunks with nearby parent context
- useful when a single chunk is too narrow to interpret correctly

This especially matters for:

- clauses
- definitions
- policy exceptions
- anything where neighboring text changes meaning

Relevant code:

- `RagConfig.parent_context_enabled` in `core/models.py`

### 4.24 Graph Retrieval

UI settings:

- `graph_enabled`
- `graph_hops`
- `graph_source_boost`

Effect:

- enables chunk/source relationship expansion beyond plain vector similarity
- helps pull in related clauses and neighboring evidence
- useful when important meaning is split across sections or files

Interpretation:

- more hops
  - broader relational expansion
  - more chance of useful linked evidence
  - also more chance of noise
- higher source boost
  - more influence from graph-related sources

Relevant code:

- `RagConfig.graph_enabled` in `core/models.py`
- backend initialization in `core/controller.py`

### 4.25 Late Interaction

UI settings:

- `late_interaction_enabled`
- `late_interaction_weight`

Effect:

- adds an extra scoring layer after initial retrieval
- helps refine relevance using more detailed token-level or late-stage matching logic, depending on backend

Tradeoff:

- can improve ranking quality
- costs more compute

Relevant code:

- `RagConfig.late_interaction_enabled` in `core/models.py`

### 4.26 Quick Q&A Style

UI setting:

- `Broader Analysis`
- `Pinpoint Quick`

Effect:

- only matters for `Quick Q&A`
- `Pinpoint Quick` forces the `Native` backend

Relevant code:

- `resolve_effective_rag_backend()` in `web_ui/actions/files_actions.py`

### 4.27 JSON Template

UI settings:

- template file
- target key
- JSON task mode
- replace existing items

Effect:

- defines where generated or extracted items go inside the JSON structure

Mode behavior:

- `Standard Generation`
  - generates new items matching the inferred schema
- `Exhaustive Extraction`
  - walks every chunk and injects verified extracted items

Relevant code:

- `generate_json_batch()` in `core/controller.py`
- `generate_exhaustive_extraction()` in `core/controller.py`

## 5. Stop button behavior

The Files stop button is cooperative rather than a hard kill.

What this means:

- the current unit of work usually finishes first
- partial results may still be exported
- this is especially useful if a prompt was wrong or the task is taking too long

Relevant code:

- `request_stop_files_task()` in `web_ui/actions/files_actions.py`
- `stop_document_generation()` in `core/controller.py`

## 6. What most strongly affects output quality

If the goal is quality rather than speed, the most important settings are usually:

1. source quality
2. prompt clarity
3. `Grounding Style`
4. `Top K`
5. `Min Score`
6. `Parent Context`
7. `Graph Retrieval`
8. `Review Depth`
9. backend choice

Practical guidance:

- use `File-based` when factual grounding matters most
- use `Balanced` when you still want readable synthesis
- raise `Top K` if the system is missing relevant sources
- lower `Min Score` slightly if weak-but-important clauses are not showing up
- keep `Parent Context` and `Graph Retrieval` on for policy/comparison work
- use `Thorough` for complex recommendation tasks

## 7. Known limitations

Current important limitations:

- `Creative` mode reduces file grounding because it maps to `pure`
- charts still depend on usable grounded context; they are not guaranteed unless a flowchart fallback is possible
- `Quick Q&A` returns text and citations, but not export files
- if retrieval misses important sources, the generated document can still be biased even when all files were uploaded

## 8. Simple mental model

The easiest way to think about `Work With Files` is:

1. put files into RAG
2. choose the kind of output you want
3. choose how tightly the result should stick to the files
4. tune retrieval breadth and depth
5. let the chosen task run on that indexed evidence
6. export the final artifact
