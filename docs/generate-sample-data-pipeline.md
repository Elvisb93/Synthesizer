# Generate Sample Data Pipeline

This document explains how the `Generate Sample Data` workflow currently works in the Gradio app, step by step, and what each setting changes in the final output.

Primary code paths:

- UI: `web_ui/app.py`
- Gradio action wiring: `web_ui/actions/data_actions.py`
- Config building: `core/app_config.py`
- Runtime controller: `core/controller.py`
- Row generation graph: `core/row_agent.py`
- Dependency resolution: `core/prompt_builder.py`
- Validation: `core/validator.py`

## 1. What the feature does

`Generate Sample Data` creates synthetic tabular rows from either:

- a schema you define from scratch
- AI-suggested fields from a plain-language prompt
- an imported CSV/JSON that becomes the base schema for enrichment

The main output is a set of generated rows that can then be:

- previewed in the UI
- exported as CSV
- exported as JSON
- exported as SQL
- exported as a versioned Power BI run
- reviewed with a simple quality report

## 2. End-to-end pipeline

### Step 1: Describe the dataset or import existing data

You can start in 2 ways:

- write a prompt and let the app suggest fields
- import a CSV or JSON file and use its columns as the base

The `Import CSV/JSON` path stores imported rows in session state and infers a first-pass schema from the file headers and data types.

If privacy masking is enabled, the imported preview and model-facing context use masked values while export restores the original imported fields later.

Relevant code:

- `import_data_file()` in `web_ui/actions/data_actions.py`
- `infer_field_records_from_dataframe()` in `web_ui/adapters.py`
- `sanitize_imported_records()` in `web_ui/adapters.py`

### Step 2: Build or refine the schema

The editable schema grid is the working definition of the dataset.

Each row in the grid becomes a `ColumnDefinition` with:

- name
- type
- prompt instruction
- duplicate policy
- optional hidden constraints when those are populated through the richer field-record structure

The visible Gradio grid is a simplified view, while the adapters normalize it into the richer internal representation.

Relevant code:

- `field_records_to_grid_dataframe()` in `web_ui/adapters.py`
- `field_record_to_definition()` in `web_ui/adapters.py`
- `field_records_to_columns()` in `web_ui/adapters.py`

### Step 3: Save the schema into session state

When you click `Save Rows`, the app:

1. reads the visible grid
2. normalizes each row
3. validates names and duplicates
4. stores the final schema in `session.fields`

Relevant code:

- `save_grid_rows()` in `web_ui/actions/data_actions.py`

### Step 4: Build the runtime config

When you click `Generate Data`, the UI settings are converted into a `GeneratorConfig`.

This config includes:

- model and provider settings
- row count
- similarity threshold
- max retries
- optional imported data
- RAG/retrieval settings

Even though this tab is not a file workflow, row generation can still use RAG context if configured.

Relevant code:

- `generate_data()` in `web_ui/actions/data_actions.py`
- `build_generator_config()` in `core/app_config.py`

### Step 5: Initialize the controller

The app creates a fresh `GeneratorController` and initializes:

- the LLM client
- the RAG service
- the uniqueness validator
- execution order for the columns

Execution order matters because fields can depend on earlier fields using `@[column_name]`.

Relevant code:

- `GeneratorController.initialize()` in `core/controller.py`
- `get_execution_order()` in `core/prompt_builder.py`

### Step 6: Resolve field dependencies

Before any rows are generated, the app topologically sorts columns based on `@[column_name]` references in the prompt instructions.

This means:

- fields with no dependencies are generated first
- dependent fields are generated after the fields they reference
- circular references are rejected

Example:

- `email` prompt: `Business email for @[first_name] @[last_name]`

Here `first_name` and `last_name` must exist before `email`.

Relevant code:

- `get_dependencies()` in `core/prompt_builder.py`
- `get_execution_order()` in `core/prompt_builder.py`

### Step 7: Choose generation mode

The controller then decides between 2 runtime modes:

#### A. Enrichment mode

If imported data exists, the controller uses the imported rows as starting context and tries to enrich each row.

Behavior:

- target row count becomes the number of imported rows
- imported values are preserved
- fields marked with `(Imported)` are not regenerated
- missing/generated fields are added around the imported context

Relevant code:

- `_run_generation_loop()` in `core/controller.py`
- `generate_row(initial_context=...)` in `core/controller.py`

#### B. Fresh generation mode

If there is no imported data, the controller generates rows from scratch until it reaches the requested target count.

Relevant code:

- `_run_generation_loop()` in `core/controller.py`

### Step 8: Pre-fill deterministic fields

Before the row agent runs, the controller fills any deterministic columns it can handle without the LLM.

This currently includes:

- `Auto Increment (ID)`
- `Faker / Deterministic`

Behavior:

- auto-increment fields get the next sequential row number
- faker fields use the selected Faker provider if possible

Relevant code:

- `generate_row()` in `core/controller.py`

### Step 9: Generate the remaining fields with the row agent

For fields that are not already filled, the app runs a LangGraph row-generation workflow.

That workflow:

1. builds prompts for the missing columns
2. includes current row context
3. optionally retrieves RAG context relevant to the current field
4. generates values
5. runs an LLM semantic-consistency check
6. attempts correction if needed

Relevant code:

- `create_row_generator_graph()` in `core/row_agent.py`

### Step 10: Apply post-generation guardrails

After the row agent returns a row, the controller runs hard checks:

- field value exists
- regex validation
- uniqueness validation

Only after a row passes those checks are its values committed into the uniqueness history.

Relevant code:

- `generate_row()` in `core/controller.py`
- `validate_regex()` in `core/validator.py`
- `is_unique()` in `core/validator.py`
- `commit()` in `core/validator.py`

### Step 11: Retry or skip when generation fails

The controller behaves differently depending on the mode:

- fresh generation:
  - failed rows are retried until enough rows are completed or failure threshold is hit
- enrichment mode:
  - a failed row is skipped and the run moves on

There is also a hard safety limit:

- after 10 consecutive failures, generation aborts

Relevant code:

- `_run_generation_loop()` in `core/controller.py`

### Step 12: Stream progress to Gradio

While generation is running, the UI shows:

- rows completed
- current row
- retry count
- last event
- elapsed time
- recent log messages

Relevant code:

- `_generation_progress_markdown()` in `web_ui/actions/data_actions.py`
- `generate_data()` in `web_ui/actions/data_actions.py`

### Step 13: Save partial results continuously

During generation, the Gradio session is updated with the current completed rows.

This is why:

- the stop button can preserve partial output
- partial exports still work after a stop

Relevant code:

- `generate_data()` in `web_ui/actions/data_actions.py`

### Step 14: Preview and export

After the run, the app:

- builds a dataframe preview
- keeps generated rows in session state
- allows CSV/JSON/SQL export

Relevant code:

- `export_generated_data()` in `web_ui/actions/data_actions.py`
- exporters in `core/exporters`

### Step 15: Optional quality review

If you click `Review Quality`, the system computes simple per-column quality metrics:

- null count
- diversity score
- top frequent values

Relevant code:

- `review_generated_data_quality()` in `web_ui/actions/data_actions.py`
- `QualityAnalyzer.analyze()` in `core/analytics.py`

## 3. How the row-generation engine works internally

### 3.1 Imported context vs blank row

Each row starts as either:

- an imported row dict
- or an empty dict

This initial row context is carried into prompt construction and helps dependent fields stay coherent.

Relevant code:

- `generate_row()` in `core/controller.py`

### 3.2 Dependency-aware prompts

Prompt instructions can reference earlier fields using `@[column_name]`.

During prompt construction:

- any known referenced values are interpolated into the instruction
- the rest of the current row is included as `Current Row Context`

This helps coherence even without explicit linking.

Relevant code:

- `construct_prompt()` in `core/prompt_builder.py`
- `_construct_prompt()` in `core/row_agent.py`

### 3.3 Optional RAG retrieval inside row generation

For each column, the row agent can query the RAG backend using:

- column name
- column instruction
- current row context

This means dataset generation can be grounded against indexed source material when RAG is configured.

Relevant code:

- `_construct_prompt()` in `core/row_agent.py`
- `LLMClient.retrieve_context()` in `core/llm_client.py`

### 3.4 Semantic validation loop

After initial generation, the row agent asks the model if the row makes semantic sense.

If not:

- it asks the model to correct the row
- loops back into validation
- gives up after 3 correction attempts

Relevant code:

- `validate_semantics()` in `core/row_agent.py`
- `correct_row()` in `core/row_agent.py`

### 3.5 Uniqueness behavior

Uniqueness works in 2 layers:

- exact duplicate hash check for all fields
- semantic similarity check for `Long Text` when sentence-transformers is available

The semantic threshold comes from the global `similarity_threshold` setting unless overridden.

Relevant code:

- `is_unique()` in `core/validator.py`

### 3.6 Regex behavior

Regex validation can use either:

- raw regex patterns
- shortcut names such as:
  - `email`
  - `phone`
  - `zip`
  - `postcode`
  - `date`
  - `ipv4`

Relevant code:

- `validate_regex()` in `core/validator.py`

## 4. What each setting changes

This section focuses on settings that affect `Generate Sample Data`.

### 4.1 AI prompt

UI setting:

- main prompt textbox

Effect:

- only affects schema suggestion, not direct row generation
- used by `Generate Fields` to propose a schema
- if imported data exists, the sample row context is also included during schema suggestion

Relevant code:

- `suggest_fields()` in `web_ui/actions/data_actions.py`
- `generate_schema()` path in `LLMClient`

### 4.2 Generate Fields

UI action:

- button

Effect:

- asks the model to return a schema in JSON
- may use a heuristic fallback if the model returns too few fields or unusable output
- merges new suggestions into the current schema by field name

Relevant code:

- `suggest_fields()` in `web_ui/actions/data_actions.py`
- `_generate_schema_fallback()` and `_generate_heuristic_schema()` in `core/llm_client.py`

### 4.3 Example buttons

UI actions:

- `Customer Contacts`
- `Support Tickets`
- `Insurance Inbox`

Effect:

- they are just prompt starters for common dataset ideas
- they do not directly alter runtime generation logic beyond setting the prompt

### 4.4 Import CSV/JSON

UI setting:

- file upload

Effect:

- switches the workflow into enrichment mode if imported rows are present
- infers a starting schema from the file headers
- marks inferred columns as `(Imported)`

Important consequence:

- imported fields are preserved and not regenerated unless you change their prompt/type behavior later

Relevant code:

- `import_data_file()` in `web_ui/actions/data_actions.py`
- `infer_field_records_from_dataframe()` in `web_ui/adapters.py`

### 4.5 Row Name

Schema field:

- `name`

Effect:

- becomes the output column name
- also becomes the dependency reference name for `@[column_name]`

Important:

- names must be unique
- duplicate names are rejected at save time

### 4.6 Type

Schema field:

- `Short Text`
- `Long Text`
- `Numeric`
- `Categorical`
- `Boolean`
- `Auto Increment (ID)`
- `Faker / Deterministic`

Effect:

- changes how the field is prompted
- changes which constraints are relevant
- changes how uniqueness behaves
- can bypass the LLM for deterministic types

Behavior by type:

- `Short Text`
  - general short values
  - exact duplicate blocking if duplicates are not allowed
- `Long Text`
  - narrative/longer values
  - exact duplicate blocking plus semantic similarity blocking when available
- `Numeric`
  - numeric-like output expected
- `Categorical`
  - best used with explicit options
- `Boolean`
  - binary values
- `Auto Increment (ID)`
  - generated sequentially, no LLM needed
- `Faker / Deterministic`
  - generated with Faker provider when possible

Relevant code:

- `ColumnType` in `core/models.py`
- `generate_row()` in `core/controller.py`

### 4.7 Prompt Instruction

Schema field:

- free text

Effect:

- this is the most important field-level control over row content
- used to tell the model what each field should contain
- supports `@[column_name]` references for dependent generation

Special case:

- `(Imported)` means the field should be treated as existing imported context rather than regenerated

Examples:

- `Business email for @[first_name] @[last_name]`
- `Detailed customer complaint consistent with @[issue_type]`

### 4.8 Allow Duplicates

Schema field:

- checkbox

Effect:

- if unchecked, exact duplicates are blocked
- for `Long Text`, semantically similar text may also be blocked
- if checked, the validator skips uniqueness enforcement for that column

Relevant code:

- `generate_row()` in `core/controller.py`
- `is_unique()` in `core/validator.py`

### 4.9 Numeric range constraints

Schema fields:

- `min_value`
- `max_value`

Effect:

- included in the internal column constraints
- intended to guide generation and validation logic

Important note:

- in the current Gradio grid-first workflow, these richer per-field constraints are not visible in the simplified table itself
- they matter when present in the richer field-record structure

### 4.10 Length constraints

Schema fields:

- `min_length`
- `max_length`

Effect:

- included in prompt construction for the field
- especially relevant for short vs long text shaping

### 4.11 Categorical options

Schema field:

- comma-separated options

Effect:

- included in the prompt as an explicit allowed set
- most important for `Categorical` fields

Without options:

- the model may still output category-like values, but with less control

### 4.12 Regex pattern

Schema field:

- regex or shortcut

Effect:

- checked after generation
- if the generated value fails regex, the row is rejected

This increases failure/retry rates when patterns are strict.

### 4.13 Faker provider

Schema field:

- provider name like `name`, `email`, `phone_number`, `company`, `uuid4`

Effect:

- only used for `Faker / Deterministic`
- bypasses the LLM for that field when Faker is available

Relevant code:

- `FAKER_PROVIDERS` in `core/models.py`
- `generate_row()` in `core/controller.py`

### 4.14 Number of rows

UI setting:

- `num_rows`

Effect:

- target number of rows for fresh generation
- ignored in enrichment mode, where the target becomes the imported row count

Relevant code:

- `build_generator_config()` in `core/app_config.py`
- `_run_generation_loop()` in `core/controller.py`

### 4.15 Similarity threshold

UI setting:

- `similarity_threshold`

Effect:

- controls semantic deduplication strictness for `Long Text`

Tradeoff:

- lower threshold
  - stricter duplicate filtering
  - more retries/failures
- higher threshold
  - more variation allowed
  - greater risk of near-duplicate long text

Relevant code:

- `is_unique()` in `core/validator.py`

### 4.16 Max retries

UI setting:

- `max_retries`

Effect:

- included in config, but the current row-generation loop mostly uses fixed internal retry behavior:
  - row-agent semantic correction loop: up to 3 attempts
  - generation abort safety: 10 consecutive failures

So this setting currently has less visible impact on sample-data generation than its name suggests.

Important note:

- this is a real behavior gap between UI expectation and runtime impact

### 4.17 Model

UI setting:

- `model_id`

Effect:

- chooses which model handles:
  - schema suggestion
  - row generation
  - semantic validation
  - row correction

This is one of the strongest quality levers in the whole workflow.

### 4.18 Provider and credentials

UI settings:

- provider
- API key
- Azure endpoint
- Azure deployment

Effect:

- determines which backend LLM client is created
- affects availability, latency, cost, and output quality

Relevant code:

- `LLMClient._init_chat_model()` in `core/llm_client.py`

### 4.19 Retrieval / RAG settings

These are shared global settings and can affect sample-data generation whenever row prompts retrieve context.

Important ones:

- `rag_backend`
- `collection_name`
- `top_k`
- `min_score`
- `max_context_chars`
- `embedding_model`
- `source_filter`
- OCR settings
- `hybrid_search_enabled`
- `rerank_enabled`
- `summary_first_enabled`
- `parent_context_enabled`
- `graph_enabled`
- `late_interaction_enabled`

Effect in this tab:

- these matter only if row generation is expected to pull grounded context from indexed sources
- if you are generating purely synthetic rows with no useful RAG collection, they matter much less

Relevant code:

- `LLMClient.retrieve_context()` in `core/llm_client.py`
- `initialize_rag()` in `core/controller.py`

## 5. Import and enrichment behavior

This is one of the most important parts of the feature.

When you import CSV/JSON:

1. imported rows are stored
2. columns are inferred
3. inferred fields get prompt instruction `(Imported)`
4. generation runs in enrichment mode
5. each imported row becomes the starting context for one generated row

This means:

- imported values are preserved
- extra fields can be generated around them
- the final output row count normally matches the import row count

## 6. Stop button behavior

The stop button is cooperative rather than a hard kill.

What this means:

- the current row attempt usually finishes first
- completed rows are preserved
- partial export remains available after stopping

Relevant code:

- `request_stop_data_generation()` in `web_ui/actions/data_actions.py`
- `stop_generation()` in `core/controller.py`

## 7. What the export buttons do

### Export CSV

- writes rows in schema column order
- best for spreadsheets and downstream tabular tools

Relevant code:

- `export_csv()` in `core/exporters/csv_exporter.py`

### Export JSON

- writes a list of row objects
- best for API and programmatic use

Relevant code:

- `export_json()` in `core/exporters/json_exporter.py`

### Export SQL

- writes `INSERT INTO synthetic_data (...) VALUES (...);`
- best for quick seeding of a SQL-like dataset

Important note:

- table name is currently fixed to `synthetic_data` in the Gradio path

Relevant code:

- `export_sql()` in `core/exporters/sql_exporter.py`

### Export Power BI Run

- writes a timestamped run folder under the selected destination
- includes `data.csv`, `schema.json`, and `metadata.json`
- appends successful runs to `index.csv`
- never overwrites prior run folders
- warns when the schema changes from the previous run for the same dataset name

Recommended use:

- choose a local OneDrive or SharePoint-synced folder as the destination
- connect Power BI to the folder or to `index.csv` for audit/history reporting

Relevant code:

- `export_power_bi_run()` in `core/exporters/power_bi_exporter.py`
- `export_power_bi_data()` in `web_ui/actions/data_actions.py`

## 8. What the quality review measures

The quality review is intentionally lightweight.

Per column it reports:

- diversity score
- null count
- most frequent values

This is useful for spotting:

- overly repetitive output
- too many blanks
- low-variety categories

It does not currently measure:

- factual correctness
- semantic realism
- cross-row coherence
- distribution quality beyond simple uniqueness/frequency

## 9. What most strongly affects output quality

If your goal is better sample data quality, the strongest levers are usually:

1. schema quality
2. prompt-instruction quality
3. dependency design with `@[column]`
4. type choice
5. duplicate policy
6. regex/constraint strictness
7. model/provider quality
8. whether imported context is clean and useful
9. similarity threshold for long text

Practical guidance:

- use clear field names
- keep prompt instructions concrete
- use `@[field]` when relationships matter
- use `Categorical` with explicit options for controlled values
- use `Faker / Deterministic` for obvious boilerplate fields
- do not over-constrain every field at once, or retries will spike

## 10. Known limitations

Current important limitations:

- the simplified schema grid does not expose every hidden constraint field directly
- `max_retries` has limited practical effect in the current row-generation runtime
- semantic validation is LLM-based, so it is helpful but not guaranteed
- imported schema inference is intentionally simple:
  - integers/floats -> `Numeric`
  - booleans -> `Boolean`
  - everything else -> `Short Text`
- quality review is basic and not a full evaluation system

## 11. Simple mental model

The easiest way to think about `Generate Sample Data` is:

1. define the columns
2. decide which fields are deterministic vs model-generated
3. define relationships between fields
4. generate rows
5. reject bad or duplicate rows
6. keep only accepted rows
7. export the final dataset
