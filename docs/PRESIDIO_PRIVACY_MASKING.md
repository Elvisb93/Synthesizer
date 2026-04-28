# Presidio Privacy Masking

This document explains how Synthesizer uses Microsoft Presidio to reduce the chance that imported personal or business-sensitive data is sent to the model during sample-data enrichment.

It describes the current implementation in this repo. It is not a claim of perfect anonymization.

## Scope

The Presidio-backed masking flow is used in the `Generate Sample Data` import/enrichment path:

- [web_ui/adapters.py](../web_ui/adapters.py)
- [web_ui/actions/data_actions.py](../web_ui/actions/data_actions.py)
- [web_ui/state.py](../web_ui/state.py)

The main goals are:

- mask likely sensitive values before AI generation
- preserve enough business context for useful summaries/replies
- restore original imported values on export
- block generation if imported files are in raw mode

## Current Flow

### 1. Import

When a user imports CSV/JSON data, the app stores:

- `raw_imported_data`: the untouched source rows
- `imported_data`: the masked rows used for preview and AI context
- `import_mask_mappings`: per-row placeholder-to-original mappings used for decode-on-export

Relevant code:

- [web_ui/actions/data_actions.py](../web_ui/actions/data_actions.py)
- [web_ui/state.py](../web_ui/state.py)

### 2. Masking

Masking is performed in:

- [mask_imported_records](../web_ui/adapters.py)
- [sanitize_imported_records](../web_ui/adapters.py)

The implementation combines:

- Presidio built-in entity detection
- custom Presidio pattern recognizers
- request-level context derived from column names
- targeted fallback regex detection
- placeholder mapping such as `<NAME_1_1>`, `<EMAIL_1>`, `<ORG_1_1>`

### 3. Pre-generation rebuild

Immediately before `Generate Fields` and `Generate Data`, the app rebuilds the masked rows from `raw_imported_data` again instead of trusting older in-memory masked state.

This prevents stale session data from bypassing improved masking logic after a code change or config change.

Relevant code:

- [web_ui/actions/data_actions.py](../web_ui/actions/data_actions.py)

### 4. Guardrail

If imported rows exist and privacy mode is not `Mask likely personal values`, the app blocks generation instead of sending raw imported rows to the model.

### 5. Export decode

After generation, imported columns are restored from `raw_imported_data`, while generated columns remain generated.

Relevant code:

- [restore_original_imported_columns](../web_ui/actions/data_actions.py)

## How Presidio Is Used

### Analyzer engine

Synthesizer creates a local `AnalyzerEngine` with a spaCy NLP backend and context enhancement.

Current behavior:

- prefers `en_core_web_lg`, then `en_core_web_md`, then `en_core_web_sm` if available locally
- uses `LemmaContextAwareEnhancer`
- passes request-level context built from column names such as `sender_email`, `policy_number`, `date_of_birth`

This follows Presidio’s documented context-enhancement model, where surrounding text and metadata can increase recognizer confidence.

Official docs:

- Context enhancement: https://microsoft.github.io/presidio/tutorial/06_context/

### Built-in entities

The app relies on Presidio built-in coverage where useful, especially for:

- `EMAIL_ADDRESS`
- `PHONE_NUMBER`
- `DATE_TIME`
- `ORGANIZATION`
- `URL`
- `PERSON`

Official docs:

- Supported entities: https://microsoft.github.io/presidio/supported_entities/

### Custom recognizers

The app adds custom pattern recognizers to cover domain-specific or weakly structured data, including:

- identifiers such as policy/member/claim/account-like IDs
- more flexible date formats
- organization phrases
- role titles

This follows Presidio’s documented approach for extending the analyzer with custom recognizers and context terms.

Official docs:

- Adding recognizers: https://microsoft.github.io/presidio/analyzer/adding_recognizers/
- Recognizer development / best practices: https://microsoft.github.io/presidio/analyzer/developing_recognizers/

## Placeholder Strategy

Detected values are replaced with stable placeholders, for example:

- `<NAME_1>`
- `<EMAIL_1>`
- `<ORG_1_1>`
- `<PHONE_1_1>`
- `<IDENTIFIER_1_1>`
- `<DATE_1_1>`

These placeholders are then reused in free-text fields so that:

- the model sees masked context rather than raw values
- row-level relationships still remain understandable
- export can restore the original imported values later

## Context Preservation

The masking layer intentionally tries to preserve useful non-PII context, for example:

- task descriptions
- workflow intent
- role titles when possible
- business instructions

The app also maintains an allow-list for known non-PII business phrases to reduce false positives.

This is a tradeoff:

- stricter masking reduces leakage risk
- looser masking preserves more business meaning

## Current Guardrails

The current implementation includes these guardrails:

- imported files default to `Mask likely personal values`
- imported-file generation is blocked in raw mode
- masked rows are rebuilt from `raw_imported_data` immediately before AI calls
- privacy leak checks run against masked rows before generation
- tests cover prompt-facing masking and export restoration

Related tests:

- [tests/test_privacy_backend.py](../tests/test_privacy_backend.py)
- [tests/test_web_ui_privacy_import_export.py](../tests/test_web_ui_privacy_import_export.py)

## Limitations

This system improves privacy protection, but it does not guarantee perfect detection for every possible email format or every possible sensitive token.

Known limitations:

- Presidio is still probabilistic and recognizer-driven
- unusual entity formats may require more custom recognizers
- highly domain-specific business identifiers may require additional patterns or deny-lists
- free-text entity detection can still trade off false negatives vs false positives

This is consistent with Presidio’s own recognizer-development guidance: strong results usually require tuning, context, and domain-specific recognizers rather than only default settings.

## Future Hardening Options

Reasonable next steps if stronger protection is needed:

- add more insurance/HR-specific identifier recognizers
- add YAML-configured no-code recognizers for domain patterns
- add an external recognizer model with stronger free-text entity recall
- separate `decoded export` and `masked export` as explicit user options
- log privacy-audit failures with redacted diagnostics only

Useful Presidio references:

- Context enhancement: https://microsoft.github.io/presidio/tutorial/06_context/
- Adding recognizers: https://microsoft.github.io/presidio/analyzer/adding_recognizers/
- Recognizer development / best practices: https://microsoft.github.io/presidio/analyzer/developing_recognizers/
- Supported entities: https://microsoft.github.io/presidio/supported_entities/
- No-code/YAML recognizers: https://microsoft.github.io/presidio/tutorial/24_no_code/
