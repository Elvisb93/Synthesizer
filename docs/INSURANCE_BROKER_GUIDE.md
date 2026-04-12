# Insurance Broker Guide

## Purpose of This Document

This document explains the project in plain language for non-technical readers. It is written for an insurance broker or employee benefits business that works across multiple schemes, insurers, and document types.

The application is not a pre-built broking administration system. It is an AI-assisted workbench that can:

- generate realistic synthetic data sets
- enrich existing data with new fields
- read and search documents
- answer questions against imported files
- draft grounded reports and summaries
- turn unstructured content into structured JSON outputs

In a brokerage setting, this makes it useful for early-stage automation, internal research, document triage, reporting, communication support, and controlled experimentation.

## Executive Summary

At a high level, the project has two main working areas:

- `Data Generation`: creates or enriches structured datasets such as member records, policy administration samples, claims-support logs, scheme communications, or test data for demonstrations and training.
- `Files`: reads uploaded business documents and lets a user ask questions, generate reports, or extract structured outputs from those documents.

The platform is designed to work with local AI models or cloud AI providers. It can also work with scanned PDFs by using OCR when needed.

For an insurance broker, the application can support:

- benefits and insurance document review
- open-enrollment and renewal communications analysis
- policy and scheme summarisation
- creation of sample datasets without using live customer data
- extraction of key information from mixed files into a structured format
- generation of broker-ready reports, briefs, and action plans

## What Problems It Can Help Solve

This guide is relevant where teams struggle with one or more of the following:

- too many documents arriving in different formats
- manual review of insurer documents, policy summaries, and employee-benefit communications
- the need to test workflows without exposing real personal data
- repeated requests for summaries, action lists, and stakeholder-ready reports
- difficulty converting free-text material into a reusable structured format
- inconsistent answers when teams manually search large volumes of policy or scheme content

## Core Product Areas

## 1. Data Generation Workspace

This workspace is used to create structured data tables.

It supports two operating styles:

- `New generation`: create a brand-new dataset from scratch.
- `Enrichment`: import an existing CSV or JSON file, keep the original columns, and ask the system to create additional columns based on the imported content.

### Key features in this workspace

#### AI connection and model selection

What it does:
- connects the app to an AI model
- supports local and cloud model providers
- lets the user test the connection before running work

How it works:
- the user selects a provider and model
- the app checks that the selected service is reachable
- once connected, the same model can be used for schema generation, row generation, document tasks, and file-based analysis

Insurance broker scenario:
- a broker operations team can run the application on a local model for privacy-sensitive experiments, or a cloud model for faster drafting and larger tasks

#### Manual column design

What it does:
- lets users define the structure of a dataset column by column

How it works:
- the user adds columns and chooses the type of information required
- supported column types include short text, long text, numeric, categorical, boolean, auto-increment IDs, and deterministic fields such as fake names or emails

Insurance broker scenario:
- create a sample membership file with fields such as `member_id`, `scheme_name`, `cover_type`, `renewal_month`, `insurer`, `premium_band`, `query_reason`, and `broker_note`

#### AI schema generation

What it does:
- turns a plain-English description into a proposed data structure

How it works:
- the user describes the desired output in normal business language
- the AI suggests columns, field types, and generation instructions
- if data has already been imported, the AI can use the existing headers and sample values as context and suggest only relevant additional fields

Insurance broker scenario:
- a user types: "Create a benefits support inbox dataset covering enrollment questions, dependent changes, payroll deduction issues, and policy clarification requests"
- the system proposes a ready-made schema instead of requiring manual setup

#### Synthetic row generation

What it does:
- creates realistic-looking records based on the selected schema

How it works:
- the app generates row values in dependency order
- deterministic fields such as IDs, names, or emails can be created without using the AI for every field
- AI-generated fields are prompted using the row context, making outputs more coherent across columns

Insurance broker scenario:
- generate a training dataset of employee-benefit inquiries for service-desk rehearsal, workflow testing, or sales demonstrations without using live member data

#### Import and enrichment

What it does:
- extends an existing file instead of replacing it

How it works:
- a CSV or JSON file is imported
- the app reads the current structure and row count
- imported columns are preserved
- new AI-generated columns are added alongside the original data

Insurance broker scenario:
- import a scheme communication log and add fields such as `issue_category`, `urgency`, `recommended_owner`, `response_tone`, or `likely_follow_up_required`

#### Bulk row-by-row analysis of imported files

What it does:
- uses the data generation workspace as an analysis and extraction tool for large imported datasets

How it works:
- the user imports a CSV or JSON file containing existing records
- each imported row becomes the context for AI generation
- instead of creating brand-new rows from scratch, the app can work line by line across the imported file and generate new analytical columns for every record
- this means one source column, such as an email body or broker note, can be read and analysed row by row at scale

Insurance broker scenario:
- import thousands of benefit, policy, claims-support, or broker-service emails in CSV format
- add new output columns such as `sentiment`, `category`, `sub_category`, `likely_priority`, `customer_intent`, `insurer_mentioned`, `recommended_owner`, `response_summary`, or `follow_up_required`
- let the system process the file record by record so the output becomes a structured analysis table

Why this matters:
- this is one of the most practical uses for broker operations teams
- it turns a flat communication export into something searchable, measurable, and easier to triage
- it can support inbox analysis, service review, trend spotting, and downstream reporting

#### Validation and guardrails

What it does:
- reduces low-quality, duplicate, or inconsistent outputs

How it works:
- exact duplicate checking is applied to shorter values
- semantic duplicate checking can be used for longer text so near-identical wording is filtered out
- regex rules can enforce patterns like email, phone, postcode, or date formats
- cross-column logic can enforce relationships between fields

Insurance broker scenario:
- ensure generated email addresses are valid
- keep IDs unique
- make sure a `renewal_end_date` is after a `renewal_start_date`
- reduce repeated benefit-query text across a synthetic inbox sample

#### Stop, retry, and safe failure behaviour

What it does:
- gives users control during long runs

How it works:
- generation can be stopped gracefully
- completed rows are preserved
- rows that fail validation are retried or skipped instead of breaking the whole run

Insurance broker scenario:
- if a user sees that a generated dataset is going in the wrong direction, they can stop the run without losing already-created sample records

#### Quality analysis

What it does:
- provides a simple quality view of generated output

How it works:
- after generation, the app can analyse each column for diversity, null values, and the most repeated values

Insurance broker scenario:
- check whether a sample dataset is varied enough to represent different policy questions, insurer responses, or employee-benefit situations

#### Export options for structured data

What it does:
- lets teams save outputs in common formats

How it works:
- generated data can be exported as CSV, JSON, SQL, PDF report, or PDF narrative

Insurance broker scenario:
- export CSV for spreadsheet review
- export JSON for downstream system testing
- export SQL for development demonstrations
- export PDF narrative for a stakeholder sample pack

## 2. Files Workspace

This workspace is used for document intelligence rather than row-based dataset generation.

It is especially relevant for insurance brokers because much of the work sits inside documents such as:

- policy wording
- benefit summaries
- renewal packs
- insurer emails
- provider brochures
- meeting notes
- spreadsheets
- client update packs
- scanned PDFs and image-based files

### Supported source inputs

The file workspace can index a broad mix of material, including:

- PDF
- text and markdown
- CSV and JSON
- Excel files
- HTML pages
- DOCX files
- images such as PNG and JPG
- web URLs

### Key features in this workspace

#### File indexing and searchable knowledge base

What it does:
- turns uploaded files into a searchable working knowledge base

How it works:
- files are imported into the workspace
- the app splits them into searchable chunks
- those chunks are stored in a retrieval index
- future questions or document-generation tasks can pull back relevant passages from the uploaded material

Insurance broker scenario:
- load insurer product summaries, open-enrollment guides, internal process notes, and benefit emails into one temporary research set for a client team

#### OCR for scanned documents

What it does:
- helps the system read scanned or image-heavy files

How it works:
- if a PDF has poor text extraction, OCR can be used automatically or forced on
- this means image-based or scan-based policy documents can still be searched and used

Insurance broker scenario:
- review an older scanned scheme booklet or image-based meeting scan without manually retyping the content

#### Quick Q&A with citations

What it does:
- answers questions based on imported files

How it works:
- the user asks a question in plain language
- the app retrieves the most relevant passages from the indexed files
- the AI produces an answer using those passages
- the answer can include citations back to source files and page references

Insurance broker scenario:
- ask: "What are the main enrollment issues raised in these benefit emails?"
- ask: "Which file mentions dependent eligibility for children under 26?"
- ask: "Where is the guidance on claims submission for dental and vision?"

#### Document Engine

What it does:
- creates longer business documents from the imported material

How it works:
- the user enters an instruction such as a summary, proposal, briefing note, or action plan
- the app creates an outline
- it writes the document section by section
- it can keep track of consistency across sections
- it can resume from a saved checkpoint if generation is interrupted
- final outputs can be exported to PDF or DOCX

Insurance broker scenario:
- generate a client-ready renewal briefing from several insurer files
- draft a policy comparison summary across carriers
- produce a management memo from a bundle of benefit administration emails and supporting spreadsheets

#### Three document strategies

The document engine supports three styles:

- `Hybrid`: combines file-grounded facts with broader synthesis. This is useful for business commentary or recommendations.
- `Factual by doc`: stays tightly grounded in the uploaded files. This is useful where accuracy to source wording matters more than creative interpretation.
- `Creative`: produces freer drafting with minimal grounding. This is useful for first-draft communications or narrative concepts.

Insurance broker scenario:

- use `Factual by doc` for a compliance-sensitive benefits summary
- use `Hybrid` for an executive brief with recommendations
- use `Creative` for a first-pass client communication concept that will later be reviewed by a broker

#### Adjustable length, audience, tone, and quality

What it does:
- lets a user control how the document should read

How it works:
- users can choose page length or let the AI decide
- users can set the intended audience and tone
- users can choose `Fast` or `Thorough` quality mode

Insurance broker scenario:
- create a short executive note for leadership
- create a more thorough internal policy digest for operations
- change tone from professional client-facing language to direct internal action planning

#### One-click document presets

What it does:
- speeds up common tasks

How it works:
- the app includes ready-made bundles such as `Executive Brief`, `Policy Draft`, `Action Plan`, and `Meeting Summary`

Insurance broker scenario:
- produce a board-level brief on renewal risks
- create an internal action plan after an insurer service issue
- summarise a benefits vendor meeting for client managers

#### Charts and flowcharts in generated documents

What it does:
- adds optional visual elements to long-form outputs

How it works:
- when enabled, the system can create grounded charts and an optional flowchart for the generated document

Insurance broker scenario:
- include simple visuals in a report showing issue categories, process stages, or document-derived trends for stakeholder presentations

#### Structured JSON generation

What it does:
- turns unstructured or semi-structured content into a known JSON format

How it works:
- the user selects a JSON template file
- the user chooses the target location inside that template
- the app fills that structure with generated or extracted items

This supports two modes:

- `Standard Generation`: creates a chosen number of items to fit the template
- `Exhaustive Extraction`: works through all indexed chunks and extracts structured items from every relevant document segment

Insurance broker scenario:
- convert mixed insurer documents into a standard internal format for downstream review
- populate a structured FAQ object from a benefits communication pack
- extract question-and-answer pairs from policy and enrollment material for chatbot or portal prototypes

#### Export of generated documents and JSON

What it does:
- saves file-based outputs in shareable forms

How it works:
- long-form documents can be exported to PDF and DOCX
- structured JSON results can be exported to a JSON file

Insurance broker scenario:
- give a client-facing PDF summary to account managers
- provide a DOCX draft to compliance or marketing for editing
- send JSON to a prototype portal, workflow engine, or internal analytics process

#### Preset task library

What it does:
- stores repeatable instructions for common tasks

How it works:
- users can save, reuse, and delete preset prompts for file-based work

Insurance broker scenario:
- save common instructions such as "Summarise key renewal changes", "Extract action items, owners, and deadlines", and "Draft a client update email from uploaded files"

## 3. Reporting, Monitoring, and Configuration Features

These features support day-to-day usability and operational control.

#### Save and load configuration

What it does:
- saves the setup for reuse

How it works:
- a user can save the current configuration and reload it later

Insurance broker scenario:
- keep a reusable setup for benefits-email analysis, one for renewal brief generation, and one for synthetic policy data creation

#### Usage and cost estimation

What it does:
- gives visibility into AI usage

How it works:
- the app tracks prompt and completion tokens
- it estimates cost based on configurable pricing
- it also estimates savings from fields generated without AI

Insurance broker scenario:
- compare the likely cost of generating a large synthetic dataset versus using deterministic placeholder fields where suitable

#### Progress and status logging

What it does:
- shows what the system is doing during a run

How it works:
- users can see logs, progress updates, completion status, and basic retrieval metrics for file-based tasks

Insurance broker scenario:
- support a live demo for stakeholders where transparency of process matters

## Detailed Insurance Broker Use Cases

The following examples show how the application could be used in a broker environment.

### Use case 1: Synthetic employee-benefits service inbox

Goal:
- create a safe sample dataset that looks like a real benefits team inbox

How it would be used:
- define or auto-generate columns for sender details, scheme type, issue category, urgency, and email body
- generate realistic but non-live examples of enrollment questions, policy clarification requests, payroll deduction issues, dependent updates, and claim-support requests

Business value:
- training
- workflow testing
- service-model design
- demo data for client presentations

### Use case 2: Enrichment of broker communication logs

Goal:
- add classification and triage fields to an existing file

How it would be used:
- import a CSV of broker or benefits communications
- generate extra columns such as sentiment, issue type, probable owner, insurer involved, or recommended next action

Business value:
- faster triage
- operational insights
- better prioritisation of client queries

### Use case 2A: Bulk analysis of imported email or communication exports

Goal:
- analyse a large existing file row by row rather than manually reviewing each communication

How it would be used:
- import a CSV export containing hundreds or thousands of emails, notes, or service interactions
- keep the original columns such as sender, subject, date, and message body
- add AI-generated columns for sentiment, category, complaint type, policy theme, product line, urgency, or recommended next action
- export the enriched file for review in Excel, BI tooling, or another workflow

Business value:
- converts unstructured communication history into structured management information
- helps identify recurring issues across schemes, insurers, or benefit types
- supports service quality reviews and operational decision-making

### Use case 3: Renewal and policy pack summarisation

Goal:
- turn multiple documents into a stakeholder-ready summary

How it would be used:
- upload insurer renewal documents, pricing summaries, benefit guides, and email commentary
- run the document engine in `Hybrid` or `Factual by doc` mode
- export a PDF or DOCX summary

Business value:
- reduces manual reading time
- accelerates internal review
- helps account managers prepare client conversations

### Use case 4: File-based question answering

Goal:
- locate answers across a mixed set of files without reading each one manually

How it would be used:
- upload policy documents, plan summaries, and support emails
- ask questions in plain language
- review the cited answers and source references

Business value:
- faster fact-finding
- less dependence on one individual remembering where a rule or clause was stored

### Use case 5: Standardising unstructured content into JSON

Goal:
- convert messy source material into a repeatable structured output

How it would be used:
- choose a JSON template for a broker intake form, FAQ library, or issue register
- run standard generation or exhaustive extraction
- export the finished JSON for reuse elsewhere

Business value:
- useful for portal prototypes
- useful for knowledge-base seeding
- useful for integrating unstructured documents into future workflow tools

### Use case 6: Executive brief for leadership or clients

Goal:
- create a concise document from multiple source files

How it would be used:
- apply the `Executive Brief` preset
- set audience to leadership, client stakeholders, or operations
- generate a short document with key findings, risks, and next steps

Business value:
- speeds up preparation for renewal meetings, service reviews, and scheme-change discussions

### Use case 7: Action planning after scheme or policy changes

Goal:
- turn a set of source documents into a follow-up plan

How it would be used:
- upload revised insurer documents, meeting notes, and internal emails
- use the `Action Plan` preset
- generate a phase-based summary of tasks, owners, milestones, and required communications

Business value:
- supports implementation planning after benefit changes, new insurer onboarding, or process updates

## End-to-End Example Journeys

### Journey A: Create a training dataset for a benefits support team

1. Connect to an AI model.
2. Use plain English to describe the target dataset.
3. Review the proposed columns.
4. Generate rows.
5. Analyse diversity and repetition.
6. Export as CSV or PDF narrative.

### Journey A2: Analyse thousands of imported emails line by line

1. Export emails or service interactions into CSV format.
2. Import the file into the `Data Generation` workspace.
3. Preserve the original columns as imported data.
4. Add new AI-generated analysis columns such as sentiment, category, priority, and recommended owner.
5. Run enrichment so the AI processes the imported file row by row.
6. Export the enriched results for triage, reporting, or dashboarding.

### Journey B: Analyse a bundle of insurer and benefits documents

1. Switch to the `Files` workspace.
2. Import PDFs, spreadsheets, emails, or URLs.
3. Ask questions in `Quick Q&A` mode to test the knowledge base.
4. Switch to `Document Engine`.
5. Generate an executive brief or action plan.
6. Export to PDF or DOCX.

### Journey C: Turn policy documents into structured output

1. Import the source files.
2. Select `Structured JSON`.
3. Choose a JSON template and target key.
4. Run standard generation or exhaustive extraction.
5. Review the preview.
6. Export the JSON result.

## Why This Matters for an Insurance Broker

For a broker dealing with multiple insurance lines and employee-benefit schemes, the main value is not one single feature. It is the combination of:

- document understanding
- grounded answer generation
- structured data creation
- synthetic data generation
- row-by-row analysis of imported datasets
- rapid report drafting

Together, these capabilities can support both front-office and back-office work, including:

- client servicing
- operations support
- document review
- training and onboarding
- prototype knowledge management
- controlled AI experimentation without relying on live customer records

## Strengths

- broad input support across tabular data and documents
- useful for both structured and unstructured work
- can operate with local AI models
- supports scanned documents through OCR
- provides citations for file-based answers
- includes export options that are easy to share with business users
- can create representative datasets without needing live member data

## Important Boundaries and Caveats

The application should be described accurately to business stakeholders, with clear boundaries around what it does and does not do.

- It is not a full policy administration platform.
- It is not a claims system.
- It is not a compliance approval engine.
- Outputs still require business review, especially for regulated insurance content.
- Grounded answers depend on the quality and completeness of uploaded files.
- Synthetic data is representative, not a substitute for actuarial or regulatory source data.
- Creative document modes are suitable for drafting, not final sign-off.

For insurer and employee-benefit use, the safest positioning is:

- a productivity and knowledge-assistance layer
- a document and data synthesis tool
- a prototype foundation for future broker workflows

## Suggested Positioning for Stakeholder Demos

When presenting this project to an insurance broker audience, it can be framed as:

"A configurable AI workbench that helps broker teams generate safe sample data, understand complex document packs, answer questions from imported files, and produce structured or narrative outputs for insurance and employee-benefit workflows."

## Conclusion

This project already demonstrates meaningful value for insurance brokers and employee-benefits teams. Its strongest value is in reducing manual effort across document-heavy and communication-heavy processes while still allowing users to keep control over prompts, files, outputs, and review.

For a broker managing a broad range of insurance and benefits schemes, the application can support early automation in:

- scheme communication analysis
- large-scale CSV enrichment and classification
- policy and renewal document review
- executive reporting
- structured extraction
- internal knowledge support
- safe synthetic data generation

In short, it is best understood as a flexible AI-assisted operations and document intelligence tool that can be shaped around broking and benefits use cases.
