# Synthetic Data Generator (Local LLM)

A modern desktop application for generating synthetic tabular data using local Large Language Models (via LM Studio). Designed for privacy, modularity, and strict semantic uniqueness.

## 🚀 What This App Does

This application solves the problem of generating high-quality, privacy-safe synthetic data for testing and development. Unlike simple random generators, it uses an LLM to create context-aware content (e.g., "A positive review for a blender") while enforcing strict data integrity rules.

### Key Features
*   **Modern UI**: Built with **Flet** (Flutter for Python) featuring a responsive layout, dark mode, and intuitive controls.
*   **Local LLM Integration**: Connects to LM Studio (`localhost:1234`), ensuring no data leaves your machine.
*   **Strict Uniqueness**:
    *   **Short Text**: Enforces exact match uniqueness (SHA256).
    *   **Short Text**: Enforces exact match uniqueness (SHA256).
    *   **Long Text**: Uses `sentence-transformers` to calculate semantic similarity. Rejects generated text that is too similar (e.g., >85%) to existing rows, preventing repetitive AI outputs.
*   **Prompt Engineering with LangChain**: Uses LangChain for structured schema generation and robust prompt management.
*   **Dynamic Configuration**: Users can define columns, types, and prompts at runtime.
*   **Resilient Generation**: Automatically retries failed or duplicate generations up to a configured limit before skipping.
*   **Configuration Persistence**: Save and load generation schemas to JSON files for easy reuse.
*   **User-Friendly Logic**: Support for Natural Language constraints (e.g., "after @[Date]") and Regex shortcuts ("phone", "email").
*   **In-App Cookbook**: Built-in documentation with real-world examples to get you started fast.

## 🗺️ Code Map for Agents & Developers

This project is structured to separate concerns between the UI, the orchestration logic, and the external AI integration.

### Directory Structure

```
synthetic_data_gen/
├── core/                 # Business logic and backend systems
│   ├── controller.py     # ORCHESTRATOR: Manages the generation loop
│   ├── llm_client.py     # INTEGRATION: Wrapper for LM Studio API
│   ├── models.py         # DATA: Pydantic models (Schema definitions)
│   └── validator.py      # LOGIC: Uniqueness and semantic checks
├── gui/                  # User Interface
│   └── flet_app.py       # UI: Main Flet application logic
├── main.py               # BOOTSTRAP: App execution entry point
├── README.md             # DOCS: This file
└── requirements.txt      # DEPS: Python dependencies
```

### Key Components & Responsibilities

#### `core/controller.py` - `GeneratorController`
*   **Role**: The "Brain" of the application.
*   ** Responsibilities**:
    *   Initialize LLM and Validator concepts.
    *   Run the generation loop (threaded).
    *   Construct prompts for each column based on constraints.
    *   Handle the retry logic when the Validator rejects a value.
    *   Export data to CSV.
*   **Key Methods**: `generate_row()`, `start_generation_thread()`.

#### `core/validator.py` - `UniquenessValidator`
*   **Role**: The "Gatekeeper" ensuring data quality.
*   **Responsibilities**:
    *   Maintain a history of generated hashes (for exact match).
    *   Maintain a history of long-text values.
    *   Load `sentence-transformers` model (`all-MiniLM-L6-v2`) for local embedding.
    *   `is_unique(text)`: Returns `False` if text is a duplicate or semantically too similar.

#### `core/llm_client.py` - `LLMClient`
*   **Role**: The "Connector" to the AI.
*   **Responsibilities**:
    *   Abstract the `openai` library calls.
    *   `list_models()`: Fetch available models from LM Studio.
    *   `generate_completion()`: Send prompts and return raw text.

#### `gui/flet_app.py` - `FletApp`
*   **Role**: The User Interface.
*   **Responsibilities**:
    *   Renders the modern Flet UI with responsive cards and layout.
    *   Manage `ColumnControl` list (add/remove inputs).
    *   **Threading**: Receives updates from the background Controller thread via `queue` (polling with `page.run_task` or timer) to update the UI without freezing.
    *   **Model Selection**: Dynamically fetches model list from `LLMClient`.
    *   **File Operations**: Handles Save/Load/Import/Export using native system file dialogs.

---

## 📅 Roadmap & Completed Features

Plans for enhancements to make this tool a production-grade asset.

### Phase 2: Usability & Persistence (Recommended Skills: python-pro, ui-ux-designer)
- [x] **Template System**:
    -   Add "Save Configuration" and "Load Configuration" buttons.
    -   Persist column definitions and prompts to JSON files.
    -   Allow users to build a library of reusable schemas (e.g., "User Profile", "Transaction Log").
- [x] **Modern UI Rewrite**:
    -   Migrated from Tkinter to Flet for a polished, responsive interface.
    -   Implemented dark mode and card-based layout.

### Phase 3: Advanced Context & Prompting (Recommended Skills: prompt-engineering-patterns, python-pro)
- [x] **Inter-Column Dependencies**:
    -   Enable column prompts to reference values from previously generated columns using a syntax like `@[column_name]`.
    -   **Example**:
        -   Column 1 (`animal`): "Generate an animal species that lives in Africa."
        -   Column 2 (`lifespan`): "How long does @[animal] live?"
        -   Column 3 (`analysis`): "Can @[animal] who lives @[lifespan] live longer elsewhere?"
    -   **Requirement**: Implement dependency graph resolution to prioritize column generation order to ensure dependencies are resolved before generation.

### Phase 4: Advanced Validation & Logic (Recommended Skills: sql-pro, python-testing-patterns)
- [x] **Regex Constraints**: Add regex validation to `validator.py`.
- [x] **Logic Constraints**: Support cross-column logic (e.g., `End Date` > `Start Date`).
- [x] **Max Consecutive Rejections**: Configurable limit for consecutive row rejections (e.g., stop after 50 failures) to prevent infinite loops during validation.
- [x] **Per-Column Similarity**: Allow custom similarity thresholds per column to accommodate interaction limits (e.g., looser checks for localized answers).
- [x] **New Export Formats**: Support JSON export and SQL `INSERT` statement generation.

### Phase 5: Hybrid Engine (Performance) (Recommended Skills: python-performance-optimization)
- [x] **Faker Integration**:
    -   Add a new `ColumnType`: `DETERMINISTIC`.
    -   Integrate `Faker` library for fields like Names, Emails, Addresses, and Dates.
    -   **Benefit**: drastically faster generation and zero token usage for simple fields.

### Phase 6: AI-Assisted Configuration (Recommended Skills: prompt-engineering, llm-app-patterns)
- [x] **"Magic" Schema Generator**:
    -   Add a text input for high-level intent (e.g., *"Make a dataset for a customer support ticketing system with priorities and sentiment"*).
    -   Use the LLM to automatically generate the `ColumnDefinitions` and `PromptInstructions` based on this request.
    -   **Context-Aware**: The schema generation agent must be prompted to utilize the `@[column_name]` syntax to create meaningful dependencies between columns where appropriate (e.g., automatically linking a "Country" column to a "City" column).

### Phase 7: Data Import & Enrichment (Recommended Skills: data-engineer, rag-implementation)
- [x] **Import Existing Data**:
    -   Support importing CSV and JSON files via `pandas`.
    -   **Schema Extraction Mode**: Automatically creates schema columns from imported file headers.
    -   **Augmentation Mode**: Keep imported rows as basic context, allowing new AI-generated columns to reference imported values (e.g., generate a bio for an imported name).

### Phase 8: Data Quality & Reporting (Recommended Skills: pdf, data-scientist, data-storytelling)
- [x] **Data Quality Metrics**:
    -   **Logic**: Calculate stats post-generation to score dataset health.
        -   *Diversity Score*: Ratio of unique values to total rows.
        -   *Redundancy Check*: Identify top 5 frequent values to spot repetition.
        -   *Semantic Spread*: Use embeddings to measure how "different" the text outputs are from each other.
- [x] **PDF Export**:
    -   Generate a visual PDF report summarizing the Data Quality Metrics.
    -   **Narrative Report Mode**: Option to export data in a document structure (not a table). Each row becomes a titled section with paragraphs, suitable for research/logic outputs.

### Phase 9: User Experience & Help (Recommended Skills: technical-writer, ui-ux-pro-max)
- [x] **In-App Documentation**:
    -   Added a dedicated "Docs / Help" window with tabs for Basics, Types, and Advanced features.
    -   **Cookbook**: Includes a gallery of real-world examples (e.g., Inventory, Logistics) for copy-pasting.
- [x] **Natural Language Logic**:
    -   Users can write constraints in English (e.g., `after @[Date]`, `longer than 5`) instead of Python syntax.
- [x] **Regex Shortcuts**:
    -   Simplified validation with presets: `phone`, `email`, `zip`, `date`, `ipv4`.

---

## ⚡ Quick Start

### Prerequisites
1.  **LM Studio**: Installed and running a local server (`http://localhost:1234/v1`).
2.  **Python 3.10+**.

### Installation
```bash
pip install -r requirements.txt
```

### Running
```bash
python main.py
```

### AI Configuration
The app supports multiple AI providers:
1. **LM Studio (Local)**: Runs locally on your machine (default, `http://localhost:1234/v1`)
2. **OpenAI**: Requires API key from [platform.openai.com](https://platform.openai.com)
3. **Google Gemini**: Requires API key from [ai.google.dev](https://ai.google.dev)
4. **OpenRouter**: Requires API key from [openrouter.ai](https://openrouter.ai)
5. **GitHub Models**: Requires GitHub token with access to [GitHub Models](https://github.com/marketplace/models)
6. **Azure OpenAI**: Requires API key, Azure endpoint, and deployment name from [Azure OpenAI Service](https://azure.microsoft.com/products/ai-services/openai-service)

To configure:
1. Open the app and locate the **"🤖 AI Configuration"** section
2. Select your provider from the dropdown
3. Enter your API key (if using a cloud provider)
4. **For Azure OpenAI only**: Enter your Azure endpoint (e.g., `https://your-resource.openai.azure.com`) and deployment name (e.g., `gpt-4`, `gpt-35-turbo`)
5. Click **"Test Connection"** to verify
6. Choose a model from the **"Model ID"** dropdown (not needed for Azure - uses deployment name)

### Using the "Magic Generator"
1.  Ensure you have a model loaded in **LM Studio** and the local server is running (`Start Server`).
2.  In the app, look for the **"✨ Magic Generator"** section.
3.  Type a description of the dataset you want (e.g., *"A list of sci-fi planets with names, inhabitants, and climate"*).
4.  Click **"Auto-Generate Schema"**.
5.  The app will automatically populate the columns and prompts for you!
