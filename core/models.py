from enum import Enum
from typing import List, Optional, Any, Dict, Union
from pydantic import BaseModel, Field

class ColumnType(str, Enum):
    SHORT_TEXT = "Short Text"
    LONG_TEXT = "Long Text"
    NUMERIC = "Numeric"
    CATEGORICAL = "Categorical"
    BOOLEAN = "Boolean"
    AUTO_INCREMENT = "Auto Increment (ID)"
    DETERMINISTIC = "Faker / Deterministic"

class AIProvider(str, Enum):
    LM_STUDIO = "LM Studio"
    OPENAI = "OpenAI"
    GOOGLE_GEMINI = "Google Gemini"
    OPENROUTER = "OpenRouter"
    GITHUB_MODELS = "GitHub Models"
    AZURE_OPENAI = "Azure OpenAI"

FAKER_PROVIDERS = [
    "name", "email", "phone_number", "address", "city", 
    "country", "company", "job", "date_of_birth", 
    "credit_card_number", "ipv4", "user_agent", "uuid4", 
    "ean", "sentence", "text"
]

class ColumnConstraints(BaseModel):
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    options: List[str] = Field(default_factory=list) # For Categorical
    min_length: int = 10 # For LongText, strictly enforced?
    max_length: int = 2000 
    regex_pattern: Optional[str] = None # Phase 4: Regex validation
    expression: Optional[str] = None # Phase 4: Cross-column logic (e.g. "this > @[age]")
    similarity_threshold: Optional[float] = None # Phase 4: Per-column override
    faker_provider: Optional[str] = None # Phase 5: Faker provider name (e.g. 'email')
    allow_duplicates: bool = False # New: Allow duplicates for this column

class ColumnDefinition(BaseModel):
    name: str
    type: ColumnType
    constraints: ColumnConstraints = Field(default_factory=ColumnConstraints)
    prompt_instruction: str = "" # Specific instruction for this column

class GeneratorConfig(BaseModel):
    model_id: str
    provider: AIProvider = AIProvider.LM_STUDIO  # New: Provider selection
    api_key: Optional[str] = None  # New: API key for cloud providers
    api_base_url: str = "http://localhost:1234/v1"  # Will be overridden based on provider
    # Azure-specific fields
    azure_endpoint: Optional[str] = None  # Azure OpenAI endpoint
    azure_deployment: Optional[str] = None  # Azure deployment name
    num_rows: int = 500
    similarity_threshold: float = 0.85 # 0.0 to 1.0, higher means stricter uniqueness (less similar allowed)
    max_retries: int = 50 # Phase 4: Configurable retry limit
    existing_data: Optional[List[Dict[str, Any]]] = None # Phase 7: Imported data for enrichment
    
    # Token Pricing (Phase 10: Metrics)
    input_price_per_1m: float = 0.15  # Default: ~$0.15 per 1M tokens (approx GPT-4o-mini input)
    output_price_per_1m: float = 0.60 # Default: ~$0.60 per 1M tokens (approx GPT-4o-mini output)
    
class RowData(BaseModel):
    data: Dict[str, Any]
    
    def get_value(self, column_name: str) -> Any:
        return self.data.get(column_name)

# --- Pydantic Models for LangChain Output Parsing (Moved from llm_client.py) ---
class ColumnConstraintsSchema(BaseModel):
    min_value: Optional[float] = Field(None, description="Minimum value for numeric columns")
    max_value: Optional[float] = Field(None, description="Maximum value for numeric columns")
    options: List[str] = Field(default_factory=list, description="List of options for Categorical columns")
    min_length: int = Field(10, description="Minimum length for text columns")
    max_length: int = Field(2000, description="Maximum length for text columns")
    regex_pattern: Optional[str] = Field(None, description="Regex pattern for validation")
    allow_duplicates: bool = Field(False, description="Set to True for low-cardinality fields or if repetition is expected")
    faker_provider: Optional[str] = Field(None, description="Faker provider if type is Deterministic")

class ColumnSchema(BaseModel):
    name: str = Field(description="Name of the column")
    type: ColumnType = Field(description="Type of the column")
    prompt_instruction: str = Field(description="Instruction for generation. Use @[col] for dependencies.")
    constraints: ColumnConstraintsSchema = Field(default_factory=ColumnConstraintsSchema, description="Constraints for the column")

class Schema(BaseModel):
    columns: List[ColumnSchema] = Field(description="List of column definitions")
