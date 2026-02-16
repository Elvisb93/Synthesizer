"""
Pydantic schemas for LangChain structured output parsing.

These models define the expected output format from the AI when generating
column schemas via the Magic Generator. They are separate from domain models
because they serve the AI interface, not the core business logic.
"""
from typing import List, Optional
from pydantic import BaseModel, Field

from .models import ColumnType


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
