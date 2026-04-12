"""
Core business logic package for Synthetic Data Generator.

Public API exports for models, controller, and utilities.
"""
from .models import (
    ColumnType,
    AIProvider,
    RagProvider,
    RagBackend,
    OcrMode,
    ColumnConstraints,
    ColumnDefinition,
    RagConfig,
    GeneratorConfig,
    RowData,
    FAKER_PROVIDERS
)
from .controller import GeneratorController
from .llm_client import LLMClient
from .validator import UniquenessValidator
from .analytics import QualityAnalyzer
from .rag import RagService
from .json_parser import (
    load_template,
    resolve_target_array,
    infer_item_schema,
    inject_item,
    clear_target_array,
    export_template,
)

__all__ = [
    # Models
    "ColumnType",
    "AIProvider",
    "RagProvider",
    "RagBackend",
    "OcrMode",
    "ColumnConstraints",
    "ColumnDefinition",
    "RagConfig",
    "GeneratorConfig",
    "RowData",
    "FAKER_PROVIDERS",
    # Core Classes
    "GeneratorController",
    "LLMClient",
    "UniquenessValidator",
    "QualityAnalyzer",
    "RagService",
    # JSON Template Utilities
    "load_template",
    "resolve_target_array",
    "infer_item_schema",
    "inject_item",
    "clear_target_array",
    "export_template",
]
