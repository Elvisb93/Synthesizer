import sys
import logging

def verify():
    print("Verifying imports...")
    try:
        import tkinter
        from openai import OpenAI
        from pydantic import BaseModel
        import numpy
        print("Standard dependencies OK.")
    except ImportError as e:
        print(f"MISSING DEPENDENCY: {e}")
        return

    try:
        from core.models import GeneratorConfig, ColumnDefinition, ColumnType
        from core.llm_client import LLMClient
        from core.validator import UniquenessValidator
        from core.controller import GeneratorController
        print("Internal modules import OK.")
        
        # Test Validator init (checks sentence-transformers)
        config = GeneratorConfig(model_id="test")
        validator = UniquenessValidator(config)
        print(f"Validator initialized. HAS_SENTENCE_TRANSFORMERS: {validator.HAS_SENTENCE_TRANSFORMERS if hasattr(validator, 'HAS_SENTENCE_TRANSFORMERS') else 'Unknown'}")
        
    except ImportError as e:
        print(f"INTERNAL IMPORT ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
