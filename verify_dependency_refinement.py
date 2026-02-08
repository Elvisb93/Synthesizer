
import json
import logging
from unittest.mock import MagicMock
from core.llm_client import LLMClient
from core.models import GeneratorConfig, AIProvider

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dependency_refinement():
    print("--- Starting Dependency Refinement Verification ---")
    
    # Mock Config
    config = GeneratorConfig(
        provider=AIProvider.LM_STUDIO,
        api_key="test",
        model_id="test-model"
    )
    
    client = LLMClient(config)
    
    # Initial Schema (Unlinked)
    initial_schema = [
        {"name": "product_name", "type": "Short Text", "prompt_instruction": "A product name", "constraints": {}},
        {"name": "category", "type": "Categorical", "prompt_instruction": "A product category", "constraints": {"options": ["Electronics", "Home"]}}
    ]
    
    # Refined Schema (Linked with @[product_name])
    refined_schema = [
        {"name": "product_name", "type": "Short Text", "prompt_instruction": "A product name", "constraints": {}},
        {"name": "category", "type": "Categorical", "prompt_instruction": "Category for @[product_name]", "constraints": {"options": ["Electronics", "Home"]}}
    ]
    
    # Mock generate_completion
    # Call 1: Generate Initial Schema
    # Call 2: Refine Dependencies
    client.generate_completion = MagicMock(side_effect=[
        json.dumps(initial_schema),
        json.dumps(refined_schema)
    ])
    
    print("Invoking generate_schema...")
    result = client.generate_schema("Create products with categories")
    
    print("\n--- Result ---")
    print(json.dumps(result, indent=2))
    
    # Assertions
    print("\n--- Verification ---")
    
    # Check calls
    print(f"LLM Call Count: {client.generate_completion.call_count}")
    if client.generate_completion.call_count == 2:
        print("✅ SUCCESS: LLM was called twice (Initial + Refinement).")
    else:
        print(f"❌ FAILURE: Expected 2 calls, got {client.generate_completion.call_count}.")

    # Check prompts
    call_args = client.generate_completion.call_args_list
    if len(call_args) >= 2:
        refine_prompt = call_args[1][0][0]
        if "Analyze the relationships" in refine_prompt:
             print("✅ SUCCESS: Refinement prompt detected.")
        else:
             print("❌ FAILURE: Wrong prompt for second call.")
             
    # Check Result
    final_prompt = result[1]['prompt_instruction']
    if "@[product_name]" in final_prompt:
        print("✅ SUCCESS: Final schema contains dependency link.")
    else:
        print(f"❌ FAILURE: Dependency link missing in: '{final_prompt}'")

if __name__ == "__main__":
    test_dependency_refinement()
