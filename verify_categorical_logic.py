
import json
import logging
from unittest.mock import MagicMock
from core.llm_client import LLMClient
from core.models import GeneratorConfig, AIProvider

logging.basicConfig(level=logging.INFO)

def test_categorical_logic():
    print("--- Testing Categorical Logic Verification ---")
    
    config = GeneratorConfig(provider=AIProvider.LM_STUDIO, api_key="test", model_id="test")
    client = LLMClient(config)
    
    # 1. Bad Schema: Categorical, No Options, No Link
    bad_schema = [
        {"name": "id", "type": "Auto Increment (ID)", "prompt_instruction": "", "constraints": {}},
        {"name": "category", "type": "Categorical", "prompt_instruction": "A category", "constraints": {"options": []}} 
    ]
    
    # 2. Good Schema: Fixed with Link
    good_schema = [
        {"name": "id", "type": "Auto Increment (ID)", "prompt_instruction": "", "constraints": {}},
        {"name": "category", "type": "Categorical", "prompt_instruction": "Category for @[id]", "constraints": {"options": []}}
    ]
    
    client.generate_completion = MagicMock(side_effect=[
        json.dumps(bad_schema),
        json.dumps(good_schema)
    ])
    
    print("Invoking generate_schema...")
    result = client.generate_schema("Test intent")
    
    print("\n--- Result ---")
    print(json.dumps(result, indent=2))
    
    # Assertions
    if client.generate_completion.call_count >= 2:
        print("✅ SUCCESS: Repair loop triggered for bad Categorical column.")
        
        # Check if prompt mention the specific issue
        call_args = client.generate_completion.call_args_list
        repair_prompt = call_args[1][0][0]
        if "Categorical columns MUST have either a specific list of 'options'" in repair_prompt:
             print("✅ SUCCESS: Repair prompt contains specific Categorical instructions.")
        else:
             print("❌ FAILURE: Repair prompt missing specific instructions.")
             print(repair_prompt)
    else:
        print("❌ FAILURE: Repair loop NOT triggered.")

if __name__ == "__main__":
    test_categorical_logic()
