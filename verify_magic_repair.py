
import json
import logging
from unittest.mock import MagicMock
from core.llm_client import LLMClient
from core.models import GeneratorConfig, AIProvider

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_magic_repair():
    print("--- Starting Magic Repair Verification ---")
    
    # Mock Config
    config = GeneratorConfig(
        provider=AIProvider.LM_STUDIO,
        api_key="test",
        model_id="test-model"
    )
    
    client = LLMClient(config)
    
    # Bad Schema (Missing prompt for 'role' which is Categorical)
    bad_schema = [
        {"name": "id", "type": "Auto Increment (ID)", "prompt_instruction": "", "constraints": {}},
        {"name": "role", "type": "Categorical", "prompt_instruction": "", "constraints": {"options": ["Admin", "User"]}},
        {"name": "bio", "type": "Short Text", "prompt_instruction": "A short bio", "constraints": {}}
    ]
    
    # Good Schema (Fixed)
    good_schema = [
        {"name": "id", "type": "Auto Increment (ID)", "prompt_instruction": "", "constraints": {}},
        {"name": "role", "type": "Categorical", "prompt_instruction": "Select a role randomly", "constraints": {"options": ["Admin", "User"]}},
        {"name": "bio", "type": "Short Text", "prompt_instruction": "A short bio", "constraints": {}}
    ]
    
    # Mock generate_completion to return bad first, then good
    client.generate_completion = MagicMock(side_effect=[
        json.dumps(bad_schema),
        json.dumps(good_schema)
    ])
    
    print("Invoking generate_schema...")
    result = client.generate_schema("Create a user table with roles")
    
    print("\n--- Result ---")
    print(json.dumps(result, indent=2))
    
    # Assertions
    print("\n--- Verification ---")
    if client.generate_completion.call_count == 2:
        print("✅ SUCCESS: LLM was called twice (Repair loop triggered).")
    else:
        print(f"❌ FAILURE: LLM called {client.generate_completion.call_count} times. Expected 2.")
        
    # Check if repair prompt contained the bad json
    call_args = client.generate_completion.call_args_list
    if len(call_args) >= 2:
        second_call_prompt = call_args[1][0][0] # args[0] is prompt
        if "Here is the incomplete JSON you generated" in second_call_prompt:
             print("✅ SUCCESS: Repair prompt contained the context of bad JSON.")
        else:
             print("❌ FAILURE: Repair prompt did NOT contain bad JSON context.")
             print(f"Prompt was: {second_call_prompt}")

    if result[1]['prompt_instruction'] == "Select a role randomly":
        print("✅ SUCCESS: Final schema has corrected prompt.")
    else:
        print("❌ FAILURE: Final schema missing prompt.")

if __name__ == "__main__":
    test_magic_repair()
