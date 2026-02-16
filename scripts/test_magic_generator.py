
import logging
import json
import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

from core.llm_client import LLMClient
from core.models import GeneratorConfig, AIProvider
from core.schema_agent import create_schema_generator_graph

# Configure Logging to see Hygiene messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_magic_generation():
    print("Initializing LLM Client...")
    # Using local-model defaults
    config = GeneratorConfig(
        model_id="gpt-oss-20b", # Adjust if your local model ID is different
        provider=AIProvider.LM_STUDIO, 
        api_key=None
    )
    
    try:
        client = LLMClient(config)
        if not client.check_connection():
            print("ERROR: Could not connect to Local LLM at http://localhost:1234/v1")
            return
    except Exception as e:
        print(f"ERROR initializing client: {e}")
        return

    print("Creating Schema Generator Graph...")
    graph = create_schema_generator_graph(client)
    
    prompt = "Generate an amazon customer database"
    print(f"Invoking graph with prompt: '{prompt}'...")
    
    try:
        final_state = graph.invoke({"user_intent": prompt})
        schema = final_state.get('schema')
        
        if not schema:
            print("FAILED: No schema generated.")
            print(f"Error Content: {final_state.get('error')}")
            return

        print("\n--- GENERATED SCHEMA ---")
        print(json.dumps(schema.model_dump(), indent=2))
        print("------------------------\n")
        
        # --- VERIFICATION CHECKS ---
        print("Running Verification Checks...")
        failures = []
        
        for col in schema.columns:
            # Check 1: Min Length
            if col.type != "Long Text":
                if col.constraints.min_length and col.constraints.min_length > 1:
                     failures.append(f"FAIL: Column '{col.name}' has min_length={col.constraints.min_length} (Expected 0 or 1)")
            
            # Check 2: Categorical Options
            if col.type == "Categorical":
                if not col.constraints.options:
                    failures.append(f"FAIL: Categorical Column '{col.name}' has NO options.")
            
            # Check 3: Uniqueness
            # Heuristic: Gender/Country should allow duplicates
            if col.name.lower() in ["gender", "country", "state", "city"]:
                if not col.constraints.allow_duplicates:
                    failures.append(f"FAIL: Column '{col.name}' has allow_duplicates=False (Expected True)")

        if failures:
            print("\n❌ VERIFICATION FAILED:")
            for f in failures:
                print(f"  - {f}")
        else:
            print("\n✅ VERIFICATION PASSED! All hygiene checks succeeded.")

    except Exception as e:
        print(f"CRASHED during execution: {e}")

if __name__ == "__main__":
    test_magic_generation()
