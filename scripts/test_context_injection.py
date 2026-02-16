
import logging
import json
import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

from core.llm_client import LLMClient
from core.models import GeneratorConfig, AIProvider
from core.schema_agent import create_schema_generator_graph
from core.row_agent import create_row_generator_graph

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_full_flow():
    print("--- 1. INITIALIZATION ---")
    config = GeneratorConfig(
        model_id="gpt-oss-20b",
        provider=AIProvider.LM_STUDIO, 
        api_key=None
    )
    
    try:
        client = LLMClient(config)
        if not client.check_connection():
            print("ERROR: Could not connect to Local LLM at http://localhost:1234/v1")
            return
    except Exception as e:
        print(f"ERROR: {e}")
        return

    # --- PART A: Magic Generator ---
    print("\n--- 2. MAGIC GENERATION (Schema) ---")
    schema_graph = create_schema_generator_graph(client)
    
    # Prompt specifically designed to test context: "City" and "Country"
    prompt = "Generate a simple dataset with City and Country columns"
    print(f"Prompt: '{prompt}'")
    
    final_state = schema_graph.invoke({"user_intent": prompt})
    schema = final_state.get('schema')
    
    if not schema:
        print("FAILED: No schema generated.")
        return

    print("Generated Schema Columns:")
    for col in schema.columns:
        print(f" - {col.name} ({col.type})")
        # Ensure correct order for test: force City before Country if needed
        # (Though usually LLM generates logical order, or we rely on insertion order)

    # --- PART B: Row Generation (Context Test) --- 
    print("\n--- 3. ROW GENERATION (Context Test) ---")
    row_graph = create_row_generator_graph(client)
    
    # We will generate 3 rows to see if Country matches City
    for i in range(1, 4):
        print(f"\nGenerating Row {i}...")
        
        # Blank state
        row_state = {
            "row_data": {},
            "columns": schema.columns,
            "errors": [],
            "attempt_count": 0,
            "is_valid": False
        }
        
        result = row_graph.invoke(row_state)
        row = result['row_data']
        print(f"Refult: {json.dumps(row, indent=2)}")
        
        # Verify Context
        city = row.get("City", "Unknown")
        country = row.get("Country", "Unknown")
        print(f"Check: Is {city} in {country}?")

if __name__ == "__main__":
    test_full_flow()
