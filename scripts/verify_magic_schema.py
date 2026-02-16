import logging
import json
from core.llm_client import LLMClient
from core.models import GeneratorConfig, ColumnType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_magic_generation():
    print("--- Starting Magic Schema Generation Test ---")
    
    # Config
    config = GeneratorConfig(model_id="local-model", api_base_url="http://localhost:1234/v1")
    client = LLMClient(config)
    
    # Check connection
    if not client.check_connection():
        print("❌ Error: Could not connect to LM Studio at http://localhost:1234/v1")
        print("Please ensure LM Studio is running and the server is started.")
        return

    print("✅ Connected to LM Studio.")
    
    # Test Prompts
    prompts = [
        "A customer support ticket system with priorities, categories, and customer sentiment.",
        "A list of RPG characters with names, classes, stats, and a brief backstory."
    ]
    
    for prompt in prompts:
        print(f"\n🔮 Testing Prompt: '{prompt}'")
        try:
            schema = client.generate_schema(prompt)
            
            if not schema:
                print("❌ Failed: Returned empty schema.")
                continue
                
            print(f"✅ Generated {len(schema)} columns:")
            for col in schema:
                print(f"  - {col.get('name')} ({col.get('type')}): {col.get('prompt_instruction')[:50]}...")
                if col.get('constraints'):
                    print(f"    Constraints: {col.get('constraints')}")
            
            # Basic Validation
            assert isinstance(schema, list)
            assert len(schema) > 0
            assert "name" in schema[0]
            assert "type" in schema[0]
            
        except Exception as e:
            print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_magic_generation()
