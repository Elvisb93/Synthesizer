
import json
from core.models import ColumnDefinition, ColumnType, ColumnConstraints, GeneratorConfig
from core.controller import GeneratorController

def test_circular_dependency():
    print("--- Testing Schema Validation ---")
    
    # Load JSON
    with open('test.json', 'r') as f:
        data = json.load(f)
        
    cols_data = data.get('columns', [])
    columns = []
    
    # Convert JSON to ColumnDefinition objects
    for c in cols_data:
        # Map string type to Enum
        ctype_str = c['type']
        # Simple lookup for this test
        for member in ColumnType:
            if member.value == ctype_str:
                ctype = member
                break
        
        constraints = ColumnConstraints(**c['constraints'])
        col = ColumnDefinition(
            name=c['name'],
            type=ctype,
            prompt_instruction=c['prompt_instruction'],
            constraints=constraints
        )
        columns.append(col)
        
    controller = GeneratorController()
    
    try:
        # This calls _get_execution_order which performs the topological sort
        controller.initialize(GeneratorConfig(model_id="test"), columns)
        print("✅ Schema loaded successfully (No cycles detected).")
    except ValueError as e:
        print(f"❌ Validation Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

if __name__ == "__main__":
    test_circular_dependency()
