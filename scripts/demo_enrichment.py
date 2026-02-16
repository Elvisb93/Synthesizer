import pandas as pd
from core.models import GeneratorConfig, ColumnDefinition, ColumnType
from core.controller import GeneratorController
from unittest.mock import MagicMock

def run_demo():
    print("=== Data Enrichment Demo ===")
    
    # 1. Create a dummy CSV
    csv_file = "demo_input.csv"
    data = [
        {"Product": "Gaming Laptop", "Price": 1500},
        {"Product": "Mechanical Keyboard", "Price": 100},
        {"Product": "Wireless Mouse", "Price": 50}
    ]
    pd.DataFrame(data).to_csv(csv_file, index=False)
    print(f"1. Created '{csv_file}' with {len(data)} rows.")

    # 2. Simulate Import
    print("2. Importing data...")
    df = pd.read_csv(csv_file)
    existing_data = df.to_dict(orient='records')
    
    # 3. Configure Controller
    print("3. Configuring Generator...")
    config = GeneratorConfig(
        model_id="local-model",
        existing_data=existing_data
    )
    
    # Define Columns: 2 Imported, 1 Generated
    columns = [
        ColumnDefinition(name="Product", type=ColumnType.SHORT_TEXT, prompt_instruction="(Imported)"),
        ColumnDefinition(name="Price", type=ColumnType.NUMERIC, prompt_instruction="(Imported)"),
        # The AI Column
        ColumnDefinition(
            name="Marketing_Slogan", 
            type=ColumnType.SHORT_TEXT, 
            prompt_instruction="Write a catchy 3-word slogan for @[Product] that costs @[Price]."
        )
    ]
    
    controller = GeneratorController()
    controller.initialize(config, columns)
    
    # MOCK the LLM to avoid needing a local server running for this quick demo
    # If you have a server running, you can comment this out!
    print("   (Mocking LLM for speed/reliability in demo)")
    controller.llm_client = MagicMock()
    def mock_gen(prompt):
        if "Gaming Laptop" in prompt: return "Power Overwhelming Now"
        if "Mechanical Keyboard" in prompt: return "Click Clack Boom"
        if "Wireless Mouse" in prompt: return "Cut The Cord"
        return "Buy This Thing"
    controller.llm_client.generate_completion.side_effect = mock_gen
    
    # 4. Run Generation
    print("4. Running Enrichment...")
    controller._run_generation_loop()
    
    # 5. Show Results
    print("\n=== Results ===")
    for i, row in enumerate(controller.generated_rows):
        print(f"Row {i+1}: {row.data}")
        
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    run_demo()
