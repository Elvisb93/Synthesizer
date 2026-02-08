import logging
import time
import sys
from core.models import GeneratorConfig, ColumnDefinition, ColumnType, ColumnConstraints
from core.controller import GeneratorController

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_generation():
    print("Initializing Controller...")
    controller = GeneratorController()
    
    # Configuration matches user request
    config = GeneratorConfig(
        model_id="gpt-oss-20b", # Corrected model ID
        api_base_url="http://localhost:1234/v1",
        num_rows=3, # Small batch for testing
        similarity_threshold=0.85
    )
    
    # Define some sample columns
    columns = [
        ColumnDefinition(
            name="id",
            type=ColumnType.AUTO_INCREMENT
        ),
        ColumnDefinition(
            name="product_name", 
            type=ColumnType.SHORT_TEXT,
            prompt_instruction="A futuristic gadget name"
        ),
        ColumnDefinition(
            name="review",
            type=ColumnType.LONG_TEXT,
            prompt_instruction="A short positive review of the gadget",
            constraints=ColumnConstraints(min_length=20, max_length=100)
        ),
        ColumnDefinition(
            name="score",
            type=ColumnType.NUMERIC,
            prompt_instruction="A rating from 1 to 10"
        )
    ]
    
    print(f"Connecting to LM Studio with model '{config.model_id}'...")
    controller.initialize(config, columns)
    
    # Check connection first
    if controller.llm_client.check_connection():
        print("Successfully connected to LM Studio!")
    else:
        print("ERROR: Could not connect to LM Studio at http://localhost:1234/v1")
        print("Please ensure LM Studio is running and the server is started.")
        return

    print("Starting generation...")
    # We can't use the threaded loop easily in a simple script without a wait, 
    # but controller.generate_row() is synchronous per row.
    
    rows = []
    for i in range(config.num_rows):
        print(f"Generating row {i+1}/{config.num_rows}...")
        row = controller.generate_row()
        if row:
            rows.append(row)
            print(f"Row {i+1} Generated: {row.data}")
        else:
            print(f"Row {i+1} Failed.")
            
    print("\nGeneration Complete.")
    print(f"Generated {len(rows)} rows.")
    
    if rows:
        export_path = "test_output.csv"
        controller.generated_rows = rows
        controller.export_csv(export_path)
        print(f"Data exported to {export_path}")

if __name__ == "__main__":
    test_generation()
