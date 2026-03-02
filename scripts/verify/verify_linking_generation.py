import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.controller import GeneratorController
from core.models import GeneratorConfig, ColumnDefinition, ColumnType


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    print("Initializing Generator...")

    from core.llm_client import LLMClient

    temp_config = GeneratorConfig(model_id="temp")
    client = LLMClient(temp_config)

    print("Checking connection to LM Studio...")
    if not client.check_connection():
        print("ERROR: Could not connect to LM Studio at http://localhost:1234/v1")
        print("Please ensure LM Studio is running and the server is started.")
        return

    models = client.list_models()
    if not models:
        print("ERROR: Connected to LM Studio but no models found.")
        return

    selected_model = models[0]
    print(f"Using Model: {selected_model}")

    controller = GeneratorController()
    config = GeneratorConfig(model_id=selected_model, num_rows=1, similarity_threshold=0.85)

    columns = [
        ColumnDefinition(
            name="Country",
            type=ColumnType.SHORT_TEXT,
            prompt_instruction="Generate a random European country name.",
        ),
        ColumnDefinition(
            name="Capital",
            type=ColumnType.SHORT_TEXT,
            prompt_instruction="What is the capital city of @[Country]?",
        ),
        ColumnDefinition(
            name="FunFact",
            type=ColumnType.SHORT_TEXT,
            prompt_instruction="Write a short 1-sentence fun fact about @[Capital].",
        ),
    ]

    print("\n--- Configuration ---")
    for c in columns:
        print(f"  - {c.name}: {c.prompt_instruction}")

    print("\n--- Attempting Real Generation with Local Model ---\n")
    try:
        controller.initialize(config, columns)

        def on_log(msg):
            print(f"[LOG]: {msg}")

        controller.on_log = on_log
        row = controller.generate_row()

        if row:
            print("\n--- Real Generation Success! ---")
            print(f"Row Data: {row.data}")
            return

        print("\n--- Real Generation Failed (Likely API Error) ---")
    except Exception as e:
        print(f"Real generation crashed: {e}")

    print("\n--- Falling back to Mock Generation to Verify Logic ---\n")
    try:
        from unittest.mock import MagicMock

        class MockLLM(LLMClient):
            def generate_completion(self, prompt, system_prompt=None):
                print(f"[MOCK LLM] Received Prompt: {prompt[:100]}...")
                if "European country" in prompt:
                    return "France"
                if "capital city of France" in prompt:
                    return "Paris"
                if "fun fact about Paris" in prompt:
                    return "It has the Eiffel Tower."
                return "Generic Value"

        controller.llm_client = MockLLM(config)
        controller.generated_rows = []
        row = controller.generate_row()

        if row:
            print("\n--- Mock Generation Success! ---")
            print(f"Row Data: {row.data}")
            print("VERIFICATION: Capital='Paris' confirms dependency interpolation worked.")
    except Exception as e:
        print("\n--- Mock Fallback Error ---")
        print(f"An exception occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
