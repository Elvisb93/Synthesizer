import unittest
from unittest.mock import MagicMock, patch
from core.models import GeneratorConfig, ColumnDefinition, ColumnType, ColumnConstraints
from core.controller import GeneratorController

class TestEnrichment(unittest.TestCase):
    def setUp(self):
        self.controller = GeneratorController()
        # Mock LLM Client
        self.controller.llm_client = MagicMock()
        self.controller.validator = MagicMock()
        self.controller.validator.validate_regex.return_value = True
        self.controller.validator.validate_logic.return_value = True
        self.controller.validator.is_unique.return_value = True

    def test_enrichment_logic(self):
        # 1. Define Existing Data
        existing_data = [
            {"name": "Alice", "role": "Engineer"},
            {"name": "Bob", "role": "Artist"}
        ]
        
        # 2. Config with existing data
        config = GeneratorConfig(
            model_id="test", 
            num_rows=2, 
            existing_data=existing_data
        )
        
        # 3. Define Schema (Imported Cols + New Col)
        cols = [
            ColumnDefinition(name="name", type=ColumnType.SHORT_TEXT, prompt_instruction="(Imported)"),
            ColumnDefinition(name="role", type=ColumnType.SHORT_TEXT, prompt_instruction="(Imported)"),
            # New Column dependent on existing ones
            ColumnDefinition(
                name="bio", 
                type=ColumnType.LONG_TEXT, 
                prompt_instruction="Write a bio for @[name] the @[role]."
            )
        ]
        
        self.controller.initialize(config, cols)
        
        # Override with Mock AFTER initialize
        self.controller.llm_client = MagicMock()
        
        # Mock LLM response to verify prompt construction
        def side_effect(prompt, system_prompt=None):
            if "Review this data row" in prompt:
                return "VALID"
            if "Alice" in prompt and "Engineer" in prompt:
                return "Alice is a great Engineer."
            if "Bob" in prompt and "Artist" in prompt:
                return "Bob paints nice clouds."
            return "Generic Bio"
            
        self.controller.llm_client.generate_completion.side_effect = side_effect
        
        # 4. Run Generation
        self.controller._run_generation_loop()
        
        # 5. Assertions
        self.assertEqual(len(self.controller.generated_rows), 2)
        
        row1 = self.controller.generated_rows[0].data
        self.assertEqual(row1["name"], "Alice")
        self.assertEqual(row1["bio"], "Alice is a great Engineer.")
        
        row2 = self.controller.generated_rows[1].data
        self.assertEqual(row2["name"], "Bob")
        self.assertEqual(row2["bio"], "Bob paints nice clouds.")
        
        print("\nTest Enrichment Passed!")

if __name__ == '__main__':
    unittest.main()
