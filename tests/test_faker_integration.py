import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.models import GeneratorConfig, ColumnDefinition, ColumnType, ColumnConstraints
from core.controller import GeneratorController

class TestFakerIntegration(unittest.TestCase):
    def setUp(self):
        self.controller = GeneratorController()
        # Mock LLM client to avoid API calls, though we shouldn't hit it for Deterministic
        self.controller.llm_client = "Mocked" 

    def test_faker_initialization(self):
        """Test that Faker is initialized in the controller."""
        self.assertIsNotNone(self.controller.fake)

    def test_deterministic_generation_email(self):
        """Test generating a deterministic email column."""
        columns = [
            ColumnDefinition(
                name="user_email",
                type=ColumnType.DETERMINISTIC,
                constraints=ColumnConstraints(faker_provider="email")
            )
        ]
        config = GeneratorConfig(model_id="test", num_rows=5)
        
        self.controller.initialize(config, columns)
        self.controller._run_generation_loop() # Run synchronously
        
        self.assertEqual(len(self.controller.generated_rows), 5)
        for row in self.controller.generated_rows:
            email = row.data["user_email"]
            print(f"Generated Email: {email}")
            self.assertIn("@", email)
            self.assertIn(".", email)

    def test_deterministic_generation_name(self):
        """Test generating a deterministic name column."""
        columns = [
            ColumnDefinition(
                name="full_name",
                type=ColumnType.DETERMINISTIC,
                constraints=ColumnConstraints(faker_provider="name")
            )
        ]
        config = GeneratorConfig(model_id="test", num_rows=5)
        
        self.controller.initialize(config, columns)
        self.controller._run_generation_loop()
        
        self.assertEqual(len(self.controller.generated_rows), 5)
        for row in self.controller.generated_rows:
            name = row.data["full_name"]
            print(f"Generated Name: {name}")
            self.assertTrue(len(name) > 1)

    def test_deterministic_generation_invalid_provider(self):
        """Test fallback when an invalid provider is specified."""
        columns = [
            ColumnDefinition(
                name="unknown_col",
                type=ColumnType.DETERMINISTIC,
                constraints=ColumnConstraints(faker_provider="INVALID_PROVIDER_NAME_XYZ")
            )
        ]
        config = GeneratorConfig(model_id="test", num_rows=1)
        
        self.controller.initialize(config, columns)
        self.controller._run_generation_loop()
        
        # specific logic in controller falls back to name() or error logging
        # Let's check what happened. Based on implementation, it logs error and uses name().
        self.assertEqual(len(self.controller.generated_rows), 1)
        val = self.controller.generated_rows[0].data["unknown_col"]
        print(f"Fallback Value: {val}")
        self.assertTrue(isinstance(val, str))
        self.assertTrue(len(val) > 0)

if __name__ == '__main__':
    unittest.main()
