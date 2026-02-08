import sys
import os
import unittest
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.controller import GeneratorController
from core.models import GeneratorConfig, ColumnDefinition, ColumnType

class MockLLMClient:
    def __init__(self):
        self.generate_completion = MagicMock(side_effect=self._mock_response)
        
    def _mock_response(self, prompt):
        if "Value A" in prompt:
            return "Result_A"
        if "Value B" in prompt:
            return "Result_B"
        return "Generic_Result"

class TestDependencyResolution(unittest.TestCase):
    
    def setUp(self):
        self.controller = GeneratorController()
        self.config = GeneratorConfig(model_id="test")
        
    def test_linear_dependency(self):
        """Test simple A -> B dependency."""
        cols = [
            ColumnDefinition(name="B", type=ColumnType.SHORT_TEXT, prompt_instruction="Depends on @[A]"),
            ColumnDefinition(name="A", type=ColumnType.SHORT_TEXT, prompt_instruction="Value A")
        ]
        
        self.controller.initialize(self.config, cols)
        
        # Verify Order
        order = [c.name for c in self.controller.execution_order]
        self.assertEqual(order, ["A", "B"])
        
        # Verify Prompt Interpolation (Manually running logic to simulate)
        # We need to mock the LLM client to run generate_row fully
        self.controller.llm_client = MockLLMClient()
        # Mock validator to always return True
        self.controller.validator = MagicMock()
        self.controller.validator.is_unique.return_value = True
        
        row = self.controller.generate_row()
        
        # Check that we called LLM with interpolated prompt for B
        # The prompt for B should contain "Result_A" (the mocked output of A)
        # We check the arguments passed to generate_completion
        calls = self.controller.llm_client.generate_completion.call_args_list
        
        # Call 1: A
        self.assertIn("Value A", calls[0][0][0])
        # Call 2: B
        prompt_b = calls[1][0][0]
        self.assertIn("Depends on Result_A", prompt_b)

    def test_multiple_dependency(self):
        """Test C -> A, B dependency."""
        cols = [
            ColumnDefinition(name="C", type=ColumnType.SHORT_TEXT, prompt_instruction="Uses @[A] and @[B]"),
            ColumnDefinition(name="A", type=ColumnType.SHORT_TEXT, prompt_instruction="Value A"),
            ColumnDefinition(name="B", type=ColumnType.SHORT_TEXT, prompt_instruction="Value B")
        ]
        
        self.controller.initialize(self.config, cols)
        
        order = [c.name for c in self.controller.execution_order]
        # A and B must be before C
        self.assertLess(order.index("A"), order.index("C"))
        self.assertLess(order.index("B"), order.index("C"))
        
        # Run
        self.controller.llm_client = MockLLMClient()
        self.controller.validator = MagicMock()
        self.controller.validator.is_unique.return_value = True
        
        self.controller.generate_row()
        
        calls = self.controller.llm_client.generate_completion.call_args_list
        # Find prompt for C
        prompt_c = next(c[0][0] for c in calls if "Uses" in c[0][0])
        self.assertIn("Result_A", prompt_c)
        self.assertIn("Result_B", prompt_c)

    def test_circular_dependency(self):
        """Test A -> B -> A cycle detection."""
        cols = [
            ColumnDefinition(name="A", type=ColumnType.SHORT_TEXT, prompt_instruction="Uses @[B]"),
            ColumnDefinition(name="B", type=ColumnType.SHORT_TEXT, prompt_instruction="Uses @[A]")
        ]
        
        with self.assertRaises(ValueError) as context:
            self.controller.initialize(self.config, cols)
        
        self.assertIn("Circular dependency", str(context.exception))

if __name__ == '__main__':
    unittest.main()
