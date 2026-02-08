import sys
import os
import unittest
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.models import GeneratorConfig, ColumnDefinition, ColumnType, ColumnConstraints
from core.validator import UniquenessValidator
from core.controller import GeneratorController

class TestPhase4(unittest.TestCase):
    def setUp(self):
        self.config = GeneratorConfig(model_id="test", num_rows=5, max_retries=3)
        self.validator = UniquenessValidator(self.config)
        self.controller = GeneratorController()
        
    def test_regex_validation(self):
        print("\n--- Testing Regex Validation ---")
        pattern = r"^\d{3}-\d{2}-\d{4}$" # SSN format
        
        valid = self.validator.validate_regex("123-45-6789", pattern)
        invalid = self.validator.validate_regex("123456789", pattern)
        
        print(f"Valid '123-45-6789': {valid}")
        print(f"Invalid '123456789': {invalid}")
        
        self.assertTrue(valid)
        self.assertFalse(invalid)

    def test_logic_validation(self):
        print("\n--- Testing Logic Validation ---")
        expression = "this > row_data.get('age')"
        
        row_data = {'age': 25}
        
        valid_val = 30
        invalid_val = 20
        
        res_valid = self.validator.validate_logic(valid_val, expression, row_data)
        res_invalid = self.validator.validate_logic(invalid_val, expression, row_data)
        
        print(f"Check 30 > 25: {res_valid}")
        print(f"Check 20 > 25: {res_invalid}")
        
        self.assertTrue(res_valid)
        self.assertFalse(res_invalid)
        
        # Test simplified syntax if we supported it (currently uses python syntax in eval)
        # Note: In validator we implemented: re.sub(r'@\[(.*?)\]', ...)
        # So "this > @[age]" becomes "this > row_data.get('age')"
        
        expr_friendly = "this > @[age]"
        res_friendly = self.validator.validate_logic(30, expr_friendly, row_data)
        print(f"Check friendly syntax 'this > @[age]': {res_friendly}")
        self.assertTrue(res_friendly)

    def test_max_retries_and_export(self):
        print("\n--- Testing Max Retries & Export ---")
        
        # Setup controller with mock LLM
        mock_llm = MagicMock()
        # Side effect: always return "duplicate"
        mock_llm.generate_completion.return_value = "dup"
        
        self.controller.llm_client = mock_llm
        self.controller.config = self.config
        
        # Define a column that requires uniqueness
        col = ColumnDefinition(
            name="col1", 
            type=ColumnType.LONG_TEXT,
            prompt_instruction="gen",
            constraints=ColumnConstraints(min_length=1)
        )
        self.controller.columns = [col]
        self.controller.execution_order = [col]
        
        # Pre-seed validator with "dup" so it fails uniqueness
        self.controller.validator = self.validator
        self.validator.seen_hashes.add("dup") # Actually hashing logic is: hashlib.sha256("dup".encode()).hexdigest()
        import hashlib
        h = hashlib.sha256("dup".encode()).hexdigest()
        self.validator.seen_hashes.add(h)
        
        # This should fail after 3 retries (config.max_retries)
        self.controller.log = MagicMock() # Suppress logs or check them
        row = self.controller.generate_row()
        
        print(f"Row result (should be None): {row}")
        self.assertIsNone(row)
        
        # Verify call count
        # Initial attempt + 3 retries = 4 calls? Or 3 total?
        # Logic: while attempts < max_retries: attempts += 1 ...
        # max_retries = 3. 
        # 1. attempts=1. call.
        # 2. attempts=2. call.
        # 3. attempts=3. call. 
        # Stop.
        self.assertEqual(mock_llm.generate_completion.call_count, 3)

if __name__ == '__main__':
    unittest.main()
