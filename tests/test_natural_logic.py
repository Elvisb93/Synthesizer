
import unittest
from core.validator import UniquenessValidator
from core.models import GeneratorConfig

class TestNaturalLogic(unittest.TestCase):
    def setUp(self):
        self.config = GeneratorConfig(model_id="test")
        self.validator = UniquenessValidator(self.config)

    def test_natural_phrases(self):
        row_data = {"age": 20, "start_date": "2023-01-01"}
        
        # Greater/Less
        self.assertTrue(self.validator.validate_logic(25, "greater than @[age]", row_data))
        self.assertFalse(self.validator.validate_logic(15, "greater than @[age]", row_data))
        self.assertTrue(self.validator.validate_logic(15, "less than @[age]", row_data))
        
        # After/Before (String comparison)
        self.assertTrue(self.validator.validate_logic("2023-02-01", "after @[start_date]", row_data))
        self.assertTrue(self.validator.validate_logic("2022-01-01", "before @[start_date]", row_data))
        
        # Equality
        self.assertTrue(self.validator.validate_logic("A", "equals 'A'", row_data))
        self.assertTrue(self.validator.validate_logic("A", "is equal to 'A'", row_data))
        self.assertTrue(self.validator.validate_logic("A", "is not 'B'", row_data))
        
        # Length
        self.assertTrue(self.validator.validate_logic("Hello", "longer than 3", row_data))
        self.assertFalse(self.validator.validate_logic("Hi", "longer than 3", row_data))
        self.assertTrue(self.validator.validate_logic("Hi", "shorter than 3", row_data))

    def test_mixed_syntax(self):
        # Ensure 'this' still works
        row_data = {"val": 10}
        self.assertTrue(self.validator.validate_logic(15, "this > @[val]", row_data))

if __name__ == '__main__':
    unittest.main()
