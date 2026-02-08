
import unittest
from core.validator import UniquenessValidator
from core.models import GeneratorConfig

class TestRegexShortcuts(unittest.TestCase):
    def setUp(self):
        self.config = GeneratorConfig(model_id="test")
        self.validator = UniquenessValidator(self.config)

    def test_shortcuts(self):
        # Email
        self.assertTrue(self.validator.validate_regex("test@example.com", "email"))
        self.assertFalse(self.validator.validate_regex("invalid-email", "email"))

        # Phone
        self.assertTrue(self.validator.validate_regex("123-456-7890", "phone"))
        self.assertTrue(self.validator.validate_regex("(123) 456-7890", "phone"))
        self.assertFalse(self.validator.validate_regex("12345", "phone"))

        # Zip
        self.assertTrue(self.validator.validate_regex("90210", "zip"))
        self.assertTrue(self.validator.validate_regex("10001-1234", "postcode"))
        self.assertFalse(self.validator.validate_regex("ABCDE", "zip"))
        
        # IPv4
        self.assertTrue(self.validator.validate_regex("192.168.1.1", "ipv4"))
        # The simple regex validates structure (N.N.N.N), not value range (0-255).
        # So 999.999.999.999 is structurally valid for this regex.
        self.assertTrue(self.validator.validate_regex("999.999.999.999", "ipv4")) 
        self.assertFalse(self.validator.validate_regex("invalid-ip", "ipv4"))

    def test_raw_regex_still_works(self):
        self.assertTrue(self.validator.validate_regex("ABC", r"^[A-Z]{3}$"))
        self.assertFalse(self.validator.validate_regex("123", r"^[A-Z]{3}$"))

if __name__ == '__main__':
    unittest.main()
