from core.models import GeneratorConfig
from core.validator import UniquenessValidator


def _validator() -> UniquenessValidator:
    return UniquenessValidator(GeneratorConfig(model_id="test"))


def test_regex_shortcuts_and_raw_patterns():
    validator = _validator()

    assert validator.validate_regex("test@example.com", "email")
    assert not validator.validate_regex("invalid-email", "email")

    assert validator.validate_regex("123-456-7890", "phone")
    assert validator.validate_regex("(123) 456-7890", "phone")
    assert not validator.validate_regex("12345", "phone")

    assert validator.validate_regex("90210", "zip")
    assert validator.validate_regex("10001-1234", "postcode")
    assert not validator.validate_regex("ABCDE", "zip")

    assert validator.validate_regex("192.168.1.1", "ipv4")
    # Shortcut validates structure, not range.
    assert validator.validate_regex("999.999.999.999", "ipv4")
    assert not validator.validate_regex("invalid-ip", "ipv4")

    assert validator.validate_regex("ABC", r"^[A-Z]{3}$")
    assert not validator.validate_regex("123", r"^[A-Z]{3}$")


def test_regex_explicit_pattern_validation():
    validator = _validator()
    pattern = r"^\d{3}-\d{2}-\d{4}$"

    assert validator.validate_regex("123-45-6789", pattern)
    assert not validator.validate_regex("123456789", pattern)


def test_logic_python_and_friendly_reference_syntax():
    validator = _validator()
    row_data = {"age": 25}

    assert validator.validate_logic(30, "this > row_data.get('age')", row_data)
    assert not validator.validate_logic(20, "this > row_data.get('age')", row_data)
    assert validator.validate_logic(30, "this > @[age]", row_data)


def test_logic_natural_language_phrases():
    validator = _validator()
    row_data = {"age": 20, "start_date": "2023-01-01", "val": 10}

    assert validator.validate_logic(25, "greater than @[age]", row_data)
    assert not validator.validate_logic(15, "greater than @[age]", row_data)
    assert validator.validate_logic(15, "less than @[age]", row_data)

    assert validator.validate_logic("2023-02-01", "after @[start_date]", row_data)
    assert validator.validate_logic("2022-01-01", "before @[start_date]", row_data)

    assert validator.validate_logic("A", "equals 'A'", row_data)
    assert validator.validate_logic("A", "is equal to 'A'", row_data)
    assert validator.validate_logic("A", "is not 'B'", row_data)

    assert validator.validate_logic("Hello", "longer than 3", row_data)
    assert not validator.validate_logic("Hi", "longer than 3", row_data)
    assert validator.validate_logic("Hi", "shorter than 3", row_data)
    assert validator.validate_logic(15, "this > @[val]", row_data)
