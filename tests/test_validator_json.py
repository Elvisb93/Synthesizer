"""
Tests for UniquenessValidator JSON extensions — extract_strings_for_hashing,
is_unique_json, and commit_json for nested JSON deduplication.
"""
import unittest

from core.models import GeneratorConfig
from core.validator import UniquenessValidator


class TestExtractStringsForHashing(unittest.TestCase):
    """Test the recursive path-value string extractor."""

    def test_flat_dict(self):
        obj = {"name": "Alice", "age": 30, "active": True}
        result = UniquenessValidator.extract_strings_for_hashing(obj)
        self.assertEqual(len(result), 3)
        self.assertIn("name: Alice", result)
        self.assertIn("age: 30", result)
        self.assertIn("active: True", result)

    def test_nested_dict(self):
        obj = {"user": {"name": "Bob", "role": "admin"}}
        result = UniquenessValidator.extract_strings_for_hashing(obj)
        self.assertEqual(len(result), 2)
        self.assertIn("user.name: Bob", result)
        self.assertIn("user.role: admin", result)

    def test_deeply_nested(self):
        obj = {"a": {"b": {"c": "deep_value"}}}
        result = UniquenessValidator.extract_strings_for_hashing(obj)
        self.assertEqual(len(result), 1)
        self.assertIn("a.b.c: deep_value", result)

    def test_list_inside_dict(self):
        obj = {"tags": ["python", "ai"]}
        result = UniquenessValidator.extract_strings_for_hashing(obj)
        self.assertEqual(len(result), 2)
        self.assertIn("tags[0]: python", result)
        self.assertIn("tags[1]: ai", result)

    def test_list_of_dicts(self):
        obj = {"items": [{"id": 1}, {"id": 2}]}
        result = UniquenessValidator.extract_strings_for_hashing(obj)
        self.assertIn("items[0].id: 1", result)
        self.assertIn("items[1].id: 2", result)

    def test_empty_dict(self):
        result = UniquenessValidator.extract_strings_for_hashing({})
        self.assertEqual(result, [])

    def test_numeric_values(self):
        obj = {"score": 9.5, "count": 42}
        result = UniquenessValidator.extract_strings_for_hashing(obj)
        self.assertIn("score: 9.5", result)
        self.assertIn("count: 42", result)

    def test_mixed_types(self):
        obj = {
            "id": 1,
            "name": "Test",
            "active": False,
            "metadata": {"type": "demo"},
            "tags": ["a"],
        }
        result = UniquenessValidator.extract_strings_for_hashing(obj)
        self.assertEqual(len(result), 5)


class TestIsUniqueJson(unittest.TestCase):
    """Test JSON uniqueness checking with path-based flattening."""

    def setUp(self):
        config = GeneratorConfig(model_id="test-model", similarity_threshold=0.85)
        self.validator = UniquenessValidator(config)

    def test_first_item_always_unique(self):
        obj = {"name": "Alice", "role": "admin"}
        self.assertTrue(self.validator.is_unique_json(obj))

    def test_exact_duplicate_rejected(self):
        obj = {"name": "Alice", "role": "admin"}
        self.validator.commit_json(obj)
        self.assertFalse(self.validator.is_unique_json(obj))

    def test_different_items_accepted(self):
        obj1 = {"name": "Alice", "role": "admin"}
        obj2 = {"name": "Bob", "role": "user"}
        self.validator.commit_json(obj1)
        self.assertTrue(self.validator.is_unique_json(obj2))

    def test_empty_dict_is_unique(self):
        # Edge case: empty dict should be considered "unique" (no crash)
        self.assertTrue(self.validator.is_unique_json({}))

    def test_clear_resets_history(self):
        obj = {"name": "Alice"}
        self.validator.commit_json(obj)
        self.assertFalse(self.validator.is_unique_json(obj))
        self.validator.clear()
        self.assertTrue(self.validator.is_unique_json(obj))


class TestCommitJson(unittest.TestCase):
    """Test JSON commit to uniqueness history."""

    def setUp(self):
        config = GeneratorConfig(model_id="test-model", similarity_threshold=0.85)
        self.validator = UniquenessValidator(config)

    def test_commit_adds_to_hash_set(self):
        obj = {"key": "value"}
        initial_count = len(self.validator.seen_hashes)
        self.validator.commit_json(obj)
        self.assertEqual(len(self.validator.seen_hashes), initial_count + 1)

    def test_commit_empty_dict_noop(self):
        initial_count = len(self.validator.seen_hashes)
        self.validator.commit_json({})
        self.assertEqual(len(self.validator.seen_hashes), initial_count)


if __name__ == "__main__":
    unittest.main()
