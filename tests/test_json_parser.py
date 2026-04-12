"""
Tests for core.json_parser — JSON template loading, path resolution,
schema inference, and item injection.
"""
import json
import os
import tempfile
import unittest

from core.json_parser import (
    load_template,
    resolve_target_array,
    infer_item_schema,
    inject_item,
    clear_target_array,
    export_template,
)


class TestLoadTemplate(unittest.TestCase):
    def test_load_valid_json(self):
        data = {"name": "test", "items": [{"id": 1}]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            result = load_template(path)
            self.assertEqual(result, data)
        finally:
            os.unlink(path)

    def test_load_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            load_template("/nonexistent/path/template.json")

    def test_load_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{invalid json")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                load_template(path)
        finally:
            os.unlink(path)

    def test_load_non_dict_root(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([1, 2, 3], f)
            path = f.name
        try:
            with self.assertRaises(ValueError):
                load_template(path)
        finally:
            os.unlink(path)


class TestResolveTargetArray(unittest.TestCase):
    def test_depth_1(self):
        template = {"nodes": [{"id": 1}]}
        result = resolve_target_array(template, "nodes")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)

    def test_depth_2(self):
        template = {"data": {"messages": [{"text": "hello"}]}}
        result = resolve_target_array(template, "data.messages")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "hello")

    def test_depth_3(self):
        template = {"app": {"users": {"history": [{"action": "login"}]}}}
        result = resolve_target_array(template, "app.users.history")
        self.assertEqual(len(result), 1)

    def test_empty_path(self):
        with self.assertRaises(ValueError):
            resolve_target_array({}, "")

    def test_nonexistent_key(self):
        with self.assertRaises(ValueError):
            resolve_target_array({"a": 1}, "b")

    def test_path_to_non_list(self):
        template = {"data": {"name": "test"}}
        with self.assertRaises(ValueError):
            resolve_target_array(template, "data.name")

    def test_path_through_non_dict(self):
        template = {"data": "string_value"}
        with self.assertRaises(ValueError):
            resolve_target_array(template, "data.nested")


class TestInferItemSchema(unittest.TestCase):
    def test_infer_from_populated_array(self):
        items = [{"name": "Alice", "age": 30, "active": True, "score": 9.5}]
        model = infer_item_schema(items)
        self.assertIsNotNone(model)

        # Validate that the model can accept data
        instance = model.model_validate({"name": "Bob", "age": 25, "active": False, "score": 8.0})
        self.assertEqual(instance.name, "Bob")

    def test_infer_from_empty_array(self):
        result = infer_item_schema([])
        self.assertIsNone(result)

    def test_infer_with_non_dict_items(self):
        result = infer_item_schema(["string1", "string2"])
        self.assertIsNone(result)

    def test_infer_nested_dict(self):
        items = [{"id": 1, "metadata": {"type": "test"}, "tags": ["a", "b"]}]
        model = infer_item_schema(items)
        self.assertIsNotNone(model)
        # Should not crash on nested structures
        instance = model.model_validate(items[0])
        self.assertEqual(instance.id, 1)


class TestInjectItem(unittest.TestCase):
    def test_inject_appends(self):
        template = {"items": [{"id": 1}]}
        inject_item(template, "items", {"id": 2})
        self.assertEqual(len(template["items"]), 2)
        self.assertEqual(template["items"][1]["id"], 2)

    def test_inject_preserves_static_properties(self):
        template = {"version": "1.0", "config": {"mode": "test"}, "items": []}
        inject_item(template, "items", {"id": 1})
        self.assertEqual(template["version"], "1.0")
        self.assertEqual(template["config"]["mode"], "test")
        self.assertEqual(len(template["items"]), 1)


class TestClearTargetArray(unittest.TestCase):
    def test_clear_removes_all(self):
        template = {"items": [{"id": 1}, {"id": 2}, {"id": 3}]}
        clear_target_array(template, "items")
        self.assertEqual(len(template["items"]), 0)

    def test_clear_preserves_structure(self):
        template = {"name": "test", "items": [1, 2, 3]}
        clear_target_array(template, "items")
        self.assertEqual(template["name"], "test")
        self.assertIn("items", template)


class TestExportTemplate(unittest.TestCase):
    def test_export_creates_file(self):
        template = {"items": [{"id": 1}]}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.json")
            export_template(template, path)
            self.assertTrue(os.path.exists(path))

            with open(path, "r") as f:
                loaded = json.load(f)
            self.assertEqual(loaded, template)

    def test_export_creates_parent_dirs(self):
        template = {"items": []}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "output.json")
            export_template(template, path)
            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
