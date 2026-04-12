"""
Tests for core.json_agent — LangGraph JSON object generation state machine.
Uses mocked LLM client to test generation, validation, correction, and
R010-compliant JSON extraction.
"""
import json
import unittest
from unittest.mock import MagicMock, patch
from typing import Optional, Dict, Any

from pydantic import BaseModel

from core.json_agent import _parse_json_from_response, create_json_generator_graph


class TestParseJsonFromResponse(unittest.TestCase):
    """Test the R010-compliant JSON extraction helper."""

    def test_clean_json(self):
        result = _parse_json_from_response('{"name": "Alice", "age": 30}')
        self.assertEqual(result, {"name": "Alice", "age": 30})

    def test_json_with_markdown_fences(self):
        text = '```json\n{"name": "Bob"}\n```'
        result = _parse_json_from_response(text)
        self.assertEqual(result, {"name": "Bob"})

    def test_json_with_reasoning_trace_before(self):
        text = (
            "Let me think about this. The schema requires name and age fields. "
            "I'll generate a diverse entry.\n\n"
            '{"name": "Charlie", "age": 25}'
        )
        result = _parse_json_from_response(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Charlie")

    def test_json_with_long_reasoning_trace(self):
        reasoning = "This is a long reasoning trace. " * 50
        text = reasoning + '\n{"id": 42, "status": "active"}'
        result = _parse_json_from_response(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], 42)

    def test_empty_response(self):
        self.assertIsNone(_parse_json_from_response(""))
        self.assertIsNone(_parse_json_from_response(None))

    def test_invalid_json(self):
        result = _parse_json_from_response("this is not json at all")
        self.assertIsNone(result)

    def test_json_array_not_dict(self):
        # Should return None since we only want dicts
        result = _parse_json_from_response('[1, 2, 3]')
        self.assertIsNone(result)

    def test_nested_json(self):
        text = '{"user": {"name": "Eve", "roles": ["admin"]}, "active": true}'
        result = _parse_json_from_response(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["user"]["name"], "Eve")


class TestJsonGeneratorGraph(unittest.TestCase):
    """Test the LangGraph state machine with mocked LLM."""

    def _make_mock_llm(self, responses):
        """Create a mock LLM client with predefined responses."""
        mock = MagicMock()
        mock.generate_completion = MagicMock(side_effect=responses)
        return mock

    def test_successful_generation_no_schema(self):
        """Generate a valid JSON object without Pydantic schema validation."""
        mock_llm = self._make_mock_llm(['{"name": "TestItem", "value": 42}'])
        graph = create_json_generator_graph(mock_llm, schema_model=None)

        result = graph.invoke({
            "item_data": {},
            "schema_description": '{"name": "string", "value": "int"}',
            "template_context": "",
            "errors": [],
            "attempt_count": 0,
            "is_valid": False,
        })

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["item_data"]["name"], "TestItem")
        self.assertEqual(result["item_data"]["value"], 42)

    def test_successful_generation_with_schema(self):
        """Generate and validate against a Pydantic schema."""
        class TestSchema(BaseModel):
            name: str
            value: int

        mock_llm = self._make_mock_llm(['{"name": "Validated", "value": 99}'])
        graph = create_json_generator_graph(mock_llm, schema_model=TestSchema)

        result = graph.invoke({
            "item_data": {},
            "schema_description": '{"name": "string", "value": "int"}',
            "template_context": "",
            "errors": [],
            "attempt_count": 0,
            "is_valid": False,
        })

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["item_data"]["name"], "Validated")

    def test_correction_on_invalid_output(self):
        """Test that invalid output triggers correction and retry."""
        class StrictSchema(BaseModel):
            name: str
            value: int

        responses = [
            '{"name": "Bad"}',  # Missing 'value' field -> validation fails
            '{"name": "Fixed", "value": 10}',  # Corrected output
        ]
        mock_llm = self._make_mock_llm(responses)
        graph = create_json_generator_graph(mock_llm, schema_model=StrictSchema)

        result = graph.invoke({
            "item_data": {},
            "schema_description": '{"name": "string", "value": "int"}',
            "template_context": "",
            "errors": [],
            "attempt_count": 0,
            "is_valid": False,
        })

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["item_data"]["value"], 10)

    def test_max_retries_exceeded(self):
        """Test that generation gives up after max retries."""
        class StrictSchema(BaseModel):
            name: str
            value: int

        # All responses are missing required 'value' field
        responses = [
            '{"name": "Bad1"}',  # Generate: fails validation
            '{"name": "Bad2"}',  # Correct attempt 1: still fails
            '{"name": "Bad3"}',  # Correct attempt 2: still fails
            '{"name": "Bad4"}',  # Correct attempt 3: still fails -> give up
        ]
        mock_llm = self._make_mock_llm(responses)
        graph = create_json_generator_graph(mock_llm, schema_model=StrictSchema)

        result = graph.invoke({
            "item_data": {},
            "schema_description": '{"name": "string", "value": "int"}',
            "template_context": "",
            "errors": [],
            "attempt_count": 0,
            "is_valid": False,
        })

        self.assertFalse(result["is_valid"])

    def test_reasoning_trace_handling(self):
        """Test R010: LLM outputs reasoning before JSON."""
        reasoning_response = (
            "Let me generate a proper JSON object. "
            "The schema requires name and value.\n\n"
            '{"name": "R010Test", "value": 7}'
        )
        mock_llm = self._make_mock_llm([reasoning_response])
        graph = create_json_generator_graph(mock_llm, schema_model=None)

        result = graph.invoke({
            "item_data": {},
            "schema_description": '{"name": "string", "value": "int"}',
            "template_context": "",
            "errors": [],
            "attempt_count": 0,
            "is_valid": False,
        })

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["item_data"]["name"], "R010Test")


if __name__ == "__main__":
    unittest.main()
