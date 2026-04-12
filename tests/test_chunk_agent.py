"""
Tests for core.chunk_agent — LangGraph chunk extraction and critique pipeline.
Uses mocked LLM client to test extraction, critique, and edge cases.
"""
import json
import unittest
from unittest.mock import MagicMock

from core.chunk_agent import (
    _parse_json_array_from_response,
    create_chunk_extraction_graph,
)


class TestParseJsonArrayFromResponse(unittest.TestCase):
    """Test JSON array extraction helper."""

    def test_clean_array(self):
        text = '[{"instruction": "Q1", "response": "A1"}]'
        result = _parse_json_array_from_response(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["instruction"], "Q1")

    def test_array_with_markdown(self):
        text = '```json\n[{"instruction": "Q", "response": "A"}]\n```'
        result = _parse_json_array_from_response(text)
        self.assertEqual(len(result), 1)

    def test_array_with_reasoning_trace(self):
        text = (
            "Let me analyze this text carefully.\n\n"
            '[{"instruction": "What is X?", "response": "X is Y."}]'
        )
        result = _parse_json_array_from_response(text)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

    def test_single_object_wrapped(self):
        text = '{"instruction": "Q", "response": "A"}'
        result = _parse_json_array_from_response(text)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

    def test_empty_response(self):
        self.assertIsNone(_parse_json_array_from_response(""))
        self.assertIsNone(_parse_json_array_from_response(None))

    def test_invalid_json(self):
        result = _parse_json_array_from_response("not json at all")
        self.assertIsNone(result)

    def test_multiple_items(self):
        text = json.dumps([
            {"instruction": "Q1", "response": "A1"},
            {"instruction": "Q2", "response": "A2"},
            {"instruction": "Q3", "response": "A3"},
        ])
        result = _parse_json_array_from_response(text)
        self.assertEqual(len(result), 3)

    def test_filters_non_dict_items(self):
        text = '[{"instruction": "Q", "response": "A"}, "not a dict", 42]'
        result = _parse_json_array_from_response(text)
        self.assertEqual(len(result), 1)


class TestChunkExtractionGraph(unittest.TestCase):
    """Test the extraction + critique LangGraph with mocked LLM."""

    def _make_mock_llm(self, responses):
        mock = MagicMock()
        mock.generate_completion = MagicMock(side_effect=responses)
        return mock

    def test_successful_extraction_and_critique(self):
        """Full pipeline: extract 2 pairs, critique keeps 1."""
        extraction_response = json.dumps([
            {"instruction": "What is Python?", "response": "Python is a programming language."},
            {"instruction": "What is Java?", "response": "Java is a coffee brand."},
        ])
        critique_response = json.dumps([
            {"index": 0, "grounded": True, "reason": "Directly stated in text."},
            {"index": 1, "grounded": False, "reason": "Not mentioned in source text."},
        ])

        mock_llm = self._make_mock_llm([extraction_response, critique_response])
        graph = create_chunk_extraction_graph(mock_llm)

        result = graph.invoke({
            "chunk_text": "Python is a programming language used for web development.",
            "chunk_metadata": {"source": "test.pdf", "page": 1},
            "extracted_pairs": [],
            "verified_pairs": [],
            "errors": [],
        })

        self.assertEqual(len(result["verified_pairs"]), 1)
        self.assertEqual(result["verified_pairs"][0]["instruction"], "What is Python?")

    def test_all_pairs_pass_critique(self):
        """All extracted pairs are grounded."""
        extraction_response = json.dumps([
            {"instruction": "Q1", "response": "A1"},
            {"instruction": "Q2", "response": "A2"},
        ])
        critique_response = json.dumps([
            {"index": 0, "grounded": True, "reason": "OK"},
            {"index": 1, "grounded": True, "reason": "OK"},
        ])

        mock_llm = self._make_mock_llm([extraction_response, critique_response])
        graph = create_chunk_extraction_graph(mock_llm)

        result = graph.invoke({
            "chunk_text": "Some factual content here.",
            "chunk_metadata": {},
            "extracted_pairs": [],
            "verified_pairs": [],
            "errors": [],
        })

        self.assertEqual(len(result["verified_pairs"]), 2)

    def test_empty_chunk(self):
        """Empty chunk text should produce no pairs."""
        mock_llm = self._make_mock_llm(["[]", "[]"])
        graph = create_chunk_extraction_graph(mock_llm)

        result = graph.invoke({
            "chunk_text": "",
            "chunk_metadata": {},
            "extracted_pairs": [],
            "verified_pairs": [],
            "errors": [],
        })

        self.assertEqual(len(result["verified_pairs"]), 0)

    def test_malformed_extraction_output(self):
        """Graceful handling of unparseable LLM output."""
        mock_llm = self._make_mock_llm(["this is not json", "also not json"])
        graph = create_chunk_extraction_graph(mock_llm)

        result = graph.invoke({
            "chunk_text": "Some text content.",
            "chunk_metadata": {},
            "extracted_pairs": [],
            "verified_pairs": [],
            "errors": [],
        })

        # Should not crash — returns empty
        self.assertEqual(len(result["verified_pairs"]), 0)

    def test_critique_failure_keeps_all_pairs(self):
        """If critique parsing fails, conservatively keep extracted pairs."""
        extraction_response = json.dumps([
            {"instruction": "Q1", "response": "A1"},
        ])
        # Critique returns unparseable output
        critique_response = "I can't process this as JSON"

        mock_llm = self._make_mock_llm([extraction_response, critique_response])
        graph = create_chunk_extraction_graph(mock_llm)

        result = graph.invoke({
            "chunk_text": "Some factual content here.",
            "chunk_metadata": {},
            "extracted_pairs": [],
            "verified_pairs": [],
            "errors": [],
        })

        # Fallback: keeps all because critique failed
        self.assertEqual(len(result["verified_pairs"]), 1)


if __name__ == "__main__":
    unittest.main()
