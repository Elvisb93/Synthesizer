import unittest
from unittest.mock import MagicMock

from core.llm_client import LLMClient
from core.models import GeneratorConfig


class TestLLMClientConnection(unittest.TestCase):
    def test_check_connection_returns_true_for_non_empty_response(self):
        client = LLMClient.__new__(LLMClient)
        client.chat_model = MagicMock()
        client.chat_model.invoke.return_value = MagicMock(content="yes")

        self.assertTrue(client.check_connection())

    def test_check_connection_returns_false_for_empty_response(self):
        client = LLMClient.__new__(LLMClient)
        client.chat_model = MagicMock()
        client.chat_model.invoke.return_value = MagicMock(content="   ")

        self.assertFalse(client.check_connection())

    def test_check_connection_returns_false_on_invoke_error(self):
        client = LLMClient.__new__(LLMClient)
        client.chat_model = MagicMock()
        client.chat_model.invoke.side_effect = RuntimeError("boom")

        self.assertFalse(client.check_connection())


class TestLLMClientInitialization(unittest.TestCase):
    def test_init_requires_api_key_for_cloud_provider(self):
        with self.assertRaises(ValueError):
            LLMClient(GeneratorConfig(model_id="test-model", provider="OpenAI"))


if __name__ == "__main__":
    unittest.main()
