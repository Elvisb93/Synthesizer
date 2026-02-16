from unittest.mock import MagicMock

from core.controller import GeneratorController
from core.models import ColumnDefinition, ColumnType, GeneratorConfig, RagConfig


class _FakeValidator:
    def validate_regex(self, value, pattern):
        return True

    def is_unique(self, value, field_type=None):
        return True

    def commit(self, value, field_type=None):
        return None


class _FakeLLMClient:
    def __init__(self, config):
        self.config = config
        self.prompts = []
        self.rag_service = MagicMock()
        self.rag_service.search.return_value = [{"id": "c1"}]
        self.rag_service.format_hits.return_value = "RAG_CTX_SNIPPET"

    def retrieve_context(self, query: str, top_k=None) -> str:
        rag_cfg = self.config.rag
        hits = self.rag_service.search(
            query,
            top_k=top_k,
            min_score=rag_cfg.min_score,
            source_filter=rag_cfg.source_filter,
        )
        return self.rag_service.format_hits(hits, max_context_chars=rag_cfg.max_context_chars)

    def generate_completion(self, prompt: str, system_prompt: str = "You are a helpful data generator."):
        self.prompts.append(prompt)
        if prompt.startswith("Review this data row:"):
            return "VALID"
        return "Grounded Value"


def test_controller_generation_injects_rag_context_end_to_end():
    controller = GeneratorController()
    rag_cfg = RagConfig(enabled=True, source_filter="manual.pdf")
    controller.config = GeneratorConfig(model_id="test-model", rag=rag_cfg)

    col = ColumnDefinition(
        name="answer",
        type=ColumnType.SHORT_TEXT,
        prompt_instruction="Answer based on reference docs",
    )
    controller.columns = [col]
    controller.execution_order = [col]
    controller.generated_rows = []
    controller.validator = _FakeValidator()

    fake_llm = _FakeLLMClient(controller.config)
    controller.llm_client = fake_llm

    row = controller.generate_row()

    assert row is not None
    generation_prompt = fake_llm.prompts[0]
    assert "Retrieved Context (use only if relevant):" in generation_prompt
    assert "RAG_CTX_SNIPPET" in generation_prompt

    fake_llm.rag_service.search.assert_called()
    _, kwargs = fake_llm.rag_service.search.call_args
    assert kwargs["source_filter"] == "manual.pdf"
