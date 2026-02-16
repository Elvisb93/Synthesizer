from core.controller import GeneratorController
from core.models import GeneratorConfig, RagConfig
from core.rag.models import RetrievedChunk


class _FakeLLM:
    def retrieve_context(self, query: str, top_k=None):
        return "Context about employee benefits and enrollment deadline."

    def generate_completion(self, prompt: str, system_prompt: str = ""):
        return "The main request is enrollment updates and employee notification."


class _FakeRagService:
    def search(self, query, top_k=5, min_score=0.25, source_filter=None):
        return [
            RetrievedChunk(
                chunk_id="c1",
                text="sample",
                score=0.91,
                metadata={"source": "benefits_email_narative.pdf", "page": 16},
            )
        ]


def test_ask_files_returns_answer_and_citations():
    controller = GeneratorController()
    controller.config = GeneratorConfig(model_id="local", rag=RagConfig())
    controller.llm_client = _FakeLLM()
    controller.rag_service = _FakeRagService()

    result = controller.ask_files("Summarize the employee issue")

    assert "answer" in result
    assert "citations" in result
    assert result["citations"][0]["source"] == "benefits_email_narative.pdf"


def test_ask_files_rejects_empty_prompt():
    controller = GeneratorController()
    controller.config = GeneratorConfig(model_id="local", rag=RagConfig())
    controller.llm_client = _FakeLLM()
    controller.rag_service = _FakeRagService()

    result = controller.ask_files("   ")
    assert result.get("error") == "Prompt is empty."
