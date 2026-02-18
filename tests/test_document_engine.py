from core.controller import GeneratorController
from core.models import DocumentEngineConfig, GeneratorConfig, RagConfig
from core.document_engine.validators import validate_chunk
from core.exporters.document_docx_exporter import DocumentDocxExporter
from core.exporters.document_pdf_exporter import DocumentPDFExporter
from core.rag.models import RetrievedChunk


class _FakeLLM:
    def __init__(self):
        self.calls = []

    def generate_completion(self, prompt: str, system_prompt: str = ""):
        self.calls.append((system_prompt, prompt))
        if "Return ONLY JSON" in prompt:
            return (
                '{"topic":"Team handbook","audience":"General","total_target_words":600,'
                '"sections":[{"title":"Overview","purpose":"Set scope","target_words":250},'
                '{"title":"Implementation","purpose":"Provide practical details","target_words":350}]}'
            )
        if prompt.startswith("Update this rolling summary"):
            return "The document covers the requested structure and introduces implementation details."
        if prompt.startswith("Review for contradictions"):
            return "OK"
        return " ".join(["Generated content sentence."] * 140)


class _PatchLLM(_FakeLLM):
    def __init__(self):
        super().__init__()
        self.consistency_calls = 0

    def generate_completion(self, prompt: str, system_prompt: str = ""):
        self.calls.append((system_prompt, prompt))
        if "Return ONLY JSON" in prompt:
            return (
                '{"topic":"Patch test","audience":"General","total_target_words":450,'
                '"sections":[{"title":"Part One","purpose":"First","target_words":220},'
                '{"title":"Part Two","purpose":"Second","target_words":230}]}'
            )
        if prompt.startswith("Update this rolling summary"):
            return "Summary updated with section details."
        if prompt.startswith("Review for contradictions"):
            self.consistency_calls += 1
            if self.consistency_calls == 1:
                return "Define the core term before first reuse."
            return "OK"
        return " ".join(["Generated content sentence."] * 120)


class _FakeRagService:
    top_k = 5
    min_score = 0.25
    max_context_chars = 3000

    def search(self, query, top_k=5, min_score=0.25, source_filter=None):
        return [
            RetrievedChunk(
                chunk_id="1",
                text="Context snippet about implementation outcomes.",
                score=0.88,
                metadata={"source": "brief.pdf", "page": 2},
            )
        ]

    def format_hits(self, hits, max_context_chars=3000):
        return "[1] source=brief.pdf, page=2, score=0.880\nContext snippet about implementation outcomes."


def test_generate_document_hybrid_mode_returns_text_and_citations():
    controller = GeneratorController()
    controller.config = GeneratorConfig(
        model_id="local",
        rag=RagConfig(),
        document_engine=DocumentEngineConfig(mode="hybrid", target_words=600),
    )
    controller.llm_client = _FakeLLM()
    controller.rag_service = _FakeRagService()
    controller.document_orchestrator = None

    result = controller.generate_document(
        "Write a team handbook",
        target_words=600,
        audience="Engineering",
        tone="professional",
        mode="hybrid",
        resume=False,
    )

    assert "error" not in result
    assert result.get("mode") == "hybrid"
    assert result.get("text")
    assert result.get("citations")


def test_generate_document_pure_mode_works_without_rag():
    controller = GeneratorController()
    controller.config = GeneratorConfig(
        model_id="local",
        document_engine=DocumentEngineConfig(mode="pure", target_words=500),
    )
    controller.llm_client = _FakeLLM()
    controller.rag_service = None
    controller.document_orchestrator = None

    result = controller.generate_document(
        "Create a long-form narrative",
        target_words=500,
        audience="General",
        tone="conversational",
        mode="pure",
        resume=False,
    )

    assert "error" not in result
    assert result.get("mode") == "pure"
    assert len(result.get("text", "")) > 100


def test_validator_detects_next_section_overflow():
    text = "This section covers key ideas.\n\nImplementation Plan\nWe now begin next section content."
    result = validate_chunk(
        text,
        target_words=80,
        tail_content="Previous context only.",
        next_section_title="Implementation Plan",
    )
    assert not result.is_valid
    assert any("next section" in r.lower() or "overflow" in r.lower() for r in result.reasons)


def test_consistency_patch_forward_is_injected_into_generation_prompt():
    controller = GeneratorController()
    controller.config = GeneratorConfig(
        model_id="local",
        document_engine=DocumentEngineConfig(mode="pure", target_words=450, consistency_check_interval=1),
    )
    patch_llm = _PatchLLM()
    controller.llm_client = patch_llm
    controller.rag_service = None
    controller.document_orchestrator = None

    result = controller.generate_document(
        "Write two coherent sections",
        target_words=450,
        mode="pure",
        resume=False,
    )
    assert "error" not in result
    assert any("Consistency patch:" in p for _, p in patch_llm.calls)


def test_reference_formatters_group_by_section():
    chunks = [
        {
            "section_title": "Overview",
            "citations": [
                {"source": "docA.pdf", "page": 1, "score": 0.91},
                {"source": "docA.pdf", "page": 1, "score": 0.91},
            ],
        },
        {
            "section_title": "Details",
            "citations": [{"source": "docB.pdf", "page": 4, "score": 0.88}],
        },
    ]
    docx_lines = DocumentDocxExporter._format_references(chunks)
    pdf_lines = DocumentPDFExporter._format_references(chunks)

    assert any("Overview:" == line for line in docx_lines)
    assert any("Details:" == line for line in docx_lines)
    assert any(line.startswith("Overview:") for line in pdf_lines)
    assert any("docA.pdf" in line for line in pdf_lines)
