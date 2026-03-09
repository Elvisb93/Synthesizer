from core.controller import GeneratorController
from core.models import DocumentEngineConfig, GeneratorConfig, RagConfig
from core.document_engine.orchestrator import DocumentOrchestrator
from core.document_engine.validators import validate_chunk
from core.charts.generator import DocumentChartGenerator
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
        if '{"chunk":' in prompt:
            return '{"chunk":"Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence."}'
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
        if '{"chunk":' in prompt:
            return '{"chunk":"Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence. Generated content sentence."}'
        return " ".join(["Generated content sentence."] * 120)


class _ChartingLLM(_FakeLLM):
    def generate_completion(self, prompt: str, system_prompt: str = ""):
        self.calls.append((system_prompt, prompt))
        if "Create grounded chart specs from context." in prompt:
            return (
                '{"charts":[{"title":"Payments by Method","chart_type":"bar","x_label":"Method","y_label":"Amount",'
                '"labels":["Bank Transfer","Card"],"values":[700.0,702.27],'
                '"caption":"Split by method for INV-2026-031.",'
                '"evidence_sources":["invoice.xlsx"]}]}'
            )
        return super().generate_completion(prompt, system_prompt)


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


class _FailingRagService:
    top_k = 5
    min_score = 0.25
    max_context_chars = 3000

    def search(self, query, top_k=5, min_score=0.25, source_filter=None):
        raise RuntimeError("Collection synthesizer_default not found")

    def format_hits(self, hits, max_context_chars=3000):
        return ""


class _ChartRagService:
    top_k = 5
    min_score = 0.25
    max_context_chars = 3000

    def search(self, query, top_k=5, min_score=0.25, source_filter=None):
        return [
            RetrievedChunk(
                chunk_id="1",
                text="Bank Transfer 700.0 Card 702.27 Total 1402.27 INV-2026-031",
                score=0.92,
                metadata={"source": "invoice.xlsx", "page": 1},
            )
        ]

    def format_hits(self, hits, max_context_chars=3000):
        return (
            "[1] source=invoice.xlsx, page=1, score=0.920\n"
            "Bank Transfer 700.0\n"
            "Card 702.27\n"
            "Total 1402.27\n"
            "INV-2026-031\n"
        )


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


def test_quality_mode_fast_runtime_tuning_prefers_speed():
    tuning = GeneratorController._build_document_runtime_tuning(
        target_words=500,
        requested_auto=False,
        quality_mode="Fast",
        cfg_max_chunk_words=500,
        cfg_min_chunk_words=220,
        cfg_max_retries=3,
        cfg_consistency_check_interval=12,
    )
    assert tuning["fast_mode"] is True
    assert tuning["max_retries"] == 1
    assert tuning["consistency_check_interval"] == 0
    assert tuning["hard_max_words"] == 0


def test_quality_mode_thorough_runtime_tuning_prefers_strictness():
    tuning = GeneratorController._build_document_runtime_tuning(
        target_words=500,
        requested_auto=False,
        quality_mode="Thorough",
        cfg_max_chunk_words=500,
        cfg_min_chunk_words=220,
        cfg_max_retries=3,
        cfg_consistency_check_interval=12,
    )
    assert tuning["fast_mode"] is False
    assert tuning["max_retries"] >= 4
    assert tuning["consistency_check_interval"] > 0
    assert tuning["hard_max_words"] == 0


def test_generate_document_handles_rag_retrieval_failure_gracefully():
    controller = GeneratorController()
    controller.config = GeneratorConfig(
        model_id="local",
        rag=RagConfig(),
        document_engine=DocumentEngineConfig(mode="hybrid", target_words=600),
    )
    controller.llm_client = _FakeLLM()
    controller.rag_service = _FailingRagService()
    controller.document_orchestrator = None

    result = controller.generate_document(
        "Write a practical memo",
        target_words=600,
        audience="General",
        tone="professional",
        mode="hybrid",
        quality_mode="Fast",
        resume=False,
    )

    assert "error" not in result
    assert result.get("text")


def test_extract_chunk_text_strips_reasoning_prefix():
    orch = DocumentOrchestrator(_FakeLLM(), None)
    raw = (
        "Thinking Process:\\n\\n"
        "1. Analyze prompt\\n"
        "2. Plan output\\n\\n"
        "{\"chunk\":\"Final section prose only.\"}"
    )
    text = orch._extract_chunk_text(raw)
    assert text == "Final section prose only."


def test_extract_chunk_text_prefers_last_valid_chunk_json():
    orch = DocumentOrchestrator(_FakeLLM(), None)
    raw = (
        'Thinking Process:\\n'
        'Schema reminder: {"chunk":"<final section prose>"}\\n'
        'More notes here.\\n\\n'
        '{"chunk":"Final clean prose only."}'
    )
    text = orch._extract_chunk_text(raw)
    assert text == "Final clean prose only."


def test_sanitize_chunk_for_publish_removes_meta_lines():
    dirty = (
        "Thinking Process:\n"
        "Constraint: Do not include chain-of-thought.\n"
        "The instruction says to output JSON.\n"
        "This section provides operational recommendations based on KPI trends.\n"
    )
    cleaned = DocumentOrchestrator._sanitize_chunk_for_publish(dirty)
    assert "Thinking Process" not in cleaned
    assert "Constraint:" not in cleaned
    assert "instruction says" not in cleaned.lower()
    assert "operational recommendations" in cleaned.lower()


def test_sanitize_chunk_for_publish_strips_numbered_word_artifact():
    dirty = (
        "Introduction *Word Count Check:* 1 This 2 section 3 explains 4 the 5 report 6 output 7 quality 8.\n"
        "9 It 10 should 11 be 12 readable."
    )
    cleaned = DocumentOrchestrator._sanitize_chunk_for_publish(dirty)
    assert "word count check" not in cleaned.lower()
    assert "1 This" not in cleaned
    assert "section explains the report output quality" in cleaned.lower()


def test_controller_fallback_title_from_prompt_is_user_friendly():
    title = GeneratorController._fallback_title_from_prompt(
        "Create a clean executive finance report using all available sources"
    )
    assert "report" in title.lower()
    assert len(title) <= 80


def test_validate_chunk_allows_long_natural_section_when_no_max_ratio():
    text = " ".join(["Natural analysis sentence."] * 180)
    result = validate_chunk(
        text,
        target_words=80,
        tail_content="",
        next_section_title="",
        min_ratio=0.8,
        max_ratio=None,
    )
    assert result.is_valid


def test_document_length_enforcement_treats_target_as_floor_not_ceiling():
    controller = GeneratorController()
    result = {"text": " ".join(["sentence."] * 700)}
    updated = controller._enforce_document_length_bounds(
        result,
        target_words=500,
        audience="General",
        tone="professional",
    )
    assert updated["final_word_count"] == 700
    assert updated["text"] == result["text"]


def test_generate_document_with_chart_enabled_builds_chart_artifacts(monkeypatch, tmp_path):
    controller = GeneratorController()
    controller.config = GeneratorConfig(
        model_id="local",
        rag=RagConfig(),
        document_engine=DocumentEngineConfig(mode="hybrid", target_words=500, chart_enabled=True, max_charts=2),
    )
    controller.llm_client = _ChartingLLM()
    controller.rag_service = _ChartRagService()
    controller.document_orchestrator = None
    controller.document_chart_generator = DocumentChartGenerator(controller.llm_client)

    def _fake_render(self, spec, index, out_dir):
        from PIL import Image

        path = tmp_path / f"chart_{index+1}.png"
        Image.new("RGB", (640, 360), color=(250, 250, 250)).save(path)
        return str(path)

    monkeypatch.setattr(DocumentChartGenerator, "_render_chart", _fake_render)

    result = controller.generate_document(
        "Create an invoice analysis report",
        target_words=500,
        audience="Finance",
        tone="professional",
        mode="hybrid",
        resume=False,
    )

    assert "error" not in result
    assert result.get("charts")
    assert len(result["charts"]) >= 1
