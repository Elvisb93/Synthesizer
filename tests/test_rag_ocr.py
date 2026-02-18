from pathlib import Path
import time

from core.rag.parsers.hybrid_pdf_parser import HybridPdfParser
from core.rag.service import RagService


class _FakeOcrEngine:
    def __init__(self, text: str = "OCR fallback text for scanned content"):
        self.text = text

    def extract_text(self, image):
        return self.text, 0.91


def _sample_pdf() -> str:
    root = Path(__file__).resolve().parent.parent
    path = root / "examples" / "test coffee.pdf"
    assert path.exists()
    return str(path)


def test_gap_detection_finds_large_vertical_space_regions():
    parser = HybridPdfParser(ocr_mode="auto", ocr_gap_multiplier=2.0)
    blocks = [
        {"text": "Line 1", "rect": (10.0, 780.0, 200.0, 790.0)},
        {"text": "Line 2", "rect": (10.0, 760.0, 200.0, 770.0)},
        {"text": "Line 3", "rect": (10.0, 500.0, 200.0, 510.0)},
        {"text": "Line 4", "rect": (10.0, 480.0, 200.0, 490.0)},
    ]

    regions = parser._detect_gap_regions(blocks)
    assert regions
    assert len(regions) >= 1


def test_auto_mode_uses_ocr_when_native_text_threshold_not_met():
    svc = RagService(
        collection_name=f"ocr_auto_{int(time.time())}",
        qdrant_url=":memory:",
        qdrant_api_key=None,
        embedding_model="BAAI/bge-small-en-v1.5",
        top_k=3,
        min_score=0.1,
        max_context_chars=1000,
        ocr_mode="auto",
        ocr_min_extracted_chars=100000,
    )
    svc.parser._ocr_engine = _FakeOcrEngine()

    report = svc.ingest_documents([_sample_pdf()], force_reindex=True)
    assert report.files_processed == 1
    assert report.ocr_pages_total > 0
    assert report.ocr_pages_full > 0


def test_ocr_mode_on_respects_max_pages_limit():
    svc = RagService(
        collection_name=f"ocr_maxpages_{int(time.time())}",
        qdrant_url=":memory:",
        qdrant_api_key=None,
        embedding_model="BAAI/bge-small-en-v1.5",
        top_k=3,
        min_score=0.1,
        max_context_chars=1000,
        ocr_mode="on",
        ocr_max_pages=0,
    )
    svc.parser._ocr_engine = _FakeOcrEngine()

    report = svc.ingest_documents([_sample_pdf()], force_reindex=True)
    assert report.files_processed == 1
    assert report.ocr_pages_total == 0
