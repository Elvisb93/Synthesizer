from pathlib import Path

from core.rag.models import ParsedDocument, RetrievedChunk
from core.rag.parsers.router_parser import RouterParser
from core.rag.service import RagService


class _FakePdfParser:
    def parse(self, path: str) -> ParsedDocument:
        return ParsedDocument(source=path, source_type="pdf", pages=["pdf text"])


def test_router_parser_parses_text_file(tmp_path: Path):
    p = tmp_path / "notes.txt"
    p.write_text("Alpha beta gamma", encoding="utf-8")

    parser = RouterParser(pdf_parser=_FakePdfParser())
    parsed = parser.parse(str(p))

    assert parsed.source_type == "txt"
    assert parsed.pages
    assert "Alpha beta gamma" in parsed.pages[0]


def test_router_parser_parses_html_file(tmp_path: Path):
    p = tmp_path / "page.html"
    p.write_text("<html><body><h1>Title</h1><p>Hello world</p></body></html>", encoding="utf-8")

    parser = RouterParser(pdf_parser=_FakePdfParser())
    parsed = parser.parse(str(p))

    assert parsed.source_type == "html"
    assert parsed.pages
    assert "Hello world" in parsed.pages[0]


class _FakeRetriever:
    def retrieve(self, query, *, top_k, min_score, source_filter=None, record_type=None):
        if record_type == "doc_summary":
            return [
                RetrievedChunk(
                    chunk_id="sum-1",
                    text="summary",
                    score=0.95,
                    metadata={"source": "a.pdf", "record_type": "doc_summary"},
                )
            ]
        if record_type == "chunk" and source_filter == "a.pdf":
            return [
                RetrievedChunk(
                    chunk_id="a-1",
                    text="Dense hit text",
                    score=0.9,
                    metadata={"source": "a.pdf", "page": 1, "parent_text": "Parent context section for page one"},
                )
            ]
        return []

    def retrieve_lexical(self, query, *, top_k, source_filter=None, record_type=None):
        if record_type == "chunk" and source_filter == "a.pdf":
            return [
                RetrievedChunk(
                    chunk_id="a-1",
                    text="Dense hit text",
                    score=2.0,
                    metadata={"source": "a.pdf", "page": 1},
                )
            ]
        return []


def test_search_uses_summary_first_hybrid_and_parent_context():
    svc = object.__new__(RagService)
    svc.top_k = 5
    svc.min_score = 0.25
    svc.max_context_chars = 3000
    svc.summary_first_enabled = True
    svc.summary_top_k = 2
    svc.dense_top_k = 5
    svc.lexical_top_k = 5
    svc.hybrid_search_enabled = True
    svc.rerank_enabled = True
    svc.parent_context_enabled = True
    svc.parent_context_max_chars = 20
    svc.retriever = _FakeRetriever()

    hits = RagService.search(svc, "dense query terms", top_k=3, min_score=0.2, source_filter=None)

    assert hits
    assert hits[0].metadata.get("source") == "a.pdf"
    assert "parent_excerpt" in hits[0].metadata
    assert len(hits[0].metadata["parent_excerpt"]) <= 20
