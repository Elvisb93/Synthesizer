from core.rag.graph import ShadowGraphIndex
from core.rag.late_interaction import LateInteractionScorer
from core.rag.models import ParsedDocument, RetrievedChunk
from core.rag.service import RagService


class _FakeRetrieverForGraph:
    def retrieve(self, query, *, top_k, min_score, source_filter=None, record_type=None):
        if record_type == "chunk" and source_filter == "neighbor.pdf":
            return [
                RetrievedChunk(
                    chunk_id="n1",
                    text="Payroll schedule policy details",
                    score=0.7,
                    metadata={"source": "neighbor.pdf", "page": 1},
                )
            ]
        return []

    def retrieve_lexical(self, query, *, top_k, source_filter=None, record_type=None):
        return []


class _FakeRouterParser:
    def parse(self, source: str) -> ParsedDocument:
        return ParsedDocument(source=source, source_type="txt", pages=["fallback parser content"])


def test_shadow_graph_related_sources():
    graph = ShadowGraphIndex(enabled=True)
    graph.upsert("a.pdf", "Payroll schedule with HR policy and timeline")
    graph.upsert("b.pdf", "HR policy update and timeline for payroll operations")
    graph.upsert("c.pdf", "Unrelated astronomy notes")

    related = graph.related_sources(["a.pdf"], hops=1, limit=5)
    assert "b.pdf" in related
    assert "c.pdf" not in related


def test_late_interaction_scores_relevant_text_higher():
    query = "employee benefits policy timeline"
    relevant = "The employee benefits policy timeline is outlined in this section."
    irrelevant = "This paragraph discusses marine biology and coral reef ecology."

    s_rel = LateInteractionScorer.score(query, relevant)
    s_irr = LateInteractionScorer.score(query, irrelevant)
    assert s_rel > s_irr


def test_search_graph_expands_sources_and_applies_graph_boost():
    svc = object.__new__(RagService)
    svc.top_k = 5
    svc.min_score = 0.2
    svc.max_context_chars = 3000
    svc.summary_first_enabled = False
    svc.summary_top_k = 2
    svc.dense_top_k = 5
    svc.lexical_top_k = 5
    svc.hybrid_search_enabled = False
    svc.rerank_enabled = False
    svc.parent_context_enabled = False
    svc.graph_enabled = True
    svc.graph_hops = 1
    svc.graph_source_boost = 0.15
    svc.late_interaction_enabled = False
    svc.late_interaction_weight = 0.0
    svc.retriever = _FakeRetrieverForGraph()
    svc.graph_index = ShadowGraphIndex(enabled=True)
    svc.graph_index.upsert("seed.pdf", "Payroll schedule policy")
    svc.graph_index.upsert("neighbor.pdf", "Policy and payroll update")

    hits = RagService.search(svc, "policy", top_k=3, min_score=0.2, source_filter="seed.pdf")
    assert hits
    assert hits[0].metadata.get("source") == "neighbor.pdf"
    assert "graph_boost" in hits[0].metadata


def test_docling_mode_falls_back_to_router_when_unavailable():
    svc = object.__new__(RagService)
    svc.parser_mode = "docling"
    svc._docling_parser = None
    svc.router_parser = _FakeRouterParser()
    svc.parser = _FakeRouterParser()

    parsed = RagService._parse_source(svc, "somefile.txt")
    assert parsed.source_type == "txt"
    assert parsed.pages and "fallback" in parsed.pages[0]
