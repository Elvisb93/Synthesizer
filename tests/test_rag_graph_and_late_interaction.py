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
        if record_type == "chunk" and source_filter == "benefits.pdf":
            return [
                RetrievedChunk(
                    chunk_id="b1",
                    text="Employee benefits enrollment timeline and policy details",
                    score=0.72,
                    metadata={"source": "benefits.pdf", "page": 2},
                )
            ]
        return []

    def retrieve_lexical(self, query, *, top_k, source_filter=None, record_type=None):
        return []


class _FakeRouterParser:
    def parse(self, source: str) -> ParsedDocument:
        return ParsedDocument(source=source, source_type="txt", pages=["fallback parser content"])


class _FakeRetrieverForChunkGraph:
    def retrieve(self, query, *, top_k, min_score, source_filter=None, record_type=None):
        if record_type != "chunk" or source_filter != "policy.pdf":
            return []
        return [
            RetrievedChunk(
                chunk_id="policy.pdf::p1::c0",
                text="Base price and premium overview for the policy.",
                score=0.72,
                metadata={"source": "policy.pdf", "page": 1},
            ),
            RetrievedChunk(
                chunk_id="policy.pdf::p1::c1",
                text="Exclusion clause modifies the final payable amount under the policy.",
                score=0.63,
                metadata={"source": "policy.pdf", "page": 1},
            ),
        ]

    def retrieve_lexical(self, query, *, top_k, source_filter=None, record_type=None):
        return []


def test_shadow_graph_related_sources():
    graph = ShadowGraphIndex(enabled=True)
    graph.upsert("a.pdf", "Payroll schedule with HR policy and timeline")
    graph.upsert("b.pdf", "HR policy update and timeline for payroll operations")
    graph.upsert("c.pdf", "Unrelated astronomy notes")

    related = graph.related_sources(["a.pdf"], hops=1, limit=5)
    assert "b.pdf" in related
    assert "c.pdf" not in related


def test_shadow_graph_tracks_chunk_neighbors_and_queries_related_chunks():
    graph = ShadowGraphIndex(enabled=True)
    graph.upsert_source("policy.pdf", "Premium policy and exclusion clauses")
    graph.upsert_chunk("policy.pdf", "policy.pdf::p1::c0", "Premium policy amount and deductible details")
    graph.upsert_chunk(
        "policy.pdf",
        "policy.pdf::p1::c1",
        "Exclusion clause changes the deductible and payout amount",
        neighbor_chunk_ids=["policy.pdf::p1::c0"],
    )

    matched = graph.query_chunks("premium policy details", limit=5)
    related = graph.related_chunks(["policy.pdf::p1::c0"], hops=1, limit=5)
    stats = graph.stats()

    assert "policy.pdf::p1::c0" in matched
    assert "policy.pdf::p1::c1" in related
    assert stats.chunks == 2
    assert stats.chunk_edges == 1


def test_late_interaction_scores_relevant_text_higher():
    query = "employee benefits policy timeline"
    relevant = "The employee benefits policy timeline is outlined in this section."
    irrelevant = "This paragraph discusses marine biology and coral reef ecology."

    s_rel = LateInteractionScorer.score(query, relevant)
    s_irr = LateInteractionScorer.score(query, irrelevant)
    assert s_rel > s_irr


def test_late_interaction_prefers_ordered_phrase_over_shuffled_terms():
    query = "employee benefits policy timeline"
    ordered = "The employee benefits policy timeline is outlined in this section."
    shuffled = "Timeline policy benefits for employee programs are discussed here."

    s_ordered = LateInteractionScorer.score(query, ordered)
    s_shuffled = LateInteractionScorer.score(query, shuffled)
    assert s_ordered > s_shuffled


def test_late_interaction_prefers_compact_match_window():
    query = "benefits enrollment deadline"
    compact = "The benefits enrollment deadline is Friday."
    dispersed = (
        "Benefits are covered in the introduction. "
        "Several unrelated sections follow with operational notes. "
        "The enrollment handbook appears later. "
        "The final appendix mentions the deadline."
    )

    s_compact = LateInteractionScorer.score(query, compact)
    s_dispersed = LateInteractionScorer.score(query, dispersed)
    assert s_compact > s_dispersed


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


def test_search_graph_can_seed_sources_from_query_entities():
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
    svc.graph_index.upsert("benefits.pdf", "Employee benefits enrollment timeline and policy update")
    svc.graph_index.upsert("astronomy.pdf", "Deep space telescope notes and nebula observations")

    hits = RagService.search(svc, "benefits enrollment timeline", top_k=3, min_score=0.2, source_filter=None)
    assert hits
    assert hits[0].metadata.get("source") == "benefits.pdf"
    assert "graph_boost" in hits[0].metadata


def test_search_graph_applies_chunk_boost_to_related_clause_hits():
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
    svc.retriever = _FakeRetrieverForChunkGraph()
    svc.graph_index = ShadowGraphIndex(enabled=True)
    svc.graph_index.upsert_source("policy.pdf", "Premium policy and exclusion clauses")
    svc.graph_index.upsert_chunk("policy.pdf", "policy.pdf::p1::c0", "Base price and premium overview for the policy.")
    svc.graph_index.upsert_chunk(
        "policy.pdf",
        "policy.pdf::p1::c1",
        "Exclusion clause modifies the final payable amount under the policy.",
        neighbor_chunk_ids=["policy.pdf::p1::c0"],
    )

    hits = RagService.search(svc, "policy price exclusion", top_k=3, min_score=0.2, source_filter=None)

    assert hits
    by_id = {hit.chunk_id: hit for hit in hits}
    assert "policy.pdf::p1::c0" in by_id
    assert "policy.pdf::p1::c1" in by_id
    assert "graph_chunk_boost" in by_id["policy.pdf::p1::c1"].metadata
    assert by_id["policy.pdf::p1::c1"].metadata.get("graph_related_chunk") is True


def test_docling_mode_falls_back_to_router_when_unavailable():
    svc = object.__new__(RagService)
    svc.parser_mode = "docling"
    svc._docling_parser = None
    svc.router_parser = _FakeRouterParser()
    svc.parser = _FakeRouterParser()

    parsed = RagService._parse_source(svc, "somefile.txt")
    assert parsed.source_type == "txt"
    assert parsed.pages and "fallback" in parsed.pages[0]
