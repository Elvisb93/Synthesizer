from core.rag.retriever import RagRetriever
from core.rag.models import RetrievedChunk


class _FakeEmbedder:
    def embed_documents(self, texts):
        return [[0.1, 0.2] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.2]


class _FakeStore:
    def upsert_chunks(self, chunks, vectors):
        return len(chunks)

    def search(self, query_vector, top_k, min_score, source_filter=None):
        return [
            RetrievedChunk(
                chunk_id="a",
                text="Alpha context",
                score=0.9,
                metadata={"source": "doc1.pdf", "page": 2},
            )
        ]

    def count(self):
        return 1

    def clear(self):
        return None


def test_retriever_formats_hits_with_budget():
    retriever = RagRetriever(_FakeEmbedder(), _FakeStore())
    hits = retriever.retrieve("query", top_k=3, min_score=0.2)
    text = retriever.format_hits(hits, max_context_chars=200)
    assert "Alpha context" in text
    assert "doc1.pdf" in text
