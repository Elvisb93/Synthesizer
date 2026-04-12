from core.models import GeneratorConfig, RagBackend, RagConfig
from core.rag.chunking.semantic_double_buffer import SemanticDoubleBufferChunker
from core.rag.models import ParsedDocument, RetrievedChunk
from core.rag.retriever import RagRetriever


class _FakeEmbedder:
    def embed_documents(self, texts):
        return [[0.1, 0.2] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.2]


class _FakeStore:
    def upsert_chunks(self, chunks, vectors):
        return len(chunks)

    def search(self, query_vector, top_k, min_score, source_filter=None, record_type=None):
        return [
            RetrievedChunk(
                chunk_id="a",
                text="Alpha context",
                score=0.9,
                metadata={"source": "doc1.pdf", "page": 2},
            )
        ]

    def search_lexical(self, query_text, top_k, source_filter=None, record_type=None):
        return []

    def count(self):
        return 1

    def clear(self):
        return None


def test_generator_config_serializes_rag_block():
    cfg = GeneratorConfig(
        model_id="local-model",
        rag=RagConfig(
            enabled=True,
            backend=RagBackend.LLAMA_INDEX,
            collection_name="test_collection",
            top_k=3,
        ),
    )

    data = cfg.model_dump()
    assert data["rag"]["enabled"] is True
    assert data["rag"]["backend"] == RagBackend.LLAMA_INDEX
    assert data["rag"]["collection_name"] == "test_collection"
    assert data["rag"]["top_k"] == 3


def test_semantic_double_buffer_chunker_generates_chunks_with_metadata():
    doc = ParsedDocument(
        source="sample.pdf",
        pages=["One. Two. Three. Four. Five. Six. Seven. Eight."],
    )

    chunker = SemanticDoubleBufferChunker(
        window_sentences=3,
        overlap_sentences=1,
        buffer_sentences=1,
    )
    chunks = chunker.chunk(doc)

    assert len(chunks) >= 2
    assert chunks[0].metadata["source"] == "sample.pdf"
    assert chunks[0].metadata["page"] == 1
    assert chunks[0].text


def test_retriever_formats_hits_with_budget():
    retriever = RagRetriever(_FakeEmbedder(), _FakeStore())
    hits = retriever.retrieve("query", top_k=3, min_score=0.2)
    text = retriever.format_hits(hits, max_context_chars=200)
    assert "Alpha context" in text
    assert "doc1.pdf" in text
