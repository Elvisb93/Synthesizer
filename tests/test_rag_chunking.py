from core.rag.chunking.semantic_double_buffer import SemanticDoubleBufferChunker
from core.rag.models import ParsedDocument


def test_semantic_double_buffer_chunker_generates_chunks_with_metadata():
    doc = ParsedDocument(
        source="sample.pdf",
        pages=[
            "One. Two. Three. Four. Five. Six. Seven. Eight.",
        ],
    )

    chunker = SemanticDoubleBufferChunker(window_sentences=3, overlap_sentences=1, buffer_sentences=1)
    chunks = chunker.chunk(doc)

    assert len(chunks) >= 2
    assert chunks[0].metadata["source"] == "sample.pdf"
    assert chunks[0].metadata["page"] == 1
    assert chunks[0].text
