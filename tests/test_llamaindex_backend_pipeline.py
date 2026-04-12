from types import SimpleNamespace

from core.rag.backends.llamaindex_backend import LlamaIndexRagService
from core.rag.models import ParsedDocument


def _build_service(tmp_path):
    return LlamaIndexRagService(
        collection_name="test_llama_pipeline",
        qdrant_url=":memory:",
        qdrant_api_key=None,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        top_k=5,
        min_score=0.1,
        max_context_chars=2000,
        cache_path=str(tmp_path / "rag_cache.json"),
    )


def _build_llm_service(tmp_path):
    return LlamaIndexRagService(
        collection_name="test_llama_pipeline_llm",
        qdrant_url=":memory:",
        qdrant_api_key=None,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        top_k=5,
        min_score=0.1,
        max_context_chars=2000,
        cache_path=str(tmp_path / "rag_cache_llm.json"),
        llm_model_name="local-model",
        llm_base_url="http://127.0.0.1:1234/v1",
        llm_api_key="lm-studio",
    )


def test_build_chunk_documents_preserves_local_metadata(tmp_path):
    service = _build_service(tmp_path)
    parsed = ParsedDocument(
        source=str(tmp_path / "sample.txt"),
        pages=["Alpha beta gamma. Delta epsilon zeta."],
        source_type="file",
    )

    documents = service._build_chunk_documents(parsed)

    assert len(documents) == 1
    assert documents[0].doc_id.endswith("::page::1")
    assert documents[0].metadata["source"] == parsed.source
    assert documents[0].metadata["record_type"] == "chunk"
    assert "parent_text" in documents[0].excluded_embed_metadata_keys


def test_ingest_documents_records_pipeline_nodes_and_summary(tmp_path, monkeypatch):
    source = tmp_path / "sample.txt"
    source.write_text("Alpha beta gamma. Delta epsilon zeta.", encoding="utf-8")
    service = _build_service(tmp_path)

    parsed = ParsedDocument(
        source=str(source),
        pages=["Alpha beta gamma. Delta epsilon zeta."],
        page_metadata=[{}],
        source_type="file",
    )
    chunk_node = SimpleNamespace(
        node_id="chunk-1",
        id_="chunk-1",
        text="Alpha beta gamma.",
        metadata={
            "chunk_id": "chunk-1",
            "source": str(source),
            "source_type": "file",
            "page": 1,
            "record_type": "chunk",
            "parent_text": "Alpha beta gamma. Delta epsilon zeta.",
        },
        get_content=lambda metadata_mode=None: "Alpha beta gamma.",
    )

    monkeypatch.setattr(service, "_parse_source", lambda path: parsed)
    monkeypatch.setattr(service, "_run_chunk_ingestion", lambda documents: [chunk_node])
    inserted = []
    monkeypatch.setattr(service, "_insert_summary_node", lambda node: inserted.append(node))

    report = service.ingest_documents([str(source)])
    chunks = service.get_all_chunks(source_filter=str(source))

    assert report.errors == []
    assert report.files_processed == 1
    assert report.chunks_created == 2
    assert report.vectors_upserted == 2
    assert len(inserted) == 1
    assert [chunk.metadata["record_type"] for chunk in chunks] == ["chunk", "doc_summary"]


def test_llamaindex_status_reports_pipeline_enabled(tmp_path):
    service = _build_service(tmp_path)

    status = service.get_status()

    assert status["backend"] == "llamaindex"
    assert status["ingestion_pipeline_enabled"] is True
    assert status["response_synthesis_enabled"] is False


def test_metadata_extraction_is_skipped_for_large_ingests(tmp_path):
    service = _build_llm_service(tmp_path)
    documents = [SimpleNamespace(text="A" * 1000) for _ in range(7)]

    assert service._should_use_metadata_extraction(documents) is False
    assert service.get_status()["metadata_extraction_enabled"] is False
    assert service.get_status()["metadata_extraction_last_reason"] == "skipped_large_ingest_doc_count>6"


def test_run_chunk_ingestion_uses_fast_path_when_metadata_is_skipped(tmp_path, monkeypatch):
    service = _build_llm_service(tmp_path)
    documents = [SimpleNamespace(text="A" * 1000) for _ in range(7)]
    requested_flags = []

    class PipelineStub:
        def run(self, documents, show_progress=False):
            return [SimpleNamespace(node_id="chunk-1", id_="chunk-1", metadata={"chunk_id": "chunk-1"})]

    monkeypatch.setattr(service, "_persist_ingestion_cache", lambda: None)

    def fake_get_ingestion_pipeline(include_metadata_extractors=None):
        requested_flags.append(include_metadata_extractors)
        return PipelineStub()

    monkeypatch.setattr(service, "_get_ingestion_pipeline", fake_get_ingestion_pipeline)

    nodes = service._run_chunk_ingestion(documents)

    assert len(nodes) == 1
    assert requested_flags == [False]
    status = service.get_status()
    assert status["metadata_extraction_enabled"] is False
    assert status["metadata_extraction_last_reason"] == "skipped_large_ingest_doc_count>6"
