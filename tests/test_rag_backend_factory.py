from unittest.mock import patch

from core.models import RagBackend
from core.rag.factory import create_rag_backend
from core.rag.service import RagService


def test_factory_returns_native_rag_service_for_default_backend():
    backend = create_rag_backend(
        backend=RagBackend.NATIVE,
        collection_name="test_collection",
        qdrant_url=":memory:",
        qdrant_api_key=None,
        embedding_model="BAAI/bge-small-en-v1.5",
        top_k=5,
        min_score=0.25,
        max_context_chars=3000,
    )

    assert isinstance(backend, RagService)


def test_factory_routes_llamaindex_backend_selection():
    sentinel = object()
    with patch("core.rag.factory.LlamaIndexRagService", return_value=sentinel) as mock_backend:
        backend = create_rag_backend(
            backend=RagBackend.LLAMA_INDEX,
            collection_name="test_collection",
            qdrant_url=":memory:",
            qdrant_api_key=None,
            embedding_model="BAAI/bge-small-en-v1.5",
            top_k=5,
            min_score=0.25,
            max_context_chars=3000,
        )

    assert backend is sentinel
    mock_backend.assert_called_once()
