"""
Tests for RAG sequential chunk access — scroll_all (QdrantVectorStore)
and get_all_chunks (RagService).
"""
import unittest
from unittest.mock import MagicMock, patch

from core.rag.models import ChunkRecord


class TestQdrantScrollAll(unittest.TestCase):
    """Test QdrantVectorStore.scroll_all via mocked _scroll_points."""

    def _make_mock_store(self, scroll_data):
        """Create a mock QdrantVectorStore with patched _scroll_points."""
        from core.rag.stores.qdrant_store import QdrantVectorStore

        with patch.object(QdrantVectorStore, "__init__", lambda self, **kw: None):
            store = QdrantVectorStore.__new__(QdrantVectorStore)
            store.collection_name = "test_collection"
            store._scroll_points = MagicMock(return_value=scroll_data)
            return store

    def test_scroll_all_returns_chunk_records(self):
        raw = [
            ("chunk_1", "Hello world", {"source": "doc.pdf", "page": 1}),
            ("chunk_2", "Second chunk", {"source": "doc.pdf", "page": 2}),
        ]
        store = self._make_mock_store(raw)
        result = store.scroll_all()

        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], ChunkRecord)
        self.assertEqual(result[0].chunk_id, "chunk_1")
        self.assertEqual(result[0].text, "Hello world")
        self.assertEqual(result[1].chunk_id, "chunk_2")

    def test_scroll_all_empty_collection(self):
        store = self._make_mock_store([])
        result = store.scroll_all()
        self.assertEqual(result, [])

    def test_scroll_all_passes_source_filter(self):
        store = self._make_mock_store([])
        store.scroll_all(source_filter="specific.pdf")
        store._scroll_points.assert_called_once_with(
            source_filter="specific.pdf",
            record_type="chunk",
            limit=10_000,
        )

    def test_scroll_all_defaults_to_chunk_record_type(self):
        store = self._make_mock_store([])
        store.scroll_all()
        store._scroll_points.assert_called_once_with(
            source_filter=None,
            record_type="chunk",
            limit=10_000,
        )


class TestRagServiceGetAllChunks(unittest.TestCase):
    """Test RagService.get_all_chunks via mocked store."""

    def test_get_all_chunks_delegates_to_store(self):
        mock_store = MagicMock()
        mock_chunks = [
            ChunkRecord(chunk_id="c1", text="text1", metadata={}),
            ChunkRecord(chunk_id="c2", text="text2", metadata={}),
        ]
        mock_store.scroll_all.return_value = mock_chunks

        # Minimal mock of RagService (avoid full initialization)
        from core.rag.service import RagService
        service = object.__new__(RagService)
        service.store = mock_store

        result = service.get_all_chunks()
        self.assertEqual(len(result), 2)
        mock_store.scroll_all.assert_called_once_with(
            source_filter=None,
            record_type="chunk",
            limit=10_000,
        )

    def test_get_all_chunks_with_source_filter(self):
        mock_store = MagicMock()
        mock_store.scroll_all.return_value = []

        from core.rag.service import RagService
        service = object.__new__(RagService)
        service.store = mock_store

        result = service.get_all_chunks(source_filter="report.pdf")
        self.assertEqual(result, [])
        mock_store.scroll_all.assert_called_once_with(
            source_filter="report.pdf",
            record_type="chunk",
            limit=10_000,
        )


if __name__ == "__main__":
    unittest.main()
