from typing import List, Optional

from core.rag.interfaces import VectorStore
from core.rag.models import ChunkRecord, RetrievedChunk


class QdrantVectorStore(VectorStore):
    def __init__(self, *, url: str, collection_name: str, api_key: Optional[str] = None):
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError("qdrant-client is required for local vector storage") from exc

        if url in (":memory:", "memory://", "local-memory"):
            self._client = QdrantClient(path=":memory:")
        else:
            self._client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name

    def _ensure_collection(self, vector_size: int) -> None:
        from qdrant_client import models as qm

        collections = self._client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if exists:
            return

        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
        )

    def upsert_chunks(self, chunks: List[ChunkRecord], vectors: List[List[float]]) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")

        from qdrant_client import models as qm

        vector_size = len(vectors[0]) if vectors and vectors[0] else 0
        if vector_size <= 0:
            return 0

        self._ensure_collection(vector_size)

        points = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            payload = {"text": chunk.text, **chunk.metadata}
            point_id = abs(hash(f"{chunk.chunk_id}::{idx}")) % (2**63 - 1)
            points.append(
                qm.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        self._client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def search(self, query_vector: List[float], top_k: int, min_score: float, source_filter: Optional[str] = None) -> List[RetrievedChunk]:
        if not query_vector:
            return []

        from qdrant_client import models as qm

        query_filter = None
        if source_filter:
            query_filter = qm.Filter(
                must=[qm.FieldCondition(key="source", match=qm.MatchValue(value=source_filter))]
            )

        limit = max(1, top_k)
        try:
            hits = self._client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=query_filter,
            )
        except AttributeError:
            query_result = self._client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                query_filter=query_filter,
            )
            hits = getattr(query_result, "points", [])

        results: List[RetrievedChunk] = []
        for item in hits:
            score = float(item.score or 0.0)
            if score < min_score:
                continue

            payload = item.payload or {}
            text = str(payload.get("text", ""))
            metadata = {k: v for k, v in payload.items() if k != "text"}
            results.append(
                RetrievedChunk(
                    chunk_id=str(item.id),
                    text=text,
                    score=score,
                    metadata=metadata,
                )
            )

        return results

    def count(self) -> int:
        try:
            count_result = self._client.count(collection_name=self.collection_name, exact=True)
            return int(count_result.count)
        except Exception:
            return 0

    def clear(self) -> None:
        self._client.delete_collection(self.collection_name)
