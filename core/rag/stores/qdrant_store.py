import hashlib
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

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

    @staticmethod
    def _stable_point_id(raw_id: str) -> int:
        digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()
        return int(digest[:16], 16)

    def _build_filter(self, source_filter: Optional[str], record_type: Optional[str]):
        from qdrant_client import models as qm

        must = []
        if source_filter:
            must.append(qm.FieldCondition(key="source", match=qm.MatchValue(value=source_filter)))
        if record_type:
            must.append(qm.FieldCondition(key="record_type", match=qm.MatchValue(value=record_type)))
        if not must:
            return None
        return qm.Filter(must=must)

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
        for chunk, vector in zip(chunks, vectors):
            payload = {"text": chunk.text, **chunk.metadata}
            point_id = self._stable_point_id(chunk.chunk_id)
            points.append(
                qm.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        self._client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def search(
        self,
        query_vector: List[float],
        top_k: int,
        min_score: float,
        source_filter: Optional[str] = None,
        record_type: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        if not query_vector:
            return []

        query_filter = self._build_filter(source_filter=source_filter, record_type=record_type)

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
                    chunk_id=str(payload.get("chunk_id") or item.id),
                    text=text,
                    score=score,
                    metadata=metadata,
                )
            )

        return results

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", (text or "").lower())

    def _scroll_points(self, *, source_filter: Optional[str], record_type: Optional[str], limit: int) -> List[Tuple[str, str, Dict]]:
        points: List[Tuple[str, str, Dict]] = []
        offset = None
        page_size = 256
        qfilter = self._build_filter(source_filter=source_filter, record_type=record_type)

        while len(points) < limit:
            batch, offset = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qfilter,
                with_vectors=False,
                with_payload=True,
                limit=min(page_size, max(1, limit - len(points))),
                offset=offset,
            )
            if not batch:
                break
            for item in batch:
                payload = item.payload or {}
                text = str(payload.get("text", "")).strip()
                if not text:
                    continue
                chunk_id = str(payload.get("chunk_id") or item.id)
                metadata = {k: v for k, v in payload.items() if k != "text"}
                points.append((chunk_id, text, metadata))
            if offset is None:
                break
        return points

    def search_lexical(
        self,
        query_text: str,
        top_k: int,
        source_filter: Optional[str] = None,
        record_type: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        query_terms = self._tokenize(query_text)
        if not query_terms:
            return []

        docs = self._scroll_points(
            source_filter=source_filter,
            record_type=record_type,
            limit=max(200, top_k * 80),
        )
        if not docs:
            return []

        doc_tfs = []
        doc_freq: Counter = Counter()
        lengths = []
        for _, text, _ in docs:
            tokens = self._tokenize(text)
            tf = Counter(tokens)
            doc_tfs.append(tf)
            lengths.append(max(1, len(tokens)))
            for term in set(query_terms):
                if tf.get(term, 0) > 0:
                    doc_freq[term] += 1

        n_docs = max(1, len(docs))
        avgdl = max(1.0, sum(lengths) / len(lengths))
        k1 = 1.5
        b = 0.75

        scored: List[RetrievedChunk] = []
        for idx, (chunk_id, text, metadata) in enumerate(docs):
            tf = doc_tfs[idx]
            dl = lengths[idx]
            score = 0.0
            for term in query_terms:
                freq = tf.get(term, 0)
                if freq <= 0:
                    continue
                df = doc_freq.get(term, 0)
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                numer = freq * (k1 + 1.0)
                denom = freq + k1 * (1.0 - b + b * (dl / avgdl))
                score += idf * (numer / max(1e-9, denom))
            if score <= 0.0:
                continue
            scored.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    score=float(score),
                    metadata=metadata,
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[: max(1, top_k)]

    def count(self) -> int:
        try:
            count_result = self._client.count(collection_name=self.collection_name, exact=True)
            return int(count_result.count)
        except Exception:
            return 0

    def has_source(self, source: str) -> bool:
        if not source:
            return False
        try:
            filt = self._build_filter(source_filter=source, record_type=None)
            count_result = self._client.count(collection_name=self.collection_name, count_filter=filt, exact=True)
            return int(count_result.count) > 0
        except Exception:
            return False

    def clear(self) -> None:
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            return

    def scroll_all(
        self,
        *,
        source_filter: Optional[str] = None,
        record_type: Optional[str] = None,
        limit: int = 10_000,
    ) -> List[ChunkRecord]:
        """Return all chunks sequentially via paginated scroll.

        Reuses the internal _scroll_points infrastructure and converts
        raw tuples into ChunkRecord objects for the public API.
        """
        raw_points = self._scroll_points(
            source_filter=source_filter,
            record_type=record_type or "chunk",
            limit=limit,
        )
        return [
            ChunkRecord(chunk_id=cid, text=text, metadata=meta)
            for cid, text, meta in raw_points
        ]
