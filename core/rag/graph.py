import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set


@dataclass
class GraphStats:
    sources: int = 0
    entities: int = 0
    edges: int = 0
    chunks: int = 0
    chunk_edges: int = 0


class ShadowGraphIndex:
    """
    Lightweight local GraphRAG scaffold.

    Nodes:
    - source documents
    - extracted entities/themes
    Edges:
    - source <-> entity if entity appears in source
    """

    ENTITY_RE = re.compile(
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)|([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})|(\d{4}-\d{2}-\d{2})"
    )
    TOKEN_RE = re.compile(r"[a-z0-9]{4,}")
    STOPWORDS = {
        "this",
        "that",
        "with",
        "from",
        "have",
        "were",
        "will",
        "about",
        "which",
        "their",
        "there",
        "these",
        "those",
        "document",
        "summary",
        "section",
        "table",
        "data",
    }

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        self._source_to_entities: Dict[str, Set[str]] = defaultdict(set)
        self._entity_to_sources: Dict[str, Set[str]] = defaultdict(set)
        self._source_to_chunks: Dict[str, Set[str]] = defaultdict(set)
        self._chunk_to_source: Dict[str, str] = {}
        self._chunk_to_entities: Dict[str, Set[str]] = defaultdict(set)
        self._entity_to_chunks: Dict[str, Set[str]] = defaultdict(set)
        self._chunk_neighbors: Dict[str, Set[str]] = defaultdict(set)

    @classmethod
    def _extract_entities(cls, text: str, max_themes: int = 24) -> Set[str]:
        out: Set[str] = set()
        for m in cls.ENTITY_RE.finditer(text or ""):
            token = next((g for g in m.groups() if g), "")
            token = (token or "").strip()
            if len(token) >= 3:
                out.add(token.lower())

        # Add coarse "themes" from frequent content tokens.
        freq: Dict[str, int] = defaultdict(int)
        for tok in cls.TOKEN_RE.findall((text or "").lower()):
            if tok in cls.STOPWORDS:
                continue
            freq[tok] += 1
        ranked = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:max_themes]
        for tok, _ in ranked:
            out.add(f"theme:{tok}")
        return out

    def upsert(self, source: str, text: str, extra_tokens: Iterable[str] | None = None) -> None:
        self.reset_source(source)
        self.upsert_source(source, text, extra_tokens=extra_tokens)

    def upsert_source(self, source: str, text: str, extra_tokens: Iterable[str] | None = None) -> None:
        if not self.enabled or not source:
            return

        entities = self._extract_entities(text)
        for token in extra_tokens or []:
            t = (token or "").strip().lower()
            if t:
                entities.add(f"meta:{t}")

        self._source_to_entities[source] = entities
        for ent in entities:
            self._entity_to_sources[ent].add(source)

    def reset_source(self, source: str) -> None:
        if not self.enabled or not source:
            return

        old_entities = self._source_to_entities.pop(source, set())
        for ent in old_entities:
            srcs = self._entity_to_sources.get(ent)
            if srcs and source in srcs:
                srcs.remove(source)
                if not srcs:
                    self._entity_to_sources.pop(ent, None)

        for chunk_id in list(self._source_to_chunks.pop(source, set())):
            self._remove_chunk(chunk_id)

    def _remove_chunk(self, chunk_id: str) -> None:
        source = self._chunk_to_source.pop(chunk_id, "")
        if source:
            chunks = self._source_to_chunks.get(source)
            if chunks and chunk_id in chunks:
                chunks.remove(chunk_id)
                if not chunks:
                    self._source_to_chunks.pop(source, None)

        entities = self._chunk_to_entities.pop(chunk_id, set())
        for ent in entities:
            chunk_ids = self._entity_to_chunks.get(ent)
            if chunk_ids and chunk_id in chunk_ids:
                chunk_ids.remove(chunk_id)
                if not chunk_ids:
                    self._entity_to_chunks.pop(ent, None)

        neighbors = self._chunk_neighbors.pop(chunk_id, set())
        for neighbor in neighbors:
            linked = self._chunk_neighbors.get(neighbor)
            if linked and chunk_id in linked:
                linked.remove(chunk_id)
                if not linked:
                    self._chunk_neighbors.pop(neighbor, None)

    def upsert_chunk(
        self,
        source: str,
        chunk_id: str,
        text: str,
        *,
        extra_tokens: Iterable[str] | None = None,
        neighbor_chunk_ids: Iterable[str] | None = None,
    ) -> None:
        if not self.enabled or not source or not chunk_id:
            return

        if chunk_id in self._chunk_to_source:
            self._remove_chunk(chunk_id)

        entities = self._extract_entities(text)
        for token in extra_tokens or []:
            t = (token or "").strip().lower()
            if t:
                entities.add(f"meta:{t}")

        self._chunk_to_source[chunk_id] = source
        self._source_to_chunks[source].add(chunk_id)
        self._chunk_to_entities[chunk_id] = entities
        for ent in entities:
            self._entity_to_chunks[ent].add(chunk_id)

        for neighbor in neighbor_chunk_ids or []:
            nb = str(neighbor or "").strip()
            if not nb or nb == chunk_id:
                continue
            self._chunk_neighbors[chunk_id].add(nb)
            self._chunk_neighbors[nb].add(chunk_id)

    @staticmethod
    def _entity_weight(entity: str) -> float:
        if entity.startswith("meta:"):
            return 0.85
        if entity.startswith("theme:"):
            return 0.60
        return 1.00

    def query_sources(self, query: str, *, limit: int = 20) -> List[str]:
        if not self.enabled or not query.strip():
            return []

        query_entities = self._extract_entities(query, max_themes=12)
        if not query_entities:
            return []

        scores: Dict[str, float] = defaultdict(float)
        for entity in query_entities:
            weight = self._entity_weight(entity)
            for src in self._entity_to_sources.get(entity, set()):
                scores[src] += weight

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [src for src, _ in ranked[: max(1, limit)]]

    def query_chunks(self, query: str, *, limit: int = 40, source_filter: str | None = None) -> List[str]:
        if not self.enabled or not query.strip():
            return []

        query_entities = self._extract_entities(query, max_themes=12)
        if not query_entities:
            return []

        scores: Dict[str, float] = defaultdict(float)
        for entity in query_entities:
            weight = self._entity_weight(entity)
            for chunk_id in self._entity_to_chunks.get(entity, set()):
                if source_filter:
                    chunk_source = self._chunk_to_source.get(chunk_id, "")
                    if chunk_source != source_filter:
                        continue
                scores[chunk_id] += weight

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [chunk_id for chunk_id, _ in ranked[: max(1, limit)]]

    def related_sources(self, seeds: Iterable[str], hops: int = 1, limit: int = 20) -> List[str]:
        if not self.enabled:
            return []

        seed_set = {s for s in seeds if s}
        if not seed_set:
            return []

        hops = max(1, int(hops))
        visited_sources: Set[str] = set(seed_set)
        scores: Dict[str, int] = defaultdict(int)
        queue = deque([(seed, 0) for seed in seed_set])

        while queue:
            source, depth = queue.popleft()
            if depth >= hops:
                continue
            for ent in self._source_to_entities.get(source, set()):
                neighbors = self._entity_to_sources.get(ent, set())
                for nb in neighbors:
                    if nb in seed_set:
                        continue
                    scores[nb] += 1
                    if nb not in visited_sources:
                        visited_sources.add(nb)
                        queue.append((nb, depth + 1))

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [src for src, _ in ranked[: max(1, limit)]]

    def related_chunks(
        self,
        seeds: Iterable[str],
        *,
        hops: int = 1,
        limit: int = 40,
        source_filter: str | None = None,
    ) -> List[str]:
        if not self.enabled:
            return []

        seed_set = {str(s or "").strip() for s in seeds if str(s or "").strip()}
        if not seed_set:
            return []

        hops = max(1, int(hops))
        visited_chunks: Set[str] = set(seed_set)
        scores: Dict[str, float] = defaultdict(float)
        queue = deque([(seed, 0) for seed in seed_set])

        while queue:
            chunk_id, depth = queue.popleft()
            if depth >= hops:
                continue

            neighbor_weights: Dict[str, float] = defaultdict(float)

            for nb in self._chunk_neighbors.get(chunk_id, set()):
                neighbor_weights[nb] += 1.2

            for ent in self._chunk_to_entities.get(chunk_id, set()):
                ent_weight = self._entity_weight(ent)
                for nb in self._entity_to_chunks.get(ent, set()):
                    if nb == chunk_id:
                        continue
                    neighbor_weights[nb] += 0.35 * ent_weight

            for nb, weight in neighbor_weights.items():
                if nb in seed_set:
                    continue
                if source_filter:
                    chunk_source = self._chunk_to_source.get(nb, "")
                    if chunk_source != source_filter:
                        continue
                scores[nb] += weight
                if nb not in visited_chunks:
                    visited_chunks.add(nb)
                    queue.append((nb, depth + 1))

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [chunk_id for chunk_id, _ in ranked[: max(1, limit)]]

    def chunk_source(self, chunk_id: str) -> str:
        return self._chunk_to_source.get(chunk_id, "")

    def clear(self) -> None:
        self._source_to_entities.clear()
        self._entity_to_sources.clear()
        self._source_to_chunks.clear()
        self._chunk_to_source.clear()
        self._chunk_to_entities.clear()
        self._entity_to_chunks.clear()
        self._chunk_neighbors.clear()

    def stats(self) -> GraphStats:
        edges = sum(len(v) for v in self._source_to_entities.values())
        chunk_edges = sum(len(v) for v in self._chunk_neighbors.values()) // 2
        return GraphStats(
            sources=len(self._source_to_entities),
            entities=len(self._entity_to_sources),
            edges=edges,
            chunks=len(self._chunk_to_source),
            chunk_edges=chunk_edges,
        )
