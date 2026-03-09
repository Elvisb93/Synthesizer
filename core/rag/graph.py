import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set


@dataclass
class GraphStats:
    sources: int = 0
    entities: int = 0
    edges: int = 0


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
        if not self.enabled or not source:
            return

        # Remove old edges.
        old_entities = self._source_to_entities.get(source, set())
        for ent in old_entities:
            srcs = self._entity_to_sources.get(ent)
            if srcs and source in srcs:
                srcs.remove(source)
                if not srcs:
                    self._entity_to_sources.pop(ent, None)

        entities = self._extract_entities(text)
        for token in extra_tokens or []:
            t = (token or "").strip().lower()
            if t:
                entities.add(f"meta:{t}")

        self._source_to_entities[source] = entities
        for ent in entities:
            self._entity_to_sources[ent].add(source)

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

    def stats(self) -> GraphStats:
        edges = sum(len(v) for v in self._source_to_entities.values())
        return GraphStats(
            sources=len(self._source_to_entities),
            entities=len(self._entity_to_sources),
            edges=edges,
        )
