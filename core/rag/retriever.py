from typing import List, Optional

from .interfaces import Embedder, Retriever, VectorStore
from .models import RetrievedChunk


class RagRetriever(Retriever):
    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, *, top_k: int, min_score: float, source_filter: Optional[str] = None) -> List[RetrievedChunk]:
        vector = self.embedder.embed_query(query)
        return self.store.search(vector, top_k=top_k, min_score=min_score, source_filter=source_filter)

    def format_hits(self, hits: List[RetrievedChunk], max_context_chars: int) -> str:
        if not hits or max_context_chars <= 0:
            return ""

        lines: List[str] = []
        current = 0
        for i, hit in enumerate(hits, start=1):
            source = hit.metadata.get("source", "unknown")
            page = hit.metadata.get("page", "?")
            header = f"[{i}] source={source}, page={page}, score={hit.score:.3f}\n"
            body = hit.text.strip()
            block = header + body + "\n"

            if current + len(block) > max_context_chars:
                remaining = max_context_chars - current
                if remaining > 0:
                    lines.append(block[:remaining])
                break

            lines.append(block)
            current += len(block)

        return "\n".join(lines).strip()
