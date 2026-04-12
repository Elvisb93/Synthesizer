from typing import List, Optional, Protocol

from core.rag.models import ChunkRecord, IngestReport, RetrievedChunk


class RagBackendProtocol(Protocol):
    top_k: int
    min_score: float
    max_context_chars: int

    def ingest_documents(self, paths: List[str], *, force_reindex: bool = False) -> IngestReport:
        ...

    def search(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        source_filter: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        ...

    def format_hits(self, hits: List[RetrievedChunk], max_context_chars: Optional[int] = None) -> str:
        ...

    def get_status(self) -> dict:
        ...

    def clear_collection(self) -> None:
        ...

    def get_all_chunks(
        self,
        *,
        source_filter: Optional[str] = None,
        limit: int = 10_000,
    ) -> List[ChunkRecord]:
        ...
