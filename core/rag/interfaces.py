from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from .models import ChunkRecord, ParsedDocument, RetrievedChunk


class DocumentParser(ABC):
    @abstractmethod
    def parse(self, path: str) -> ParsedDocument:
        raise NotImplementedError


class Chunker(ABC):
    @abstractmethod
    def chunk(self, doc: ParsedDocument) -> List[ChunkRecord]:
        raise NotImplementedError


class Embedder(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError


class VectorStore(ABC):
    @abstractmethod
    def upsert_chunks(self, chunks: List[ChunkRecord], vectors: List[List[float]]) -> int:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int,
        min_score: float,
        source_filter: Optional[str] = None,
        record_type: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        raise NotImplementedError

    def search_lexical(
        self,
        query_text: str,
        top_k: int,
        source_filter: Optional[str] = None,
        record_type: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        return []

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def has_source(self, source: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    def scroll_all(
        self,
        *,
        source_filter: Optional[str] = None,
        record_type: Optional[str] = None,
        limit: int = 10_000,
    ) -> List[ChunkRecord]:
        """Return all chunks sequentially. Override in concrete stores."""
        return []


class Retriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        min_score: float,
        source_filter: Optional[str] = None,
        record_type: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        raise NotImplementedError

    @abstractmethod
    def format_hits(self, hits: List[RetrievedChunk], max_context_chars: int) -> str:
        raise NotImplementedError


class OcrEngine(ABC):
    @abstractmethod
    def extract_text(self, image) -> Tuple[str, Optional[float]]:
        raise NotImplementedError
