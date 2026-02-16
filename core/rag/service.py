import os
import time
from typing import List, Optional

from .cache import IngestionCache
from .chunking.semantic_double_buffer import SemanticDoubleBufferChunker
from .embeddings.fastembed_embedder import FastEmbedEmbedder
from .models import IngestReport, RetrievedChunk
from .parsers.pdfium_parser import PdfiumParser
from .retriever import RagRetriever
from .stores.qdrant_store import QdrantVectorStore


class RagService:
    def __init__(
        self,
        *,
        collection_name: str,
        qdrant_url: str,
        qdrant_api_key: Optional[str],
        embedding_model: str,
        top_k: int,
        min_score: float,
        max_context_chars: int,
        cache_path: Optional[str] = None,
    ):
        self.top_k = top_k
        self.min_score = min_score
        self.max_context_chars = max_context_chars
        safe_collection = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in collection_name)
        effective_cache_path = cache_path or f".rag_cache_{safe_collection}.json"
        self.cache = IngestionCache(cache_path=effective_cache_path)

        self.parser = PdfiumParser()
        self.chunker = SemanticDoubleBufferChunker()
        self.embedder = FastEmbedEmbedder(model_name=embedding_model)
        self.store = QdrantVectorStore(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=collection_name,
        )
        self.retriever = RagRetriever(self.embedder, self.store)

    def ingest_documents(self, paths: List[str], *, force_reindex: bool = False) -> IngestReport:
        report = IngestReport()
        started = time.time()

        for path in paths:
            if not path or not os.path.exists(path):
                report.errors.append(f"File not found: {path}")
                continue

            if not force_reindex and self.cache.is_unchanged(path) and self.store.has_source(path):
                report.files_skipped += 1
                continue

            try:
                parsed = self.parser.parse(path)
                chunks = self.chunker.chunk(parsed)
                texts = [c.text for c in chunks]
                vectors = self.embedder.embed_documents(texts) if texts else []
                upserted = self.store.upsert_chunks(chunks, vectors)

                report.files_processed += 1
                report.chunks_created += len(chunks)
                report.vectors_upserted += upserted

                self.cache.mark(path)
                self.cache.save()
            except Exception as exc:
                report.errors.append(f"{path}: {exc}")

        report.duration_seconds = time.time() - started
        return report

    def search(self, query: str, *, top_k: Optional[int] = None, min_score: Optional[float] = None, source_filter: Optional[str] = None) -> List[RetrievedChunk]:
        return self.retriever.retrieve(
            query,
            top_k=top_k if top_k is not None else self.top_k,
            min_score=min_score if min_score is not None else self.min_score,
            source_filter=source_filter,
        )

    def format_hits(self, hits: List[RetrievedChunk], max_context_chars: Optional[int] = None) -> str:
        budget = max_context_chars if max_context_chars is not None else self.max_context_chars
        return self.retriever.format_hits(hits, max_context_chars=budget)

    def get_status(self) -> dict:
        return {
            "collection_size": self.store.count(),
            "top_k": self.top_k,
            "min_score": self.min_score,
            "max_context_chars": self.max_context_chars,
        }

    def clear_collection(self) -> None:
        self.store.clear()
