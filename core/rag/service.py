import os
import re
import time
from typing import Dict, List, Optional, Set

from .cache import IngestionCache
from .chunking.semantic_double_buffer import SemanticDoubleBufferChunker
from .embeddings.fastembed_embedder import FastEmbedEmbedder
from .graph import ShadowGraphIndex
from .late_interaction import LateInteractionScorer
from .models import ChunkRecord, IngestReport, ParsedDocument, RetrievedChunk
from .parsers.docling_parser import DoclingParser
from .parsers.hybrid_pdf_parser import HybridPdfParser
from .parsers.router_parser import RouterParser
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
        ocr_mode: str = "off",
        ocr_dpi: int = 150,
        ocr_max_pages: int = 20,
        ocr_max_regions_per_page: int = 8,
        ocr_region_padding_px: int = 18,
        ocr_gap_multiplier: float = 2.5,
        ocr_min_extracted_chars: int = 60,
        ocr_timeout_ms_per_page: int = 4000,
        parser_mode: str = "auto",
        hybrid_search_enabled: bool = True,
        rerank_enabled: bool = True,
        summary_first_enabled: bool = True,
        summary_top_k: int = 3,
        dense_top_k: int = 12,
        lexical_top_k: int = 12,
        parent_context_enabled: bool = True,
        parent_context_max_chars: int = 1200,
        graph_enabled: bool = True,
        graph_hops: int = 1,
        graph_source_boost: float = 0.08,
        late_interaction_enabled: bool = True,
        late_interaction_weight: float = 0.2,
        cache_path: Optional[str] = None,
    ):
        self.top_k = top_k
        self.min_score = min_score
        self.max_context_chars = max_context_chars
        self.parser_mode = (parser_mode or "auto").strip().lower()
        self.hybrid_search_enabled = bool(hybrid_search_enabled)
        self.rerank_enabled = bool(rerank_enabled)
        self.summary_first_enabled = bool(summary_first_enabled)
        self.summary_top_k = max(1, int(summary_top_k))
        self.dense_top_k = max(1, int(dense_top_k))
        self.lexical_top_k = max(1, int(lexical_top_k))
        self.parent_context_enabled = bool(parent_context_enabled)
        self.parent_context_max_chars = max(200, int(parent_context_max_chars))
        self.graph_enabled = bool(graph_enabled)
        self.graph_hops = max(1, int(graph_hops))
        self.graph_source_boost = max(0.0, float(graph_source_boost))
        self.late_interaction_enabled = bool(late_interaction_enabled)
        self.late_interaction_weight = min(1.0, max(0.0, float(late_interaction_weight)))
        safe_collection = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in collection_name)
        effective_cache_path = cache_path or f".rag_cache_{safe_collection}.json"
        self.cache = IngestionCache(cache_path=effective_cache_path)

        self.parser = HybridPdfParser(
            ocr_mode=ocr_mode,
            ocr_dpi=ocr_dpi,
            ocr_max_pages=ocr_max_pages,
            ocr_max_regions_per_page=ocr_max_regions_per_page,
            ocr_region_padding_px=ocr_region_padding_px,
            ocr_gap_multiplier=ocr_gap_multiplier,
            ocr_min_extracted_chars=ocr_min_extracted_chars,
            ocr_timeout_ms_per_page=ocr_timeout_ms_per_page,
        )
        self.router_parser = RouterParser(pdf_parser=self.parser)
        self._docling_parser: Optional[DoclingParser] = None
        self._docling_error: Optional[str] = None
        if self.parser_mode == "docling":
            try:
                self._docling_parser = DoclingParser()
            except Exception as exc:
                self._docling_error = str(exc)
                # Degrade gracefully to auto routing if Docling is unavailable.
                self.parser_mode = "auto"
        self.chunker = SemanticDoubleBufferChunker()
        self.embedder = FastEmbedEmbedder(model_name=embedding_model)
        self.store = QdrantVectorStore(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=collection_name,
        )
        self.retriever = RagRetriever(self.embedder, self.store)
        self.graph_index = ShadowGraphIndex(enabled=self.graph_enabled)

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        return set(re.findall(r"[a-z0-9]+", (text or "").lower()))

    @staticmethod
    def _infer_doc_purpose(source: str, text: str) -> str:
        lowered = (text or "").lower()
        ext = os.path.splitext(source)[1].lower()
        if ext in {".xlsx", ".xls", ".csv"}:
            return "tabular_data"
        if "invoice" in lowered:
            return "invoice"
        if "contract" in lowered or "agreement" in lowered:
            return "contract"
        if "action item" in lowered or "next steps" in lowered:
            return "action_or_plan"
        return "general_document"

    @staticmethod
    def _infer_security_label(text: str) -> str:
        lowered = (text or "").lower()
        pii_markers = ("ssn", "social security", "passport", "date of birth", "email", "phone")
        return "pii_detected" if any(marker in lowered for marker in pii_markers) else "internal"

    def _build_doc_summary(self, parsed: ParsedDocument) -> str:
        pages = [p.strip() for p in parsed.pages if p and p.strip()]
        if not pages:
            return ""
        joined = " ".join(pages)
        pieces = re.split(r"(?<=[.!?])\s+", joined)
        summary = " ".join(pieces[:12]).strip()
        if len(summary) < 120:
            summary = joined[:2500].strip()
        return summary[:4000]

    def _enrich_chunks(self, parsed: ParsedDocument, chunks: List[ChunkRecord]) -> List[ChunkRecord]:
        out: List[ChunkRecord] = []
        for chunk in chunks:
            page_num = int(chunk.metadata.get("page") or 0)
            parent_text = ""
            if 1 <= page_num <= len(parsed.pages):
                parent_text = (parsed.pages[page_num - 1] or "").strip()
            parent_text = parent_text[: max(1000, self.parent_context_max_chars * 2)]

            combined_text = f"{chunk.text}\n{parent_text}".strip()
            metadata = {
                **chunk.metadata,
                "chunk_id": chunk.chunk_id,
                "record_type": "chunk",
                "source_type": parsed.source_type or "file",
                "source": parsed.source,
                "parent_text": parent_text,
                "doc_purpose": self._infer_doc_purpose(parsed.source, combined_text),
                "security_label": self._infer_security_label(combined_text),
            }
            out.append(ChunkRecord(chunk_id=chunk.chunk_id, text=chunk.text, metadata=metadata))
        return out

    def _summary_chunk(self, parsed: ParsedDocument) -> Optional[ChunkRecord]:
        summary = self._build_doc_summary(parsed)
        if not summary:
            return None
        chunk_id = f"{parsed.source}::summary"
        return ChunkRecord(
            chunk_id=chunk_id,
            text=summary,
            metadata={
                "chunk_id": chunk_id,
                "record_type": "doc_summary",
                "source": parsed.source,
                "source_type": parsed.source_type or "file",
                "page": 0,
                "doc_purpose": self._infer_doc_purpose(parsed.source, summary),
                "security_label": self._infer_security_label(summary),
            },
        )

    def _parse_source(self, source: str) -> ParsedDocument:
        if self.parser_mode == "pdf_only":
            return self.parser.parse(source)
        if self.parser_mode == "docling":
            # Docling path for local files, router path for URLs.
            if source.lower().startswith(("http://", "https://")):
                return self.router_parser.parse(source)
            if self._docling_parser is not None:
                try:
                    return self._docling_parser.parse(source)
                except Exception:
                    # Fallback to robust router parser on conversion edge-cases.
                    return self.router_parser.parse(source)
        return self.router_parser.parse(source)

    def ingest_documents(self, paths: List[str], *, force_reindex: bool = False) -> IngestReport:
        report = IngestReport()
        started = time.time()

        for source in paths:
            is_url = bool(source and source.lower().startswith(("http://", "https://")))
            if not source or (not is_url and not os.path.exists(source)):
                report.errors.append(f"Source not found: {source}")
                continue

            if not is_url and not force_reindex and self.cache.is_unchanged(source) and self.store.has_source(source):
                report.files_skipped += 1
                continue

            try:
                parsed = self._parse_source(source)
                chunks = self._enrich_chunks(parsed, self.chunker.chunk(parsed))
                summary_chunk = self._summary_chunk(parsed)
                if summary_chunk is not None:
                    chunks.append(summary_chunk)

                # Keep a lightweight "shadow graph" for association retrieval.
                joined = " ".join([p for p in parsed.pages if p]).strip()
                graph_tokens = [
                    str(parsed.source_type or ""),
                    str(parsed.metadata.get("source_type") or ""),
                    str(parsed.metadata.get("title") or ""),
                    str(parsed.metadata.get("content_type") or ""),
                ]
                self.graph_index.upsert(parsed.source, joined[:12000], extra_tokens=graph_tokens)

                chunks = [c for c in chunks if c.text and c.text.strip()]
                texts = [c.text for c in chunks]
                vectors = self.embedder.embed_documents(texts) if texts else []
                upserted = self.store.upsert_chunks(chunks, vectors)

                report.files_processed += 1
                report.chunks_created += len(chunks)
                report.vectors_upserted += upserted

                for meta in parsed.page_metadata:
                    if meta.get("ocr_used"):
                        report.ocr_pages_total += 1
                        if meta.get("ocr_scope") == "full_page":
                            report.ocr_pages_full += 1
                    report.ocr_regions_total += int(meta.get("ocr_regions_count") or 0)
                    if meta.get("ocr_error"):
                        report.ocr_failures += 1

                if not is_url:
                    self.cache.mark(source)
                    self.cache.save()
            except Exception as exc:
                report.errors.append(f"{source}: {exc}")

        report.duration_seconds = time.time() - started
        return report

    def search(self, query: str, *, top_k: Optional[int] = None, min_score: Optional[float] = None, source_filter: Optional[str] = None) -> List[RetrievedChunk]:
        wanted_k = top_k if top_k is not None else self.top_k
        threshold = min_score if min_score is not None else self.min_score
        candidate_sources: List[str] = []
        graph_boost_sources: Set[str] = set()

        if self.summary_first_enabled:
            summary_hits = self.retriever.retrieve(
                query,
                top_k=max(self.summary_top_k, wanted_k),
                min_score=0.0,
                source_filter=source_filter,
                record_type="doc_summary",
            )
            for hit in summary_hits:
                src = str(hit.metadata.get("source") or "").strip()
                if src and src not in candidate_sources:
                    candidate_sources.append(src)

        graph_enabled = bool(getattr(self, "graph_enabled", False))
        graph_hops = max(1, int(getattr(self, "graph_hops", 1)))
        graph_source_boost = max(0.0, float(getattr(self, "graph_source_boost", 0.0)))
        graph_index = getattr(self, "graph_index", None)

        if graph_enabled and graph_index is not None:
            seed_sources = list(candidate_sources)
            if source_filter and source_filter not in seed_sources:
                seed_sources.append(source_filter)
            if seed_sources:
                related = graph_index.related_sources(
                    seed_sources,
                    hops=graph_hops,
                    limit=max(10, wanted_k * 5),
                )
                for src in related:
                    if src not in candidate_sources:
                        candidate_sources.append(src)
                        graph_boost_sources.add(src)

        dense_hits: List[RetrievedChunk] = []
        lexical_hits: List[RetrievedChunk] = []

        search_sources = candidate_sources or ([source_filter] if source_filter else [])
        if search_sources:
            for src in search_sources:
                dense_hits.extend(
                    self.retriever.retrieve(
                        query,
                        top_k=max(wanted_k, self.dense_top_k),
                        min_score=threshold,
                        source_filter=src,
                        record_type="chunk",
                    )
                )
                if self.hybrid_search_enabled:
                    lexical_hits.extend(
                        self.retriever.retrieve_lexical(
                            query,
                            top_k=max(wanted_k, self.lexical_top_k),
                            source_filter=src,
                            record_type="chunk",
                        )
                    )
        else:
            dense_hits = self.retriever.retrieve(
                query,
                top_k=max(wanted_k, self.dense_top_k),
                min_score=threshold,
                source_filter=source_filter,
                record_type="chunk",
            )
            if self.hybrid_search_enabled:
                lexical_hits = self.retriever.retrieve_lexical(
                    query,
                    top_k=max(wanted_k, self.lexical_top_k),
                    source_filter=source_filter,
                    record_type="chunk",
                )

        hits = self._fuse_hits(dense_hits, lexical_hits, top_k=wanted_k)
        if self.rerank_enabled:
            hits = self._rerank(query, hits)
        if graph_boost_sources and graph_source_boost > 0:
            for hit in hits:
                src = str(hit.metadata.get("source") or "")
                if src in graph_boost_sources:
                    hit.score += graph_source_boost
                    hit.metadata["graph_boost"] = graph_source_boost
            hits.sort(key=lambda x: x.score, reverse=True)
        return hits[: max(1, wanted_k)]

    def _fuse_hits(self, dense_hits: List[RetrievedChunk], lexical_hits: List[RetrievedChunk], *, top_k: int) -> List[RetrievedChunk]:
        by_id: Dict[str, RetrievedChunk] = {}
        dense_max = max([h.score for h in dense_hits], default=1.0)
        lex_max = max([h.score for h in lexical_hits], default=1.0)

        for h in dense_hits:
            md = dict(h.metadata)
            md["dense_score"] = h.score
            md.setdefault("lexical_score", 0.0)
            existing = by_id.get(h.chunk_id)
            if existing is None or h.score > existing.score:
                by_id[h.chunk_id] = RetrievedChunk(
                    chunk_id=h.chunk_id,
                    text=h.text,
                    score=h.score,
                    metadata=md,
                )

        for h in lexical_hits:
            existing = by_id.get(h.chunk_id)
            if existing is None:
                by_id[h.chunk_id] = RetrievedChunk(
                    chunk_id=h.chunk_id,
                    text=h.text,
                    score=0.0,
                    metadata={"dense_score": 0.0, "lexical_score": h.score, **h.metadata},
                )
            else:
                existing.metadata["lexical_score"] = h.score

        fused: List[RetrievedChunk] = []
        for hit in by_id.values():
            dense_norm = float(hit.metadata.get("dense_score", 0.0)) / max(1e-9, dense_max)
            lex_norm = float(hit.metadata.get("lexical_score", 0.0)) / max(1e-9, lex_max)
            fused_score = 0.65 * dense_norm + 0.35 * lex_norm
            hit.score = fused_score
            if self.parent_context_enabled:
                parent_text = (hit.metadata.get("parent_text") or "").strip()
                if parent_text:
                    hit.metadata["parent_excerpt"] = parent_text[: self.parent_context_max_chars]
            fused.append(hit)

        fused.sort(key=lambda x: x.score, reverse=True)
        return fused[: max(1, top_k * 3)]

    def _rerank(self, query: str, hits: List[RetrievedChunk]) -> List[RetrievedChunk]:
        if not hits:
            return hits
        q_tokens = self._tokenize(query)
        for hit in hits:
            body_tokens = self._tokenize(hit.text)
            overlap = len(q_tokens.intersection(body_tokens)) / max(1, len(q_tokens))
            dense = float(hit.metadata.get("dense_score", 0.0))
            lex = float(hit.metadata.get("lexical_score", 0.0))
            base_score = 0.55 * hit.score + 0.30 * min(1.0, dense) + 0.10 * min(1.0, lex) + 0.05 * overlap
            late_enabled = bool(getattr(self, "late_interaction_enabled", False))
            late_weight = min(1.0, max(0.0, float(getattr(self, "late_interaction_weight", 0.0))))
            if late_enabled and late_weight > 0:
                li = LateInteractionScorer.score(query, hit.text)
                hit.metadata["late_interaction_score"] = li
                hit.score = (1.0 - late_weight) * base_score + late_weight * li
            else:
                hit.score = base_score
        hits.sort(key=lambda x: x.score, reverse=True)
        return hits

    def format_hits(self, hits: List[RetrievedChunk], max_context_chars: Optional[int] = None) -> str:
        budget = max_context_chars if max_context_chars is not None else self.max_context_chars
        return self.retriever.format_hits(hits, max_context_chars=budget)

    def get_status(self) -> dict:
        gstats = self.graph_index.stats()
        return {
            "collection_size": self.store.count(),
            "top_k": self.top_k,
            "min_score": self.min_score,
            "max_context_chars": self.max_context_chars,
            "ocr_mode": getattr(self.parser, "ocr_mode", "off"),
            "parser_mode": self.parser_mode,
            "hybrid_search_enabled": self.hybrid_search_enabled,
            "summary_first_enabled": self.summary_first_enabled,
            "rerank_enabled": self.rerank_enabled,
            "graph_enabled": self.graph_enabled,
            "graph_sources": gstats.sources,
            "graph_entities": gstats.entities,
            "late_interaction_enabled": self.late_interaction_enabled,
            "docling_error": self._docling_error,
        }

    def clear_collection(self) -> None:
        self.store.clear()
