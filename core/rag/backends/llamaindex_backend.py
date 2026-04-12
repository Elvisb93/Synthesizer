import json
import os
import re
import time
from typing import Dict, Iterable, List, Optional, Set
from uuid import NAMESPACE_URL, uuid5

from core.rag.cache import IngestionCache
from core.rag.models import ChunkRecord, IngestReport, ParsedDocument, RetrievedChunk
from core.rag.parsers.docling_parser import DoclingParser
from core.rag.parsers.hybrid_pdf_parser import HybridPdfParser
from core.rag.parsers.router_parser import RouterParser


class LlamaIndexRagService:
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
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.embedding_model_name = embedding_model
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
        effective_cache_path = cache_path or f".rag_cache_{safe_collection}_llamaindex.json"
        self.cache = IngestionCache(cache_path=effective_cache_path)
        self._manifest_path = f".rag_manifest_{safe_collection}_llamaindex.json"
        self._manifest = self._load_manifest()

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
                self.parser_mode = "auto"

        self._client = None
        self._vector_store = None
        self._embed_model = None
        self._index = None
        self._small_to_big_enabled = False

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        return set(re.findall(r"[a-z0-9]+", (text or "").lower()))

    @staticmethod
    def _stable_uuid(value: str) -> str:
        return str(uuid5(NAMESPACE_URL, value))

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

    def _load_manifest(self) -> Dict[str, List[Dict[str, object]]]:
        if not os.path.exists(self._manifest_path):
            return {}
        try:
            with open(self._manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                cleaned: Dict[str, List[Dict[str, object]]] = {}
                for source, entries in data.items():
                    if isinstance(entries, list):
                        cleaned[str(source)] = [e for e in entries if isinstance(e, dict)]
                return cleaned
        except Exception:
            return {}
        return {}

    def _save_manifest(self) -> None:
        with open(self._manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, indent=2, ensure_ascii=True)

    def _replace_manifest_source(self, source: str, records: List[ChunkRecord]) -> None:
        self._manifest[source] = [
            {
                "chunk_id": r.chunk_id,
                "text": r.text,
                "metadata": r.metadata,
            }
            for r in records
        ]
        self._save_manifest()

    def _remove_manifest_source(self, source: str) -> None:
        if source in self._manifest:
            del self._manifest[source]
            self._save_manifest()

    def _clear_manifest(self) -> None:
        self._manifest = {}
        if os.path.exists(self._manifest_path):
            try:
                os.remove(self._manifest_path)
            except OSError:
                pass

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError("qdrant-client is required for the LlamaIndex RAG backend") from exc

        if self.qdrant_url in (":memory:", "memory://", "local-memory"):
            self._client = QdrantClient(path=":memory:")
        else:
            self._client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        return self._client

    def _collection_exists(self) -> bool:
        try:
            collections = self._get_client().get_collections().collections
            return any(c.name == self.collection_name for c in collections)
        except Exception:
            return False

    def _build_filter(self, source_filter: Optional[str], record_type: Optional[str]):
        try:
            from qdrant_client import models as qm
        except ImportError:
            return None

        must = []
        if source_filter:
            must.append(qm.FieldCondition(key="source", match=qm.MatchValue(value=source_filter)))
        if record_type:
            must.append(qm.FieldCondition(key="record_type", match=qm.MatchValue(value=record_type)))
        if not must:
            return None
        return qm.Filter(must=must)

    def _count(self, *, source_filter: Optional[str] = None, record_type: Optional[str] = None) -> int:
        if not self._collection_exists():
            return 0
        try:
            filt = self._build_filter(source_filter=source_filter, record_type=record_type)
            result = self._get_client().count(
                collection_name=self.collection_name,
                count_filter=filt,
                exact=True,
            )
            return int(result.count)
        except Exception:
            return 0

    def _delete_source_records(self, source: str) -> None:
        if not source or not self._collection_exists():
            self._remove_manifest_source(source)
            return
        try:
            from qdrant_client import models as qm

            filt = self._build_filter(source_filter=source, record_type=None)
            if filt is None:
                return
            self._get_client().delete(
                collection_name=self.collection_name,
                points_selector=qm.FilterSelector(filter=filt),
            )
        except Exception:
            pass
        self._remove_manifest_source(source)

    def _get_vector_store(self):
        if self._vector_store is not None:
            return self._vector_store
        try:
            from llama_index.vector_stores.qdrant import QdrantVectorStore
        except ImportError as exc:
            raise RuntimeError(
                "Install llama-index-vector-stores-qdrant to use the LlamaIndex RAG backend."
            ) from exc

        self._vector_store = QdrantVectorStore(
            collection_name=self.collection_name,
            client=self._get_client(),
        )
        return self._vector_store

    def _get_embed_model(self):
        if self._embed_model is not None:
            return self._embed_model
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "Install llama-index-embeddings-huggingface to use the LlamaIndex RAG backend."
            ) from exc

        self._embed_model = HuggingFaceEmbedding(model_name=self.embedding_model_name)
        return self._embed_model

    def _create_index(self, nodes=None):
        from llama_index.core import StorageContext, VectorStoreIndex

        storage_context = StorageContext.from_defaults(vector_store=self._get_vector_store())
        return VectorStoreIndex(
            nodes=list(nodes or []),
            storage_context=storage_context,
            embed_model=self._get_embed_model(),
        )

    def _get_index(self):
        if self._index is not None:
            return self._index
        from llama_index.core import VectorStoreIndex

        self._index = VectorStoreIndex.from_vector_store(
            vector_store=self._get_vector_store(),
            embed_model=self._get_embed_model(),
        )
        return self._index

    def _build_metadata_filters(self, *, source_filter: Optional[str], record_type: Optional[str]):
        filters = []
        try:
            from llama_index.core.vector_stores.types import ExactMatchFilter, MetadataFilters
        except ImportError:
            return None

        if source_filter:
            filters.append(ExactMatchFilter(key="source", value=source_filter))
        if record_type:
            filters.append(ExactMatchFilter(key="record_type", value=record_type))
        if not filters:
            return None
        return MetadataFilters(filters=filters)

    def _get_node_parser(self):
        try:
            from llama_index.core.node_parser import SentenceWindowNodeParser

            self._small_to_big_enabled = True
            return SentenceWindowNodeParser.from_defaults(
                window_size=2,
                window_metadata_key="window",
                original_text_metadata_key="original_text",
            )
        except Exception:
            from llama_index.core.node_parser import SentenceSplitter

            self._small_to_big_enabled = False
            return SentenceSplitter(chunk_size=512, chunk_overlap=64)

    def _node_text(self, node) -> str:
        metadata = dict(getattr(node, "metadata", {}) or {})
        if metadata.get("window"):
            return str(metadata["window"]).strip()
        if metadata.get("parent_text"):
            return str(metadata["parent_text"]).strip()
        try:
            from llama_index.core.schema import MetadataMode

            return str(node.get_content(metadata_mode=MetadataMode.NONE)).strip()
        except Exception:
            return str(getattr(node, "text", "")).strip()

    def _node_to_hit(self, item) -> RetrievedChunk:
        node = getattr(item, "node", item)
        metadata = dict(getattr(node, "metadata", {}) or {})
        score = float(getattr(item, "score", 0.0) or 0.0)
        chunk_id = str(metadata.get("chunk_id") or getattr(node, "node_id", "") or "")
        if not chunk_id:
            chunk_id = str(getattr(node, "node_id", ""))
        return RetrievedChunk(
            chunk_id=chunk_id,
            text=self._node_text(node),
            score=score,
            metadata=metadata,
        )

    def _python_filter_hits(
        self,
        hits: List[RetrievedChunk],
        *,
        source_filter: Optional[str],
        record_type: Optional[str],
    ) -> List[RetrievedChunk]:
        out: List[RetrievedChunk] = []
        for hit in hits:
            if source_filter and str(hit.metadata.get("source") or "") != source_filter:
                continue
            if record_type and str(hit.metadata.get("record_type") or "") != record_type:
                continue
            out.append(hit)
        return out

    def _retrieve_dense(
        self,
        query: str,
        *,
        top_k: int,
        source_filter: Optional[str] = None,
        record_type: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        if not query.strip():
            return []
        if self._count() <= 0:
            return []

        retriever_kwargs = {"similarity_top_k": max(1, top_k)}
        metadata_filters = self._build_metadata_filters(source_filter=source_filter, record_type=record_type)
        if metadata_filters is not None:
            retriever_kwargs["filters"] = metadata_filters

        try:
            retriever = self._get_index().as_retriever(**retriever_kwargs)
            raw_hits = retriever.retrieve(query)
        except Exception:
            return []

        hits = [self._node_to_hit(item) for item in raw_hits]
        if metadata_filters is None:
            hits = self._python_filter_hits(hits, source_filter=source_filter, record_type=record_type)
        return hits

    def _parse_source(self, source: str) -> ParsedDocument:
        if self.parser_mode == "pdf_only":
            return self.parser.parse(source)
        if self.parser_mode == "docling":
            if source.lower().startswith(("http://", "https://")):
                return self.router_parser.parse(source)
            if self._docling_parser is not None:
                try:
                    return self._docling_parser.parse(source)
                except Exception:
                    return self.router_parser.parse(source)
        return self.router_parser.parse(source)

    def _build_nodes(self, parsed: ParsedDocument):
        from llama_index.core import Document
        from llama_index.core.schema import TextNode

        page_docs = []
        manifest_records: List[ChunkRecord] = []
        for page_num, page_text in enumerate(parsed.pages, start=1):
            cleaned = (page_text or "").strip()
            if not cleaned:
                continue
            parent_text = cleaned[: max(1000, self.parent_context_max_chars * 2)]
            metadata = {
                "source": parsed.source,
                "source_type": parsed.source_type or "file",
                "page": page_num,
                "record_type": "chunk",
                "parent_text": parent_text,
                "doc_purpose": self._infer_doc_purpose(parsed.source, cleaned),
                "security_label": self._infer_security_label(cleaned),
            }
            page_docs.append(Document(text=cleaned, metadata=metadata))

        parser = self._get_node_parser()
        chunk_nodes = parser.get_nodes_from_documents(page_docs) if page_docs else []
        for idx, node in enumerate(chunk_nodes, start=1):
            metadata = dict(getattr(node, "metadata", {}) or {})
            metadata.setdefault("source", parsed.source)
            metadata.setdefault("source_type", parsed.source_type or "file")
            metadata.setdefault("record_type", "chunk")
            metadata.setdefault("page", metadata.get("page") or 1)
            metadata.setdefault("chunk_id", f"{parsed.source}::chunk::{idx}")
            try:
                node.id_ = self._stable_uuid(str(metadata["chunk_id"]))
            except Exception:
                pass
            node.metadata = metadata
            manifest_records.append(
                ChunkRecord(
                    chunk_id=str(metadata["chunk_id"]),
                    text=self._node_text(node),
                    metadata=metadata,
                )
            )

        summary_nodes = []
        summary = self._build_doc_summary(parsed)
        if summary:
            summary_id = f"{parsed.source}::summary"
            summary_nodes.append(
                TextNode(
                    text=summary,
                    id_=self._stable_uuid(summary_id),
                    metadata={
                        "chunk_id": summary_id,
                        "source": parsed.source,
                        "source_type": parsed.source_type or "file",
                        "page": 0,
                        "record_type": "doc_summary",
                        "doc_purpose": self._infer_doc_purpose(parsed.source, summary),
                        "security_label": self._infer_security_label(summary),
                    },
                )
            )

        return chunk_nodes + summary_nodes, manifest_records

    def _has_source(self, source: str) -> bool:
        if not source:
            return False
        if source in self._manifest:
            return True
        return self._count(source_filter=source) > 0

    def ingest_documents(self, paths: List[str], *, force_reindex: bool = False) -> IngestReport:
        report = IngestReport()
        started = time.time()

        for source in paths:
            is_url = bool(source and source.lower().startswith(("http://", "https://")))
            if not source or (not is_url and not os.path.exists(source)):
                report.errors.append(f"Source not found: {source}")
                continue

            if not is_url and not force_reindex and self.cache.is_unchanged(source) and self._has_source(source):
                report.files_skipped += 1
                continue

            try:
                parsed = self._parse_source(source)
                nodes, manifest_records = self._build_nodes(parsed)
                if not nodes:
                    report.errors.append(f"{source}: no indexable content found")
                    continue

                if force_reindex or self._has_source(source):
                    self._delete_source_records(source)

                if self._index is None and self._count() == 0:
                    self._index = self._create_index(nodes)
                else:
                    self._get_index().insert_nodes(nodes)

                self._replace_manifest_source(source, manifest_records)
                report.files_processed += 1
                report.chunks_created += len(nodes)
                report.vectors_upserted += len(nodes)

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

    def search(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        source_filter: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        wanted_k = top_k if top_k is not None else self.top_k
        threshold = min_score if min_score is not None else self.min_score
        candidate_sources: List[str] = []

        if self.summary_first_enabled:
            summary_hits = self._retrieve_dense(
                query,
                top_k=max(self.summary_top_k * 4, wanted_k * 2),
                source_filter=source_filter,
                record_type="doc_summary",
            )
            for hit in summary_hits[: self.summary_top_k]:
                src = str(hit.metadata.get("source") or "").strip()
                if src and src not in candidate_sources:
                    candidate_sources.append(src)

        dense_hits: List[RetrievedChunk] = []
        if candidate_sources:
            for src in candidate_sources:
                dense_hits.extend(
                    self._retrieve_dense(
                        query,
                        top_k=max(self.dense_top_k, wanted_k * 3),
                        source_filter=src,
                        record_type="chunk",
                    )
                )
        else:
            dense_hits = self._retrieve_dense(
                query,
                top_k=max(self.dense_top_k, wanted_k * 3),
                source_filter=source_filter,
                record_type="chunk",
            )

        by_id: Dict[str, RetrievedChunk] = {}
        for hit in dense_hits:
            if hit.score < threshold:
                continue
            existing = by_id.get(hit.chunk_id)
            if existing is None or hit.score > existing.score:
                if self.parent_context_enabled:
                    parent_text = (hit.metadata.get("parent_text") or "").strip()
                    if parent_text:
                        hit.metadata["parent_excerpt"] = parent_text[: self.parent_context_max_chars]
                by_id[hit.chunk_id] = hit

        hits = list(by_id.values())
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[: max(1, wanted_k)]

    def format_hits(self, hits: List[RetrievedChunk], max_context_chars: Optional[int] = None) -> str:
        budget = max_context_chars if max_context_chars is not None else self.max_context_chars
        if not hits or budget <= 0:
            return ""

        lines: List[str] = []
        current = 0
        for i, hit in enumerate(hits, start=1):
            source = hit.metadata.get("source", "unknown")
            page = hit.metadata.get("page", "?")
            header = f"[{i}] source={source}, page={page}, score={hit.score:.3f}\n"
            body = hit.text.strip()
            parent_excerpt = (hit.metadata.get("parent_excerpt") or "").strip()
            if parent_excerpt:
                body = f"{body}\n[Parent Context]\n{parent_excerpt}"
            block = header + body + "\n"

            if current + len(block) > budget:
                remaining = budget - current
                if remaining > 0:
                    lines.append(block[:remaining])
                break

            lines.append(block)
            current += len(block)

        return "\n".join(lines).strip()

    def get_status(self) -> dict:
        return {
            "backend": "llamaindex",
            "collection_size": self._count(),
            "top_k": self.top_k,
            "min_score": self.min_score,
            "max_context_chars": self.max_context_chars,
            "ocr_mode": getattr(self.parser, "ocr_mode", "off"),
            "parser_mode": self.parser_mode,
            "hybrid_search_enabled": False,
            "summary_first_enabled": self.summary_first_enabled,
            "rerank_enabled": False,
            "graph_enabled": False,
            "graph_sources": 0,
            "graph_entities": 0,
            "late_interaction_enabled": False,
            "docling_error": self._docling_error,
            "small_to_big_enabled": self._small_to_big_enabled,
        }

    def clear_collection(self) -> None:
        if self._collection_exists():
            try:
                self._get_client().delete_collection(self.collection_name)
            except Exception:
                pass
        self._index = None
        self._clear_manifest()

    def get_all_chunks(
        self,
        *,
        source_filter: Optional[str] = None,
        limit: int = 10_000,
    ) -> List[ChunkRecord]:
        out: List[ChunkRecord] = []
        for source, records in self._manifest.items():
            if source_filter and source != source_filter:
                continue
            for record in records:
                out.append(
                    ChunkRecord(
                        chunk_id=str(record.get("chunk_id") or ""),
                        text=str(record.get("text") or ""),
                        metadata=dict(record.get("metadata") or {}),
                    )
                )
                if len(out) >= max(1, limit):
                    return out
        return out
