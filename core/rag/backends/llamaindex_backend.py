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

from .local_openai_llm import LocalOpenAICompatibleLLM


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
        llm_model_name: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_temperature: float = 0.0,
        llm_context_window: int = 16384,
        llm_num_output: int = 768,
        llm_enabled: bool = True,
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
        self.llm_model_name = (llm_model_name or "").strip() or None
        self.llm_base_url = (llm_base_url or "").strip() or None
        self.llm_api_key = (llm_api_key or "lm-studio").strip() or "lm-studio"
        self.llm_temperature = float(llm_temperature)
        self.llm_context_window = max(1024, int(llm_context_window))
        self.llm_num_output = max(128, int(llm_num_output))
        self.llm_enabled = bool(llm_enabled and self.llm_model_name and self.llm_base_url)
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
        self._llm = None
        self._llm_error: Optional[str] = None
        self._metadata_extractors = None
        self._metadata_extraction_error: Optional[str] = None
        self._ingestion_cache = None
        self._prepare_nodes_transform = None
        self._response_synthesizers: Dict[str, object] = {}
        self._pipeline = None
        self._pipeline_uses_metadata_extractors = False
        self._index = None
        self._small_to_big_enabled = False
        self._ingestion_cache_path = f".rag_ingestion_cache_{safe_collection}_llamaindex.json"
        self._metadata_extraction_max_documents = 6
        self._metadata_extraction_max_chars = 18_000
        self._last_metadata_extraction_enabled: Optional[bool] = None
        self._last_metadata_extraction_reason: Optional[str] = None

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

    def _get_ingestion_cache(self):
        if self._ingestion_cache is not None:
            return self._ingestion_cache
        try:
            from llama_index.core.ingestion.cache import IngestionCache as LlamaIngestionCache
        except ImportError as exc:
            raise RuntimeError("Install llama-index-core to use the LlamaIndex ingestion pipeline.") from exc

        if os.path.exists(self._ingestion_cache_path):
            self._ingestion_cache = LlamaIngestionCache.from_persist_path(self._ingestion_cache_path)
        else:
            self._ingestion_cache = LlamaIngestionCache()
        return self._ingestion_cache

    def _persist_ingestion_cache(self) -> None:
        if self._ingestion_cache is None:
            return
        self._ingestion_cache.persist(self._ingestion_cache_path)

    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        if not self.llm_enabled:
            return None
        try:
            self._llm = LocalOpenAICompatibleLLM(
                model_name=self.llm_model_name,
                base_url=self.llm_base_url,
                api_key=self.llm_api_key,
                temperature=self.llm_temperature,
                context_window=self.llm_context_window,
                num_output=self.llm_num_output,
            )
            return self._llm
        except Exception as exc:
            self._llm_error = str(exc)
            return None

    def _get_metadata_extractors(self):
        if self._metadata_extractors is not None:
            return self._metadata_extractors
        llm = self._get_llm()
        if llm is None:
            self._metadata_extractors = []
            return self._metadata_extractors
        try:
            from llama_index.core.extractors import (
                KeywordExtractor,
                QuestionsAnsweredExtractor,
                SummaryExtractor,
                TitleExtractor,
            )
        except ImportError:
            self._metadata_extractors = []
            return self._metadata_extractors

        self._metadata_extractors = [
            TitleExtractor(llm=llm, nodes=3, num_workers=1),
            SummaryExtractor(llm=llm, summaries=["self"], num_workers=1),
            KeywordExtractor(llm=llm, keywords=5, num_workers=1),
            QuestionsAnsweredExtractor(llm=llm, questions=3, embedding_only=True, num_workers=1),
        ]
        return self._metadata_extractors

    def _build_ingestion_transformations(self, include_metadata_extractors: bool) -> List[object]:
        transforms: List[object] = [self._get_node_parser()]
        if include_metadata_extractors:
            transforms.extend(self._get_metadata_extractors())
        transforms.extend(
            [
                self._get_prepare_nodes_transform(),
                self._get_embed_model(),
            ]
        )
        return transforms

    def _get_ingestion_pipeline(self, include_metadata_extractors: Optional[bool] = None):
        wants_metadata = self.llm_enabled if include_metadata_extractors is None else bool(include_metadata_extractors)
        if self._pipeline is not None and self._pipeline_uses_metadata_extractors == wants_metadata:
            return self._pipeline
        try:
            from llama_index.core.ingestion import IngestionPipeline
        except ImportError as exc:
            raise RuntimeError("Install llama-index-core to use the LlamaIndex ingestion pipeline.") from exc

        self._pipeline = IngestionPipeline(
            transformations=self._build_ingestion_transformations(wants_metadata),
            vector_store=self._get_vector_store(),
            cache=self._get_ingestion_cache(),
        )
        self._pipeline_uses_metadata_extractors = wants_metadata
        return self._pipeline

    def _get_response_synthesizer(self, response_mode: str):
        llm = self._get_llm()
        if llm is None:
            return None
        if response_mode in self._response_synthesizers:
            return self._response_synthesizers[response_mode]
        try:
            from llama_index.core import get_response_synthesizer
        except ImportError as exc:
            raise RuntimeError("Install llama-index-core to use LlamaIndex response synthesis.") from exc

        synthesizer = get_response_synthesizer(llm=llm, response_mode=response_mode)
        self._response_synthesizers[response_mode] = synthesizer
        return synthesizer

    def _get_prepare_nodes_transform(self):
        if self._prepare_nodes_transform is not None:
            return self._prepare_nodes_transform
        try:
            from llama_index.core.schema import TransformComponent
        except ImportError as exc:
            raise RuntimeError("Install llama-index-core to use the LlamaIndex ingestion pipeline.") from exc

        class PrepareIngestedNodes(TransformComponent):
            def __call__(self, nodes, **kwargs):
                for node in nodes:
                    metadata = dict(getattr(node, "metadata", {}) or {})
                    metadata.setdefault(
                        "chunk_id",
                        str(getattr(node, "node_id", "") or getattr(node, "id_", "") or ""),
                    )
                    node.metadata = metadata

                    excluded_embed = list(getattr(node, "excluded_embed_metadata_keys", []) or [])
                    excluded_llm = list(getattr(node, "excluded_llm_metadata_keys", []) or [])
                    for key in ("parent_text", "window", "original_text"):
                        if key not in excluded_embed:
                            excluded_embed.append(key)
                    for key in ("original_text", "questions_this_excerpt_can_answer", "excerpt_keywords"):
                        if key not in excluded_llm:
                            excluded_llm.append(key)
                    node.excluded_embed_metadata_keys = excluded_embed
                    node.excluded_llm_metadata_keys = excluded_llm
                return nodes

        self._prepare_nodes_transform = PrepareIngestedNodes()
        return self._prepare_nodes_transform

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
        def stable_id(chunk_index: int, document) -> str:
            metadata = dict(getattr(document, "metadata", {}) or {})
            source = str(metadata.get("source") or "")
            page = str(metadata.get("page") or "0")
            record_type = str(metadata.get("record_type") or "chunk")
            source_doc_id = str(getattr(document, "doc_id", "") or f"{source}::{page}::{record_type}")
            return self._stable_uuid(f"{source_doc_id}::chunk::{chunk_index}")

        try:
            from llama_index.core.node_parser import SentenceWindowNodeParser

            self._small_to_big_enabled = True
            return SentenceWindowNodeParser.from_defaults(
                window_size=2,
                window_metadata_key="window",
                original_text_metadata_key="original_text",
                id_func=stable_id,
            )
        except Exception:
            from llama_index.core.node_parser import SentenceSplitter

            self._small_to_big_enabled = False
            return SentenceSplitter(chunk_size=512, chunk_overlap=64, id_func=stable_id)

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

    def _hit_to_node_with_score(self, hit: RetrievedChunk):
        try:
            from llama_index.core.schema import NodeWithScore, TextNode
        except ImportError:
            return None

        node = TextNode(
            text=hit.text,
            id_=self._stable_uuid(hit.chunk_id or f"{hit.metadata.get('source', 'chunk')}::{hit.metadata.get('page', '?')}"),
            metadata=dict(hit.metadata or {}),
        )
        return NodeWithScore(node=node, score=float(hit.score))

    def _synthesize_from_hits(self, query: str, hits: List[RetrievedChunk], *, response_mode: str) -> str:
        if not hits:
            return ""
        synthesizer = self._get_response_synthesizer(response_mode)
        if synthesizer is None:
            return ""

        nodes = [self._hit_to_node_with_score(hit) for hit in hits]
        nodes = [node for node in nodes if node is not None]
        if not nodes:
            return ""
        try:
            response = synthesizer.synthesize(query=query, nodes=nodes)
            return str(response).strip()
        except Exception as exc:
            self._llm_error = str(exc)
            return ""

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

    def _build_chunk_documents(self, parsed: ParsedDocument):
        from llama_index.core import Document

        page_docs = []
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
            doc = Document(text=cleaned, metadata=metadata, doc_id=f"{parsed.source}::page::{page_num}")
            doc.excluded_embed_metadata_keys = ["parent_text"]
            page_docs.append(doc)
        return page_docs

    def _build_summary_node(self, parsed: ParsedDocument):
        from llama_index.core.schema import TextNode

        summary = self._build_doc_summary(parsed)
        if not summary:
            return None
        summary_id = f"{parsed.source}::summary"
        node = TextNode(
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
        node.excluded_embed_metadata_keys = ["chunk_id"]
        return node

    def _set_metadata_extraction_state(self, enabled: bool, reason: str) -> None:
        self._last_metadata_extraction_enabled = bool(enabled)
        self._last_metadata_extraction_reason = reason

    def _should_use_metadata_extraction(self, documents) -> bool:
        if not self.llm_enabled:
            self._set_metadata_extraction_state(False, "disabled_without_local_llm")
            return False
        doc_count = len(documents or [])
        total_chars = 0
        for document in documents or []:
            total_chars += len(str(getattr(document, "text", "") or ""))
        if doc_count > self._metadata_extraction_max_documents:
            self._set_metadata_extraction_state(
                False,
                f"skipped_large_ingest_doc_count>{self._metadata_extraction_max_documents}",
            )
            return False
        if total_chars > self._metadata_extraction_max_chars:
            self._set_metadata_extraction_state(
                False,
                f"skipped_large_ingest_chars>{self._metadata_extraction_max_chars}",
            )
            return False
        self._set_metadata_extraction_state(True, "enabled_for_small_ingest")
        return True

    def _manifest_record_from_node(self, node) -> ChunkRecord:
        metadata = dict(getattr(node, "metadata", {}) or {})
        chunk_id = str(metadata.get("chunk_id") or getattr(node, "node_id", "") or getattr(node, "id_", "") or "")
        return ChunkRecord(
            chunk_id=chunk_id,
            text=self._node_text(node),
            metadata=metadata,
        )

    def _run_chunk_ingestion(self, documents):
        if not documents:
            return []
        include_metadata_extractors = self._should_use_metadata_extraction(documents)
        self._metadata_extraction_error = None
        try:
            nodes = self._get_ingestion_pipeline(
                include_metadata_extractors=include_metadata_extractors
            ).run(documents=list(documents), show_progress=False)
        except Exception as exc:
            if not include_metadata_extractors:
                raise
            self._metadata_extraction_error = str(exc)
            self._set_metadata_extraction_state(False, "fallback_without_metadata_after_error")
            nodes = self._get_ingestion_pipeline(include_metadata_extractors=False).run(
                documents=list(documents),
                show_progress=False,
            )
        self._persist_ingestion_cache()
        return list(nodes)

    def _insert_summary_node(self, node) -> None:
        if node is None:
            return
        self._get_index().insert_nodes([node])

    def _clear_ingestion_state(self) -> None:
        self._pipeline = None
        self._ingestion_cache = None
        self._prepare_nodes_transform = None
        self._metadata_extractors = None
        self._response_synthesizers = {}
        self._pipeline_uses_metadata_extractors = False
        self._last_metadata_extraction_enabled = None
        self._last_metadata_extraction_reason = None
        if os.path.exists(self._ingestion_cache_path):
            try:
                os.remove(self._ingestion_cache_path)
            except OSError:
                pass

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
                chunk_documents = self._build_chunk_documents(parsed)
                if not chunk_documents:
                    report.errors.append(f"{source}: no indexable content found")
                    continue
                if force_reindex or self._has_source(source):
                    self._delete_source_records(source)
                chunk_nodes = self._run_chunk_ingestion(chunk_documents)
                if not chunk_nodes:
                    report.errors.append(f"{source}: no indexable content found")
                    continue
                summary_node = self._build_summary_node(parsed)
                if summary_node is not None:
                    self._insert_summary_node(summary_node)

                manifest_records = [self._manifest_record_from_node(node) for node in chunk_nodes]
                if summary_node is not None:
                    manifest_records.append(self._manifest_record_from_node(summary_node))
                self._replace_manifest_source(source, manifest_records)
                report.files_processed += 1
                created = len(chunk_nodes) + (1 if summary_node is not None else 0)
                report.chunks_created += created
                report.vectors_upserted += created

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

    def _default_qa_response_mode(self, query: str) -> str:
        lowered = (query or "").lower()
        broader_markers = (
            "summarize",
            "summary",
            "compare",
            "difference",
            "recommend",
            "analyze",
            "analysis",
            "overview",
            "brief",
            "risks",
            "next steps",
        )
        if any(marker in lowered for marker in broader_markers):
            return "refine"
        return "compact"

    def answer_query(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        source_filter: Optional[str] = None,
        response_mode: Optional[str] = None,
    ) -> Optional[dict]:
        mode = (response_mode or self._default_qa_response_mode(query)).strip().lower()
        hits = self.search(
            query,
            top_k=top_k if top_k is not None else max(self.top_k, 6),
            min_score=min_score,
            source_filter=source_filter,
        )
        if not hits:
            return {
                "answer": "",
                "context": "",
                "citations": [],
                "response_mode": mode,
            }

        synthesized = self._synthesize_from_hits(query, hits, response_mode=mode)
        if not synthesized:
            return None
        return {
            "answer": synthesized,
            "context": self.format_hits(hits),
            "citations": [
                {
                    "source": hit.metadata.get("source", "unknown"),
                    "page": hit.metadata.get("page", "?"),
                    "score": hit.score,
                }
                for hit in hits
            ],
            "response_mode": mode,
        }

    def prepare_document_context(
        self,
        query: str,
        *,
        source_filter: Optional[str] = None,
        document_mode: Optional[str] = None,
    ) -> Optional[dict]:
        hits = self.search(
            query,
            top_k=max(self.top_k, 8),
            min_score=max(0.0, min(self.min_score, 0.15)),
            source_filter=source_filter,
        )
        if not hits:
            return {
                "context": "",
                "citations": [],
                "response_mode": "tree_summarize",
            }

        raw_context = self.format_hits(hits, max_context_chars=max(self.max_context_chars, 4500))
        synthesized = self._synthesize_from_hits(query, hits, response_mode="tree_summarize")
        context = raw_context
        if synthesized:
            context = f"[Grounding Summary]\n{synthesized}\n\n[Evidence Snippets]\n{raw_context}".strip()
        return {
            "context": context,
            "citations": [
                {
                    "source": hit.metadata.get("source", "unknown"),
                    "page": hit.metadata.get("page", "?"),
                    "score": hit.score,
                }
                for hit in hits
            ],
            "response_mode": "tree_summarize",
            "document_mode": document_mode or "hybrid",
        }

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
            "ingestion_pipeline_enabled": True,
            "metadata_extraction_configured": self.llm_enabled,
            "metadata_extraction_enabled": (
                self.llm_enabled
                if self._last_metadata_extraction_enabled is None
                else self._last_metadata_extraction_enabled
            ),
            "metadata_extraction_last_reason": self._last_metadata_extraction_reason,
            "metadata_extraction_max_documents": self._metadata_extraction_max_documents,
            "metadata_extraction_max_chars": self._metadata_extraction_max_chars,
            "response_synthesis_enabled": self.llm_enabled,
            "metadata_extraction_error": self._metadata_extraction_error,
            "llamaindex_llm_error": self._llm_error,
        }

    def clear_collection(self) -> None:
        if self._collection_exists():
            try:
                self._get_client().delete_collection(self.collection_name)
            except Exception:
                pass
        self._index = None
        self._clear_ingestion_state()
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
