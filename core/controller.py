"""
GeneratorController — Orchestrates synthetic data generation.

Manages the generation lifecycle: initialization, row generation (fresh + enrichment),
threading, and stop control. Delegates to specialized modules for:
- Prompt construction: core.prompt_builder
- Export: core.exporters
- Metrics: core.metrics
- Validation: core.validator
- Quality analysis: core.analytics
"""
import logging
import time
import threading
import hashlib
from typing import List, Callable, Optional, Dict, Any

from .models import GeneratorConfig, ColumnDefinition, RowData, ColumnType
from .llm_client import LLMClient
from .validator import UniquenessValidator
from .analytics import QualityAnalyzer
from .exporters import (
    PDFReportGenerator,
    DocumentPDFExporter,
    DocumentDocxExporter,
    export_csv,
    export_json,
    export_sql,
)
from .prompt_builder import get_dependencies, get_execution_order, construct_prompt
from .metrics import calculate_metrics
import pandas as pd
from .rag.service import RagService
from .document_engine import DocumentGenerationOptions, DocumentMode, DocumentOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneratorController:
    def __init__(self):
        self.config = GeneratorConfig(model_id="local-model")  # Default, will be updated
        self.columns: List[ColumnDefinition] = []
        self.execution_order: List[ColumnDefinition] = []  # Sorted columns
        self.generated_rows: List[RowData] = []
        
        self.llm_client: Optional[LLMClient] = None
        self.rag_service: Optional[RagService] = None
        self.validator: Optional[UniquenessValidator] = None
        self.analyzer = QualityAnalyzer()
        self.pdf_exporter = PDFReportGenerator()
        self.document_pdf_exporter = DocumentPDFExporter()
        self.document_docx_exporter = DocumentDocxExporter()
        self.document_orchestrator: Optional[DocumentOrchestrator] = None
        self.document_result: Optional[Dict[str, Any]] = None
        
        # Metrics Tracking
        self.metrics_data = {
            "faker_cols": 0,
            "llm_cols": 0,
        }
        
        # Faker for deterministic columns
        try:
            from faker import Faker
            self.fake = Faker()
        except ImportError:
            self.fake = None
            self.log("Warning: Faker library not found. Deterministic columns will fail.")
        
        self.is_running = False
        self.stop_requested = False
        
        # Callbacks
        self.on_progress: Optional[Callable[[int, int], None]] = None
        self.on_log: Optional[Callable[[str], None]] = None
        self.on_finished: Optional[Callable[[], None]] = None
        
    def initialize(self, config: GeneratorConfig, columns: List[ColumnDefinition]):
        self.config = config
        self.columns = columns
        self.llm_client = LLMClient(config, on_log=self.log)
        self.initialize_rag()
        self.validator = UniquenessValidator(config)
        self.generated_rows = []
        self.document_result = None
        self.stop_requested = False
        
        # Reset run metrics
        self.metrics_data = {
            "faker_cols": 0,
            "llm_cols": 0,
        }
        
        # Calculate execution order based on dependencies
        try:
            self.execution_order = get_execution_order(columns, log_fn=self.log)
            self.log(f"Execution order resolved: {[c.name for c in self.execution_order]}")
        except Exception as e:
            self.log(f"Dependency Error: {e}")
            raise e

    def initialize_rag(self):
        rag_cfg = self.config.rag
        if not rag_cfg:
            self.rag_service = None
            if self.llm_client:
                self.llm_client.set_rag_service(None)
            return

        try:
            self.rag_service = RagService(
                collection_name=rag_cfg.collection_name,
                qdrant_url=rag_cfg.qdrant_url,
                qdrant_api_key=rag_cfg.qdrant_api_key,
                embedding_model=rag_cfg.embedding_model,
                top_k=rag_cfg.top_k,
                min_score=rag_cfg.min_score,
                max_context_chars=rag_cfg.max_context_chars,
                ocr_mode=rag_cfg.ocr_mode.value if hasattr(rag_cfg.ocr_mode, "value") else str(rag_cfg.ocr_mode),
                ocr_dpi=rag_cfg.ocr_dpi,
                ocr_max_pages=rag_cfg.ocr_max_pages,
                ocr_max_regions_per_page=rag_cfg.ocr_max_regions_per_page,
                ocr_region_padding_px=rag_cfg.ocr_region_padding_px,
                ocr_gap_multiplier=rag_cfg.ocr_gap_multiplier,
                ocr_min_extracted_chars=rag_cfg.ocr_min_extracted_chars,
                ocr_timeout_ms_per_page=rag_cfg.ocr_timeout_ms_per_page,
            )
            if self.llm_client:
                self.llm_client.set_rag_service(self.rag_service)
            self.log("RAG initialized successfully.")
        except Exception as e:
            self.rag_service = None
            if self.llm_client:
                self.llm_client.set_rag_service(None)
            self.log(f"RAG initialization failed: {e}")

        if self.llm_client:
            self.document_orchestrator = DocumentOrchestrator(self.llm_client, self.rag_service)

    def ingest_documents(self, paths: List[str], force_reindex: bool = False) -> Dict[str, Any]:
        if not self.rag_service:
            return {"error": "RAG is not configured."}

        report = self.rag_service.ingest_documents(paths, force_reindex=force_reindex)
        return report.model_dump()

    def search_context(self, query: str, top_k: Optional[int] = None, source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.rag_service:
            return []
        hits = self.rag_service.search(query, top_k=top_k, source_filter=source_filter)
        return [h.model_dump() for h in hits]

    def clear_rag_collection(self) -> None:
        if self.rag_service:
            self.rag_service.clear_collection()

    def get_rag_status(self) -> Dict[str, Any]:
        if not self.rag_service:
            return {"enabled": False}
        status = self.rag_service.get_status()
        status["enabled"] = True
        status["collection_name"] = self.config.rag.collection_name if self.config.rag else ""
        status["source_filter"] = self.config.rag.source_filter if self.config.rag else None
        status["ocr_mode"] = self.config.rag.ocr_mode.value if self.config.rag else "off"
        status["ocr_dpi"] = self.config.rag.ocr_dpi if self.config.rag else 150
        return status

    def set_runtime_config(self, config: GeneratorConfig) -> None:
        self.config = config
        self.llm_client = LLMClient(config, on_log=self.log)
        self.initialize_rag()

    def generate_document(
        self,
        prompt: str,
        *,
        target_words: int = 1400,
        audience: str = "General",
        tone: str = "professional",
        mode: str = "hybrid",
        resume: bool = True,
    ) -> Dict[str, Any]:
        if not prompt or not prompt.strip():
            return {"error": "Prompt is empty."}
        if not self.llm_client:
            return {"error": "LLM client is not initialized."}

        if not self.document_orchestrator:
            self.document_orchestrator = DocumentOrchestrator(self.llm_client, self.rag_service)

        mode_map = {
            "hybrid": DocumentMode.HYBRID,
            "strict_grounded": DocumentMode.STRICT_GROUNDED,
            "pure": DocumentMode.PURE,
        }
        safe_mode = mode_map.get((mode or "hybrid").strip().lower(), DocumentMode.HYBRID)

        doc_cfg = self.config.document_engine if self.config and self.config.document_engine else None
        options = DocumentGenerationOptions(
            prompt=prompt.strip(),
            target_words=target_words if target_words > 0 else (doc_cfg.target_words if doc_cfg else 1400),
            audience=audience or (doc_cfg.audience if doc_cfg else "General"),
            tone=tone or (doc_cfg.tone if doc_cfg else "professional"),
            mode=safe_mode,
            max_chunk_words=doc_cfg.max_chunk_words if doc_cfg else 500,
            min_chunk_words=doc_cfg.min_chunk_words if doc_cfg else 220,
            max_retries=doc_cfg.max_retries if doc_cfg else 3,
            consistency_check_interval=doc_cfg.consistency_check_interval if doc_cfg else 12,
            resume=resume,
        )

        self.stop_requested = False
        basis = f"{prompt.strip()}|{options.mode.value}|{options.target_words}|{options.audience}|{options.tone}"
        digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]
        job_id = f"doc_{digest}"

        try:
            result = self.document_orchestrator.run(
                job_id=job_id,
                options=options,
                on_log=self.log,
                on_progress=self.on_progress,
                should_stop=lambda: self.stop_requested,
            )
            self.document_result = result
            return result
        except Exception as e:
            self.log(f"Document generation failed: {e}")
            return {"error": str(e)}

    def stop_document_generation(self):
        self.stop_requested = True
        self.log("Stopping document generation...")

    def export_document_pdf(self, filepath: str):
        if not self.document_result:
            raise ValueError("No generated document available. Run document generation first.")
        self.document_pdf_exporter.export(
            title=self.document_result.get("title", "Generated Document"),
            outline=self.document_result.get("outline", {}),
            text=self.document_result.get("text", ""),
            output_path=filepath,
            chunks=self.document_result.get("chunks", []),
        )
        self.log(f"Document PDF exported to {filepath}")

    def export_document_docx(self, filepath: str):
        if not self.document_result:
            raise ValueError("No generated document available. Run document generation first.")
        self.document_docx_exporter.export(
            title=self.document_result.get("title", "Generated Document"),
            outline=self.document_result.get("outline", {}),
            text=self.document_result.get("text", ""),
            output_path=filepath,
            chunks=self.document_result.get("chunks", []),
        )
        self.log(f"Document DOCX exported to {filepath}")

    def ask_files(self, prompt: str) -> Dict[str, Any]:
        if not prompt.strip():
            return {"error": "Prompt is empty."}
        if not self.llm_client:
            return {"error": "LLM client is not initialized."}
        if not self.rag_service:
            return {"error": "RAG service is not configured."}

        context = self.llm_client.retrieve_context(prompt)
        if not context.strip():
            status = self.get_rag_status()
            self.log(
                "RAG query returned no context "
                f"(collection_size={status.get('collection_size', 0)}, "
                f"source_filter={status.get('source_filter')}, "
                f"min_score={status.get('min_score')})"
            )
            return {
                "answer": "I could not find relevant context in the imported files. Try a more specific query.",
                "context": "",
                "citations": [],
            }

        top_k = self.config.rag.top_k if self.config.rag else 5
        min_score = self.config.rag.min_score if self.config.rag else 0.25
        source_filter = self.config.rag.source_filter if self.config.rag else None

        hits = self.rag_service.search(
            prompt,
            top_k=top_k,
            min_score=min_score,
            source_filter=source_filter,
        )
        if not hits and source_filter:
            hits = self.rag_service.search(prompt, top_k=top_k, min_score=min_score, source_filter=None)
        if not hits and min_score > 0.0:
            hits = self.rag_service.search(prompt, top_k=top_k, min_score=0.0, source_filter=None)

        citations = []
        for hit in hits:
            citations.append(
                {
                    "source": hit.metadata.get("source", "unknown"),
                    "page": hit.metadata.get("page", "?"),
                    "score": hit.score,
                }
            )

        prompt_text = (
            "Use only the provided context from imported files. "
            "If context is insufficient, clearly say so.\n\n"
            f"Context:\n{context}\n\n"
            f"User Task:\n{prompt}\n\n"
            "Return a concise answer and keep it grounded in the context."
        )
        answer = self.llm_client.generate_completion(prompt_text, system_prompt="You are a file-grounded assistant.")

        return {
            "answer": answer or "No answer returned by model.",
            "context": context,
            "citations": citations,
        }
        
    def log(self, message: str):
        logger.info(message)
        if self.on_log:
            self.on_log(message)

    def generate_row(self, initial_context: Optional[Dict[str, Any]] = None) -> Optional[RowData]:
        from core.row_agent import create_row_generator_graph
        
        # Prepare initial data
        row_data: Dict[str, Any] = initial_context.copy() if initial_context else {}
        
        # Pre-fill deterministic logic (Faker, Auto-Increment) BEFORE Agent
        for col in self.execution_order:
            if col.name in row_data:
                continue
            
            if col.type == ColumnType.AUTO_INCREMENT:
                row_data[col.name] = len(self.generated_rows) + 1
                self.metrics_data["faker_cols"] += 1
            
            elif col.type == ColumnType.DETERMINISTIC and self.fake:
                provider = col.constraints.faker_provider or "name"
                try:
                    if hasattr(self.fake, provider):
                        func = getattr(self.fake, provider)
                        row_data[col.name] = str(func())
                    else:
                        row_data[col.name] = self.fake.name()
                    self.metrics_data["faker_cols"] += 1
                except:
                    pass

        # If all cols are filled (purely deterministic), skip agent
        if len(row_data) == len(self.columns):
            return RowData(data=row_data)

        # Filter out columns that are strictly "(Imported)"
        agent_cols = [
            col for col in self.execution_order 
            if col.prompt_instruction != "(Imported)" 
            and col.name not in row_data
        ]
        
        # Track LLM columns
        self.metrics_data["llm_cols"] += len(agent_cols)

        # Agentic Generation for LLM columns
        try:
            agent = create_row_generator_graph(self.llm_client)
            
            initial_state = {
                "row_data": row_data,
                "columns": agent_cols,
                "errors": [],
                "attempt_count": 0,
                "is_valid": False
            }
            
            final_state = agent.invoke(initial_state)
            
            result_data = final_state['row_data']
            is_valid = final_state['is_valid']
            
            if not is_valid and final_state['attempt_count'] >= 3:
                self.log(f"Row generation failed validation: {final_state.get('errors')}")
                return None
                
            # Post-Agent Guardrails: Uniqueness/Regex checks
            for col in self.columns:
                if col.prompt_instruction == "(Imported)":
                    continue

                val = result_data.get(col.name)
                if not val:
                    return None
                
                # Regex
                if not self.validator.validate_regex(val, col.constraints.regex_pattern):
                    self.log(f"Regex failed for {col.name}: {val}")
                    return None
                    
                # Uniqueness
                if not col.constraints.allow_duplicates:
                    if not self.validator.is_unique(val, field_type=col.type.value):
                        self.log(f"Duplicate value for {col.name}: {val}")
                        return None
                        
            # Commit unique values
            for col in self.columns:
                if not col.constraints.allow_duplicates:
                    val = result_data.get(col.name)
                    self.validator.commit(val, field_type=col.type.value)

            return RowData(data=result_data)

        except Exception as e:
            self.log(f"Agent error: {e}")
            return None

    def start_generation_thread(self):
        """Starts generation in a separate thread."""
        self.is_running = True
        t = threading.Thread(target=self._run_generation_loop)
        t.start()
        
    def stop_generation(self):
        print("DEBUG: Controller stop_generation called")
        self.stop_requested = True
        self.log("Stopping generation...")

    def _run_generation_loop(self):
        target_count = len(self.config.existing_data) if self.config.existing_data else self.config.num_rows
        self.log(f"Starting generation attempt for {target_count} rows...")
        
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 10
        
        self.metrics_data["start_time"] = time.time()
        self.metrics_data["end_time"] = None
        self.metrics_data["last_row_time"] = self.metrics_data["start_time"]
        self.metrics_data["total_attempts"] = 0
        self.metrics_data["failed_attempts"] = 0
        
        # Mode A: Enrichment (Existing Data)
        if self.config.existing_data:
            for i, context_row in enumerate(self.config.existing_data):
                if self.stop_requested:
                    break
                
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    self.log(f"CRITICAL: Aborting generation after {MAX_CONSECUTIVE_FAILURES} consecutive failures. Please check your constraints.")
                    break

                self.metrics_data["total_attempts"] += 1
                row = self.generate_row(initial_context=context_row)
                if self.stop_requested:
                    break
                
                if row:
                    self.generated_rows.append(row)
                    self.metrics_data["last_row_time"] = time.time()
                    self.log(f"Generated row {len(self.generated_rows)}/{target_count}")
                    consecutive_failures = 0
                else:
                    self.metrics_data["failed_attempts"] += 1
                    self.log(f"Row {i+1} enrichment failed. Skipping.")
                    consecutive_failures += 1
                
                if self.on_progress:
                    self.on_progress(len(self.generated_rows), target_count)
                    
        # Mode B: Fresh Generation
        else:
            while len(self.generated_rows) < target_count:
                if self.stop_requested:
                    break
                
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    self.log(f"CRITICAL: Aborting generation after {MAX_CONSECUTIVE_FAILURES} consecutive failures. Please check your constraints.")
                    break
                    
                self.metrics_data["total_attempts"] += 1
                row = self.generate_row()
                if self.stop_requested:
                    break
                
                if row:
                    self.generated_rows.append(row)
                    self.metrics_data["last_row_time"] = time.time()
                    self.log(f"Generated row {len(self.generated_rows)}/{target_count}")
                    consecutive_failures = 0
                    if self.on_progress:
                        self.on_progress(len(self.generated_rows), target_count)
                else:
                    self.metrics_data["failed_attempts"] += 1
                    self.log("Row generation failed. Retrying...")
                    consecutive_failures += 1
        
        self.metrics_data["end_time"] = time.time()
        self.is_running = False
        self.log("Generation finished.")
        if self.on_finished:
            self.on_finished()

    # --- Export Delegation ---
    # These thin wrappers maintain backward compatibility with FletApp
    
    def export_csv(self, filepath: str):
        export_csv(filepath, self.generated_rows, self.columns, log_fn=self.log)

    def export_json(self, filepath: str):
        export_json(filepath, self.generated_rows, log_fn=self.log)

    def export_sql(self, filepath: str, table_name: str = "synthetic_data"):
        export_sql(filepath, self.generated_rows, table_name=table_name, log_fn=self.log)

    def analyze_quality(self) -> Dict[str, Any]:
        """Runs quality analysis on the generated data."""
        if not self.generated_rows:
            return {}
        try:
            df = pd.DataFrame([r.data for r in self.generated_rows])
            return self.analyzer.analyze(df)
        except Exception as e:
            self.log(f"Analysis failed: {e}")
            return {}

    def export_pdf_report(self, filepath: str):
        try:
            metrics = self.analyze_quality()
            self.pdf_exporter.generate_quality_report(metrics, filepath)
            self.log(f"PDF Quality Report exported to {filepath}")
        except Exception as e:
            self.log(f"PDF Export failed: {e}")

    def export_narrative_pdf(self, filepath: str):
        if not self.generated_rows:
            return
            
        try:
            df = pd.DataFrame([r.data for r in self.generated_rows])
            cols = df.columns.tolist()
            if not cols:
                return
                
            title_col = cols[0]
            body_cols = cols[1:]
            
            self.pdf_exporter.generate_narrative_export(df, title_col, body_cols, filepath)
            self.log(f"Narrative PDF exported to {filepath}")
        except Exception as e:
            self.log(f"Narrative Export failed: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Calculate real-time metrics — delegates to core.metrics."""
        return calculate_metrics(
            config=self.config,
            generated_rows=self.generated_rows,
            llm_client=self.llm_client,
            run_metrics=self.metrics_data,
        )
