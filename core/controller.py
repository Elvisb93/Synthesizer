import re
import logging
import time
import threading
from typing import List, Callable, Optional, Dict, Any, Set
from .models import GeneratorConfig, ColumnDefinition, RowData, ColumnType
from .llm_client import LLMClient
from .validator import UniquenessValidator
from .analytics import QualityAnalyzer
from .exporters import PDFReportGenerator
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeneratorController:
    def __init__(self):
        self.config = GeneratorConfig(model_id="local-model") # Default, wil be updated
        self.columns: List[ColumnDefinition] = []
        self.execution_order: List[ColumnDefinition] = [] # Sorted columns
        self.generated_rows: List[RowData] = []
        
        self.llm_client: Optional[LLMClient] = None
        self.llm_client: Optional[LLMClient] = None
        self.validator: Optional[UniquenessValidator] = None
        self.analyzer = QualityAnalyzer()
        self.pdf_exporter = PDFReportGenerator()
        
        # Phase 5: Faker
        try:
            from faker import Faker
            self.fake = Faker()
        except ImportError:
            self.fake = None
            self.log("Warning: Faker library not found. Deterministic columns will fail.")
        
        self.is_running = False
        self.stop_requested = False
        
        # Callbacks
        self.on_progress: Optional[Callable[[int, int], None]] = None # current, total
        self.on_log: Optional[Callable[[str], None]] = None
        self.on_finished: Optional[Callable[[], None]] = None
        
    def initialize(self, config: GeneratorConfig, columns: List[ColumnDefinition]):
        self.config = config
        self.columns = columns
        self.llm_client = LLMClient(config, on_log=self.log)
        self.validator = UniquenessValidator(config)
        self.generated_rows = []
        self.stop_requested = False
        
        # Calculate execution order based on dependencies
        try:
            self.execution_order = self._get_execution_order(columns)
            self.log(f"Execution order resolved: {[c.name for c in self.execution_order]}")
        except Exception as e:
            self.log(f"Dependency Error: {e}")
            raise e
        
    def log(self, message: str):
        logger.info(message)
        if self.on_log:
            self.on_log(message)

    def _get_dependencies(self, prompt: str) -> Set[str]:
        """Extract valid column references from prompt like @[col_name]."""
        matches = re.findall(r'@\[(.*?)\]', prompt)
        return set(matches)

    def _get_execution_order(self, columns: List[ColumnDefinition]) -> List[ColumnDefinition]:
        """Topological sort of columns based on prompt dependencies."""
        col_map = {col.name: col for col in columns}
        adj_list: Dict[str, Set[str]] = {col.name: set() for col in columns}
        in_degree: Dict[str, int] = {col.name: 0 for col in columns}
        
        # Build Graph
        for col in columns:
            deps = self._get_dependencies(col.prompt_instruction)
            for dep_name in deps:
                if dep_name in col_map:
                    # dep_name must be generated BEFORE col
                    # Edge: dep_name -> col
                    adj_list[dep_name].add(col.name)
                    in_degree[col.name] += 1
                else:
                    self.log(f"Warning: Column '{col.name}' references unknown column '{dep_name}'. Ignoring.")

        # Kahn's Algorithm
        queue = [name for name, deg in in_degree.items() if deg == 0]
        sorted_cols = []
        
        while queue:
            node = queue.pop(0)
            sorted_cols.append(col_map[node])
            
            for neighbor in adj_list[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                    
        if len(sorted_cols) != len(columns):
            raise ValueError("Circular dependency detected in column prompts!")
            
        return sorted_cols

    def _construct_prompt(self, col: ColumnDefinition, row_data: Dict[str, Any]) -> str:
        """Create a prompt for a single column value, interpolating dependencies."""
        
        # Interpolate dependencies
        instruction = col.prompt_instruction
        deps = self._get_dependencies(instruction)
        for dep in deps:
            if dep in row_data:
                val = str(row_data[dep])
                instruction = instruction.replace(f"@[{dep}]", val)
        
        constraints_text = ""
        if col.constraints.min_length:
            constraints_text += f"\n- Minimum length: {col.constraints.min_length} characters"
        if col.constraints.max_length:
            constraints_text += f"\n- Maximum length: {col.constraints.max_length} characters"
        if col.constraints.options:
             constraints_text += f"\n- Choose strictly from: {', '.join(col.constraints.options)}"
             
        prompt = (
            f"Generate a single {col.type.value} value for a database column named '{col.name}'.\n"
            f"Context/Description: {instruction}\n"
            f"Constraints: {constraints_text}\n"
            "Return ONLY the value. Do not include quotes or markdown formatting if possible."
        )
        return prompt

    def generate_row(self, initial_context: Optional[Dict[str, Any]] = None) -> Optional[RowData]:
        from core.row_agent import create_row_generator_graph
        
        # Prepare initial data
        row_data: Dict[str, Any] = initial_context.copy() if initial_context else {}
        
        # Pre-fill deterministic logic (Faker, Auto-Increment) BEFORE Agent
        # Because we want the Agent to focus on semantic LLM work.
        for col in self.execution_order:
            if col.name in row_data: continue
            
            if col.type == ColumnType.AUTO_INCREMENT:
                row_data[col.name] = len(self.generated_rows) + 1
            
            elif col.type == ColumnType.DETERMINISTIC and self.fake:
                provider = col.constraints.faker_provider or "name"
                try:
                    if hasattr(self.fake, provider):
                        func = getattr(self.fake, provider)
                        row_data[col.name] = str(func())
                    else:
                        row_data[col.name] = self.fake.name()
                except:
                    pass

        # If all cols are filled (purely deterministic), skip agent
        if len(row_data) == len(self.columns):
            return RowData(data=row_data)

        # Agentic Generation for LLM columns
        try:
            agent = create_row_generator_graph(self.llm_client)
            
            initial_state = {
                "row_data": row_data,
                "columns": self.execution_order, # Sorted for dependency context
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
                
            # Perform strict Uniqueness/Regex constraints check (Post-Agent Guardrails)
            for col in self.columns:
                val = result_data.get(col.name)
                if not val: return None
                
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
        self.stop_requested = True
        self.log("Stopping generation...")

    def _run_generation_loop(self):
        target_count = len(self.config.existing_data) if self.config.existing_data else self.config.num_rows
        self.log(f"Starting generation attempt for {target_count} rows...")
        
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 10  # Hard safeguard against infinite loops
        
        # Mode A: Enrichment (Existing Data)
        if self.config.existing_data:
            for i, context_row in enumerate(self.config.existing_data):
                if self.stop_requested:
                    break
                
                # Check safeguards
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    self.log(f"CRITICAL: Aborting generation after {MAX_CONSECUTIVE_FAILURES} consecutive failures. Please check your constraints.")
                    break

                row = self.generate_row(initial_context=context_row)
                if row:
                    self.generated_rows.append(row)
                    consecutive_failures = 0 # Reset on success
                else:
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
                    
                row = self.generate_row()
                if row:
                    self.generated_rows.append(row)
                    consecutive_failures = 0 # Reset on success
                    if self.on_progress:
                        self.on_progress(len(self.generated_rows), target_count)
                else:
                    self.log("Row generation failed. Retrying...")
                    consecutive_failures += 1
        
        self.is_running = False
        self.log("Generation finished.")
        if self.on_finished:
            self.on_finished()

    def export_csv(self, filepath: str):
        import csv
        if not self.generated_rows:
            return
            
        fieldnames = [col.name for col in self.columns]
        
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in self.generated_rows:
                    writer.writerow(row.data)
            self.log(f"Exported to {filepath}")
        except Exception as e:
            self.log(f"Export failed: {e}")

    def export_json(self, filepath: str):
        import json
        if not self.generated_rows:
            return
        
        try:
            data = [row.data for row in self.generated_rows]
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            self.log(f"Exported to {filepath}")
        except Exception as e:
            self.log(f"JSON Export failed: {e}")

    def export_sql(self, filepath: str, table_name: str = "synthetic_data"):
        if not self.generated_rows:
            return
            
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for row in self.generated_rows:
                    cols = []
                    vals = []
                    for k, v in row.data.items():
                        cols.append(k)
                        # Basic escaping
                        if isinstance(v, str):
                            safe_v = v.replace("'", "''")
                            vals.append(f"'{safe_v}'")
                        else:
                            vals.append(str(v))
                    
                    sql = f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES ({', '.join(vals)});\n"
                    f.write(sql)
            self.log(f"Exported to {filepath}")
        except Exception as e:
            self.log(f"SQL Export failed: {e}")

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
                
            # Heuristic: First column is title, rest are body
            # TODO: Make configurable? For now, this is a sensible default for "Narrative Mode"
            title_col = cols[0]
            body_cols = cols[1:]
            
            self.pdf_exporter.generate_narrative_export(df, title_col, body_cols, filepath)
            self.log(f"Narrative PDF exported to {filepath}")
        except Exception as e:
            self.log(f"Narrative Export failed: {e}")

