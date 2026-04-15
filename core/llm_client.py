import logging
import json
import re
from typing import List, Optional, Dict, Any, Callable
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from .models import GeneratorConfig, ColumnType, AIProvider
from .schemas import Schema

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Provider-to-base-url mapping
PROVIDER_BASE_URLS = {
    AIProvider.LM_STUDIO: "http://localhost:1234/v1",
    AIProvider.OPENAI: "https://api.openai.com/v1",
    AIProvider.GOOGLE_GEMINI: "https://generativelanguage.googleapis.com/v1beta/openai/",
    AIProvider.OPENROUTER: "https://openrouter.ai/api/v1",
    AIProvider.GITHUB_MODELS: "https://models.github.ai/inference",
}

class LLMClient:
    def __init__(self, config: GeneratorConfig, on_log: Optional[Callable[[str], None]] = None):
        self.config = config
        self.on_log = on_log
        self.chat_model = self._init_chat_model()
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.latency_stats = {"total_time": 0.0, "count": 0}
        self.rag_service = None
        self.rag_stats = {
            "queries": 0,
            "queries_with_hits": 0,
            "total_retrieval_time": 0.0,
            "total_context_chars": 0,
            "last_hits": 0,
        }


    def _init_chat_model(self):
        """Initialize the LangChain ChatModel based on configuration."""
        if self.config.provider == AIProvider.AZURE_OPENAI:
            if not all([self.config.api_key, self.config.azure_endpoint, self.config.azure_deployment]):
                 raise ValueError("Azure OpenAI requires api_key, azure_endpoint, and azure_deployment")
            
            return AzureChatOpenAI(
                api_key=self.config.api_key,
                azure_endpoint=self.config.azure_endpoint,
                deployment_name=self.config.azure_deployment,
                api_version="2024-02-15-preview",
                temperature=0.7
            )
        else:
            base_url = PROVIDER_BASE_URLS.get(self.config.provider, self.config.api_base_url)
            api_key = self.config.api_key
            
            if self.config.provider == AIProvider.LM_STUDIO:
                 api_key = "lm-studio"
            elif not api_key:
                 raise ValueError(f"API key is required for provider: {self.config.provider.value}")

            # Special handling for Gemini headers if needed (LangChain handles most standard auth)
            default_headers = None
            if self.config.provider == AIProvider.GOOGLE_GEMINI:
                 default_headers = {"x-goog-api-client": "synthesizer-app/1.0.0"}

            return ChatOpenAI(
                base_url=base_url,
                api_key=api_key,
                model=self.config.model_id,
                temperature=0.7,
                default_headers=default_headers
            )

    def list_models(self) -> List[str]:
        """Fetch available models."""
        try:
            from openai import OpenAI
            
            base_url = PROVIDER_BASE_URLS.get(self.config.provider, self.config.api_base_url)
            if self.config.provider == AIProvider.AZURE_OPENAI:
                 return [self.config.azure_deployment] if self.config.azure_deployment else []
            
            api_key = "lm-studio" if self.config.provider == AIProvider.LM_STUDIO else self.config.api_key
            
            client = OpenAI(base_url=base_url, api_key=api_key)
            models = client.models.list()
            return [m.id for m in models.data]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def generate_completion(self, prompt: str, system_prompt: str = "You are a helpful data generator.") -> Optional[str]:
        """Generate a raw text completion using LangChain."""
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ]
            response = self.chat_model.invoke(messages)
            
            # Track Usage
            if response.response_metadata and 'token_usage' in response.response_metadata:
                usage = response.response_metadata['token_usage']
                self.token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                self.token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
            
            # Helper for latency (implied from higher level or measured here? 
            # Controller measures "wall time" per row, but LLMClient knows actual API time.
            # LangChain doesn't always give latency in metadata.
            # We'll rely on the caller or wrap invoke.)
            
            return response.content
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return None

    def generate_completion_with_latency(self, prompt: str, system_prompt: str = "You are a helpful data generator.") -> tuple[Optional[str], float]:
        """Wrapper to measure latency."""
        import time
        start = time.time()
        result = self.generate_completion(prompt, system_prompt)
        end = time.time()
        return result, end - start

    def check_connection(self) -> bool:
        try:
            from langchain_core.messages import SystemMessage, HumanMessage

            response = self.chat_model.invoke(
                [
                    SystemMessage(content="Answer with a short confirmation."),
                    HumanMessage(content="Are you there?"),
                ]
            )
            content = getattr(response, "content", None)
            if isinstance(content, str):
                return bool(content.strip())
            return content is not None
        except Exception as exc:
            logger.error(f"Connection check failed: {exc}")
            return False

    def get_token_usage(self) -> Dict[str, int]:
        return self.token_usage

    def set_rag_service(self, rag_service) -> None:
        self.rag_service = rag_service

    def retrieve_context(self, query: str, top_k: Optional[int] = None) -> str:
        if not self.rag_service or not query:
            return ""
        try:
            import time

            self.rag_stats["queries"] += 1
            started = time.time()

            rag_cfg = self.config.rag
            min_score = rag_cfg.min_score if rag_cfg else 0.25
            max_chars = rag_cfg.max_context_chars if rag_cfg else 3000
            source_filter = rag_cfg.source_filter if rag_cfg else None

            # Primary retrieval with user-configured filters.
            hits = self.rag_service.search(query, top_k=top_k, min_score=min_score, source_filter=source_filter)

            # Fallback 1: if source filter is too strict, retry across all sources.
            if not hits and source_filter:
                hits = self.rag_service.search(query, top_k=top_k, min_score=min_score, source_filter=None)

            # Fallback 2: if threshold is too strict, retry with permissive score.
            if not hits and min_score > 0.0:
                hits = self.rag_service.search(query, top_k=top_k, min_score=0.0, source_filter=None)

            context = self.rag_service.format_hits(hits, max_context_chars=max_chars)

            elapsed = time.time() - started
            self.rag_stats["total_retrieval_time"] += elapsed
            self.rag_stats["last_hits"] = len(hits)
            if hits:
                self.rag_stats["queries_with_hits"] += 1
            self.rag_stats["total_context_chars"] += len(context)

            return context
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            return ""

    def get_rag_stats(self) -> Dict[str, Any]:
        return self.rag_stats

    def _normalize_schema_type(self, raw_type: Any) -> str:
        text = str(raw_type or "").strip()
        if not text:
            return ColumnType.SHORT_TEXT.value

        normalized = text.lower().replace("_", " ").replace("-", " ")
        aliases = {
            "short text": ColumnType.SHORT_TEXT.value,
            "shorttext": ColumnType.SHORT_TEXT.value,
            "long text": ColumnType.LONG_TEXT.value,
            "longtext": ColumnType.LONG_TEXT.value,
            "numeric": ColumnType.NUMERIC.value,
            "number": ColumnType.NUMERIC.value,
            "integer": ColumnType.NUMERIC.value,
            "float": ColumnType.NUMERIC.value,
            "categorical": ColumnType.CATEGORICAL.value,
            "category": ColumnType.CATEGORICAL.value,
            "boolean": ColumnType.BOOLEAN.value,
            "bool": ColumnType.BOOLEAN.value,
            "auto increment (id)": ColumnType.AUTO_INCREMENT.value,
            "auto increment": ColumnType.AUTO_INCREMENT.value,
            "id": ColumnType.AUTO_INCREMENT.value,
            "faker / deterministic": ColumnType.DETERMINISTIC.value,
            "faker": ColumnType.DETERMINISTIC.value,
            "deterministic": ColumnType.DETERMINISTIC.value,
        }
        return aliases.get(normalized, text if text in [t.value for t in ColumnType] else ColumnType.SHORT_TEXT.value)

    def _extract_json_payload(self, response_text: str) -> Optional[Any]:
        if not response_text:
            return None

        text = response_text.strip()
        fenced_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        candidates = [fenced_match.group(1).strip()] if fenced_match else []
        candidates.extend(re.findall(r"(\{[\s\S]*\}|\[[\s\S]*\])", text))

        for candidate in candidates or [text]:
            try:
                return json.loads(candidate)
            except Exception:
                continue
        return None

    def _coerce_schema_list(self, payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, dict):
            raw_columns = payload.get("columns", [])
        elif isinstance(payload, list):
            raw_columns = payload
        else:
            raw_columns = []

        columns: List[Dict[str, Any]] = []
        for raw_column in raw_columns:
            if not isinstance(raw_column, dict):
                continue
            name = str(raw_column.get("name", "")).strip()
            if not name:
                continue
            constraints = raw_column.get("constraints") or {}
            if not isinstance(constraints, dict):
                constraints = {}
            cleaned_constraints = {
                "options": constraints.get("options") or [],
                "regex_pattern": constraints.get("regex_pattern") or None,
                "min_value": constraints.get("min_value"),
                "max_value": constraints.get("max_value"),
                "min_length": constraints.get("min_length"),
                "max_length": constraints.get("max_length"),
                "allow_duplicates": bool(constraints.get("allow_duplicates", False)),
                "faker_provider": constraints.get("faker_provider") or None,
            }
            cleaned_constraints = {k: v for k, v in cleaned_constraints.items() if v not in (None, [], "")}
            columns.append(
                {
                    "name": name,
                    "type": self._normalize_schema_type(raw_column.get("type")),
                    "prompt_instruction": str(raw_column.get("prompt_instruction", "")).strip(),
                    "constraints": cleaned_constraints,
                }
            )
        return columns

    def _generate_schema_fallback(self, user_intent: str, context: Optional[str] = None) -> List[Dict[str, Any]]:
        fallback_prompt = (
            "Return JSON only. Format: "
            '{"columns":[{"name":"field_name","type":"Short Text","prompt_instruction":"what it should contain","constraints":{"options":["A","B"],"allow_duplicates":true}}]}\n'
            'Allowed types: "Short Text", "Long Text", "Numeric", "Categorical", "Boolean", "Auto Increment (ID)", "Faker / Deterministic".\n'
            "Rules: include at least 7 columns if requested; use @[column_name] for dependent fields; categorical fields need constraints.options; "
            "IDs/emails/phones should use allow_duplicates=false; no markdown or commentary.\n"
        )
        if context:
            fallback_prompt += f"Existing context: {context[:800]}\n"
        fallback_prompt += f"User intent: {user_intent}\n"

        raw = self.generate_completion(fallback_prompt, system_prompt="You return JSON only.")
        payload = self._extract_json_payload(raw or "")
        return self._coerce_schema_list(payload)

    def _generate_heuristic_schema(self, user_intent: str) -> List[Dict[str, Any]]:
        text = (user_intent or "").lower()

        def col(name: str, type_value: str, prompt_instruction: str, constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            return {
                "name": name,
                "type": type_value,
                "prompt_instruction": prompt_instruction,
                "constraints": constraints or {},
            }

        if "email" in text or "inbox" in text:
            return [
                col("message_id", ColumnType.AUTO_INCREMENT.value, "Unique identifier for each inbox message", {"allow_duplicates": False}),
                col("client_first_name", ColumnType.SHORT_TEXT.value, "Client first name"),
                col("client_last_name", ColumnType.SHORT_TEXT.value, "Client last name"),
                col("client_email", ColumnType.SHORT_TEXT.value, "Valid client email address based on @[client_first_name] and @[client_last_name]", {"allow_duplicates": False}),
                col("policy_number", ColumnType.SHORT_TEXT.value, "Medical insurance policy number for the client", {"allow_duplicates": False}),
                col("service_type", ColumnType.CATEGORICAL.value, "Type of insurance service the client is contacting about", {"options": ["Claim Status", "Coverage Question", "Premium Payment", "Network Provider", "Policy Update", "Pre-Authorization"], "allow_duplicates": True}),
                col("message_subject", ColumnType.SHORT_TEXT.value, "Email subject line matching @[service_type] and @[policy_number]"),
                col("message_body", ColumnType.LONG_TEXT.value, "Realistic client email body for @[service_type], consistent with @[policy_number] and @[message_subject]"),
            ]

        if "ticket" in text or "support" in text:
            return [
                col("ticket_id", ColumnType.AUTO_INCREMENT.value, "Unique sequential support ticket identifier", {"allow_duplicates": False}),
                col("issue_type", ColumnType.CATEGORICAL.value, "Primary support issue category", {"options": ["Login Problem", "Billing Question", "Bug Report", "Feature Request", "Account Update", "Service Outage"], "allow_duplicates": True}),
                col("customer_priority", ColumnType.CATEGORICAL.value, "Priority level for the ticket", {"options": ["Low", "Medium", "High", "Urgent"], "allow_duplicates": True}),
                col("customer_name", ColumnType.SHORT_TEXT.value, "Customer full name"),
                col("customer_email", ColumnType.SHORT_TEXT.value, "Customer email address for the ticket", {"allow_duplicates": False}),
                col("summary", ColumnType.SHORT_TEXT.value, "Brief issue summary related to @[issue_type]"),
                col("status", ColumnType.CATEGORICAL.value, "Current workflow status influenced by @[customer_priority]", {"options": ["Open", "Pending", "In Progress", "Resolved", "Closed"], "allow_duplicates": True}),
                col("resolution_note", ColumnType.LONG_TEXT.value, "Resolution details consistent with @[issue_type] and @[status]"),
            ]

        if "customer" in text or "contact" in text:
            return [
                col("customer_id", ColumnType.AUTO_INCREMENT.value, "Unique identifier for each customer", {"allow_duplicates": False}),
                col("first_name", ColumnType.SHORT_TEXT.value, "Customer first name"),
                col("last_name", ColumnType.SHORT_TEXT.value, "Customer last name"),
                col("email", ColumnType.SHORT_TEXT.value, "Customer email address based on @[first_name] and @[last_name]", {"allow_duplicates": False}),
                col("phone_number", ColumnType.SHORT_TEXT.value, "Customer phone number", {"allow_duplicates": False}),
                col("company", ColumnType.SHORT_TEXT.value, "Company name associated with the customer"),
                col("region", ColumnType.CATEGORICAL.value, "Customer region", {"options": ["North", "South", "East", "West", "Central"], "allow_duplicates": True}),
            ]

        return [
            col("record_id", ColumnType.AUTO_INCREMENT.value, "Unique identifier for each generated row", {"allow_duplicates": False}),
            col("title", ColumnType.SHORT_TEXT.value, "Short descriptive title for the record"),
            col("category", ColumnType.CATEGORICAL.value, "Main category for the record", {"options": ["Standard", "Priority", "Review", "Escalated", "Archived"], "allow_duplicates": True}),
            col("owner_name", ColumnType.SHORT_TEXT.value, "Name of the person associated with the record"),
            col("owner_email", ColumnType.SHORT_TEXT.value, "Email for @[owner_name]", {"allow_duplicates": False}),
            col("status", ColumnType.CATEGORICAL.value, "Current lifecycle status", {"options": ["Open", "Pending", "Approved", "Completed", "Closed"], "allow_duplicates": True}),
            col("description", ColumnType.LONG_TEXT.value, "Detailed text content consistent with @[title], @[category], and @[status]"),
        ]

    def generate_schema(self, user_intent: str, context: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate a list of ColumnDefinitions using LangGraph SchemaGeneratorAgent.
        """
        from core.schema_agent import create_schema_generator_graph
        
        logger.info(f"Generating schema for: {user_intent}")
        if self.on_log: self.on_log(f"Generating schema via LangGraph Agent...")

        if self.config.provider == AIProvider.LM_STUDIO:
            compact_columns = self._generate_schema_fallback(user_intent, context=context)
            if compact_columns:
                if self.on_log:
                    self.on_log(f"Compact schema generation returned {len(compact_columns)} column(s).")
                return compact_columns
            heuristic_columns = self._generate_heuristic_schema(user_intent)
            if heuristic_columns:
                if self.on_log:
                    self.on_log(f"Heuristic schema fallback returned {len(heuristic_columns)} column(s).")
                return heuristic_columns

        # Initialize the graph with self (providing the chat_model)
        agent = create_schema_generator_graph(self)
        
        # Initial state
        initial_state = {
            "user_intent": user_intent,
            "data_context": context,
            "messages": [],
            "schema": None,
            "error": None,
            "attempt_count": 0
        }
        
        try:
            # Invoke the graph
            final_state = agent.invoke(initial_state)
            output: Optional[Schema] = final_state.get("schema")
            error = final_state.get("error")

            if not output:
                 msg = f"Failed to generate schema after retries. Error: {error}"
                 logger.error(msg)
                 if self.on_log: self.on_log(msg)
                 fallback_columns = self._generate_schema_fallback(user_intent, context=context)
                 if fallback_columns:
                     fallback_msg = f"Schema fallback recovered {len(fallback_columns)} column(s)."
                     logger.info(fallback_msg)
                     if self.on_log: self.on_log(fallback_msg)
                     return fallback_columns
                 heuristic_columns = self._generate_heuristic_schema(user_intent)
                 if heuristic_columns:
                     heuristic_msg = f"Heuristic schema fallback recovered {len(heuristic_columns)} column(s)."
                     logger.info(heuristic_msg)
                     if self.on_log: self.on_log(heuristic_msg)
                     return heuristic_columns
                 return []

            # Convert back to list of dicts for the Controller
            schema_list = []
            for col in output.columns:
                col_dict = col.model_dump()
                # Ensure constraints is a dict and clean None values
                if "constraints" in col_dict and col_dict["constraints"]:
                     col_dict["constraints"] = {k: v for k, v in col_dict["constraints"].items() if v is not None}
                schema_list.append(col_dict)
            
            if self.on_log: self.on_log(f"Agent successfully generated {len(schema_list)} columns.")
            return schema_list

        except Exception as e:
            logger.error(f"Schema generation failed: {e}")
            if self.on_log: self.on_log(f"Error generating schema: {e}")
            fallback_columns = self._generate_schema_fallback(user_intent, context=context)
            return fallback_columns or self._generate_heuristic_schema(user_intent)
