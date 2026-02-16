import logging
import json
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
            # We don't track usage for connection checks
            self.generate_completion("Are you there?", system_prompt="Answer yes or no.")
            return True
        except:
            return False

    def get_token_usage(self) -> Dict[str, int]:
        return self.token_usage

    def generate_schema(self, user_intent: str, context: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate a list of ColumnDefinitions using LangGraph SchemaGeneratorAgent.
        """
        from core.schema_agent import create_schema_generator_graph
        
        logger.info(f"Generating schema for: {user_intent}")
        if self.on_log: self.on_log(f"Generating schema via LangGraph Agent...")

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
            return []
