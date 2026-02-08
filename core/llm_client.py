import logging
import json
import re
from typing import List, Optional, Dict, Any, Callable
from openai import OpenAI, AzureOpenAI, APIError
from .models import GeneratorConfig, ColumnType, AIProvider

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
        
        # Azure OpenAI uses a different client class
        if config.provider == AIProvider.AZURE_OPENAI:
            if not config.api_key:
                raise ValueError("API key is required for Azure OpenAI")
            if not config.azure_endpoint:
                raise ValueError("Azure endpoint is required for Azure OpenAI")
            if not config.azure_deployment:
                raise ValueError("Azure deployment name is required for Azure OpenAI")
            
            self.client = AzureOpenAI(
                api_key=config.api_key,
                api_version="2024-02-15-preview",  # Latest stable version
                azure_endpoint=config.azure_endpoint
            )
            # Store deployment name for use in completions
            self.azure_deployment = config.azure_deployment
        else:
            # Standard OpenAI client for all other providers
            base_url = PROVIDER_BASE_URLS.get(config.provider, config.api_base_url)
            
            # Determine API key
            if config.provider == AIProvider.LM_STUDIO:
                api_key = "lm-studio"  # LM Studio doesn't need a real key
            else:
                if not config.api_key:
                    raise ValueError(f"API key is required for provider: {config.provider.value}")
                api_key = config.api_key
            
            # Configure default headers (e.g., for Gemini)
            default_headers = {}
            if config.provider == AIProvider.GOOGLE_GEMINI:
                default_headers["x-goog-api-client"] = "synthesizer-app/1.0.0"
            
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                default_headers=default_headers if default_headers else None
            )

    def list_models(self) -> List[str]:
        """Fetch available models from LM Studio."""
        try:
            models = self.client.models.list()
            return [m.id for m in models.data]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def generate_completion(self, prompt: str, system_prompt: str = "You are a helpful data generator.") -> Optional[str]:
        """Generate a response for a given prompt."""
        try:
            # For Azure, use deployment name; for others, use model_id
            model_name = self.azure_deployment if self.config.provider == AIProvider.AZURE_OPENAI else self.config.model_id
            
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4096,
                timeout=120.0 # 2 minute timeout per call
            )
            return response.choices[0].message.content.strip()
        except APIError as e:
            logger.error(f"LLM API Error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected generation error: {e}")
            return None
            
    def check_connection(self) -> bool:
        try:
            models = self.list_models()
            return len(models) > 0
        except:
            return False

    def generate_schema(self, user_intent: str) -> List[Dict[str, Any]]:
        """
        Generate a list of ColumnDefinitions based on user intent.
        Returns a list of dictionaries compatible with ColumnDefinition.
        """
        system_prompt = """You are a specialized data scientist assistant.
        Your task is to convert the user's dataset description into a JSON schema for a synthetic data generator.

        **Output Format:**
        Return ONLY a raw JSON list of objects. Do not include markdown formatting, ```json blocks, or explanations.

        **Column Specification:**
        Each object in the list should have:
        - "name": (string) Column name.
        - "type": (string) One of ["Short Text", "Long Text", "Numeric", "Categorical", "Boolean", "Auto Increment (ID)", "Faker / Deterministic"].
        - "prompt_instruction": (string) A concise generation prompt for the LLM OR a Faker provider (e.g. "Generate a realistic US city").
        - "constraints": (object) Optional rules.
        
        **Supported Constraints:**
        - "min_value", "max_value" (float/int): For Numeric types.
        - "min_length", "max_length" (int): For Text types.
        - "regex_pattern" (string): For strict formatting (e.g. phone numbers).
        - "options" (list of strings): REQUIRED for "Categorical" type.
        - "allow_duplicates" (boolean): **CRITICAL**. Defaults to false (unique). Set to TRUE for low-cardinality fields (e.g. Gender, State, Country, Categories) or if the user implies repetition.
        
        **Examples:**
        1. User: "A list of 1000 users with age and gender"
           [
             {"name": "id", "type": "Auto Increment (ID)"},
             {"name": "full_name", "type": "Short Text", "prompt_instruction": "Full Name"},
             {"name": "gender", "type": "Categorical", "constraints": {"options": ["Male", "Female", "Non-binary"], "allow_duplicates": true}},
             {"name": "age", "type": "Numeric", "constraints": {"min_value": 18, "max_value": 90, "allow_duplicates": true}}
           ]
        """
        # The following part was part of the original system_prompt and is now appended to the new one.
        system_prompt += (
            "\n\n"
            "### Dependency Syntax:\n"
            "Use @[column_name] in 'prompt_instruction' to reference values from other columns.\n"
            "Example: If 'country' column exists, 'city' column prompt could be \"City in @[country]\".\n"
            "\n"
            "### Example Output:\n"
            "[\n"
            "  {\"name\": \"id\", \"type\": \"Auto Increment (ID)\", \"prompt_instruction\": \"\", \"constraints\": {}},\n"
            "  {\"name\": \"full_name\", \"type\": \"Faker / Deterministic\", \"prompt_instruction\": \"\", \"constraints\": {\"faker_provider\": \"name\"}},\n"
            "  {\"name\": \"role\", \"type\": \"Categorical\", \"prompt_instruction\": \"\", \"constraints\": {\"options\": [\"Admin\", \"User\", \"Guest\"]}},\n"
            "  {\"name\": \"bio\", \"type\": \"Long Text\", \"prompt_instruction\": \"Short bio for a @[role] named @[full_name]\", \"constraints\": {\"min_length\": 20}}\n"
            "]\n"
            "\n"
            "[\n"
            "  {\"name\": \"id\", \"type\": \"Auto Increment (ID)\", \"prompt_instruction\": \"\", \"constraints\": {}},\n"
            "  {\"name\": \"full_name\", \"type\": \"Faker / Deterministic\", \"prompt_instruction\": \"\", \"constraints\": {\"faker_provider\": \"name\"}},\n"
            "  {\"name\": \"role\", \"type\": \"Categorical\", \"prompt_instruction\": \"\", \"constraints\": {\"options\": [\"Admin\", \"User\", \"Guest\"]}},\n"
            "  {\"name\": \"bio\", \"type\": \"Long Text\", \"prompt_instruction\": \"Short bio for a @[role] named @[full_name]\", \"constraints\": {\"min_length\": 20}}\n"
            "]\n"
            "\n"
            "Do not include markdown formatting (```json) or explanations. Just the JSON array.\n"
            "IMPORTANT: Do not output your internal reasoning or chain-of-thought. Output ONLY the raw JSON.\n"
            "IMPORTANT: ENSURE every 'Short Text' or 'Long Text' column has a 'prompt_instruction'. If a column depends on another, USE the @[col] syntax."
        )

        current_prompt = f"User Intent: {user_intent}"
        
        for attempt in range(3): # Max 3 attempts (Initial + 2 Repairs)
            try:
                if attempt > 0:
                    msg = f"Schema Repair Attempt {attempt}: {current_prompt[:50]}..."
                    logger.info(msg)
                    if self.on_log: self.on_log(msg)
                else:
                    msg = f"Generating schema attempt {attempt+1}/3..."
                    logger.info(msg)
                    if self.on_log: self.on_log(msg)
                
                raw_response = self.generate_completion(current_prompt, system_prompt)
                
                if not raw_response:
                    raise ValueError("Empty response from LLM")
                
                # Clean up response (remove markdown code blocks if present)
                clean_json = raw_response.strip()
                
                # Handle potential "Reasoning" preamble commonly found in local models (e.g. <think>, OR just text before the json)
                # We look for the first '[' and last ']'
                start_idx = clean_json.find('[')
                end_idx = clean_json.rfind(']')
                
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    clean_json = clean_json[start_idx:end_idx+1]
                else:
                    # Fallback cleanups
                    if clean_json.startswith("```json"):
                        clean_json = clean_json[7:]
                    if clean_json.startswith("```"):
                        clean_json = clean_json[3:]
                    if clean_json.endswith("```"):
                        clean_json = clean_json[:-3]
                
                clean_json = clean_json.strip()
                
                # Parse JSON
                schema = json.loads(clean_json)
                
                if not isinstance(schema, list):
                    raise ValueError("Response is not a JSON array")
                
                # --- VALIDATION & REPAIR CHECK ---
                missing_prompts = []
                for col in schema:
                    ctype = col.get("type", "")
                    prompt = col.get("prompt_instruction", "")
                    
                    # Logic: Text/Numeric/Categorical types MUST have a prompt.
                    # Auto Inc doesn't need one. Faker uses it for provider name (handled as constraints now, but let's be safe).
                    if ctype in ["Short Text", "Long Text", "Numeric", "Categorical"] and (not prompt or len(prompt.strip()) < 3):
                        missing_prompts.append(col.get("name", "unknown"))
                    
                    # Extra Check for Categorical: Must have options OR a dependency link
                    if ctype == "Categorical":
                        constraints = col.get("constraints", {})
                        options = constraints.get("options", [])
                        if not options and "@[" not in prompt:
                             missing_prompts.append(f"{col.get('name', 'unknown')} (Categorical missing options/link)")

                if missing_prompts:
                    if attempt < 2:
                        msg = f"Invalid columns found: {missing_prompts}. Retrying..."
                        logger.warning(msg)
                        if self.on_log: self.on_log(msg)
                        
                        logger.warning(f"Repairing schema due to issues in: {missing_prompts}")
                        current_prompt = (
                            f"The previous schema had issues. The following columns need fixing: {missing_prompts}.\n"
                            f"Issue: Categorical columns MUST have either a specific list of 'options' in constraints OR a prompt that links to another column (e.g., 'Category for @[product_name]').\n"
                            f"Issue: Text/Numeric columns must have detailed prompts.\n"
                            f"Here is the incomplete JSON:\n"
                            f"```json\n{json.dumps(schema, indent=2)}\n```\n"
                            f"Please regenerate the FULL JSON schema, fixing these columns.\n"
                            f"Original Intent: {user_intent}"
                        )
                        continue # Retry loop
                    else:
                        logger.error("Max repair attempts reached. Returning imperfect schema.")
                
                # --- DEPENDENCY REFINEMENT (New Step) ---
                # Only run if we have multiple columns and valid schema
                if len(schema) > 1:
                     try:
                         msg = "Refining schema dependencies..."
                         logger.info(msg)
                         if self.on_log: self.on_log(msg)
                         
                         schema = self._refine_dependencies(schema, user_intent)
                     except Exception as e:
                         logger.error(f"Dependency refinement failed, using original schema: {e}")

                return schema

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON schema: {e}. Raw response: {raw_response}")
                if attempt < 2:
                     current_prompt = f"The previous JSON was invalid ({e}). Please output strictly valid JSON array."
                     continue
                return []
            except Exception as e:
                logger.error(f"Schema generation failed: {e}")
                return []
        
        return []

    def _refine_dependencies(self, schema: List[Dict[str, Any]], user_intent: str) -> List[Dict[str, Any]]:
        """
        Second pass to explicitly link columns that should be dependent.
        """
        system_prompt = (
            "You are an expert data schema optimizer.\n"
            "Your task is to review a JSON schema and add *dependencies* between columns where logical consistency is needed.\n"
            "If Column B depends on Column A (e.g. City depends on Country, or Product Category depends on Product Name), "
            "you MUST update Column B's 'prompt_instruction' to include '@[Column A]'.\n"
            "\n"
            "Example:\n"
            "Input: [{'name': 'country', ...}, {'name': 'city', 'prompt_instruction': 'Random city', ...}]\n"
            "Output: [{'name': 'country', ...}, {'name': 'city', 'prompt_instruction': 'Major city in @[country]', ...}]\n"
            "\n"
            "Return the FULL updated JSON array. Do not remove any columns. Only update 'prompt_instruction' fields to add links."
        )
        
        prompt = (
            f"User Intent: {user_intent}\n"
            f"Current Schema:\n```json\n{json.dumps(schema, indent=2)}\n```\n"
            "Analyze the relationships between these columns.\n"
            "If a column should be consistent with another (e.g. correlated data), update its prompt to use the @[col_name] syntax.\n"
            "Return the full VALID JSON array."
        )
        
        raw_response = self.generate_completion(prompt, system_prompt)
        
        if not raw_response:
            return schema # Fallback
            
        # Clean & Parse (Reuse logic or simplify)
        clean_json = raw_response.strip()
        if clean_json.startswith("```json"): clean_json = clean_json[7:]
        if clean_json.startswith("```"): clean_json = clean_json[3:]
        if clean_json.endswith("```"): clean_json = clean_json[:-3]
        clean_json = clean_json.strip()
        
        try:
            refined_schema = json.loads(clean_json)
            if isinstance(refined_schema, list) and len(refined_schema) == len(schema):
                return refined_schema
            else:
                logger.warning("Refined schema invalid or column count mismatch. Ignoring.")
                return schema
        except json.JSONDecodeError:
            logger.warning("Failed to parse refined schema. Ignoring.")
            return schema
