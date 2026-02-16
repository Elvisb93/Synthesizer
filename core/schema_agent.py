import logging
import json
from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from core.models import Schema, ColumnSchema
from core.llm_client import LLMClient  # Type hint only if needed, but we pass the client instance or model

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    user_intent: str
    data_context: Optional[str]  # Added context field
    messages: List[Any]
    schema: Optional[Schema]
    error: Optional[str]
    attempt_count: int

def create_schema_generator_graph(llm_client):
    """
    Creates a LangGraph for generating and validating schemas.
    """
    
    # --- Nodes ---
    
    def generate_schema(state: AgentState):
        """Generates or refines the schema based on intent and errors."""
        user_intent = state['user_intent']
        data_context = state.get('data_context')
        error = state.get('error')
        attempt = state.get('attempt_count', 0)
        
        parser = PydanticOutputParser(pydantic_object=Schema)
        
        # Base instructions
        base_instruction = """You are a specialized data scientist assistant.
Your task is to convert the user's dataset description into a JSON schema for a synthetic data generator.

{format_instructions}

IMPORTANT RULES:
1. "type" must be one of: "Short Text", "Long Text", "Numeric", "Categorical", "Boolean", "Auto Increment (ID)", "Faker / Deterministic".
2. If "type" is "Categorical", you MUST provide "options" in "constraints" OR use a dependency in "prompt_instruction".
3. Use @[column_name] in "prompt_instruction" to reference values from other columns for consistency (e.g., "City in @[Country]").
4. Ensure "allow_duplicates" is True for most fields (e.g. Gender, Country, City, First Name), EXCEPT for unique identifiers (ID, Email, SSN, Phone).
5. For "Phone" or "Telephone" fields, use a simple regex like "^\\+?[0-9\\-\\s]+$" to avoid generation errors. Do NOT use complex strict patterns.
6. DEFAULT "min_length" to 0 or 1. Do NOT set it to 10 unless absolutely necessary (like for an ID or long text).
7. For "Categorical", you MUST provide at least 5-10 valid `options` in the constraints.
"""

        # Inject Data Context if present
        if data_context:
            base_instruction += f"\n\nEXISTING DATA CONTEXT:\n{data_context}\n\n"
            base_instruction += "CRITICAL: Do NOT regenerate or list columns that are already present in the EXISTING DATA CONTEXT. " \
                                "Only generate NEW columns that derive from, enrich, or analyze the existing data. " \
                                "You CAN and SHOULD reference existing columns using @[ColumnName] syntax in your 'prompt_instruction'."

        if error:
            # Refinement prompt
            instruction = base_instruction + f"\n\nPREVIOUS ERROR: {error}\nPlease fix the schema based on this error."
        else:
            instruction = base_instruction

        prompt_template = PromptTemplate(
            template=instruction + "\n\nUser Intent: {user_intent}",
            input_variables=["user_intent"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt_template | llm_client.chat_model | parser
        
        try:
            output = chain.invoke({"user_intent": user_intent})
            
            # Post-processing / Hygiene
            if output and output.columns:
                filtered_columns = []
                
                # Parse context to get existing column names for safety filtering
                existing_cols = []
                if data_context:
                    import re
                    # extract explicit column names if formatted as "Column: Name (Type)"
                    # This is valid because we control the format in flet_app.py
                    existing_cols = re.findall(r"Column: (.+?) \(", data_context)
                
                for col in output.columns:
                    # Safety Filter: Reject columns that already exist
                    if col.name in existing_cols:
                        logger.warning(f"Rejecting generated column '{col.name}' because it already exists in the imported data.")
                        continue
                        
                    col_name_lower = col.name.lower()
                    
                    # 1. Force min_length = 0 unless it's strictly a "Long Text" field
                    # The AI defaults to 10 way too often, breaking things like "Country=USA" or "Gender=Male"
                    if col.type != "Long Text":
                        if col.constraints.min_length is not None and col.constraints.min_length > 0:
                            logger.info(f"Hygiene: Resetting min_length from {col.constraints.min_length} to 0 for column '{col.name}'")
                            col.constraints.min_length = 0

                    # 2. Ensure Categorical has options
                    if col.type == "Categorical":
                        if not col.constraints.options:
                            logger.warning(f"Hygiene: Categorical column '{col.name}' has no options. Injecting placeholders.")
                            col.constraints.options = ["Option A", "Option B", "Option C"]
                            col.prompt_instruction += " (Please edit options)"
                    
                    # 3. Smart Uniqueness Defaults
                    # If it looks like an ID, Email, Phone, or SSN, keep it unique.
                    # Otherwise, default allow_duplicates to True to prevent generation exhaustion.
                    is_identifier = any(x in col_name_lower for x in ["id", "num", "code", "email", "phone", "ssn", "isbn", "uuid"])
                    if col.type == "Auto Increment (ID)":
                        is_identifier = True
                    
                    if not is_identifier:
                        if not col.constraints.allow_duplicates:
                            logger.info(f"Hygiene: Setting allow_duplicates=True for non-identifier column '{col.name}'")
                            col.constraints.allow_duplicates = True
                    
                    filtered_columns.append(col)
                
                output.columns = filtered_columns

            return {"schema": output, "error": None, "attempt_count": attempt + 1}
        except Exception as e:
            return {"schema": None, "error": str(e), "attempt_count": attempt + 1}

    def validate_schema(state: AgentState):
        """Validates the generated schema for logical correctness."""
        schema = state.get('schema')
        
        if not schema:
            return {"error": "No schema generated."}
            
        if not schema.columns:
            return {"error": "Schema has 0 columns. Please generate at least one column."}

        # Basic logical check (e.g., ensuring Categorical types have options or instructions)
        for col in schema.columns:
             if col.type == "Categorical" and not col.constraints.options and "@" not in col.prompt_instruction:
                 # This is a soft check, maybe not strict error, but good for self-correction
                 pass

        return {"error": None} # Success

    # --- Edges ---
    
    def router(state: AgentState):
        """Decides whether to end or retry."""
        error = state.get('error')
        attempt = state.get('attempt_count', 0)
        
        if not error:
            return "end"
        
        if attempt >= 3:
            logger.warning(f"Max retries reached. Last error: {error}")
            return "end" # Give up
        
        return "retry"

    # --- Graph Construction ---
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("generate", generate_schema)
    workflow.add_node("validate", validate_schema)
    
    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "validate")
    
    workflow.add_conditional_edges(
        "validate",
        router,
        {
            "end": END,
            "retry": "generate"
        }
    )
    
    return workflow.compile()
