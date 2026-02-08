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
    messages: List[Any]
    schema: Optional[Schema]
    error: Optional[str]
    attempt_count: int

def create_schema_generator_graph(llm_client: 'LLMClient'):
    """
    Creates a LangGraph for generating and validating schemas.
    """
    
    # --- Nodes ---
    
    def generate_schema(state: AgentState):
        """Generates or refines the schema based on intent and errors."""
        user_intent = state['user_intent']
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
4. Ensure "allow_duplicates" is True for fields that naturally repeat (Gender, Country, etc.).
"""

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
