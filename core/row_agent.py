import logging
import re
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage

from core.models import ColumnDefinition, ColumnType
from core.llm_client import LLMClient

logger = logging.getLogger(__name__)

class RowState(TypedDict):
    row_data: Dict[str, Any]
    columns: List[ColumnDefinition]
    errors: List[str]
    attempt_count: int
    is_valid: bool

def create_row_generator_graph(llm_client: LLMClient):
    """
    Creates a LangGraph for generating, validating, and correcting a single row of data.
    """

    # --- Helpers ---
    def _construct_prompt(col: ColumnDefinition, row_data: Dict[str, Any]) -> str:
        """Helper to construct prompt for a single column (same as Controller logic)."""
        instruction = col.prompt_instruction
        # Simple dependency interpolation
        for other_col_name, val in row_data.items():
            if f"@[{other_col_name}]" in instruction:
                instruction = instruction.replace(f"@[{other_col_name}]", str(val))
        
        constraints_text = ""
        if col.constraints.options:
             constraints_text += f"\n- Options: {', '.join(col.constraints.options)}"
        
        # NEW: Inject Context from already generated fields in this row
        # This helps the LLM make coherent decisions (e.g. City -> Country) without explicit linking
        context_str = ""
        visible_data = {}
        if row_data:
            # Filter out internal keys if any, though row_data should be clean here
            visible_data = {k: v for k, v in row_data.items() if v is not None and v != ""}
            if visible_data:
                context_str = f"\nCurrent Row Context: {visible_data}"
        allowed_placeholders = sorted(
            {
                match
                for value in visible_data.values()
                if isinstance(value, str)
                for match in re.findall(r"<[A-Z0-9_]+>", value)
            }
        )

        role_hint = ""
        sender_name = row_data.get("sender_name")
        sender_email = row_data.get("sender_email")
        field_name = (col.name or "").strip().lower()
        instruction_lower = (instruction or "").strip().lower()
        if sender_name or sender_email:
            if "reply" in field_name or "reply" in instruction_lower:
                role_hint = (
                    "\nRole Guidance:"
                    f"\n- The incoming email was written by the sender ({sender_name or sender_email})."
                    "\n- Write the reply as if you are responding to that sender."
                    "\n- Do not treat organization placeholders or salutations inside the email body as the sender."
                    "\n- Keep placeholders exactly as placeholders; do not invent real identities."
                    "\n- Do not invent new placeholder tokens, IDs, case numbers, document titles, or contact details."
                )
            elif "summary" in field_name or "summary" in instruction_lower:
                role_hint = (
                    "\nRole Guidance:"
                    "\n- Summarize the sender's request or message."
                    "\n- Do not describe the recipient organization as if it authored the email."
                    "\n- Keep placeholders exactly as placeholders; do not invent real identities."
                    "\n- Do not invent new placeholder tokens, IDs, case numbers, document titles, or contact details."
                )
        placeholder_guidance = (
            "\nPlaceholder Guidance:"
            f"\n- Allowed placeholders already present in the row: {allowed_placeholders if allowed_placeholders else 'none'}."
            "\n- Reuse only placeholders already present in the current row context."
            "\n- If a fact is not present in the row context, keep the response generic rather than inventing a new placeholder."
        )

        rag_context = ""
        query = f"Column: {col.name}\nInstruction: {instruction}\nRow Context: {visible_data if row_data else {}}"
        if hasattr(llm_client, "retrieve_context"):
            rag_context = llm_client.retrieve_context(query)

        rag_block = ""
        if rag_context:
            rag_block = (
                "\nRetrieved Context (use only if relevant):\n"
                f"{rag_context}\n"
            )

        return (
            f"Generate a single {col.type.value} value for column '{col.name}'.\n"
            f"Context: {instruction}\n"
            f"{context_str}\n"
            f"{role_hint}\n"
            f"{placeholder_guidance}\n"
            f"Constraints: {constraints_text}\n"
            f"{rag_block}"
            "Return ONLY the value."
        )

    # --- Nodes ---

    def generate_row(state: RowState):
        """Generates initial values for the row if they are missing."""
        row_data = state['row_data'].copy()
        columns = state['columns']
        
        # Simple topological sort/execution order should be passed or handled. 
        # For this node, we iterate and fill missing.
        # We assume 'columns' is already sorted by dependency if needed, 
        # OR we just iterate and hope dependencies are resolved (Controller handles sort).
        
        for col in columns:
            if col.name not in row_data:
                # 1. Faker/Deterministic check
                if col.type == ColumnType.DETERMINISTIC:
                    # We might need the faker instance here, or skip. 
                    # Ideally, purely deterministic stuff is done before? 
                    # Let's assume LLM handles everything NOT deterministic or pre-filled.
                    continue 

                # 2. LLM Generation
                prompt = _construct_prompt(col, row_data)
                val = llm_client.generate_completion(prompt)
                if val:
                    row_data[col.name] = val.strip().strip('"')
        
        return {"row_data": row_data, "errors": [], "is_valid": False}

    def validate_semantics(state: RowState):
        """Checks if the row makes semantic sense using LLM."""
        row_data = state['row_data']
        # If any required keys are missing, that's an error (unless optional)
        
        # Construct a validation prompt
        data_str = ", ".join([f"{k}: {v}" for k, v in row_data.items()])
        allowed_placeholders = sorted(
            {
                match
                for value in row_data.values()
                if isinstance(value, str)
                for match in re.findall(r"<[A-Z0-9_]+>", value)
            }
        )
        prompt = (
            f"Review this data row: {{{data_str}}}.\n"
            "Check for semantic consistency. Example: 'Lion' cannot live in 'Bank'.\n"
            f"Allowed placeholders from source data: {allowed_placeholders if allowed_placeholders else 'none'}.\n"
            "Treat only explicit angle-bracket mask tokens such as <NAME_1> or <EMAIL_2> as placeholders.\n"
            "Do not treat ordinary numbers, IDs, dates, or business values as placeholders unless they use that angle-bracket token format.\n"
            "If the row contains an explicit angle-bracket placeholder token not in the allowed list, it is INVALID.\n"
            "If valid and consistent, reply 'VALID'.\n"
            "If invalid, reply 'INVALID: <reason>' specifying which field is wrong."
        )
        
        response = llm_client.generate_completion(prompt, system_prompt="You are a data quality auditor.")
        
        if response and response.strip().upper().startswith("VALID"):
            return {"is_valid": True, "errors": []}
        else:
            return {"is_valid": False, "errors": [response]}

    def correct_row(state: RowState):
        """Fixes specific fields based on validation errors."""
        row_data = state['row_data'].copy()
        errors = state['errors']
        columns = state['columns']
        attempt = state.get('attempt_count', 0)
        
        error_msg = " ".join(errors) if errors else "Unknown error"
        
        # Heuristic: Ask LLM to fix the whole row context or specific fields?
        # Let's try fixing the whole row context to ensure consistency.
        
        prompt = (
            f"The following row has semantic errors: {row_data}\n"
            f"Errors found: {error_msg}\n"
            "Please regenerate the incorrect values to make the row consistent.\n"
            "Do not introduce any new placeholder tokens that were not already present in the row.\n"
            "Return the corrected row as JSON."
        )
        
        # We need structured output here ideally.
        # Fallback: Just ask to regenerate the specfic columns mentioned?
        # Let's try a simpler approach: Ask specifically "Fix the row"
        
        response = llm_client.generate_completion(prompt)
        
        # Try to parse JSON output? Or just rely on string matching?
        # This is where strict structure calls help. 
        # For now, let's assume the LLM might give us JSON if asked nicely.
        
        # To make it robust without complicating parsing too much:
        # We'll just ask for the WHOLE valid row JSON.
        
        try:
            # Clean up json markdown
            clean = response.replace("```json", "").replace("```", "").strip()
            import json
            corrected_data = json.loads(clean)
            
            # Update row_data with corrected values
            row_data.update(corrected_data)
            
        except Exception:
            # If parsing fails, maybe just re-run generation?
            # Or mark as failed.
            pass

        return {"row_data": row_data, "attempt_count": attempt + 1}

    # --- Edges ---
    
    def router(state: RowState):
        if state['is_valid']:
            return "end"
        if state['attempt_count'] >= 3:
            return "end" # Give up
        return "correct"

    # --- Graph ---
    
    workflow = StateGraph(RowState)
    
    workflow.add_node("generate", generate_row)
    workflow.add_node("validate", validate_semantics)
    workflow.add_node("correct", correct_row)
    
    # Start -> Generate (to fill gaps) -> Validate -> Correct/End
    workflow.set_entry_point("generate")
    
    workflow.add_edge("generate", "validate")
    
    workflow.add_conditional_edges(
        "validate",
        router,
        {
            "end": END,
            "correct": "correct"
        }
    )
    
    workflow.add_edge("correct", "validate") # Loop back to validate after correction
    
    return workflow.compile()
