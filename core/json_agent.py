"""
JSON Agent — LangGraph state machine for generating single JSON objects.

Adapted from row_agent.py, this agent generates JSON objects matching an
inferred Pydantic schema, validates them, and corrects on failure.
Handles LM Studio reasoning traces before JSON output (R010).
"""
import json
import logging
from typing import TypedDict, List, Dict, Any, Optional, Type

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel

from core.llm_client import LLMClient

logger = logging.getLogger(__name__)


class JsonItemState(TypedDict):
    item_data: Dict[str, Any]
    schema_description: str
    template_context: str
    errors: List[str]
    attempt_count: int
    is_valid: bool


def _parse_json_from_response(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from LLM response, handling reasoning traces (R010).

    LM Studio models may produce reasoning text before the actual JSON payload.
    This parser tries progressively more aggressive extraction methods.
    """
    if not text:
        return None

    cleaned = text.strip().replace("```json", "").replace("```", "").strip()

    # Attempt 1: Direct parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Attempt 2: Find the last JSON object in the response (handles reasoning traces)
    last_start = cleaned.rfind("{")
    if last_start >= 0:
        # Walk forward to find the matching closing brace
        depth = 0
        for i in range(last_start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[last_start : i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        pass
                    break

    # Attempt 3: Find the FIRST complete JSON object
    first_start = cleaned.find("{")
    if first_start >= 0 and first_start != last_start:
        depth = 0
        for i in range(first_start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[first_start : i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        pass
                    break

    return None


def create_json_generator_graph(
    llm_client: LLMClient,
    schema_model: Optional[Type[BaseModel]] = None,
):
    """Creates a LangGraph for generating, validating, and correcting a single JSON object.

    Args:
        llm_client: The LLM client for text generation.
        schema_model: Optional Pydantic model class for validation. If None,
                      validation is limited to JSON well-formedness.

    Returns:
        Compiled LangGraph workflow.
    """

    def generate_item(state: JsonItemState) -> dict:
        """Generate a JSON object matching the target schema."""
        schema_desc = state["schema_description"]
        template_ctx = state.get("template_context", "")

        context_block = ""
        if template_ctx:
            context_block = (
                "\nHere is additional context from the master template "
                "(use it for consistency, not as the schema to follow):\n"
                f"{template_ctx[:2000]}\n"
            )

        prompt = (
            f"Generate a single JSON object that matches this schema:\n"
            f"{schema_desc}\n"
            f"{context_block}\n"
            "Rules:\n"
            "- Return ONLY a valid JSON object. No explanation, no markdown.\n"
            "- All fields in the schema must be present.\n"
            "- Values must be realistic and diverse.\n"
            "- Do not copy example values verbatim; generate novel content.\n"
        )

        response = llm_client.generate_completion(
            prompt,
            system_prompt="You are a precise JSON data generator. Output valid JSON only.",
        )

        parsed = _parse_json_from_response(response)
        if parsed:
            return {"item_data": parsed, "errors": [], "is_valid": False}
        else:
            return {
                "item_data": {},
                "errors": [f"Failed to parse JSON from response: {(response or '')[:200]}"],
                "is_valid": False,
            }

    def validate_item(state: JsonItemState) -> dict:
        """Validate the generated item against the Pydantic schema."""
        item_data = state["item_data"]

        if not item_data:
            return {"is_valid": False, "errors": ["Empty item data."]}

        # If we have a Pydantic model, validate against it
        if schema_model is not None:
            try:
                schema_model.model_validate(item_data)
                return {"is_valid": True, "errors": []}
            except Exception as e:
                return {"is_valid": False, "errors": [f"Schema validation failed: {e}"]}

        # Fallback: just check it's a non-empty dict with string keys
        if isinstance(item_data, dict) and len(item_data) > 0:
            return {"is_valid": True, "errors": []}

        return {"is_valid": False, "errors": ["Item data is not a valid dict."]}

    def correct_item(state: JsonItemState) -> dict:
        """Attempt to correct a failed JSON object based on errors."""
        item_data = state["item_data"]
        errors = state["errors"]
        attempt = state.get("attempt_count", 0)
        schema_desc = state["schema_description"]

        error_msg = " ".join(errors) if errors else "Unknown validation error"

        prompt = (
            f"The following JSON object failed validation:\n"
            f"{json.dumps(item_data, indent=2)}\n\n"
            f"Errors: {error_msg}\n\n"
            f"Required schema:\n{schema_desc}\n\n"
            "Please generate a corrected JSON object that fixes these errors.\n"
            "Return ONLY the corrected JSON object. No explanation.\n"
        )

        response = llm_client.generate_completion(
            prompt,
            system_prompt="You are a JSON correction agent. Fix validation errors and output valid JSON only.",
        )

        parsed = _parse_json_from_response(response)
        if parsed:
            return {"item_data": parsed, "attempt_count": attempt + 1}
        else:
            return {"item_data": item_data, "attempt_count": attempt + 1}

    def router(state: JsonItemState) -> str:
        if state["is_valid"]:
            return "end"
        if state.get("attempt_count", 0) >= 3:
            return "end"  # Give up after max attempts
        return "correct"

    # Build the graph
    workflow = StateGraph(JsonItemState)

    workflow.add_node("generate", generate_item)
    workflow.add_node("validate", validate_item)
    workflow.add_node("correct", correct_item)

    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "validate")
    workflow.add_conditional_edges(
        "validate",
        router,
        {"end": END, "correct": "correct"},
    )
    workflow.add_edge("correct", "validate")

    return workflow.compile()
