"""
Chunk Agent — LangGraph state machine for exhaustive document extraction.

Takes a raw text chunk from ingested documents and extracts factual,
strictly grounded instruction→response pairs using Self-Foveate and
Chain-of-Verification (CoVe) prompting, then filters via an
LLM-as-a-Judge critique stage.
"""
import json
import logging
from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, END

from core.llm_client import LLMClient
from core.json_agent import _parse_json_from_response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers for parsing an array of objects from LLM output
# ---------------------------------------------------------------------------

def _parse_json_array_from_response(text: str) -> Optional[List[Dict[str, Any]]]:
    """Extract a JSON array from LLM response, handling reasoning traces (R010).

    Returns a list of dicts, or None if parsing fails.
    """
    if not text:
        return None

    cleaned = text.strip().replace("```json", "").replace("```", "").strip()

    # Attempt 1: Direct parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass

    # Attempt 2: Find the outermost JSON array
    first_bracket = cleaned.find("[")
    if first_bracket >= 0:
        depth = 0
        for i in range(first_bracket, len(cleaned)):
            if cleaned[i] == "[":
                depth += 1
            elif cleaned[i] == "]":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[first_bracket : i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, list):
                            return [item for item in parsed if isinstance(item, dict)]
                    except json.JSONDecodeError:
                        pass
                    break

    # Attempt 3: Try parsing as a single object wrapped in array
    single = _parse_json_from_response(text)
    if single and isinstance(single, dict):
        return [single]

    return None


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class ChunkExtractionState(TypedDict):
    chunk_text: str
    chunk_metadata: Dict[str, Any]
    extracted_pairs: List[Dict[str, str]]
    verified_pairs: List[Dict[str, str]]
    errors: List[str]


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def create_chunk_extraction_graph(llm_client: LLMClient):
    """Create a LangGraph for extracting and verifying QA pairs from a text chunk.

    Stage 1 (extract_pairs): Self-Foveate + CoVe prompting for grounded extraction.
    Stage 2 (critique_pairs): LLM-as-a-Judge filtering for faithfulness.

    Args:
        llm_client: The LLM client for text generation.

    Returns:
        Compiled LangGraph workflow.
    """

    def extract_pairs(state: ChunkExtractionState) -> dict:
        """Extract factual instruction→response pairs from a chunk."""
        chunk_text = state["chunk_text"]

        if not chunk_text or not chunk_text.strip():
            return {"extracted_pairs": [], "errors": ["Empty chunk text."]}

        prompt = (
            "You are an expert knowledge extractor. Given the following text chunk, "
            "extract all factual instruction→response pairs.\n\n"
            "## Method\n"
            "1. **Self-Foveate**: First, identify the 3-5 most important facts, "
            "concepts, or procedures in this text.\n"
            "2. **Generate pairs**: For each, write an 'instruction' (a question or "
            "task prompt) and a 'response' (the answer, strictly from the text).\n"
            "3. **Chain-of-Verification**: For each pair, mentally verify that the "
            "response can be traced back to the source text. If not, discard it.\n\n"
            "## Rules\n"
            "- Every response MUST be grounded in the source text — no external knowledge.\n"
            "- Do NOT hallucinate or extrapolate beyond what the text says.\n"
            "- Each instruction should be self-contained (understandable without the source).\n"
            "- Return a JSON array of objects: [{\"instruction\": \"...\", \"response\": \"...\"}]\n"
            "- Return ONLY the JSON array. No explanation.\n\n"
            f"## Source Text\n```\n{chunk_text[:4000]}\n```\n"
        )

        response = llm_client.generate_completion(
            prompt,
            system_prompt=(
                "You are a precise knowledge extraction agent. "
                "Output a JSON array of instruction-response pairs only."
            ),
        )

        pairs = _parse_json_array_from_response(response)
        if pairs:
            # Filter to only keep well-formed pairs
            valid_pairs = []
            for pair in pairs:
                instr = pair.get("instruction", "").strip()
                resp = pair.get("response", "").strip()
                if instr and resp:
                    valid_pairs.append({"instruction": instr, "response": resp})
            return {"extracted_pairs": valid_pairs, "errors": []}
        else:
            return {
                "extracted_pairs": [],
                "errors": [f"Failed to parse extraction output: {(response or '')[:200]}"],
            }

    def critique_pairs(state: ChunkExtractionState) -> dict:
        """LLM-as-a-Judge: verify each pair is faithfully grounded in the source."""
        chunk_text = state["chunk_text"]
        pairs = state.get("extracted_pairs", [])

        if not pairs:
            return {"verified_pairs": []}

        # Batch critique: send all pairs at once for efficiency
        pairs_json = json.dumps(pairs, indent=2)

        prompt = (
            "You are a strict grounding verifier. For each instruction-response pair below, "
            "determine if the response is **faithfully and completely supported** by the source text.\n\n"
            "## Source Text\n```\n" + chunk_text[:3000] + "\n```\n\n"
            "## Pairs to Verify\n```json\n" + pairs_json + "\n```\n\n"
            "## Task\n"
            "Return a JSON array of verdict objects: "
            "[{\"index\": 0, \"grounded\": true/false, \"reason\": \"brief explanation\"}]\n"
            "- Set 'grounded' to true ONLY if the response is directly supported by the source.\n"
            "- Set 'grounded' to false if ANY part of the response is hallucinated or extrapolated.\n"
            "- Return ONLY the JSON array.\n"
        )

        response = llm_client.generate_completion(
            prompt,
            system_prompt="You are a strict faithfulness judge. Output a JSON array of verdicts only.",
        )

        verdicts = _parse_json_array_from_response(response)

        if verdicts:
            verified = []
            for verdict in verdicts:
                idx = verdict.get("index")
                is_grounded = verdict.get("grounded", False)
                if is_grounded and isinstance(idx, int) and 0 <= idx < len(pairs):
                    verified.append(pairs[idx])
            return {"verified_pairs": verified}
        else:
            # If critique parsing fails, conservatively keep all pairs
            # (extraction already applied CoVe self-verification)
            logger.warning("Critique parsing failed; keeping all extracted pairs as fallback.")
            return {"verified_pairs": pairs}

    # Build the graph
    workflow = StateGraph(ChunkExtractionState)

    workflow.add_node("extract", extract_pairs)
    workflow.add_node("critique", critique_pairs)

    workflow.set_entry_point("extract")
    workflow.add_edge("extract", "critique")
    workflow.add_edge("critique", END)

    return workflow.compile()
