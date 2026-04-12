"""
JSON Template Parser — Utilities for loading, parsing, and populating JSON templates.

Handles dot-notation path resolution, Pydantic schema inference from array items,
and item injection into target arrays within a master JSON structure.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, create_model

logger = logging.getLogger(__name__)


def load_template(filepath: str) -> dict:
    """Load and validate a JSON template file.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Parsed JSON as a dict.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON or is not a dict at root.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Template file not found: {filepath}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in template file: {e}")

    if not isinstance(data, dict):
        raise ValueError("Template root must be a JSON object (dict), not an array or primitive.")

    return data


def resolve_target_array(template: dict, dot_path: str) -> list:
    """Navigate a dict using dot-notation and return the target array.

    Args:
        template: The parsed JSON template dict.
        dot_path: Dot-separated path to the target array (e.g. 'data.messages', 'nodes').

    Returns:
        Reference to the target list within the template.

    Raises:
        ValueError: If the path is empty, doesn't exist, or doesn't point to a list.
    """
    if not dot_path or not dot_path.strip():
        raise ValueError("Target path cannot be empty.")

    keys = dot_path.strip().split(".")
    current: Any = template

    for i, key in enumerate(keys):
        if not isinstance(current, dict):
            traversed = ".".join(keys[:i])
            raise ValueError(
                f"Cannot traverse into '{key}' — '{traversed}' is not a dict "
                f"(got {type(current).__name__})."
            )
        if key not in current:
            traversed = ".".join(keys[: i + 1])
            raise ValueError(f"Key path '{traversed}' does not exist in the template.")
        current = current[key]

    if not isinstance(current, list):
        raise ValueError(
            f"Target path '{dot_path}' does not point to a list "
            f"(got {type(current).__name__}). The target must be a JSON array."
        )

    return current


def _python_type_for_value(value: Any) -> type:
    """Map a JSON value to a Python/Pydantic-compatible type annotation."""
    if isinstance(value, str):
        return str
    elif isinstance(value, bool):
        # bool before int because bool is a subclass of int in Python
        return bool
    elif isinstance(value, int):
        return int
    elif isinstance(value, float):
        return float
    elif isinstance(value, list):
        return List[Any]
    elif isinstance(value, dict):
        return Dict[str, Any]
    else:
        return Any


def infer_item_schema(target_array: list) -> Optional[Type[BaseModel]]:
    """Infer a Pydantic BaseModel from the first item in a target array.

    Introspects the keys and value types of the first array element to dynamically
    build a Pydantic model class using ``create_model()``.

    Args:
        target_array: The resolved target array from the template.

    Returns:
        A dynamically created Pydantic BaseModel class, or None if the array is empty
        or the first item is not a dict.
    """
    if not target_array:
        return None

    sample = target_array[0]
    if not isinstance(sample, dict):
        logger.warning(
            f"First array item is {type(sample).__name__}, not a dict. "
            "Cannot infer schema — falling back to LLM-inferred schema."
        )
        return None

    # Build field definitions: (type, default_value)
    field_definitions = {}
    for key, value in sample.items():
        py_type = _python_type_for_value(value)
        # Use ... (Ellipsis) for required fields, or provide a default
        # We'll make all fields optional with None default for flexibility
        field_definitions[key] = (Optional[py_type], None)

    model = create_model("InferredItemSchema", **field_definitions)
    logger.info(f"Inferred schema with {len(field_definitions)} fields: {list(field_definitions.keys())}")
    return model


def inject_item(template: dict, dot_path: str, item: dict) -> dict:
    """Append a validated item to the target array in the template.

    Modifies the template in-place and returns it.

    Args:
        template: The master JSON template dict.
        dot_path: Dot-notation path to the target array.
        item: The validated JSON object to append.

    Returns:
        The modified template dict (same reference, mutated in-place).
    """
    target_array = resolve_target_array(template, dot_path)
    target_array.append(item)
    return template


def clear_target_array(template: dict, dot_path: str) -> dict:
    """Remove all existing items from the target array.

    Args:
        template: The master JSON template dict.
        dot_path: Dot-notation path to the target array.

    Returns:
        The modified template dict.
    """
    target_array = resolve_target_array(template, dot_path)
    target_array.clear()
    return template


def export_template(template: dict, filepath: str) -> None:
    """Write the populated template to disk as formatted JSON.

    Args:
        template: The final JSON template dict.
        filepath: Output file path.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)

    logger.info(f"JSON template exported to {filepath}")
