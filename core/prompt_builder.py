"""
Prompt construction and dependency resolution for column generation.

Handles:
- Extracting @[col_name] dependency references from prompt instructions
- Topological sorting of columns based on dependencies (Kahn's algorithm)
- Building final prompts with interpolated dependency values
"""
import re
import logging
from typing import List, Dict, Any, Set, Optional, Callable

from .models import ColumnDefinition

logger = logging.getLogger(__name__)


def get_dependencies(prompt: str) -> Set[str]:
    """Extract valid column references from prompt like @[col_name]."""
    matches = re.findall(r'@\[(.*?)\]', prompt)
    return set(matches)


def get_execution_order(
    columns: List[ColumnDefinition],
    log_fn: Optional[Callable[[str], None]] = None,
) -> List[ColumnDefinition]:
    """Topological sort of columns based on prompt dependencies."""
    col_map = {col.name: col for col in columns}
    adj_list: Dict[str, Set[str]] = {col.name: set() for col in columns}
    in_degree: Dict[str, int] = {col.name: 0 for col in columns}

    # Build Graph
    for col in columns:
        deps = get_dependencies(col.prompt_instruction)
        for dep_name in deps:
            if dep_name in col_map:
                adj_list[dep_name].add(col.name)
                in_degree[col.name] += 1
            elif log_fn:
                log_fn(f"Warning: Column '{col.name}' references unknown column '{dep_name}'. Ignoring.")

    # Kahn's Algorithm
    queue = [name for name, deg in in_degree.items() if deg == 0]
    sorted_cols: List[ColumnDefinition] = []

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


def construct_prompt(col: ColumnDefinition, row_data: Dict[str, Any]) -> str:
    """Create a prompt for a single column value, interpolating dependencies."""

    # Interpolate dependencies
    instruction = col.prompt_instruction
    deps = get_dependencies(instruction)
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
