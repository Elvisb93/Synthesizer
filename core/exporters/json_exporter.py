"""JSON export for generated row data."""
import json
import logging
from typing import List

from ..models import RowData

logger = logging.getLogger(__name__)


def export_json(
    filepath: str,
    generated_rows: List[RowData],
    log_fn=None,
) -> None:
    """Export generated rows to a JSON file."""
    if not generated_rows:
        return

    try:
        data = [row.data for row in generated_rows]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        if log_fn:
            log_fn(f"Exported to {filepath}")
    except Exception as e:
        if log_fn:
            log_fn(f"JSON Export failed: {e}")
