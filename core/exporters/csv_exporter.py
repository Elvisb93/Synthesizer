"""CSV export for generated row data."""
import csv
import logging
from typing import List

from ..models import ColumnDefinition, RowData

logger = logging.getLogger(__name__)


def export_csv(
    filepath: str,
    generated_rows: List[RowData],
    columns: List[ColumnDefinition],
    log_fn=None,
) -> None:
    """Export generated rows to a CSV file."""
    if not generated_rows:
        return

    fieldnames = [col.name for col in columns]

    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in generated_rows:
                writer.writerow(row.data)
        if log_fn:
            log_fn(f"Exported to {filepath}")
    except Exception as e:
        if log_fn:
            log_fn(f"Export failed: {e}")
