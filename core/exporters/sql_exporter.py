"""SQL INSERT export for generated row data."""
import logging
from typing import List

from ..models import RowData

logger = logging.getLogger(__name__)


def export_sql(
    filepath: str,
    generated_rows: List[RowData],
    table_name: str = "synthetic_data",
    log_fn=None,
) -> None:
    """Export generated rows as SQL INSERT statements."""
    if not generated_rows:
        return

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for row in generated_rows:
                cols = []
                vals = []
                for k, v in row.data.items():
                    cols.append(k)
                    if isinstance(v, str):
                        safe_v = v.replace("'", "''")
                        vals.append(f"'{safe_v}'")
                    else:
                        vals.append(str(v))

                sql = f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES ({', '.join(vals)});\n"
                f.write(sql)
        if log_fn:
            log_fn(f"Exported to {filepath}")
    except Exception as e:
        if log_fn:
            log_fn(f"SQL Export failed: {e}")
