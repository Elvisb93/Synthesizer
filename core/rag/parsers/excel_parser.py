import os
from typing import List

from core.rag.interfaces import DocumentParser
from core.rag.models import ParsedDocument


class ExcelParser(DocumentParser):
    def __init__(self, max_rows_per_sheet: int = 300):
        self.max_rows_per_sheet = max(20, int(max_rows_per_sheet))

    def parse(self, path: str) -> ParsedDocument:
        try:
            import pandas as pd
        except Exception as exc:
            raise RuntimeError("pandas is required for Excel ingestion") from exc

        xls = pd.ExcelFile(path)
        pages: List[str] = []
        page_meta = []
        total_rows = 0

        for idx, sheet in enumerate(xls.sheet_names, start=1):
            df = pd.read_excel(path, sheet_name=sheet)
            total_rows += len(df)
            head = df.head(self.max_rows_per_sheet)
            sheet_text = (
                f"Workbook: {os.path.basename(path)}\n"
                f"Sheet: {sheet}\n"
                f"Rows: {len(df)} | Columns: {len(df.columns)}\n"
                f"Column names: {', '.join(map(str, df.columns.tolist()))}\n\n"
                f"{head.to_string(index=False)}"
            )
            pages.append(sheet_text)
            page_meta.append({"page": idx, "sheet_name": sheet, "sheet_rows": len(df)})

        return ParsedDocument(
            source=path,
            source_type="excel",
            pages=pages,
            page_metadata=page_meta,
            metadata={"source_type": "excel", "sheet_count": len(xls.sheet_names), "rows_total": total_rows},
        )
