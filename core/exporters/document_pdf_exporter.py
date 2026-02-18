from fpdf import FPDF
from typing import Dict, List


class DocumentPDFExporter:
    def export(self, *, title: str, outline: dict, text: str, output_path: str, chunks: List[Dict[str, object]] | None = None) -> None:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.set_font("helvetica", "B", 18)
        pdf.multi_cell(0, 10, (title or "Generated Document").encode("latin-1", "replace").decode("latin-1"), align="C")
        pdf.ln(5)

        for para in (text or "").split("\n\n"):
            cleaned = para.strip()
            if not cleaned:
                continue
            if len(cleaned.split()) <= 8 and cleaned == cleaned.title():
                pdf.set_font("helvetica", "B", 14)
                pdf.multi_cell(0, 8, cleaned.encode("latin-1", "replace").decode("latin-1"))
                pdf.ln(1)
                pdf.set_font("helvetica", "", 12)
                continue

            pdf.set_font("helvetica", "", 12)
            pdf.multi_cell(0, 7, cleaned.encode("latin-1", "replace").decode("latin-1"))
            pdf.ln(2)

        refs = self._format_references(chunks or [])
        if refs:
            pdf.add_page()
            pdf.set_font("helvetica", "B", 16)
            pdf.cell(0, 10, "References", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
            pdf.set_font("helvetica", "", 11)
            for line in refs:
                pdf.multi_cell(0, 6, line.encode("latin-1", "replace").decode("latin-1"))

        pdf.output(output_path)

    @staticmethod
    def _format_references(chunks: List[Dict[str, object]]) -> List[str]:
        grouped: Dict[str, List[str]] = {}
        for chunk in chunks:
            section = str(chunk.get("section_title", "Section"))
            citations = chunk.get("citations") or []
            if not isinstance(citations, list):
                continue
            for c in citations:
                if not isinstance(c, dict):
                    continue
                source = str(c.get("source", "unknown"))
                page = str(c.get("page", "?"))
                score = float(c.get("score", 0.0))
                grouped.setdefault(section, [])
                entry = f"{source} (page {page}, score {score:.3f})"
                if entry not in grouped[section]:
                    grouped[section].append(entry)

        lines: List[str] = []
        for section, refs in grouped.items():
            lines.append(f"{section}:")
            for ref in refs[:10]:
                lines.append(f"- {ref}")
            lines.append("")
        return lines
