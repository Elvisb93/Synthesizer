from fpdf import FPDF
from fpdf.errors import FPDFException
from typing import Dict, List


class DocumentPDFExporter:
    @staticmethod
    def _sanitize_text(text: str) -> str:
        """
        Sanitizes text to be compatible with latin-1 encoding used by FPDF standard fonts.
        Replaces common incompatible characters with ASCII approximations.
        """
        if not isinstance(text, str):
            text = str(text)
            
        replacements = {
            # Quotes
            '\u2018': "'",  # Left single quote
            '\u2019': "'",  # Right single quote
            '\u201C': '"',  # Left double quote
            '\u201D': '"',  # Right double quote
            '\u2032': "'",  # Prime (foot mark)
            '\u2033': '"',  # Double prime (inch mark)
            # Dashes and hyphens
            '\u2011': '-',  # Non-breaking hyphen  <-- KEY FIX
            '\u2012': '-',  # Figure dash
            '\u2013': '-',  # En dash
            '\u2014': '--', # Em dash
            '\u2015': '--', # Horizontal bar
            '\u00ad': '-',  # Soft hyphen
            # Bullets and others
            '\u2022': '*',  # Bullet
            '\u2023': '>',  # Triangular bullet
            '\u2026': '...', # Ellipsis
            '\u20ac': 'EUR', # Euro
            '\u2122': '(TM)', # Trademark
            '\u00a9': '(c)', # Copyright
            '\u00ae': '(R)', # Registered
            '\u00a0': ' ', # Non-breaking space
            '\u2009': ' ', # Thin space
            '\u200b': '',  # Zero-width space
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
            
        # Final fallback for any other non-latin-1 characters
        return text.encode('latin-1', 'replace').decode('latin-1')

    def _safe_multi_cell(self, pdf: FPDF, w: float, h: float, txt: str, **kwargs):
        """
        Wrapper for multi_cell that catches layout errors (like 'Not enough horizontal space')
        and attempts to recover by forcing a line break and resetting indentation.
        """
        # 1. Pre-emptive check: If closer to right margin than left margin, likely unsafe
        # 1mm tolerance
        effective_w = pdf.w - pdf.r_margin - pdf.get_x()
        if effective_w < 1: 
            pdf.ln(h)
            pdf.set_x(pdf.l_margin)

        try:
            pdf.multi_cell(w, h, txt, **kwargs)
        except FPDFException as e:
            # Catch "Not enough horizontal space to render a single character"
            # Recovery: New line, reset X, try again
            pdf.ln(h)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(w, h, txt, **kwargs)

    def export(self, *, title: str, outline: dict, text: str, output_path: str, chunks: List[Dict[str, object]] | None = None) -> None:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.set_font("helvetica", "B", 18)
        pdf.multi_cell(0, 10, self._sanitize_text(title or "Generated Document"), align="C")
        pdf.ln(5)

        for para in (text or "").split("\n\n"):
            # Ensure we start paragraphs at the left margin to avoid drift
            pdf.set_x(pdf.l_margin)
            
            cleaned = para.strip()
            if not cleaned:
                continue
            if len(cleaned.split()) <= 8 and cleaned == cleaned.title():
                pdf.set_font("helvetica", "B", 14)
                self._safe_multi_cell(pdf, 0, 8, self._sanitize_text(cleaned))
                pdf.ln(1)
                pdf.set_font("helvetica", "", 12)
                continue

            pdf.set_font("helvetica", "", 12)
            self._safe_multi_cell(pdf, 0, 7, self._sanitize_text(cleaned))
            pdf.ln(2)

        refs = self._format_references(chunks or [])
        if refs:
            pdf.add_page()
            pdf.set_font("helvetica", "B", 16)
            pdf.cell(0, 10, "References", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
            pdf.ln(2)
            pdf.set_font("helvetica", "", 11)
            for line in refs:
                # Use safe multi_cell here too just in case
                self._safe_multi_cell(pdf, 0, 6, self._sanitize_text(line))

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
