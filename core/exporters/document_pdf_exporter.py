from fpdf import FPDF
from fpdf.errors import FPDFException
from typing import Dict, List
import os

from .markdown_pdf_renderer import MarkdownPDFRenderer


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

    def export(
        self,
        *,
        title: str,
        outline: dict,
        text: str,
        output_path: str,
        chunks: List[Dict[str, object]] | None = None,
        charts: List[Dict[str, object]] | None = None,
    ) -> None:
        renderer = MarkdownPDFRenderer(
            sanitize_text=self._sanitize_text,
            safe_multi_cell=self._safe_multi_cell,
        )
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.set_font("helvetica", "B", 18)
        pdf.multi_cell(0, 10, self._sanitize_text(title or "Generated Document"), align="C")
        pdf.ln(5)

        renderer.render(pdf, text or "")

        if charts:
            self._render_charts(pdf, charts)

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

    def _render_charts(self, pdf: FPDF, charts: List[Dict[str, object]]) -> None:
        chart_items = [c for c in charts if isinstance(c, dict) and c.get("image_path")]
        if not chart_items:
            return

        pdf.add_page()
        pdf.set_font("helvetica", "B", 16)
        pdf.cell(0, 10, "Charts & Visuals", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        for idx, chart in enumerate(chart_items, start=1):
            image_path = str(chart.get("image_path", "")).strip()
            if not image_path or not os.path.exists(image_path):
                continue

            title = self._sanitize_text(str(chart.get("title", f"Chart {idx}")))
            caption = self._sanitize_text(str(chart.get("caption", "")).strip())
            sources = chart.get("evidence_sources") or []
            if isinstance(sources, list):
                source_text = ", ".join(
                    os.path.basename(str(s).strip()) or str(s).strip()
                    for s in sources
                    if str(s).strip()
                )
            else:
                source_text = ""

            if pdf.get_y() > 240:
                pdf.add_page()

            pdf.set_font("helvetica", "B", 13)
            self._safe_multi_cell(pdf, 0, 7, f"{idx}. {title}")
            pdf.set_font("helvetica", "", 10)
            if caption:
                self._safe_multi_cell(pdf, 0, 5, caption)
            if source_text:
                self._safe_multi_cell(pdf, 0, 5, f"Sources: {source_text}")
            pdf.ln(1)

            try:
                image_w = pdf.w - pdf.l_margin - pdf.r_margin
                pdf.image(image_path, w=image_w)
                pdf.ln(4)
            except Exception:
                pdf.set_font("helvetica", "I", 10)
                self._safe_multi_cell(pdf, 0, 5, f"[Chart image failed to render: {image_path}]")
                pdf.ln(2)

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
