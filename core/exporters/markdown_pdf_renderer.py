import re
from typing import Callable, List

from fpdf import FPDF


class MarkdownPDFRenderer:
    def __init__(
        self,
        *,
        sanitize_text: Callable[[str], str],
        safe_multi_cell: Callable[..., None],
    ) -> None:
        self._sanitize_text = sanitize_text
        self._safe_multi_cell = safe_multi_cell

    def render(self, pdf: FPDF, text: str) -> None:
        lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
        paragraph_buffer: List[str] = []
        code_buffer: List[str] = []
        in_code_block = False

        for raw_line in lines:
            line = raw_line
            stripped = line.strip()

            if in_code_block:
                if stripped.startswith("```"):
                    self._flush_code_block(pdf, code_buffer)
                    code_buffer = []
                    in_code_block = False
                else:
                    code_buffer.append(line.rstrip("\n"))
                continue

            if stripped.startswith("```"):
                self._flush_paragraph(pdf, paragraph_buffer)
                paragraph_buffer = []
                in_code_block = True
                continue

            if not stripped:
                self._flush_paragraph(pdf, paragraph_buffer)
                paragraph_buffer = []
                pdf.ln(2)
                continue

            heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
            if heading_match:
                self._flush_paragraph(pdf, paragraph_buffer)
                paragraph_buffer = []
                level = len(heading_match.group(1))
                self._render_heading(pdf, level, heading_match.group(2).strip())
                continue

            if re.match(r"^(\*{3,}|-{3,}|_{3,})\s*$", stripped):
                self._flush_paragraph(pdf, paragraph_buffer)
                paragraph_buffer = []
                self._render_horizontal_rule(pdf)
                continue

            blockquote_match = re.match(r"^>\s?(.*)$", stripped)
            if blockquote_match:
                self._flush_paragraph(pdf, paragraph_buffer)
                paragraph_buffer = []
                self._render_blockquote(pdf, blockquote_match.group(1).strip())
                continue

            list_match = re.match(r"^(\s*)([-*+]|\d+[.)])\s+(.+)$", line)
            if list_match:
                self._flush_paragraph(pdf, paragraph_buffer)
                paragraph_buffer = []
                indent_spaces = len(list_match.group(1))
                marker = list_match.group(2)
                body = list_match.group(3).strip()
                self._render_list_item(pdf, indent_spaces, marker, body)
                continue

            if "|" in stripped and stripped.count("|") >= 2:
                self._flush_paragraph(pdf, paragraph_buffer)
                paragraph_buffer = []
                self._render_table_like_line(pdf, stripped)
                continue

            paragraph_buffer.append(stripped)

        self._flush_paragraph(pdf, paragraph_buffer)
        if code_buffer:
            self._flush_code_block(pdf, code_buffer)

    def _flush_paragraph(self, pdf: FPDF, paragraph_buffer: List[str]) -> None:
        if not paragraph_buffer:
            return

        text = " ".join(part.strip() for part in paragraph_buffer if part.strip()).strip()
        if not text:
            return

        if len(text.split()) <= 8 and text == text.title():
            self._render_heading(pdf, 3, text)
            return

        pdf.set_x(pdf.l_margin)
        pdf.set_font("helvetica", "", 12)
        self._safe_multi_cell(pdf, 0, 7, self._sanitize_text(text))
        pdf.ln(2)

    def _render_heading(self, pdf: FPDF, level: int, text: str) -> None:
        size_by_level = {1: 17, 2: 15, 3: 13, 4: 12, 5: 11, 6: 11}
        top_space_by_level = {1: 4, 2: 3, 3: 2, 4: 2, 5: 1, 6: 1}
        size = size_by_level.get(level, 12)
        top_space = top_space_by_level.get(level, 2)

        pdf.ln(top_space)
        pdf.set_x(pdf.l_margin)
        pdf.set_font("helvetica", "B", size)
        self._safe_multi_cell(pdf, 0, 8 if level <= 3 else 7, self._sanitize_text(text))
        pdf.ln(1 if level <= 3 else 0)

    def _render_horizontal_rule(self, pdf: FPDF) -> None:
        y = pdf.get_y() + 2
        x1 = pdf.l_margin
        x2 = pdf.w - pdf.r_margin
        pdf.line(x1, y, x2, y)
        pdf.ln(5)

    def _render_blockquote(self, pdf: FPDF, text: str) -> None:
        indent = 8
        available_width = pdf.w - pdf.l_margin - pdf.r_margin - indent
        pdf.set_x(pdf.l_margin + indent)
        pdf.set_font("helvetica", "I", 11)
        self._safe_multi_cell(pdf, available_width, 6, self._sanitize_text(text))
        pdf.ln(1)

    def _render_list_item(self, pdf: FPDF, indent_spaces: int, marker: str, body: str) -> None:
        indent = min(24, max(0, indent_spaces // 2) * 4)
        left_offset = 6 + indent
        marker_text = "-" if re.match(r"^[-*+]$", marker) else marker
        content = f"{marker_text} {body}".strip()

        available_width = pdf.w - pdf.l_margin - pdf.r_margin - left_offset
        pdf.set_x(pdf.l_margin + left_offset)
        pdf.set_font("helvetica", "", 11)
        self._safe_multi_cell(pdf, available_width, 6, self._sanitize_text(content))
        pdf.ln(0.5)

    def _render_table_like_line(self, pdf: FPDF, line: str) -> None:
        # Basic markdown-table fallback: keep row readable even without full table layout.
        if re.match(r"^\|?[\s:\-]+\|[\s:\-|]*$", line):
            return
        cells = [c.strip() for c in line.strip("|").split("|")]
        row = " | ".join(cells)
        pdf.set_x(pdf.l_margin)
        pdf.set_font("courier", "", 10)
        self._safe_multi_cell(pdf, 0, 5.5, self._sanitize_text(row))
        pdf.ln(0.5)

    def _flush_code_block(self, pdf: FPDF, code_lines: List[str]) -> None:
        if not code_lines:
            return
        pdf.set_x(pdf.l_margin)
        pdf.set_font("courier", "", 10)
        for line in code_lines:
            safe_line = self._sanitize_text(line.rstrip())
            self._safe_multi_cell(pdf, 0, 5.5, safe_line if safe_line else " ")
        pdf.ln(2)
