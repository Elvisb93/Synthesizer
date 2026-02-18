from typing import Dict, List


class DocumentDocxExporter:
    def export(self, *, title: str, outline: Dict[str, object], text: str, output_path: str, chunks: List[Dict[str, object]] | None = None) -> None:
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("python-docx is required for DOCX export") from exc

        doc = Document()
        doc.add_heading(title or "Generated Document", level=0)

        sections = self._split_sections(text)
        if sections:
            for section_title, body in sections:
                doc.add_heading(section_title, level=1)
                for para in body.split("\n\n"):
                    para = para.strip()
                    if para:
                        doc.add_paragraph(para)
        else:
            for para in (text or "").split("\n\n"):
                para = para.strip()
                if para:
                    doc.add_paragraph(para)

        refs = self._format_references(chunks or [])
        if refs:
            doc.add_page_break()
            doc.add_heading("References", level=1)
            for line in refs:
                if not line:
                    doc.add_paragraph("")
                    continue
                if line.endswith(":"):
                    doc.add_heading(line[:-1], level=2)
                    continue
                doc.add_paragraph(line)

        doc.save(output_path)

    @staticmethod
    def _split_sections(text: str) -> List[tuple[str, str]]:
        lines = (text or "").splitlines()
        sections: List[tuple[str, str]] = []
        current_title = None
        current_body: List[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                current_body.append("")
                continue

            if current_title is None:
                current_title = stripped
                continue

            if stripped and stripped == stripped.title() and len(stripped.split()) <= 8 and current_body:
                sections.append((current_title, "\n".join(current_body).strip()))
                current_title = stripped
                current_body = []
                continue

            current_body.append(line)

        if current_title:
            sections.append((current_title, "\n".join(current_body).strip()))
        return [(t, b) for t, b in sections if b]

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
                ref = f"- {source} (page {page}, score {score:.3f})"
                if ref not in grouped[section]:
                    grouped[section].append(ref)

        lines: List[str] = []
        for section, refs in grouped.items():
            lines.append(f"{section}:")
            lines.extend(refs[:10])
            lines.append("")
        return lines
