import json
import math
import os
import re
import tempfile
from typing import Any, Dict, List, Optional


class DocumentChartGenerator:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.last_error: Optional[str] = None

    def generate(
        self,
        *,
        user_prompt: str,
        document_title: str,
        retrieved_context: str,
        available_sources: List[str],
        max_charts: int = 3,
        include_flowchart: bool = True,
    ) -> List[Dict[str, Any]]:
        self.last_error = None
        context = (retrieved_context or "").strip()
        if not context:
            return []

        limited_sources = [s for s in available_sources if s][:20]
        source_lines = "\n".join(f"- {s}" for s in limited_sources) if limited_sources else "- (none)"
        max_items = max(1, min(4, int(max_charts)))
        prompt = (
            "Create grounded chart specs from context.\n"
            "Return ONLY JSON in this schema:\n"
            "{\"charts\": ["
            "{\"title\": str, \"chart_type\": \"bar|line|pie|scatter|flow\", "
            "\"x_label\": str, \"y_label\": str, "
            "\"labels\": [str], \"values\": [number], "
            "\"x_values\": [number], \"y_values\": [number], "
            "\"point_labels\": [str], "
            "\"steps\": [str], "
            "\"caption\": str, \"evidence_sources\": [str]}"
            "]}\n\n"
            "Rules:\n"
            f"- Return at most {max_items} charts.\n"
            "- Prefer at least one trend/comparison chart when numeric context is available.\n"
            "- Use only values explicitly present in context.\n"
            "- Use only evidence_sources from the available source list.\n"
            "- bar/line/pie: labels and values must have same length (2..12).\n"
            "- scatter: x_values and y_values must have same length (3..20).\n"
            "- flow: provide 3..10 concise steps in process order.\n"
            "- If data is insufficient, return {\"charts\": []}.\n\n"
            f"User request:\n{user_prompt}\n\n"
            f"Document title:\n{document_title}\n\n"
            f"Available sources:\n{source_lines}\n\n"
            f"Retrieved context:\n{context}\n"
        )
        raw = self.llm_client.generate_completion(
            prompt,
            system_prompt="You are a strict data-to-chart JSON planner. Output valid JSON only.",
        )
        parsed = self._parse_json(raw)
        if not parsed:
            parsed = self._attempt_json_repair(raw)
        raw_specs: List[Any] = []
        if parsed and isinstance(parsed.get("charts"), list):
            raw_specs = list(parsed.get("charts") or [])

        validated: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_specs):
            if len(validated) >= max_items:
                break
            spec = self._normalize_spec(item)
            if not spec:
                continue
            if not self._is_grounded(spec, context, limited_sources):
                continue
            validated.append(spec)

        if not validated:
            fallback = self._fallback_tabular_metric_chart(context, limited_sources)
            if not fallback:
                fallback = self._fallback_source_relevance_chart(context, limited_sources)
            if fallback:
                validated = [fallback]
            elif include_flowchart:
                flow = self._fallback_process_flow(user_prompt, limited_sources)
                if flow:
                    validated = [flow]
                else:
                    return []
            else:
                return []

        if include_flowchart and len(validated) < max_items:
            if not any(v.get("chart_type") == "flow" for v in validated):
                flow = self._fallback_process_flow(user_prompt, limited_sources)
                if flow:
                    validated.append(flow)

        out_dir = tempfile.mkdtemp(prefix="synth_doc_charts_")
        artifacts: List[Dict[str, Any]] = []
        for idx, spec in enumerate(validated):
            if len(artifacts) >= max_items:
                break
            path = self._render_chart(spec, idx, out_dir)
            if not path:
                continue
            artifacts.append(
                {
                    "title": spec["title"],
                    "chart_type": spec["chart_type"],
                    "caption": spec.get("caption", ""),
                    "x_label": spec.get("x_label", ""),
                    "y_label": spec.get("y_label", ""),
                    "labels": spec.get("labels", []),
                    "values": spec.get("values", []),
                    "x_values": spec.get("x_values", []),
                    "y_values": spec.get("y_values", []),
                    "steps": spec.get("steps", []),
                    "evidence_sources": spec["evidence_sources"],
                    "image_path": path,
                }
            )
        return artifacts

    @staticmethod
    def _parse_json(raw: Optional[str]) -> Optional[Dict[str, Any]]:
        if not raw:
            return None
        cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(cleaned[start : end + 1])
                    return parsed if isinstance(parsed, dict) else None
                except Exception:
                    return None
        return None

    def _attempt_json_repair(self, raw: Optional[str]) -> Optional[Dict[str, Any]]:
        text = (raw or "").strip()
        if not text:
            return None
        repair_prompt = (
            "Repair the following model output into valid JSON only.\n"
            "Schema:\n"
            "{\"charts\": [{\"title\": str, \"chart_type\": \"bar|line|pie|scatter|flow\", "
            "\"x_label\": str, \"y_label\": str, \"labels\": [str], \"values\": [number], "
            "\"x_values\": [number], \"y_values\": [number], \"point_labels\": [str], "
            "\"steps\": [str], \"caption\": str, \"evidence_sources\": [str]}]}\n\n"
            "Output:\n"
            f"{text}\n"
        )
        repaired = self.llm_client.generate_completion(
            repair_prompt,
            system_prompt="You only output valid JSON.",
        )
        return self._parse_json(repaired)

    @staticmethod
    def _normalize_spec(item: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None
        chart_type = str(item.get("chart_type", "")).strip().lower()
        if chart_type not in {"bar", "line", "pie", "scatter", "flow"}:
            return None

        safe_labels: List[str] = []
        safe_values: List[float] = []
        safe_x_values: List[float] = []
        safe_y_values: List[float] = []
        safe_steps: List[str] = []

        if chart_type in {"bar", "line", "pie"}:
            labels = item.get("labels")
            values = item.get("values")
            if not isinstance(labels, list) or not isinstance(values, list):
                return None
            if len(labels) != len(values) or len(labels) < 2 or len(labels) > 12:
                return None
            for label, val in zip(labels, values):
                label_txt = str(label).strip()
                if not label_txt:
                    return None
                try:
                    num = float(val)
                except Exception:
                    return None
                if not math.isfinite(num):
                    return None
                safe_labels.append(label_txt[:80])
                safe_values.append(num)
        elif chart_type == "scatter":
            x_values = item.get("x_values")
            y_values = item.get("y_values")
            if not isinstance(x_values, list) or not isinstance(y_values, list):
                return None
            if len(x_values) != len(y_values) or len(x_values) < 3 or len(x_values) > 20:
                return None
            for xv, yv in zip(x_values, y_values):
                try:
                    x_num = float(xv)
                    y_num = float(yv)
                except Exception:
                    return None
                if not (math.isfinite(x_num) and math.isfinite(y_num)):
                    return None
                safe_x_values.append(x_num)
                safe_y_values.append(y_num)
        else:
            steps = item.get("steps")
            if not isinstance(steps, list):
                return None
            safe_steps = [str(s).strip()[:80] for s in steps if str(s).strip()]
            if len(safe_steps) < 3 or len(safe_steps) > 10:
                return None

        sources = item.get("evidence_sources") or []
        if not isinstance(sources, list):
            sources = []
        safe_sources = [str(s).strip() for s in sources if str(s).strip()]

        return {
            "title": str(item.get("title", "Chart")).strip()[:120] or "Chart",
            "chart_type": chart_type,
            "x_label": str(item.get("x_label", "")).strip()[:80],
            "y_label": str(item.get("y_label", "")).strip()[:80],
            "labels": safe_labels,
            "values": safe_values,
            "x_values": safe_x_values,
            "y_values": safe_y_values,
            "steps": safe_steps,
            "caption": str(item.get("caption", "")).strip()[:280],
            "evidence_sources": safe_sources,
        }

    @staticmethod
    def _number_tokens(num: float) -> List[str]:
        if float(num).is_integer():
            n = int(num)
            return [str(n), f"{n}.0", f"{n:.1f}", f"{n:.2f}"]
        return [f"{num}", f"{num:.1f}", f"{num:.2f}", f"{num:.3f}"]

    def _is_grounded(self, spec: Dict[str, Any], context: str, available_sources: List[str]) -> bool:
        lowered = (context or "").lower()
        chart_type = spec.get("chart_type")
        if chart_type in {"bar", "line", "pie"}:
            for value in spec["values"]:
                tokens = self._number_tokens(float(value))
                if not any(tok.lower() in lowered for tok in tokens):
                    return False
        elif chart_type == "scatter":
            for value in spec.get("x_values", []) + spec.get("y_values", []):
                tokens = self._number_tokens(float(value))
                if not any(tok.lower() in lowered for tok in tokens):
                    return False

        if available_sources:
            allowed = set(available_sources)
            kept = [s for s in spec["evidence_sources"] if s in allowed]
            if not kept:
                for src in available_sources:
                    if src in context:
                        kept.append(src)
                        break
            if not kept:
                return False
            spec["evidence_sources"] = kept
        return True

    def _render_chart(self, spec: Dict[str, Any], index: int, out_dir: str) -> Optional[str]:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            self.last_error = f"matplotlib is required for chart rendering: {exc}"
            return None

        labels = spec.get("labels", [])
        values = spec.get("values", [])
        fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=140)
        chart_type = spec["chart_type"]
        if chart_type == "bar":
            ax.bar(labels, values, color="#2A6FDB")
            ax.set_ylabel(spec.get("y_label", "Value"))
            ax.set_xlabel(spec.get("x_label", "Category"))
        elif chart_type == "line":
            ax.plot(labels, values, marker="o", color="#0D9488", linewidth=2)
            ax.set_ylabel(spec.get("y_label", "Value"))
            ax.set_xlabel(spec.get("x_label", "Category"))
            ax.grid(alpha=0.25)
        elif chart_type == "pie":
            ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
        elif chart_type == "scatter":
            x_vals = spec.get("x_values", [])
            y_vals = spec.get("y_values", [])
            ax.scatter(x_vals, y_vals, color="#B45309")
            ax.set_ylabel(spec.get("y_label", "Y"))
            ax.set_xlabel(spec.get("x_label", "X"))
            ax.grid(alpha=0.25)
        else:
            self._render_flow_chart(ax, spec.get("steps", []))

        ax.set_title(spec.get("title") or "Chart")
        fig.tight_layout()

        safe_title = re.sub(r"[^a-zA-Z0-9_-]+", "_", spec.get("title", "chart")).strip("_") or "chart"
        path = os.path.join(out_dir, f"{index+1:02d}_{safe_title}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    @staticmethod
    def _render_flow_chart(ax, steps: List[str]) -> None:
        if not steps:
            steps = ["Input", "Analyze", "Decide", "Act"]
        ax.axis("off")
        n = len(steps)
        y_top = 0.9
        y_step = 0.75 / max(1, n - 1)
        for i, step in enumerate(steps):
            y = y_top - i * y_step
            ax.text(
                0.5,
                y,
                step,
                ha="center",
                va="center",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.35", fc="#E0F2FE", ec="#0369A1", lw=1.2),
                transform=ax.transAxes,
            )
            if i < n - 1:
                y2 = y_top - (i + 1) * y_step
                ax.annotate(
                    "",
                    xy=(0.5, y2 + 0.04),
                    xytext=(0.5, y - 0.04),
                    arrowprops=dict(arrowstyle="->", color="#0369A1", lw=1.3),
                    xycoords=ax.transAxes,
                )

    @staticmethod
    def _fallback_tabular_metric_chart(context: str, available_sources: List[str]) -> Optional[Dict[str, Any]]:
        excel_chart = DocumentChartGenerator._fallback_excel_source_chart(available_sources)
        if excel_chart:
            return excel_chart

        month_pattern = re.compile(
            r"(?i)\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
            r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b"
        )
        value_pattern = re.compile(r"[-+]?\d+(?:\.\d+)?")
        points: Dict[str, float] = {}
        for line in (context or "").splitlines():
            month_match = month_pattern.search(line)
            if not month_match:
                continue
            values = [v for v in value_pattern.findall(line) if len(v) < 8]
            if not values:
                continue
            try:
                num = float(values[-1])
            except Exception:
                continue
            month = month_match.group(1).title()[:3]
            points[month] = num

        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        ordered = [(m, points[m]) for m in month_order if m in points]
        if len(ordered) >= 3:
            labels = [m for m, _ in ordered[:8]]
            values = [v for _, v in ordered[:8]]
            return {
                "title": "Monthly Trend Snapshot",
                "chart_type": "line",
                "x_label": "Month",
                "y_label": "Value",
                "labels": labels,
                "values": values,
                "caption": "Fallback trend chart extracted from month/value pairs in retrieved context.",
                "evidence_sources": available_sources[:2] if available_sources else [],
            }
        return None

    @staticmethod
    def _fallback_excel_source_chart(available_sources: List[str]) -> Optional[Dict[str, Any]]:
        try:
            import pandas as pd
        except Exception:
            return None

        for src in available_sources:
            ext = os.path.splitext(str(src).lower())[1]
            if ext not in {".xlsx", ".xls"}:
                continue
            if not os.path.exists(src):
                continue
            try:
                sheets = pd.read_excel(src, sheet_name=None)
            except Exception:
                continue
            for sheet_name, df in sheets.items():
                if df is None or df.empty:
                    continue
                frame = df.copy()
                numeric_cols = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
                if not numeric_cols:
                    continue
                label_cols = [c for c in frame.columns if c not in numeric_cols]

                if label_cols:
                    label_col = label_cols[0]
                    value_col = numeric_cols[0]
                    view = frame[[label_col, value_col]].dropna().head(10)
                    labels = [str(v).strip()[:40] for v in view[label_col].tolist() if str(v).strip()]
                    values = []
                    for v in view[value_col].tolist()[: len(labels)]:
                        try:
                            values.append(float(v))
                        except Exception:
                            values.append(float("nan"))
                    pairs = [(l, val) for l, val in zip(labels, values) if isinstance(val, float) and math.isfinite(val)]
                    if len(pairs) >= 2:
                        labels = [p[0] for p in pairs]
                        values = [p[1] for p in pairs]
                        return {
                            "title": f"{sheet_name}: {value_col} by {label_col}"[:120],
                            "chart_type": "bar",
                            "x_label": str(label_col)[:60],
                            "y_label": str(value_col)[:60],
                            "labels": labels,
                            "values": values,
                            "caption": "Fallback chart extracted directly from spreadsheet columns.",
                            "evidence_sources": [src],
                        }

                value_col = numeric_cols[0]
                series = frame[value_col].dropna().head(10).tolist()
                values = []
                for v in series:
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    if math.isfinite(fv):
                        values.append(fv)
                if len(values) >= 2:
                    labels = [f"Row {i}" for i in range(1, len(values) + 1)]
                    return {
                        "title": f"{sheet_name}: {value_col} trend"[:120],
                        "chart_type": "line",
                        "x_label": "Row",
                        "y_label": str(value_col)[:60],
                        "labels": labels,
                        "values": values,
                        "caption": "Fallback trend chart extracted directly from spreadsheet values.",
                        "evidence_sources": [src],
                    }
        return None

    @staticmethod
    def _fallback_source_relevance_chart(context: str, available_sources: List[str]) -> Optional[Dict[str, Any]]:
        rows = re.findall(r"source=([^,\n]+).*?score=([0-9]*\.?[0-9]+)", context or "", flags=re.IGNORECASE)
        if not rows:
            return None
        scores: Dict[str, float] = {}
        for src, score_txt in rows:
            src_clean = str(src).strip()
            if not src_clean:
                continue
            try:
                score = float(score_txt)
            except Exception:
                continue
            scores[src_clean] = max(score, scores.get(src_clean, -1.0))
        if not scores:
            return None

        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:6]
        labels = [os.path.basename(src) or src for src, _ in ordered]
        values = [val for _, val in ordered]
        evidence = [src for src, _ in ordered]

        # If we have an available source allowlist, keep intersection where possible.
        if available_sources:
            allowed = set(available_sources)
            kept = [src for src in evidence if src in allowed]
            if kept:
                evidence = kept

        return {
            "title": "Retrieved Source Relevance",
            "chart_type": "bar",
            "x_label": "Source",
            "y_label": "Retrieval Score",
            "labels": labels,
            "values": values,
            "caption": "Fallback chart built from retrieval scores when explicit chart data was not parsed.",
            "evidence_sources": evidence,
        }

    @staticmethod
    def _fallback_process_flow(user_prompt: str, available_sources: List[str]) -> Optional[Dict[str, Any]]:
        if not available_sources:
            return None
        return {
            "title": "Analysis Workflow",
            "chart_type": "flow",
            "x_label": "",
            "y_label": "",
            "labels": [],
            "values": [],
            "x_values": [],
            "y_values": [],
            "steps": [
                "Ingest mixed sources",
                "Retrieve relevant context",
                "Cross-source validation",
                "Quantitative synthesis",
                "Actions and recommendations",
            ],
            "caption": "Process flow used for this report generation pipeline.",
            "evidence_sources": available_sources[:3],
        }
