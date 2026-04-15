from pathlib import Path

from core.charts.generator import DocumentChartGenerator


class _FakeChartLLM:
    def generate_completion(self, prompt: str, system_prompt: str = "") -> str:
        return (
            '{"charts":[{"title":"Invoice Payments","chart_type":"bar","x_label":"Payment","y_label":"Amount",'
            '"labels":["PAY-9012","PAY-9024"],"values":[700.00,702.27],'
            '"caption":"Payment split for invoice INV-2026-031.",'
            '"evidence_sources":["examples/coffee_shop_invoice_data.xlsx"]}]}'
        )


class _InvalidChartLLM:
    def generate_completion(self, prompt: str, system_prompt: str = "") -> str:
        return "Thinking Process: I will reason step by step and not return valid JSON."


def test_chart_generator_creates_grounded_artifact(monkeypatch, tmp_path: Path):
    gen = DocumentChartGenerator(_FakeChartLLM())

    # Patch renderer to keep test lightweight and deterministic.
    def _fake_render(self, spec, index, out_dir):
        from PIL import Image

        path = Path(out_dir) / f"chart_{index+1}.png"
        img = Image.new("RGB", (640, 360), color=(240, 248, 255))
        img.save(path)
        return str(path)

    monkeypatch.setattr(DocumentChartGenerator, "_render_chart", _fake_render)

    context = (
        "[1] source=examples/coffee_shop_invoice_data.xlsx, page=3, score=0.729\n"
        "PAY-9012 INV-2026-031 Bank Transfer 700.00\n"
        "PAY-9024 INV-2026-031 Card 702.27\n"
        "Total 1402.27\n"
    )
    charts = gen.generate(
        user_prompt="Show a payments breakdown chart",
        document_title="Invoice analysis",
        retrieved_context=context,
        available_sources=["examples/coffee_shop_invoice_data.xlsx"],
        max_charts=3,
        include_flowchart=False,
    )

    assert len(charts) == 1
    chart = charts[0]
    assert chart["chart_type"] == "bar"
    assert chart["title"] == "Invoice Payments"
    assert chart["evidence_sources"] == ["examples/coffee_shop_invoice_data.xlsx"]
    assert Path(chart["image_path"]).exists()


def test_chart_generator_fallback_uses_retrieval_scores(monkeypatch):
    gen = DocumentChartGenerator(_InvalidChartLLM())

    def _fake_render(self, spec, index, out_dir):
        from PIL import Image

        path = Path(out_dir) / f"fallback_{index+1}.png"
        Image.new("RGB", (640, 360), color=(235, 245, 235)).save(path)
        return str(path)

    monkeypatch.setattr(DocumentChartGenerator, "_render_chart", _fake_render)

    context = (
        "[1] source=missing_sheet.xlsx, page=1, score=0.711\n"
        "some text\n"
        "[2] source=missing_invoice.png, page=1, score=0.630\n"
    )
    charts = gen.generate(
        user_prompt="Create chart",
        document_title="Doc",
        retrieved_context=context,
        available_sources=["missing_sheet.xlsx", "missing_invoice.png"],
        max_charts=2,
        include_flowchart=False,
    )
    assert len(charts) == 1
    assert charts[0]["title"] == "Retrieved Source Relevance"
    assert Path(charts[0]["image_path"]).exists()


def test_chart_generator_fallback_uses_excel_when_available(monkeypatch, tmp_path: Path):
    import pandas as pd
    import pytest

    pytest.importorskip("openpyxl")

    gen = DocumentChartGenerator(_InvalidChartLLM())
    sample_xlsx = tmp_path / "ops.xlsx"
    df = pd.DataFrame(
        [
            {"Month": "Jan", "Revenue": 120000},
            {"Month": "Feb", "Revenue": 135000},
            {"Month": "Mar", "Revenue": 128000},
        ]
    )
    with pd.ExcelWriter(sample_xlsx) as writer:
        df.to_excel(writer, index=False, sheet_name="Performance")

    def _fake_render(self, spec, index, out_dir):
        from PIL import Image

        path = Path(out_dir) / f"excel_fallback_{index+1}.png"
        Image.new("RGB", (640, 360), color=(248, 248, 238)).save(path)
        return str(path)

    monkeypatch.setattr(DocumentChartGenerator, "_render_chart", _fake_render)

    charts = gen.generate(
        user_prompt="Create chart",
        document_title="Doc",
        retrieved_context="[1] source=ops.xlsx, page=1, score=0.711\n",
        available_sources=[str(sample_xlsx)],
        max_charts=2,
        include_flowchart=False,
    )

    assert len(charts) == 1
    assert charts[0]["chart_type"] in {"bar", "line"}
    assert charts[0]["evidence_sources"] == [str(sample_xlsx)]
    assert Path(charts[0]["image_path"]).exists()


def test_chart_generator_can_append_flowchart(monkeypatch):
    gen = DocumentChartGenerator(_InvalidChartLLM())

    def _fake_render(self, spec, index, out_dir):
        from PIL import Image

        path = Path(out_dir) / f"chart_{index+1}.png"
        Image.new("RGB", (640, 360), color=(245, 245, 255)).save(path)
        return str(path)

    monkeypatch.setattr(DocumentChartGenerator, "_render_chart", _fake_render)

    context = (
        "[1] source=examples/coffee_shop_invoice_data.xlsx, page=1, score=0.711\n"
        "[2] source=examples/coffee_shop_invoice.png, page=1, score=0.630\n"
    )
    charts = gen.generate(
        user_prompt="Analyze and provide process flow",
        document_title="Doc",
        retrieved_context=context,
        available_sources=["examples/coffee_shop_invoice_data.xlsx", "examples/coffee_shop_invoice.png"],
        max_charts=3,
        include_flowchart=True,
    )
    assert len(charts) >= 2
    assert any(c.get("chart_type") == "flow" for c in charts)


def test_chart_generator_can_render_flowchart_without_base_chart(monkeypatch, tmp_path: Path):
    class _NoChartLLM:
        def generate_completion(self, prompt, system_prompt=None):
            return '{"charts":[]}'

    def _fake_render(self, spec, index, out_dir):
        from PIL import Image

        path = tmp_path / f"flow_{index+1}.png"
        Image.new("RGB", (640, 360), color=(240, 248, 255)).save(path)
        return str(path)

    monkeypatch.setattr(DocumentChartGenerator, "_render_chart", _fake_render)
    gen = DocumentChartGenerator(_NoChartLLM())

    charts = gen.generate(
        user_prompt="Show the analysis process as a flowchart",
        document_title="Comparison",
        retrieved_context="Source A discusses coverage limits.\nSource B discusses exclusions.",
        available_sources=["a.pdf", "b.pdf"],
        max_charts=2,
        include_flowchart=True,
    )

    assert len(charts) == 1
    assert charts[0]["chart_type"] == "flow"
    assert Path(charts[0]["image_path"]).exists()
