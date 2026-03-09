import unittest
import os
import pandas as pd
from core.exporters import PDFReportGenerator, DocumentPDFExporter

class TestPDFReportGenerator(unittest.TestCase):
    def setUp(self):
        self.exporter = PDFReportGenerator()
        self.test_quality_file = "test_quality_report.pdf"
        self.test_narrative_file = "test_narrative.pdf"
        self.test_chart_image = "test_chart_image.png"

    def tearDown(self):
        if os.path.exists(self.test_quality_file):
            os.remove(self.test_quality_file)
        if os.path.exists(self.test_narrative_file):
            os.remove(self.test_narrative_file)
        if os.path.exists(self.test_chart_image):
            os.remove(self.test_chart_image)

    def test_generate_quality_report(self):
        metrics = {
            "Column A": {
                "diversity_score": 0.85,
                "null_count": 0,
                "top_frequent": {"Value1": 5, "Value2": 3}
            },
            "Column B": {
                "diversity_score": 0.2,
                "null_count": 10,
                "top_frequent": {"HighFreq": 50}
            }
        }
        self.exporter.generate_quality_report(metrics, self.test_quality_file)
        
        self.assertTrue(os.path.exists(self.test_quality_file))
        self.assertGreater(os.path.getsize(self.test_quality_file), 0)

    def test_generate_narrative_export(self):
        df = pd.DataFrame([
            {"Title": "Row 1", "Body": "This is content for row 1."},
            {"Title": "Row 2", "Body": "This is content for row 2."}
        ])
        
        self.exporter.generate_narrative_export(df, "Title", ["Body"], self.test_narrative_file)
        
        self.assertTrue(os.path.exists(self.test_narrative_file))
        self.assertGreater(os.path.getsize(self.test_narrative_file), 0)

    def test_special_characters_export(self):
        """Test that special characters (smart quotes, em-dashes) don't crash the exporter."""
        # This title contains em-dash and smart quotes
        title = "Story with Symbols: “Hello” — World"
        text = """
        • Point 1: Smart quotes “work” now.
        • Point 2: Em-dashes — are cool.
        • Point 3: Ellipses… behave.
        """
        
        # Test Document Export
        doc_exporter = DocumentPDFExporter()
        doc_exporter.export(
            title=title,
            outline={},
            text=text,
            output_path=self.test_narrative_file
        )
        self.assertTrue(os.path.exists(self.test_narrative_file))
        
        # Test Report Export
        report_exporter = PDFReportGenerator()
        df = pd.DataFrame([{"Title": title, "Body": text}])
        report_exporter.generate_narrative_export(
            df=df, 
            title_col="Title", 
            body_cols=["Body"], 
            output_path=self.test_quality_file
        )
        self.assertTrue(os.path.exists(self.test_quality_file))

    def test_document_export_with_embedded_chart(self):
        from PIL import Image

        Image.new("RGB", (640, 360), color=(245, 245, 245)).save(self.test_chart_image)

        doc_exporter = DocumentPDFExporter()
        doc_exporter.export(
            title="Document With Chart",
            outline={},
            text="Overview\n\nThis report includes one grounded chart.",
            output_path=self.test_narrative_file,
            charts=[
                {
                    "title": "Sample Chart",
                    "caption": "Synthetic test chart",
                    "image_path": self.test_chart_image,
                    "evidence_sources": ["sample.xlsx"],
                }
            ],
        )
        self.assertTrue(os.path.exists(self.test_narrative_file))
        self.assertGreater(os.path.getsize(self.test_narrative_file), 0)

    def test_markdown_layout_export_for_document_and_narrative(self):
        markdown_text = (
            "# Executive Summary\n\n"
            "## Highlights\n"
            "- Revenue increased 18%\n"
            "- Costs held flat\n\n"
            "### SQL Snippet\n"
            "```sql\n"
            "SELECT region, SUM(amount)\n"
            "FROM sales\n"
            "GROUP BY region;\n"
            "```\n"
        )

        doc_exporter = DocumentPDFExporter()
        doc_exporter.export(
            title="Markdown Layout Test",
            outline={},
            text=markdown_text,
            output_path=self.test_narrative_file,
        )
        self.assertTrue(os.path.exists(self.test_narrative_file))
        self.assertGreater(os.path.getsize(self.test_narrative_file), 0)

        report_exporter = PDFReportGenerator()
        df = pd.DataFrame([{"Title": "Markdown Row", "Body": markdown_text}])
        report_exporter.generate_narrative_export(
            df=df,
            title_col="Title",
            body_cols=["Body"],
            output_path=self.test_quality_file,
        )
        self.assertTrue(os.path.exists(self.test_quality_file))
        self.assertGreater(os.path.getsize(self.test_quality_file), 0)

if __name__ == '__main__':
    unittest.main()
