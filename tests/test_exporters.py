import unittest
import os
import pandas as pd
from core.exporters import PDFReportGenerator

class TestPDFReportGenerator(unittest.TestCase):
    def setUp(self):
        self.exporter = PDFReportGenerator()
        self.test_quality_file = "test_quality_report.pdf"
        self.test_narrative_file = "test_narrative.pdf"

    def tearDown(self):
        if os.path.exists(self.test_quality_file):
            os.remove(self.test_quality_file)
        if os.path.exists(self.test_narrative_file):
            os.remove(self.test_narrative_file)

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

if __name__ == '__main__':
    unittest.main()
