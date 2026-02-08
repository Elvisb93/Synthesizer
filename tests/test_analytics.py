import unittest
import pandas as pd
from core.analytics import QualityAnalyzer

class TestQualityAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = QualityAnalyzer()

    def test_diversity_score(self):
        # 4 unique values out of 5 rows = 0.8
        df = pd.DataFrame({'col1': ['A', 'B', 'A', 'C', 'D']})
        report = self.analyzer.analyze(df)
        self.assertAlmostEqual(report['col1']['diversity_score'], 0.8)

    def test_redundancy_check(self):
        # 'A' appears 3 times
        df = pd.DataFrame({'col1': ['A', 'A', 'A', 'B', 'B', 'C']})
        report = self.analyzer.analyze(df)
        self.assertEqual(report['col1']['top_frequent']['A'], 3)
        self.assertEqual(report['col1']['top_frequent']['B'], 2)

    def test_null_count(self):
        df = pd.DataFrame({'col1': ['A', None, 'B', None]})
        report = self.analyzer.analyze(df)
        self.assertEqual(report['col1']['null_count'], 2)

    def test_empty_df(self):
        df = pd.DataFrame({})
        report = self.analyzer.analyze(df)
        self.assertEqual(report, {})

if __name__ == '__main__':
    unittest.main()
