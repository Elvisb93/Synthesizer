from typing import Dict, Any, List
import pandas as pd
from collections import Counter

class QualityAnalyzer:
    """
    Analyzes a DataFrame to produce quality metrics per column.
    """

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run analysis on the dataframe.
        Returns a dictionary keyed by column name with metric data.
        """
        report = {}
        total_rows = len(df)

        if total_rows == 0:
            return {}

        for col in df.columns:
            column_data = df[col]
            
            # 1. Null Count
            null_count = column_data.isnull().sum()
            
            # 2. Diversity Score (Uniqueness)
            # Convert to string to ensure consistent hashing for lists/dicts if any
            try:
                unique_val_count = column_data.nunique()
            except TypeError:
                # Fallback for unhashable types (like lists of lists)
                unique_val_count = len(column_data.astype(str).unique())
            
            diversity_score = unique_val_count / total_rows if total_rows > 0 else 0.0

            # 3. Redundancy (Top frequent)
            # value_counts is robust
            try:
                top_counts = column_data.value_counts(dropna=False).head(5)
            except Exception:
                # Fallback
                top_counts = pd.Series(Counter(column_data.astype(str)).most_common(5))
                
            redundancy_data = top_counts.to_dict()
            # Convert keys to str for JSON serialization safety if needed
            redundancy_data = {str(k): int(v) for k, v in redundancy_data.items()}

            report[col] = {
                "null_count": int(null_count),
                "diversity_score": float(diversity_score),
                "top_frequent": redundancy_data
            }
        
        return report
