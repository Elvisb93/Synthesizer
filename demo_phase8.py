import os
import pandas as pd
from core.controller import GeneratorController
from core.models import RowData

def demo_phase8():
    print("--- 🧪 Phase 8 Feature Demo ---")
    
    # 1. Setup Controller & Data
    controller = GeneratorController()
    
    # Simulate some data with duplicates to test metrics
    data = [
        {"Name": "Alice", "Role": "Admin", "City": "New York"},   
        {"Name": "Bob", "Role": "User", "City": "London"},       
        {"Name": "Charlie", "Role": "User", "City": "Paris"},    
        {"Name": "Alice", "Role": "Admin", "City": "New York"},   # Duplicate (full row)
        {"Name": "Dave", "Role": "User", "City": "London"},      # Duplicate City/Role
    ]
    controller.generated_rows = [RowData(data=d) for d in data]
    print(f"Loaded {len(data)} rows (with intentional duplicates).")
    
    # 2. Test Analysis
    print("\n--- 📊 Testing Quality Analysis ---")
    metrics = controller.analyze_quality()
    for col, stats in metrics.items():
        print(f"Column '{col}':")
        print(f"  Uniqueness: {stats['diversity_score']:.0%}")
        # Show top values to confirm count is correct
        top = stats['top_frequent']
        print(f"  Top Values: {top}")
    
    # 3. Test PDF Export
    print("\n--- 📄 Testing PDF Export ---")
    report_file = "demo_quality_report.pdf"
    narrative_file = "demo_narrative.pdf"
    
    # Clean up old
    if os.path.exists(report_file): os.remove(report_file)
    if os.path.exists(narrative_file): os.remove(narrative_file)
    
    controller.export_pdf_report(report_file)
    if os.path.exists(report_file):
        print(f"✅ Exported Quality Report: {report_file} ({os.path.getsize(report_file)} bytes)")
    else:
        print(f"❌ Failed to export Quality Report")

    controller.export_narrative_pdf(narrative_file)
    if os.path.exists(narrative_file):
        print(f"✅ Exported Narrative Report: {narrative_file} ({os.path.getsize(narrative_file)} bytes)")
    else:
        print(f"❌ Failed to export Narrative Report")

if __name__ == "__main__":
    demo_phase8()
