import os
from core.controller import GeneratorController
from core.models import RowData

def verify_integration():
    print("Initializing Controller...")
    controller = GeneratorController()
    
    # Mock data
    print("Mocking generated data...")
    controller.generated_rows = [
        RowData(data={"Name": "Alice", "Role": "Engineer", "Bio": "Loves coding."}),
        RowData(data={"Name": "Bob", "Role": "Designer", "Bio": "Loves art."}),
        RowData(data={"Name": "Charlie", "Role": "Engineer", "Bio": "Loves systems."}),
    ]
    
    # Test Quality Report
    report_path = "verify_quality_report.pdf"
    if os.path.exists(report_path): os.remove(report_path)
    
    print(f"Exporting Quality Report to {report_path}...")
    controller.export_pdf_report(report_path)
    
    if os.path.exists(report_path) and os.path.getsize(report_path) > 0:
        print("✅ Quality Report Exported Successfully.")
    else:
        print("❌ Quality Report Export Failed.")

    # Test Narrative Export
    narrative_path = "verify_narrative.pdf"
    if os.path.exists(narrative_path): os.remove(narrative_path)
    
    print(f"Exporting Narrative Report to {narrative_path}...")
    controller.export_narrative_pdf(narrative_path)
    
    if os.path.exists(narrative_path) and os.path.getsize(narrative_path) > 0:
        print("✅ Narrative Report Exported Successfully.")
    else:
        print("❌ Narrative Report Export Failed.")

    # Cleanup
    # os.remove(report_path)
    # os.remove(narrative_path)

if __name__ == "__main__":
    verify_integration()
