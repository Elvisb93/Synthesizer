from fpdf import FPDF
import pandas as pd
from typing import Dict, Any, List

class PDFReportGenerator:
    """
    Generates PDF reports for data quality and narrative exports.
    """

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """
        Sanitizes text to be compatible with latin-1 encoding used by FPDF standard fonts.
        Replaces common incompatible characters with ASCII approximations.
        """
        if not isinstance(text, str):
            text = str(text)
            
        replacements = {
            # Quotes
            '\u2018': "'",  # Left single quote
            '\u2019': "'",  # Right single quote
            '\u201C': '"',  # Left double quote
            '\u201D': '"',  # Right double quote
            '\u2032': "'",  # Prime (foot mark)
            '\u2033': '"',  # Double prime (inch mark)
            # Dashes and hyphens
            '\u2011': '-',  # Non-breaking hyphen  <-- KEY FIX
            '\u2012': '-',  # Figure dash
            '\u2013': '-',  # En dash
            '\u2014': '--', # Em dash
            '\u2015': '--', # Horizontal bar
            '\u00ad': '-',  # Soft hyphen
            # Bullets and others
            '\u2022': '*',  # Bullet
            '\u2023': '>',  # Triangular bullet
            '\u2026': '...', # Ellipsis
            '\u20ac': 'EUR', # Euro
            '\u2122': '(TM)', # Trademark
            '\u00a9': '(c)', # Copyright
            '\u00ae': '(R)', # Registered
            '\u00a0': ' ', # Non-breaking space
            '\u2009': ' ', # Thin space
            '\u200b': '',  # Zero-width space
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
            
        # Final fallback for any other non-latin-1 characters
        return text.encode('latin-1', 'replace').decode('latin-1')

    
    def generate_quality_report(self, metrics: Dict[str, Any], output_path: str):
        """
        Creates a PDF report visualizing the quality metrics.
        """
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font("helvetica", "B", 16)
        pdf.cell(0, 10, "Data Quality Report", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(10)
        
        # Metrics Loop
        for col, data in metrics.items():
            # Check for page break space (approx 60mm needed)
            if pdf.get_y() > 250:
                pdf.add_page()

            pdf.set_font("helvetica", "B", 14)
            # Background color for header
            pdf.set_fill_color(240, 240, 240)
            clean_col = self._sanitize_text(col)
            pdf.cell(0, 10, f"Column: {clean_col}", new_x="LMARGIN", new_y="NEXT", fill=True)
            
            pdf.set_font("helvetica", "", 12)
            
            # Diversity Bar
            score = data.get('diversity_score', 0)
            pdf.cell(50, 8, f"Diversity Score: {score:.1%}")
            
            # Draw a simple progress bar
            bar_x = pdf.get_x()
            bar_y = pdf.get_y() + 2
            bar_w = 50
            bar_h = 4
            
            pdf.set_fill_color(220, 220, 220)
            pdf.rect(bar_x, bar_y, bar_w, bar_h, 'F') # Background
            
            # Color based on score (Red < 30%, Yellow < 70%, Green > 70%)
            if score < 0.3:
                pdf.set_fill_color(220, 50, 50)
            elif score < 0.7:
                pdf.set_fill_color(220, 220, 50)
            else:
                pdf.set_fill_color(50, 220, 50)
                
            pdf.rect(bar_x, bar_y, bar_w * score, bar_h, 'F')
            pdf.ln(8)

            pdf.cell(0, 8, f"Null Count: {data.get('null_count', 0)}", new_x="LMARGIN", new_y="NEXT")
            
            pdf.set_font("helvetica", "I", 11)
            pdf.cell(0, 8, "Top 5 Frequent Values:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("helvetica", "", 10)
            
            frequent = data.get('top_frequent', {})
            for val, count in frequent.items():
                pdf.set_x(pdf.get_x() + 10) # Indent
                clean_val = self._sanitize_text(str(val))
                pdf.cell(0, 6, f"- {clean_val}: {count}", new_x="LMARGIN", new_y="NEXT")
            
            pdf.ln(5)
            
        try:
            pdf.output(output_path)
        except Exception as e:
            print(f"Error saving PDF: {e}")
            raise e

    def generate_narrative_export(self, df: pd.DataFrame, title_col: str, body_cols: List[str], output_path: str):
        """
        Exports row data as a document (Narrative Mode).
        Each row gets a section or page.
        """
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        for _, row in df.iterrows():
            pdf.add_page()
            
            # Check if title column exists
            title_text = "Record"
            if title_col and title_col in row:
                title_text = str(row[title_col])
                
            pdf.set_font("helvetica", "B", 18)
            pdf.multi_cell(0, 10, self._sanitize_text(title_text), align='C')
            pdf.ln(10)
            
            # Body Content
            pdf.set_font("helvetica", "", 12)
            
            for col in body_cols:
                if col in row:
                    # Header for the field if multiple body columns
                    if len(body_cols) > 1:
                        pdf.set_font("helvetica", "B", 12)
                        clean_col_header = self._sanitize_text(col)
                        pdf.cell(0, 8, f"{clean_col_header}:", new_x="LMARGIN", new_y="NEXT")
                        pdf.set_font("helvetica", "", 12)
                    
                    text = str(row[col])
                    # Sanitize text to latin-1 compatible or basic ASCII if needed
                    # FPDF2 handles utf-8 better than FPDF1, but standard fonts are still limited.
                    # For robustness in this MVP, we encode/decode to replace incompatible chars
                    text = self._sanitize_text(text)
                    
                    pdf.multi_cell(0, 6, text)
                    pdf.ln(5)
                    
        try:
            pdf.output(output_path)
        except Exception as e:
            print(f"Error saving Narrative PDF: {e}")
            raise e
