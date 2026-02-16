"""
Data-related event handlers for FletApp.

Handles: data import (CSV/JSON), data export, quality analysis.
"""
import flet as ft

from core.models import ColumnDefinition, ColumnType
from gui.utils import Dialogs, pick_file, save_file

try:
    import pandas as pd
except ImportError:
    pd = None


class DataHandlersMixin:
    """Mixin providing data import/export/analysis handlers for FletApp."""

    async def _on_import_data(self, e):
        if not pd:
            Dialogs.show_snackbar(self.page, "Error: pandas not installed.")
            return

        path = await pick_file(
            title="Import Data",
            filter_pairs=("CSV files", "*.csv", "JSON files", "*.json", "All files", "*.*")
        )
        if path:
            try:
                if path.endswith('.csv'):
                    df = pd.read_csv(path)
                else:
                    df = pd.read_json(path)

                self.imported_data = df.to_dict(orient='records')
                count = len(self.imported_data)
                self.rows_field.value = str(count)

                # Clear and create schema
                self.columns.clear()
                self.columns_list.controls.clear()

                for col_name in df.columns:
                    dtype = str(df[col_name].dtype)
                    col_type = ColumnType.NUMERIC if 'int' in dtype or 'float' in dtype else ColumnType.SHORT_TEXT

                    self._add_column(ColumnDefinition(
                        name=col_name,
                        type=col_type,
                        prompt_instruction="(Imported)"
                    ))

                Dialogs.show_snackbar(self.page, f"Imported {count} rows. Schema updated.")
                self.page.update()
            except Exception as ex:
                Dialogs.show_snackbar(self.page, f"Import error: {ex}")

    async def export_data(self, format_type):
        self.current_export_format = format_type
        ext = format_type.split('_')[0]  # pdf_report -> pdf

        if not self.controller.generated_rows:
            Dialogs.show_snackbar(self.page, "Please generate data first (0 rows).")
            return

        path = await save_file(
            title=f"Export {format_type.upper()}",
            default_name=f"export.{ext}",
            filter_pairs=(f"{ext.upper()} files", f"*.{ext}", "All files", "*.*")
        )
        if path:
            try:
                if format_type == "csv":
                    self.controller.export_csv(path)
                elif format_type == "json":
                    self.controller.export_json(path)
                elif format_type == "sql":
                    self.controller.export_sql(path)
                elif format_type == "pdf_report":
                    self.controller.export_pdf_report(path)
                elif format_type == "pdf_narrative":
                    self.controller.export_narrative_pdf(path)

                Dialogs.show_snackbar(self.page, f"Exported to {path}")
            except Exception as ex:
                Dialogs.show_snackbar(self.page, f"Export error: {ex}")

    def _handle_export(self, e, format_type):
        """Sync wrapper to schedule async export_data."""
        async def task():
            await self.export_data(format_type)
        self.page.run_task(task)

    def _on_analyze(self, e):
        metrics = self.controller.analyze_quality()
        if not metrics:
            Dialogs.show_snackbar(self.page, "No data to analyze.")
            return
        
        msg = "=== DATA QUALITY REPORT ===\n\n"
        for col, data in metrics.items():
            msg += f"COLUMN: {col}\n"
            msg += f"{'-'*30}\n"
            msg += f"  • Diversity Score: {data.get('diversity_score', 0):.1%}\n"
            msg += f"  • Null Count:      {data.get('null_count', 0)}\n"
            msg += f"  • Top Frequent Values:\n"
            for val, count in data.get('top_frequent', {}).items():
                msg += f"      - {val}: {count}\n"
            msg += "\n"

        dlg = ft.AlertDialog(
            title=ft.Text("Data Quality Analysis"),
            content=ft.Text(msg, font_family="monospace"),
            on_dismiss=lambda e: print("Dialog dismissed")
        )
        self.page.overlay.append(dlg)
        dlg.open = True
        self.page.update()
