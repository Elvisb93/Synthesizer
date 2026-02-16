import flet as ft
import tkinter as tk
from tkinter import filedialog
from typing import List

class Dialogs:
    """Helper class for UI dialogs and interactions."""
    
    @staticmethod
    def show_snackbar(page: ft.Page, message: str):
        """Show a snackbar message using overlay."""
        sb = ft.SnackBar(ft.Text(message), open=True)
        page.overlay.append(sb)
        page.update()

    @staticmethod
    def get_file_save_path(page: ft.Page, title: str, types: List[tuple], default_ext: str) -> str:
        """Helper to use Tkinter file dialog in a thread-safe way."""
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            path = filedialog.asksaveasfilename(title=title, filetypes=types, defaultextension=default_ext)
            root.destroy()
            return path
        except Exception as e:
            Dialogs.show_snackbar(page, f"Error opening file dialog: {e}")
            return ""

    @staticmethod
    def get_file_open_path(page: ft.Page, title: str, types: List[tuple]) -> str:
        """Helper to use Tkinter file dialog in a thread-safe way."""
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            path = filedialog.askopenfilename(title=title, filetypes=types)
            root.destroy()
            return path
        except Exception as e:
            Dialogs.show_snackbar(page, f"Error opening file dialog: {e}")
            return ""
