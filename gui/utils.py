import flet as ft
import asyncio
import ctypes
import ctypes.wintypes
import threading
from typing import Optional, List


class Dialogs:
    """Helper class for UI dialogs and interactions."""

    @staticmethod
    def show_snackbar(page: ft.Page, message: str):
        """Show a snackbar message using overlay."""
        sb = ft.SnackBar(ft.Text(message), open=True)
        page.overlay.append(sb)
        page.update()


# --- Native Windows File Dialogs via ctypes ---

def _win_open_file(title: str, filter_str: str) -> Optional[str]:
    """Open a native Windows file open dialog. Must run in a thread."""
    import ctypes
    import ctypes.wintypes

    OFN_FILEMUSTEXIST = 0x00001000
    OFN_PATHMUSTEXIST = 0x00000800
    OFN_EXPLORER = 0x00080000
    MAX_PATH = 260

    class OPENFILENAME(ctypes.Structure):
        _fields_ = [
            ("lStructSize", ctypes.wintypes.DWORD),
            ("hwndOwner", ctypes.wintypes.HWND),
            ("hInstance", ctypes.wintypes.HINSTANCE),
            ("lpstrFilter", ctypes.wintypes.LPCWSTR),
            ("lpstrCustomFilter", ctypes.c_wchar_p),
            ("nMaxCustFilter", ctypes.wintypes.DWORD),
            ("nFilterIndex", ctypes.wintypes.DWORD),
            ("lpstrFile", ctypes.c_wchar_p),
            ("nMaxFile", ctypes.wintypes.DWORD),
            ("lpstrFileTitle", ctypes.c_wchar_p),
            ("nMaxFileTitle", ctypes.wintypes.DWORD),
            ("lpstrInitialDir", ctypes.wintypes.LPCWSTR),
            ("lpstrTitle", ctypes.wintypes.LPCWSTR),
            ("Flags", ctypes.wintypes.DWORD),
            ("nFileOffset", ctypes.wintypes.WORD),
            ("nFileExtension", ctypes.wintypes.WORD),
            ("lpstrDefExt", ctypes.wintypes.LPCWSTR),
            ("lCustData", ctypes.wintypes.LPARAM),
            ("lpfnHook", ctypes.c_void_p),
            ("lpTemplateName", ctypes.wintypes.LPCWSTR),
            ("pvReserved", ctypes.c_void_p),
            ("dwReserved", ctypes.wintypes.DWORD),
            ("FlagsEx", ctypes.wintypes.DWORD),
        ]

    buf = ctypes.create_unicode_buffer(MAX_PATH)
    ofn = OPENFILENAME()
    ofn.lStructSize = ctypes.sizeof(OPENFILENAME)
    ofn.lpstrFilter = filter_str
    ofn.lpstrFile = ctypes.cast(buf, ctypes.c_wchar_p)
    ofn.nMaxFile = MAX_PATH
    ofn.lpstrTitle = title
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_EXPLORER

    if ctypes.windll.comdlg32.GetOpenFileNameW(ctypes.byref(ofn)):
        return buf.value
    return None


def _win_open_files(title: str, filter_str: str) -> List[str]:
    """Open a native Windows multi-select file dialog. Must run in a thread."""
    import ctypes
    import ctypes.wintypes

    OFN_FILEMUSTEXIST = 0x00001000
    OFN_PATHMUSTEXIST = 0x00000800
    OFN_EXPLORER = 0x00080000
    OFN_ALLOWMULTISELECT = 0x00000200
    MAX_CHARS = 32768

    class OPENFILENAME(ctypes.Structure):
        _fields_ = [
            ("lStructSize", ctypes.wintypes.DWORD),
            ("hwndOwner", ctypes.wintypes.HWND),
            ("hInstance", ctypes.wintypes.HINSTANCE),
            ("lpstrFilter", ctypes.wintypes.LPCWSTR),
            ("lpstrCustomFilter", ctypes.c_wchar_p),
            ("nMaxCustFilter", ctypes.wintypes.DWORD),
            ("nFilterIndex", ctypes.wintypes.DWORD),
            ("lpstrFile", ctypes.c_wchar_p),
            ("nMaxFile", ctypes.wintypes.DWORD),
            ("lpstrFileTitle", ctypes.c_wchar_p),
            ("nMaxFileTitle", ctypes.wintypes.DWORD),
            ("lpstrInitialDir", ctypes.wintypes.LPCWSTR),
            ("lpstrTitle", ctypes.wintypes.LPCWSTR),
            ("Flags", ctypes.wintypes.DWORD),
            ("nFileOffset", ctypes.wintypes.WORD),
            ("nFileExtension", ctypes.wintypes.WORD),
            ("lpstrDefExt", ctypes.wintypes.LPCWSTR),
            ("lCustData", ctypes.wintypes.LPARAM),
            ("lpfnHook", ctypes.c_void_p),
            ("lpTemplateName", ctypes.wintypes.LPCWSTR),
            ("pvReserved", ctypes.c_void_p),
            ("dwReserved", ctypes.wintypes.DWORD),
            ("FlagsEx", ctypes.wintypes.DWORD),
        ]

    buf = ctypes.create_unicode_buffer(MAX_CHARS)
    ofn = OPENFILENAME()
    ofn.lStructSize = ctypes.sizeof(OPENFILENAME)
    ofn.lpstrFilter = filter_str
    ofn.lpstrFile = ctypes.cast(buf, ctypes.c_wchar_p)
    ofn.nMaxFile = MAX_CHARS
    ofn.lpstrTitle = title
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_EXPLORER | OFN_ALLOWMULTISELECT

    if not ctypes.windll.comdlg32.GetOpenFileNameW(ctypes.byref(ofn)):
        return []

    raw = buf[:]
    parts = [p for p in raw.split("\x00") if p]
    if not parts:
        return []
    if len(parts) == 1:
        return [parts[0]]

    directory = parts[0]
    names = parts[1:]
    return [f"{directory}\\{name}" for name in names]


def _win_save_file(title: str, default_name: str,
                   filter_str: str) -> Optional[str]:
    """Open a native Windows file save dialog. Must run in a thread."""
    import ctypes
    import ctypes.wintypes

    OFN_OVERWRITEPROMPT = 0x00000002
    OFN_PATHMUSTEXIST = 0x00000800
    OFN_EXPLORER = 0x00080000
    MAX_PATH = 260

    class OPENFILENAME(ctypes.Structure):
        _fields_ = [
            ("lStructSize", ctypes.wintypes.DWORD),
            ("hwndOwner", ctypes.wintypes.HWND),
            ("hInstance", ctypes.wintypes.HINSTANCE),
            ("lpstrFilter", ctypes.wintypes.LPCWSTR),
            ("lpstrCustomFilter", ctypes.c_wchar_p),
            ("nMaxCustFilter", ctypes.wintypes.DWORD),
            ("nFilterIndex", ctypes.wintypes.DWORD),
            ("lpstrFile", ctypes.c_wchar_p),
            ("nMaxFile", ctypes.wintypes.DWORD),
            ("lpstrFileTitle", ctypes.c_wchar_p),
            ("nMaxFileTitle", ctypes.wintypes.DWORD),
            ("lpstrInitialDir", ctypes.wintypes.LPCWSTR),
            ("lpstrTitle", ctypes.wintypes.LPCWSTR),
            ("Flags", ctypes.wintypes.DWORD),
            ("nFileOffset", ctypes.wintypes.WORD),
            ("nFileExtension", ctypes.wintypes.WORD),
            ("lpstrDefExt", ctypes.wintypes.LPCWSTR),
            ("lCustData", ctypes.wintypes.LPARAM),
            ("lpfnHook", ctypes.c_void_p),
            ("lpTemplateName", ctypes.wintypes.LPCWSTR),
            ("pvReserved", ctypes.c_void_p),
            ("dwReserved", ctypes.wintypes.DWORD),
            ("FlagsEx", ctypes.wintypes.DWORD),
        ]

    buf = ctypes.create_unicode_buffer(MAX_PATH)
    if default_name:
        for i, c in enumerate(default_name[:MAX_PATH - 1]):
            buf[i] = c

    ofn = OPENFILENAME()
    ofn.lStructSize = ctypes.sizeof(OPENFILENAME)
    ofn.lpstrFilter = filter_str
    ofn.lpstrFile = ctypes.cast(buf, ctypes.c_wchar_p)
    ofn.nMaxFile = MAX_PATH
    ofn.lpstrTitle = title
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_EXPLORER

    if ctypes.windll.comdlg32.GetSaveFileNameW(ctypes.byref(ofn)):
        return buf.value
    return None


# Build filter strings: pairs of (description, pattern) separated by \0
def _make_filter(*pairs):
    """Create a Windows file dialog filter string.
    Usage: _make_filter("JSON files", "*.json", "All files", "*.*")
    """
    parts = list(pairs) + [""]
    return "\0".join(parts)


async def pick_file(title: str = "Select File",
                    filter_pairs=None) -> Optional[str]:
    """Open native Windows file open dialog. Returns path or None."""
    if filter_pairs is None:
        filter_pairs = ("All files", "*.*")
    filt = _make_filter(*filter_pairs)
    return await asyncio.to_thread(_win_open_file, title, filt)


async def pick_files(title: str = "Select Files", filter_pairs=None) -> List[str]:
    """Open native Windows multi-file dialog. Returns selected file paths."""
    if filter_pairs is None:
        filter_pairs = ("All files", "*.*")
    filt = _make_filter(*filter_pairs)
    return await asyncio.to_thread(_win_open_files, title, filt)


async def save_file(title: str = "Save File",
                    default_name: str = "",
                    filter_pairs=None) -> Optional[str]:
    """Open native Windows file save dialog. Returns path or None."""
    if filter_pairs is None:
        filter_pairs = ("All files", "*.*")
    filt = _make_filter(*filter_pairs)
    return await asyncio.to_thread(_win_save_file, title, default_name, filt)
