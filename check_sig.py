import flet as ft
import inspect

def main(page: ft.Page):
    print("Flet version:", ft.version)
    sig = inspect.signature(ft.FilePicker.__init__)
    print("FilePicker init signature:", sig)
    
    try:
        # Try initializing without args
        fp = ft.FilePicker()
        print("Success: FilePicker()")
    except Exception as e:
        print("Error: FilePicker() failed:", e)

    page.add(ft.Text("Check console for signature info"))

if __name__ == "__main__":
    ft.app(target=main)
