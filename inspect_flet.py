import flet as ft

def main(page: ft.Page):
    print("Flet version:", ft.version)
    print("FilePicker in dir(ft):", "FilePicker" in dir(ft))
    
    try:
        picker = ft.FilePicker()
        print("FilePicker instantiated successfully")
        print("FilePicker dir:", dir(picker))
    except Exception as e:
        print("FilePicker instantiation failed:", e)

    page.add(ft.Text("Debug complete. Check console."))

if __name__ == "__main__":
    ft.app(target=main)
