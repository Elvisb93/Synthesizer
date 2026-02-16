import flet as ft

def main(page: ft.Page):
    page.title = "Flet 0.28.3 FilePicker Debug"
    page.add(ft.Text(f"Running Flet version: {ft.version}"))
    
    def pick_files_result(e: ft.FilePickerResultEvent):
        page.add(ft.Text(f"Selected files: {e.files}"))

    # Standard initialization for 0.28.3
    try:
        file_picker = ft.FilePicker(on_result=pick_files_result)
        page.overlay.append(file_picker)
        page.update()
        
        page.add(
            ft.ElevatedButton(
                "Upload files (Standard API)",
                icon=ft.Icons.UPLOAD_FILE,
                on_click=lambda _: file_picker.pick_files(allow_multiple=True),
            )
        )
    except Exception as e:
        page.add(ft.Text(f"Error initializing FilePicker: {e}", color=ft.Colors.RED))

if __name__ == "__main__":
    ft.app(target=main)
