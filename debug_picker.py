import flet as ft

def main(page: ft.Page):
    page.title = "FilePicker Debug"
    
    def pick_files_result(e):
        page.add(ft.Text(f"Selected files: {e.files}"))

    # Initialize without arguments
    file_picker = ft.FilePicker()
    # Assign callback property
    file_picker.on_result = pick_files_result
    
    page.overlay.append(file_picker)
    page.update()

    page.add(
        ft.ElevatedButton(
            "Upload files",
            icon=ft.Icons.UPLOAD_FILE,
            on_click=lambda _: file_picker.pick_files(allow_multiple=True),
        )
    )

if __name__ == "__main__":
    ft.app(target=main)
