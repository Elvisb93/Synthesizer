import flet as ft

def main(page: ft.Page):
    def pick_files_result(e):
        print("Selected files:", e.files)

    pick_files_dialog = ft.FilePicker()
    pick_files_dialog.on_result = pick_files_result
    page.overlay.append(pick_files_dialog)

    page.add(
        ft.ElevatedButton(
            "Pick files",
            icon=ft.Icons.UPLOAD_FILE,
            on_click=lambda _: pick_files_dialog.pick_files(allow_multiple=True),
        )
    )

ft.app(target=main)
