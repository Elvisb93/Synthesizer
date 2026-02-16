"""Minimal FilePicker test for flet 0.80.5 - using direct return values"""
import flet as ft


async def main(page: ft.Page):
    page.title = "FilePicker Test v5"
    
    fp = ft.FilePicker()
    # In 0.80.x, FilePicker is a Service - add to overlay
    page.overlay.append(fp)
    await page.update_async()

    async def pick_clicked(e):
        files = await fp.pick_files(dialog_title="Test Pick Files")
        if files:
            for f in files:
                print(f"Picked: {f.name} at {f.path}")
        else:
            print("No files selected")

    async def save_clicked(e):
        path = await fp.save_file(
            dialog_title="Test Save",
            file_name="test.txt"
        )
        if path:
            print(f"Save to: {path}")
        else:
            print("Save cancelled")

    btn_pick = ft.ElevatedButton("Pick File", on_click=pick_clicked)
    btn_save = ft.ElevatedButton("Save File", on_click=save_clicked)
    page.add(btn_pick, btn_save)


ft.app(target=main)
