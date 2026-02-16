"""Minimal FilePicker test for flet 0.80.5"""
import flet as ft


async def main(page: ft.Page):
    page.title = "FilePicker Test v4"

    def on_result(e):
        print(f"Result: {e}")

    fp = ft.FilePicker(on_result=on_result)
    page.overlay.append(fp)
    page.update()

    btn = ft.ElevatedButton(
        "Pick File",
        on_click=lambda _: fp.pick_files(dialog_title="Test")
    )
    page.add(btn)


ft.app(target=main)
