import flet as ft

def main(page: ft.Page):
    print("Full dir(ft.FilePicker):")
    print(dir(ft.FilePicker))
    
    fp = ft.FilePicker()
    print("\nFull dir(instance):")
    print(dir(fp))
    
    # Check for likely candidates
    candidates = [a for a in dir(fp) if "on_" in a]
    print("\nEvent handlers:", candidates)

if __name__ == "__main__":
    ft.app(target=main)
