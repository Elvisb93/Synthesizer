import flet as ft
from core.models import ColumnDefinition, ColumnType, ColumnConstraints

class ColumnControl(ft.Card):
    def __init__(self, app, index: int, on_remove, col_def: ColumnDefinition = None):
        super().__init__()
        # self.app = app # Removed to prevent serialization error
        self.index = index
        self.on_remove = on_remove
        self.col_def = col_def
        
        # Initialize UI Components
        self.name_field = ft.TextField(
            label="Column Name", 
            value=f"col_{index+1}" if not col_def else col_def.name,
            width=180, 
            dense=True
        )
        self.type_dropdown = ft.Dropdown(
            label="Type",
            options=[ft.dropdown.Option(t.value) for t in ColumnType],
            value=ColumnType.SHORT_TEXT.value if not col_def else col_def.type.value,
            width=140,
            dense=True,

        )
        self.type_dropdown.on_change = self._on_type_change
        self.prompt_field = ft.TextField(
            label="Instruction / Prompt", 
            value=col_def.prompt_instruction if col_def else "",
            hint_text="e.g. 'Generate a valid US phone number'",
            expand=True,
            dense=True
        )
        self.remove_btn = ft.IconButton(
            icon=ft.Icons.DELETE, 
            icon_color=ft.Colors.RED_400, 
            tooltip="Remove Column",
            on_click=lambda e: self.on_remove(self)
        )
        
        # Advanced Options
        self.advanced_visible = False
        constraints = col_def.constraints if col_def else ColumnConstraints()
        
        self.regex_field = ft.TextField(label="Regex Pattern", value=constraints.regex_pattern or "", hint_text="e.g. ^[A-Z]{3}-\d{4}$", visible=False, dense=True)
        self.min_len_field = ft.TextField(label="Min Len", value=str(constraints.min_length), width=80, visible=False, dense=True)
        self.max_len_field = ft.TextField(label="Max Len", value=str(constraints.max_length), width=80, visible=False, dense=True)
        self.min_val_field = ft.TextField(label="Min Val", value=str(constraints.min_value) if constraints.min_value is not None else "", width=80, visible=False, dense=True)
        self.max_val_field = ft.TextField(label="Max Val", value=str(constraints.max_value) if constraints.max_value is not None else "", width=80, visible=False, dense=True)
        self.options_field = ft.TextField(label="Options (comma-separated)", value=",".join([str(x) for x in constraints.options]) if constraints.options else "", hint_text="Red, Green, Blue", visible=False, dense=True)
        self.allow_dups_cb = ft.Checkbox(label="Allow Duplicates", value=constraints.allow_duplicates, visible=False)

        self.advanced_content = ft.Column([
            ft.Row([self.regex_field], visible=False),
            ft.Row([self.options_field], visible=False),
            ft.Row([self.min_len_field, self.max_len_field], visible=False),
            ft.Row([self.min_val_field, self.max_val_field], visible=False),
            ft.Row([self.allow_dups_cb], visible=False)
        ])
        
        self.advanced_toggle = ft.TextButton("Show Advanced", on_click=self._toggle_advanced)

        self.content = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Text(f"#{index+1}", color=ft.Colors.GREY_500),
                    self.name_field,
                    self.type_dropdown,
                    self.prompt_field,
                    self.remove_btn
                ], alignment=ft.MainAxisAlignment.START),
                self.advanced_toggle,
                self.advanced_content
            ]),
            padding=10
        )
        
        # Initial visibility update
        self._update_visibility()

    def _toggle_advanced(self, e):
        self.advanced_visible = not self.advanced_visible
        self.advanced_toggle.text = "Hide Advanced" if self.advanced_visible else "Show Advanced"
        self._update_visibility()
        self.update()

    def _on_type_change(self, e):
        self._update_visibility()
        self.update()

    def _update_visibility(self):
        val = self.type_dropdown.value
        is_adv = self.advanced_visible

        is_text = val in [ColumnType.SHORT_TEXT.value, ColumnType.LONG_TEXT.value]
        self.regex_field.visible = is_adv and is_text
        self.min_len_field.visible = is_adv and is_text
        self.max_len_field.visible = is_adv and is_text
        
        self.options_field.visible = is_adv and (val == ColumnType.CATEGORICAL.value)
        
        is_numeric = str(val) == "Numeric" or val == ColumnType.NUMERIC.value
        self.min_val_field.visible = is_adv and is_numeric
        self.max_val_field.visible = is_adv and is_numeric
        
        self.allow_dups_cb.visible = is_adv

        # Update row visibility
        self.advanced_content.controls[0].visible = self.regex_field.visible
        self.advanced_content.controls[1].visible = self.options_field.visible
        self.advanced_content.controls[2].visible = any([self.min_len_field.visible, self.max_len_field.visible])
        self.advanced_content.controls[3].visible = any([self.min_val_field.visible, self.max_val_field.visible])
        self.advanced_content.controls[4].visible = self.allow_dups_cb.visible

    def get_definition(self) -> ColumnDefinition:
        col_type = ColumnType(self.type_dropdown.value)
        
        # Build constraints
        constraints = ColumnConstraints(
            regex_pattern = self.regex_field.value if self.regex_field.visible else None,
            min_length = int(self.min_len_field.value) if self.min_len_field.value and self.min_len_field.visible else 10,
            max_length = int(self.max_len_field.value) if self.max_len_field.value and self.max_len_field.visible else 2000,
            min_value = float(self.min_val_field.value) if self.min_val_field.value and self.min_val_field.visible else None,
            max_value = float(self.max_val_field.value) if self.max_val_field.value and self.max_val_field.visible else None,
            options = [x.strip() for x in self.options_field.value.split(",") if x.strip()] if self.options_field.value and self.options_field.visible else [],
            allow_duplicates = self.allow_dups_cb.value
        )

        return ColumnDefinition(
            name=self.name_field.value,
            type=col_type,
            prompt_instruction=self.prompt_field.value,
            constraints=constraints
        )
