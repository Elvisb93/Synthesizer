# Flet Development Guidelines

This document outlines best practices and common pitfalls when developing UI components with Flet in this project. Reference this when building or refactoring UI elements.

## 1. Safety Checks for UI Updates

**Problem:** Calling `.update()` on a control that has not yet been added to the `page` raises a `RuntimeError`.
**Context:** This often happens in `__init__` methods or event handlers that might trigger before the component is fully mounted.

**Best Practice:**
Always wrap update calls in initialization logic or unsure contexts with a try-except block, or check for page presence (though `try-except` is more robust across versions).

```python
# BAD
self.my_textfield.update() 

# GOOD
try:
    self.my_textfield.update()
except RuntimeError:
    # Control is not yet on the page, ignore
    pass
```

## 2. Event Handler Assignment

**Problem:** In some Flet versions, passing `on_change` (or other events) directly to the constructor of complex controls like `Dropdown` can sometimes cause issues or `TypeError` if the internal init signature doesn't match perfectly or if there are circular dependencies during init.

**Best Practice:**
Assign event handlers *after* initializing the control.

```python
# SAFER PATTERN
self.my_dropdown = ft.Dropdown(
    label="Choose Option",
    options=[...],
    value="A"
)
# Assign callback after creation
self.my_dropdown.on_change = self._on_change_handler
```

## 3. Thread Safety

**Problem:** Flet runs the GUI in a main thread. Background threads (like data generation) cannot directly update UI controls safely without `page.update()` being called in a way that respects the loop, but usually `control.update()` is thread-safe *if* the control is mounted.
**Best Practice:**

- Update data models in background threads.
- Use `queue` or similar mechanisms to pass messages to the main thread if complex UI reconstruction is needed.
- For simple property updates (text value, progress bar), direct `control.update()` from a thread works, provided the control is mounted.

## 4. Control Visibility

**Problem:** Toggling `visible` property requires an `update()` to take effect.
**Best Practice:**
Group visibility changes and call `page.update()` once if multiple controls are changing, rather than updating each individually, to reduce flicker and IPC overhead.

```python
self.loading_spinner.visible = True
self.start_btn.disabled = True
self.page.update() # Single update
```
