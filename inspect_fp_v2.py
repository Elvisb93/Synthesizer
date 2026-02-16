"""Inspect FilePicker API in flet 0.80.5"""
import flet as ft
import inspect

fp = ft.FilePicker()
print("Type:", type(fp))
print()

# Check init signature
print("__init__ signature:", inspect.signature(ft.FilePicker.__init__))
print()

# All public attrs
attrs = [m for m in dir(fp) if not m.startswith('_')]
print("Public attributes:", attrs)
print()

# Check if on_result exists as a settable property
print("Has on_result attr:", hasattr(fp, 'on_result'))
if hasattr(fp, 'on_result'):
    print("on_result value:", fp.on_result)
    print("on_result type:", type(type(fp).__dict__.get('on_result', None)))

# Check pick_files signature  
print()
print("pick_files signature:", inspect.signature(fp.pick_files))
print("save_file signature:", inspect.signature(fp.save_file))

# Check if it's async
print()
print("pick_files is coroutine?", inspect.iscoroutinefunction(fp.pick_files))
print("save_file is coroutine?", inspect.iscoroutinefunction(fp.save_file))

# Check the base control name
print()
if hasattr(fp, '_get_control_name'):
    print("Control name:", fp._get_control_name())
elif hasattr(fp, '_control_name'):
    print("Control name:", fp._control_name)
