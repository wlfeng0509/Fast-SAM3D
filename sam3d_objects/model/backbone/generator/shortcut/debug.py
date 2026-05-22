# debug.py
try:
    from sam3d_objects.model.backbone.generator.shortcut.model import ShortCut_taylorseer
    print("Class found successfully.")
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"File contains an internal error: {e}")