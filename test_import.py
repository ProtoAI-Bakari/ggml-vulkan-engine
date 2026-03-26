#!/usr/bin/env python3
import sys
sys.path.insert(0, "/home/z/AGENT")
try:
    import ggml_extension
    print("Import successful!")
    print("Available functions:", [x for x in dir(ggml_extension) if not x.startswith("_")])
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
