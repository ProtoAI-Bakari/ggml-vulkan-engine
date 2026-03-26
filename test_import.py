#!/usr/bin/env python3
import sys
sys.path.insert(0, "/home/z/AGENT")
try:
    import ggml_extension
    print("Import OK")
    print("Functions:", dir(ggml_extension))
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
