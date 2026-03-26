#!/usr/bin/env python3
"""Debug GGUF header to see all keys."""
import struct

def dump_gguf_kv(gguf_path: str, limit=50):
    """Dump all KV pairs from GGUF header."""
    with open(gguf_path, "rb") as f:
        magic = f.read(4)
        print(f"Magic: {magic}")
        version = struct.unpack("<I", f.read(4))[0]
        print(f"Version: {version}")
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        kv_count = struct.unpack("<Q", f.read(8))[0]
        print(f"Tensors: {tensor_count}, KV pairs: {kv_count}")
        
        for i in range(kv_count):
            key_len = struct.unpack("<Q", f.read(8))[0]
            key = f.read(key_len).decode("utf-8")
            value_type = struct.unpack("<I", f.read(4))[0]
            
            # Try to read value for display
            if value_type == 11:  # STRING
                val_len = struct.unpack("<Q", f.read(8))[0]
                val = f.read(val_len).decode("utf-8", errors='ignore')
                print(f"  {i}: {key} = {val[:100]}")
            elif value_type == 8:  # FLOAT32
                val = struct.unpack("<f", f.read(4))[0]
                print(f"  {i}: {key} = {val}")
            elif value_type == 4:  # UINT32
                val = struct.unpack("<I", f.read(4))[0]
                print(f"  {i}: {key} = {val}")
            elif value_type == 6:  # UINT64
                val = struct.unpack("<Q", f.read(8))[0]
                print(f"  {i}: {key} = {val}")
            elif value_type == 1:  # INT32
                val = struct.unpack("<i", f.read(4))[0]
                print(f"  {i}: {key} = {val}")
            else:
                print(f"  {i}: {key} (type={value_type}) - SKIPPING")
                # Skip based on type
                if value_type == 0: f.read(1)
                elif value_type == 1: f.read(1)
                elif value_type == 2: f.read(2)
                elif value_type == 3: f.read(2)
                elif value_type == 4: f.read(4)
                elif value_type == 5: f.read(8)
                elif value_type == 6: f.read(8)
                elif value_type == 7: f.read(8)
                elif value_type == 8: f.read(4)
                elif value_type == 9: f.read(8)
                elif value_type == 10: f.read(1)
                elif value_type == 11: f.read(struct.unpack("<Q", f.read(8))[0])
                elif value_type == 12:
                    array_type = struct.unpack("<I", f.read(4))[0]
                    count = struct.unpack("<Q", f.read(8))[0]
                    print(f"      Array: {count} items, type={array_type}")
            
            if i >= limit:
                print(f"  ... (showing first {limit+1} of {kv_count})")
                break

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        dump_gguf_kv(sys.argv[1])
    else:
        print("Usage: python debug_gguf.py <gguf_file>")
