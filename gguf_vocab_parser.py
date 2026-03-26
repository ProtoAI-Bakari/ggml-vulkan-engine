#!/usr/bin/env python3
"""GGUF header parser for vocab size extraction."""
import struct
from pathlib import Path


def extract_gguf_vocab_size(gguf_path: str) -> int:
    """
    Extract vocab size from GGUF file by parsing header.
    Returns -1 if not found or on error.
    """
    try:
        with open(gguf_path, "rb") as f:
            # Read first 5KB (should contain all KV pairs)
            data = f.read(5000)
            
            # GGUF magic number
            magic = data[0:4]
            if magic != b"GGUF":
                raise ValueError(f"Invalid GGUF file: missing GGUF magic (got {magic})")
            
            # Version
            version = struct.unpack("<I", data[4:8])[0]
            if version < 2:
                raise ValueError(f"Unsupported GGUF version: {version}")
            
            # KV count
            kv_count = struct.unpack("<Q", data[16:24])[0]
            
            # Search for vocab size keys (try multiple architectures)
            vocab_keys = [b"llama.vocab_size", b"qwen.vocab_size", b"gemma.vocab_size",
                         b"phi3.vocab_size", b"mistral.vocab_size", b"tokenizer.vocab_size"]
            
            for vocab_key in vocab_keys:
                idx = data.find(vocab_key)
                if idx >= 0:
                    # Parse key length (8 bytes before the key)
                    key_len = struct.unpack("<Q", data[idx-8:idx])[0]
                    
                    # Value type is after the key
                    value_type_offset = idx + key_len
                    if value_type_offset + 4 > len(data):
                        continue
                    
                    value_type = struct.unpack("<I", data[value_type_offset:value_type_offset+4])[0]
                    
                    # Value is after type
                    value_offset = value_type_offset + 4
                    
                    if value_type == 4:  # UINT32
                        return struct.unpack("<I", data[value_offset:value_offset+4])[0]
                    elif value_type == 1:  # INT32
                        return struct.unpack("<i", data[value_offset:value_offset+4])[0]
                    elif value_type == 5:  # UINT64
                        return struct.unpack("<Q", data[value_offset:value_offset+8])[0]
                    elif value_type == 6:  # UINT64 (alternate)
                        return struct.unpack("<Q", data[value_offset:value_offset+8])[0]
            
            return -1  # Not found
    except Exception as e:
        print(f"[ERROR] Failed to extract GGUF vocab size: {e}")
        return -1


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        vocab = extract_gguf_vocab_size(sys.argv[1])
        print(f"Vocab size: {vocab}")
    else:
        print("Usage: python gguf_vocab_parser.py <gguf_file>")
