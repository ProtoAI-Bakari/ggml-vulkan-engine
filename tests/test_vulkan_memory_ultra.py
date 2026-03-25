#!/usr/bin/env python3
"""W1: Vulkan Memory Capacity Test on M1 Ultra 128GB
Tests allocation using 2D tensors (Vulkan images have dimension limits).
"""
import torch
import gc
import time
import os

os.environ.setdefault("HK_SYSMEM", "112000000000")

assert torch.is_vulkan_available(), "Vulkan not available!"

def try_alloc_2d(rows, cols, label=""):
    """Allocate a 2D tensor on Vulkan."""
    size_mb = rows * cols * 4 / (1024**2)
    try:
        t0 = time.time()
        t = torch.zeros(rows, cols, dtype=torch.float32).to('vulkan')
        dt = time.time() - t0
        # Verify
        check = t.cpu()
        assert check.shape == (rows, cols)
        print(f"  [{label}] ({rows}x{cols}) = {size_mb:.1f} MB: OK ({dt:.2f}s)")
        return True, t, dt
    except Exception as e:
        print(f"  [{label}] ({rows}x{cols}) = {size_mb:.1f} MB: FAILED — {e}")
        return False, None, 0

print("=" * 60)
print("W1: VULKAN MEMORY CAPACITY TEST — M1 Ultra 128GB")
print(f"HK_SYSMEM={os.environ.get('HK_SYSMEM', 'not set')}")
print("=" * 60)

# --- Test 1: Actual Llama-8B MLP weight shapes ---
print("\n--- TEST 1: Allocate Llama-8B MLP weights (1 layer) ---")
mlp_shapes = [
    ("gate_proj", 14336, 4096),
    ("up_proj", 14336, 4096),
    ("down_proj", 4096, 14336),
]
layer_tensors = []
for name, r, c in mlp_shapes:
    ok, t, dt = try_alloc_2d(r, c, name)
    if ok:
        layer_tensors.append(t)
del layer_tensors
gc.collect()

# --- Test 2: Allocate ALL 32 layers of MLP weights ---
print("\n--- TEST 2: Allocate ALL 32 layers of 8B MLP weights ---")
all_weights = []
total_mb = 0
for layer_idx in range(32):
    layer_ok = True
    for name, r, c in mlp_shapes:
        ok, t, dt = try_alloc_2d(r, c, f"L{layer_idx}/{name}")
        if ok:
            all_weights.append(t)
            total_mb += r * c * 4 / (1024**2)
        else:
            layer_ok = False
            break
    if not layer_ok:
        print(f"  >> FAILED at layer {layer_idx}")
        break

print(f"\n>> Successfully allocated: {total_mb / 1024:.2f} GB ({len(all_weights)} tensors, {len(all_weights)//3} full layers)")
del all_weights
gc.collect()
time.sleep(1)

# --- Test 3: Max single 2D allocation (push dimensions) ---
print("\n--- TEST 3: Max single 2D Vulkan image ---")
# Vulkan images limited to ~16384 in one dimension typically
test_sizes = [
    (8192, 8192),    # 256 MB
    (16000, 8192),   # 500 MB
    (16000, 16000),  # 976 MB
    (8192, 32768),   # 1 GB — exceeds dim limit?
    (4096, 65536),   # 1 GB — different shape
    (16000, 32000),  # 1.9 GB
]
for r, c in test_sizes:
    ok, t, dt = try_alloc_2d(r, c, "max-2d")
    if ok:
        del t
        gc.collect()

# --- Test 4: Cumulative allocation with weight-sized chunks ---
print("\n--- TEST 4: Cumulative allocation (14336x4096 chunks = 224MB each) ---")
tensors = []
for i in range(150):  # up to ~33 GB
    ok, t, dt = try_alloc_2d(14336, 4096, f"chunk-{i}")
    if ok:
        tensors.append(t)
    else:
        break

total_gb = len(tensors) * 14336 * 4096 * 4 / (1024**3)
print(f"\n>> Max cumulative: {total_gb:.2f} GB ({len(tensors)} x 224MB chunks)")
del tensors
gc.collect()

# --- Summary ---
print("\n" + "=" * 60)
print("W1 SUMMARY:")
print(f"  Llama-8B MLP weights (32 layers, float32): 21.00 GB")
print(f"  Llama-8B attention weights (32 layers, float32): 5.00 GB")
print(f"  Llama-8B total on Vulkan: 26.00 GB")
print("=" * 60)
print("W1 COMPLETE")
