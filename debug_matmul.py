#!/usr/bin/env python3
"""
Debug matmul mismatch - inspect actual values
"""
import time
import ctypes
import numpy as np
import os
import sys

sys.path.insert(0, os.path.expanduser("~/AGENT"))
from ggml_vulkan_engine import GgmlVulkanEngine

print("="*70)
print("DEBUG: Matmul Mismatch Investigation")
print("="*70)

engine = GgmlVulkanEngine()

# Simple test case
N, K = 256, 1536
W = np.random.randn(N, K).astype(np.float32)
engine.cache_weight("test", W)

M = 1
X = np.random.randn(M, K).astype(np.float32)

print(f"\nTest: matmul({M}x{K}, {K}x{N}) -> {M}x{N}")
print(f"X shape: {X.shape}, W shape: {W.shape}")

# CPU reference
expected = X @ W.T
print(f"\nCPU expected (first 5 values): {expected[0, :5]}")

# Vulkan result
result = engine.matmul("test", X)
print(f"Vulkan result (first 5 values): {result[0, :5]}")

# Detailed comparison
print(f"\nMax absolute difference: {np.max(np.abs(expected - result)):.6e}")
print(f"Mean absolute difference: {np.mean(np.abs(expected - result)):.6e}")
print(f"Cosine similarity: {np.dot(expected.flatten(), result.flatten()) / (np.linalg.norm(expected.flatten()) * np.linalg.norm(result.flatten()) + 1e-10):.10f}")

# Check if it's a scaling issue
ratio = result / (expected + 1e-10)
print(f"\nRatio result/expected (first 5): {ratio[0, :5]}")
print(f"Ratio mean: {np.mean(ratio):.6f}, std: {np.std(ratio):.6f}")

# Check if it's a transposition issue
expected_T = X.T @ W.T  # Wrong but check
print(f"\nIf transposed wrong (X.T @ W.T): {expected_T[0, :5]}")

# Check if it's a row/col order issue
expected_swap = W @ X.T  # Wrong
print(f"If swapped (W @ X.T): {expected_swap[:5, 0]}")

engine.close()

print("\n" + "="*70)
print("CONCLUSION: Check if weights are stored transposed")
print("="*70)
