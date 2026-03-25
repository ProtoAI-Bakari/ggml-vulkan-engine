# vLLM Vulkan/Asahi Linux Fix Summary

## Problem
vLLM server on Asahi Linux (Fedora aarch64, M1 Max) was crashing when sending chat completion requests due to CUDA-specific code being called on Vulkan/CPU backend.

## Root Cause
The vLLM code assumed CUDA is always available and used  functions unconditionally, which fails on:
1. Vulkan backend (no CUDA support)
2. CPU fallback (when device memory is limited)

## Fixes Applied

### 1. gpu_model_runner.py - torch.cuda.current_stream()
**File**: 

**Issue**:  called unconditionally, fails when PyTorch not compiled with CUDA.

**Fix**: Only call  when platform is not Vulkan.

### 2. gpu_model_runner.py - sampled_token_ids_cpu attribute
**Issue**:  attribute not created when Vulkan is used.

**Fix**: Always create  attribute, regardless of platform.

### 3. gpu_model_runner.py - torch.Event() creation
**Issue**:  not supported on CPU backend.

**Fix**: Only create  when platform is not Vulkan.

### 4. gpu_model_runner.py - Event recording/synchronization
**Issue**: Code tried to call  and  on None event.

**Fix**: Check if event exists before calling methods.

### 5. gpu_input_batch.py - Event assertion
**Issue**: Assertion  fails on Vulkan.

**Fix**: Remove assertion, check if event exists before synchronizing.

## Test Results
✅ Server starts successfully
✅ Health check passes (200 OK)
✅ Chat completion requests work (200 OK)
✅ Response generated: "Hello! How can I assist you today? If"

## Files Modified
1. 
2. 

## Environment
- Host: Apple Mac Studio (M1 Max, 10 Cores)
- OS: Fedora Linux Asahi aarch64
- Memory: 32 GB Unified Memory
- vLLM Version: 0.17.2.dev4+asahi
- Backend: Vulkan (with CPU fallback due to memory limits)
