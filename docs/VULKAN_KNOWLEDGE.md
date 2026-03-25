# VULKAN vLLM KNOWLEDGE BASE - DO NOT RE-DISCOVER

## COMMIT HISTORY (what was tried, what worked)
- eb73a8290: 24 TPS Native FP16 - THE WORKING BASELINE
- e2050a8da: FP16 enabled after PyTorch C++ rebuild 
- 43fcf378a: enforce-eager, full GPU residency, float16, 8GB KV cache
- 11e2215ee: 15.4 TPS Full GPU Residency
- 76c659b6d: Partial GPU/CPU inference at 4TPS (first hybrid attempt)
- ab1b257e2: 27.61 TPS (actually CPU compute with Vulkan label)

## KNOWN BUGS (already discovered, don't re-find these)
1. PyTorch Vulkan Packing.cpp:65 rejects fp16 in image layout mode
2. bfloat16 crashes Vulkan - must convert to float16 or float32
3. int64/bool tensors crash Vulkan - convert to int32
4. Descriptor pool was 1024, fixed to 65536 in custom PyTorch build
5. CpuGpuBuffer hack (gpu=cpu) forces all compute to CPU
6. Weight interceptor must cast to float32 BEFORE .to("vulkan")
7. default_unquantized_gemm had CPU fallback that killed GPU benefit

## DTYPE SHIELD (in core.py, commit e2050a8da)
- Monkey-patches torch.Tensor.to for Vulkan
- int64/bool -> int32, unknown -> float32
- BUG: bfloat16 passes through but crashes - MUST convert to float16

## WHAT ACTUALLY NEEDS TO HAPPEN
1. Fix bfloat16 in dtype shield (core.py): bfloat16 -> float16
2. Fix hybrid gemm output dtype: cast back to input dtype before return
3. The real perf win is PREFILL (batch>1) not decode (batch=1)
4. Vulkan matmul at batch=32 is 10x faster, batch=128 is 32x faster, batch=512 is 81x faster than CPU
