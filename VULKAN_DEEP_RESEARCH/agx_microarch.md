# Apple AGX GPU Microarchitecture

## Sources
- philipturner/metal-benchmarks (GitHub)
- dougallj.github.io/applegpu/docs.html (G13 ISA reference)
- Alyssa Rosenzweig's Asahi GPU blog series
- Wikipedia Apple M1/M1 Max/M1 Ultra
- Chips and Cheese iGPU analysis

---

## GPU Core Architecture

### M1 (Base)
- **GPU Cores**: 8 (7 active in base M1)
- **Execution Units per Core**: 16 EUs
- **ALUs per EU**: 8
- **Total ALUs**: 128 EUs x 8 ALUs = 1024 ALUs
- **SIMD Width**: 32 threads per SIMD-group
- **Clock Speed**: 1.278 GHz
- **FP32 Performance**: ~2.6 TFLOPS

### M1 Max
- **GPU Cores**: 32 (24 in lower-bin)
- **Total ALUs**: ~4096
- **Memory Bandwidth**: 400 GB/s (LPDDR5, 512-bit bus)
- **SLC (System Level Cache)**: 48 MB

### M1 Ultra
- **GPU Cores**: 64 (48 in lower-bin)
- **Total ALUs**: ~8192
- **Memory Bandwidth**: 800 GB/s (two M1 Max dies via UltraFusion)
- **SLC**: 96 MB
- **Architecture**: Two M1 Max dies interconnected, appears as single GPU

---

## Per-Core Resources

### Register File
- **Size**: ~208 KB per core
- **Registers per SIMD-group**: Up to 128 general-purpose 32-bit registers (r0-r127)
- **Also available as**: 16-bit halves (r0l, r0h, etc.)
- **Uniform Registers**: 256 x 32-bit (u0-u255), shared across SIMD-group
- **Tradeoff**: More registers per thread = fewer concurrent threads (occupancy)

### Threadgroup Memory (Shared Memory)
- **Size**: ~60 KB per core (exact allocation depends on tiles/compute)
- **Note**: On compute kernels, tile memory is repurposed as threadgroup memory
- **Bank size**: 4 bytes (standard)
- **Bank count**: Unknown (not yet benchmarked; competitors typically use 32 banks)
- **Key limitation**: Unknown bank count means bank conflict avoidance must be empirical

### L1 Data Cache
- **Size**: 8 KB per core (very small!)
- **Bandwidth**: 64 bytes/cycle on-core data
- **Implication**: L1 is essentially a filter; real working set must fit in registers or threadgroup memory

### Instruction Cache
- **Size**: 12 KB per core

### Occupancy
- **Maximum threads per core**: 384 to 3072 (depending on register pressure)
- **Maximum simultaneous threads (whole GPU, M1)**: 24,576
- **Threadgroups per GPU**: Up to 24 executing in parallel (with full register files)

---

## Cache Hierarchy

### L2 Cache (Per-GPU)
- **M1**: 768 KB (shared across all GPU cores)
- **Bandwidth**: ~32 bytes/cycle per on-GPU data

### System Level Cache (SLC)
- **M1**: 8-16 MB (shared with CPU and other accelerators)
- **M1 Max**: 48 MB
- **M1 Ultra**: 96 MB
- **Note**: SLC is the last-level cache before DRAM. It's shared by CPU, GPU, Neural Engine, etc.

### Memory Bandwidth to DRAM
- **M1**: ~68 GB/s (128-bit LPDDR4X)
- **M1 Pro**: ~200 GB/s (256-bit LPDDR5)
- **M1 Max**: ~400 GB/s (512-bit LPDDR5)
- **M1 Ultra**: ~800 GB/s (1024-bit effective via UltraFusion)

### Per-Core Memory Bandwidth
- **M1 Max**: 400 GB/s / 32 cores = ~12.5 GB/s per core
- **M1 Ultra**: 800 GB/s / 64 cores = ~12.5 GB/s per core
- **This is the key bottleneck for GEMV** (batch=1 is bandwidth-bound)

---

## Compute Shader Specifics

### SIMD-Group Structure
- 32 threads execute in lockstep
- Per-thread execution mask (exec_mask, 32-bit) for divergent control flow
- Execution mask stack for nested conditionals (stored in r0l)
- Quad-group operations available (4-thread scope)

### Instruction Set Highlights (G13)
- **FMA**: Fused multiply-add in both f32 and f16
- **Transcendentals**: sin, cos, log2, exp2, rsqrt (hardware)
- **Integer**: Full 32-bit multiply-add with saturation
- **Bitfield**: Insert, extract, reverse -- useful for quant dequantization
- **SIMD operations**: shuffle, ballot, broadcast within SIMD-group
- **Variable-length encoding**: 2-12 bytes per instruction

### simdgroup_matrix (Metal Only)
- **Matrix size**: 8x8 elements
- **Hardware**: Uses existing FMA units but with optimized data routing
- **Each thread holds 2 elements** of the 8x8 matrix (64 elements / 32 threads)
- **NOT available via Vulkan** (no VK_KHR_cooperative_matrix on Honeykrisp)

---

## Optimal Workgroup Sizes for Compute

### General Recommendations
- **Must be multiple of 32** (SIMD-group width)
- **64-256 threads** is the sweet spot for most compute kernels
- **128 threads = 4 SIMD-groups** is often optimal: enough for latency hiding, not so many that register pressure kills occupancy
- **512+ threads**: Risk of register spilling; only if each thread needs few registers

### For GEMM (Matmul)
- llama.cpp uses **128 threads** for coopmat path, **256 threads** for scalar path
- On AGX, **128 threads** (4 SIMD-groups) is recommended because:
  - 4 SIMD-groups can keep ALUs busy during memory latency
  - Leaves room for register allocation
  - Fits threadgroup memory budget

### For GEMV (Batch=1)
- llama.cpp uses **32-256 threads** depending on quantization type
- Q4_K uses 16 threads per quant block, so workgroup size should be multiple of 16
- **Reduction pattern**: Subgroup-level reduction is fast; minimize shared memory reductions

---

## Known Performance Cliffs and Gotchas

### 1. Register Pressure
More registers per thread = fewer concurrent threads. Apple GPUs have no separate register file per EU; it's shared across the core. Going from 32 registers/thread to 64 registers/thread can halve occupancy.

### 2. Tiny L1 Cache (8 KB)
The L1 is small enough that most global memory accesses go to L2 or SLC. Don't rely on temporal locality for repeated global loads -- use threadgroup memory instead.

### 3. Threadgroup Memory Banks
Bank count is unknown. The standard workaround is to add padding (e.g., +1 or +4 elements per row) to shared memory arrays, as llama.cpp and MLX both do.

### 4. f16 vs f32 ALU Throughput
The GPU is scalar at all precisions, but 16-bit arithmetic should have ~2x throughput due to:
- Reduced register usage (higher occupancy)
- Potentially superscalar 16-bit execution paths

### 5. Memory Coalescing
AGX does support coalesced memory access (adjacent threads accessing adjacent memory). Non-coalesced access patterns will serialize, dramatically reducing bandwidth.

### 6. Barrier Cost
Threadgroup barriers are relatively expensive on AGX. Minimize the number of barriers per iteration. The MLX pattern of two barriers per K-iteration is about as good as it gets without double buffering.

### 7. Unified Memory
The "GPU" and "CPU" share the same physical DRAM. There is no explicit device<->host copy cost, but cache coherence must be managed. For Vulkan on Asahi, this means `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` and `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` can refer to the same physical memory.

### 8. SLC Contention
The SLC is shared across all accelerators. If the CPU is also doing heavy memory work (e.g., tokenization, sampling), SLC capacity for GPU work is reduced. Keep CPU-side work minimal during GPU inference.

---

## Theoretical Performance Bounds

### Compute-Bound (GEMM)
- M1 Ultra FP32: ~8192 ALUs * 1.278 GHz * 2 ops/FMA = ~20.9 TFLOPS
- M1 Ultra FP16: Estimated ~41.8 TFLOPS (2x f32 due to superscalar)
- For 4-bit quant GEMM: Dequant overhead reduces effective throughput

### Memory-Bound (GEMV at Batch=1)
- M1 Ultra: 800 GB/s bandwidth
- 8B model Q4_K_M weights: ~4.5 GB
- **Theoretical minimum**: 4.5 GB / 800 GB/s = 5.6 ms per full model pass
- **Theoretical max TPS**: ~178 tokens/sec (if perfectly bandwidth-limited)
- **Practical limit**: ~60-80% bandwidth utilization = 107-142 TPS

### Current Achievement: 24.8 TPS on 8B Q4_K_M via Vulkan
- Implies we're using about 14% of theoretical bandwidth capacity
- Or the bottleneck is elsewhere (pipeline overhead, synchronization, partial GPU utilization)
