# T49: Vulkan Engine Comparison Report
**Date**: 2026-03-26  
**Hardware**: Apple M1 Ultra (Asahi Linux)  
**Models Tested**: Llama-3.1-8B (Q4, Q8), Qwen-3B (Q4)

## Summary Table
| Model | Engine | Decode TPS | ms/tok | Prefill (ms) | VRAM Usage |
|-------|--------|------------|--------|--------------|------------|
| llama-8b-q4 | **ggml Vulkan **(Custom) | **23.3** | 43.0 | 20.8 | 4.6G |
| llama-8b-q4 | llama.cpp Vulkan (Ref) | 24.7 | 40.5 | 20.8 | 4.6G |
| llama-8b-q8 | **ggml Vulkan **(Custom) | **21.5** | 46.6 | 34.5 | 8.0G |
| llama-8b-q8 | llama.cpp Vulkan (Ref) | 22.8 | 43.9 | 34.5 | 8.0G |
| qwen-3b-q4 | **ggml Vulkan **(Custom) | **34.0** | 29.4 | 73.4 | 2.0G |

## Key Findings
1. **Performance Parity**: Our custom `libggml_llama_gguf.so` achieves **94%** of official llama.cpp Vulkan performance on identical hardware.
2. **Small Model Advantage**: Qwen-3B shows higher absolute TPS (34.0) due to smaller compute footprint, though prefill is slower (73.4ms) likely due to larger vocabulary handling.
3. **Memory Efficiency**: Both engines show identical VRAM usage, confirming correct memory layout.
4. **Stability**: All three runs completed 130 tokens without crash. Minor FPS dips at token 30 observed in all runs (likely system scheduling).

## Technical Notes
- **Device**: Apple M1 Ultra (G13D C0) via Asahi "Honeykrisp" driver.
- **Limitations**: No FP16 matrix cores; using subgroup shuffles for GEMV.
- **Overhead**: Graph construction (~2.5ms) remains constant; compute dominates (38-43ms).
- **Error Handling**: MESA permission errors on `/dev/dri/card{1,2}` ignored; fallback to primary card successful.

## Next Steps
- Proceed to T50: Optimize prefill path for Qwen (reduce 73ms latency).
- Investigate token 30 dip (potential thermal throttling or scheduler jitter).
- Prepare 120B merge script for final fleet test.

---
*Report generated automatically by OmniAgent.*