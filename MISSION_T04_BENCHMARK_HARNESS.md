# Mission: T04 - Benchmarking Harness
# Agent: Sub-Agent (Parallel Worker)
# Priority: HIGH
# Time Budget: 4 hours

---

## OBJECTIVE
Create automated benchmarking harness with reproducible measurements for Vulkan LLM inference.

## SUCCESS CRITERIA
- [ ] Script runs full benchmark suite with one command
- [ ] Measures: TPS (p50/p99), TTFT, total latency, GPU utilization
- [ ] Uses Vulkan timestamp queries if available
- [ ] Outputs CSV with reproducible results
- [ ] Achieves 22+ TPS baseline on 8B Q4_K_M

## WORKING DIRECTORY
**CRITICAL:** Work in `/home/z/AGENT` and `/home/z/GITDEV/vllm-vulkan`

## KEY FILES TO READ FIRST
1. `/home/z/AGENT/ggml_vulkan_engine.py` — Main Vulkan engine
2. `/home/z/AGENT/ggml_server.py` — HTTP server (if exists)
3. `/home/z/AGENT/benchmark_all.py` — Existing benchmark script
4. `/home/z/GITDEV/vllm-vulkan/vllm/platforms/vulkan.py` — Vulkan platform layer

## TASKS
1. **Explore codebase first:**
   ```bash
   cd /home/z/AGENT
   ls -la *.py
   cat benchmark_all.py  # Read existing benchmarks
   ```

2. **Create `benchmark_vulkan.py` with:**
   - Warmup phase (10 tokens)
   - Measurement phase (100 tokens, 10 runs)
   - Statistics: mean, p50, p99, std dev
   - GPU utilization via vulkaninfo or custom shader
   - Import from `ggml_vulkan_engine.py`

3. **Create `run_benchmarks.sh` wrapper:**
   - Tests all models: 0.5B, 1.5B, 3B, 8B, 32B
   - Tests all quants: Q4_K_M, Q8_0, F16
   - Tests batch sizes: 1, 4, 8, 16, 32
   - Outputs `benchmark_results.csv`

4. **Integrate with existing `ggml_server.py` or `stream_server.py`**

5. **Add Vulkan timestamp query support:**
   - Query GPU timestamps before/after dispatch
   - Measure actual GPU execution time vs wall clock

## DELIVERABLES
- `benchmark_vulkan.py` — Python benchmark script
- `run_benchmarks.sh` — Shell wrapper
- `benchmark_results.csv` — Output file
- `~/AGENT/LOGS/benchmark_*.log` — Detailed logs

## NOTES
- Use `time.perf_counter()` for wall clock
- Use Vulkan `vkGetQueryPoolResults()` for GPU timestamps
- Run on Sys0 (M1 Ultra) for consistency
- Compare against llama.cpp baseline
- Read existing `benchmark_all.py` to understand current approach

## COORDINATION
- Read `agent-comms-bridge.md` before starting
- Append results to bridge after completion
- Don't conflict with LEAD agent's work
