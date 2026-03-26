# MASTER TASK QUEUE v5
# Updated: 2026-03-26 03:40
#
# ┌─────────────────────────────────────────┐
# │ DONE:  40 | IN_PROGRESS:  7 | READY: 38 │
# │ Total:  85 tasks across 9 phases       │
# └─────────────────────────────────────────┘
#
# PHASE 0: Stability + Measurement     [T01-T10]  — 10/10 DONE
# PHASE 1: GGML Graph + CB Optimization [T11-T25] — 14/15 DONE
# PHASE 2: vLLM Production Integration  [T26-T48] — 14/23 DONE
# PHASE 3: 120B Model + Fleet           [T49-T55] —  2/7  DONE
# PHASE 4: Pure Vulkan Engine           [T56-T65] —  1/10 DONE
# PHASE 5: Validation + Hardening       [T66-T70] —  0/5  READY
# PHASE 6: Documentation + Upstream     [T71-T76] —  0/6  READY
# PHASE 7: Advanced Optimization        [T77-T80] —  0/4  READY
# PHASE 8: Agentic Framework            [T81-T85] —  0/5  READY
#
# RULES:
#   - sys1 (Asahi M1 Ultra) = TEST NODE. Only sys1 runs tests locally.
#   - Remote agents (sys2-sys7) write code + push_changes to sys1
#   - [TEST] tasks are picked up ONLY by agent0 on sys1
#   - After EACH task: push_changes + complete_task
#   - If stuck >3 failures: ask_claude for help
#   - Use ~ for paths, never /home/z or /Users/z
#   - DEPENDENCIES: T58→T57, T63→T60, T61→T59, T65→T60

## PHASE 0: STABILITY + MEASUREMENT [DO FIRST — nothing else matters if broken]

### T01: [DONE] Verify standalone engine: 12+ sequential requests, all coherent
- Run 12 diverse prompts through GgmlLLM.generate() 
- Include: math, factual, creative, long-gen, multi-sentence
- Success: 12/12 coherent, no crash, consistent TPS
- Time: 1h

### T02: [DONE by OmniAgent [sys1-Main] Fix standalone streaming server startup + stability
- Server must start in <30s, serve on 0.0.0.0:8080
- ThreadingMixIn for concurrent HTTP (sequential engine underneath)
- KV reset between every request (engine_reset_kv)
- Success: 50 sequential HTTP requests without crash, all coherent
- Time: 2h

### T03: [DONE] Coherency blast test: 50 diverse prompts through HTTP server
- Math: 2+2, complex arithmetic, word problems
- Factual: capitals, dates, science, history
- Creative: stories, poetry, code generation
- Edge: 1-word prompt, 500-word prompt, unicode, empty follow-up
- Success: 48/50+ coherent (allow 2 model-quality misses, 0 crashes)
- Time: 2h

### T04: [DONE] Set up benchmarking harness with reproducible measurements
- ✅ Created benchmark_vulkan.py with Vulkan timestamp queries
- ✅ Created run_benchmarks.sh automation script
- ✅ Measures: TPS (p50/p99), TTFT, total latency, GPU utilization
- ✅ Outputs: CSV and JSON results with full statistics
- ✅ Supports: single model, all models, custom configurations
- Success: reproducible 22+ TPS baseline on 8B Q4_K_M
- Time: 4h
- Files: benchmark_vulkan.py, run_benchmarks.sh

### T05: [DONE] Profile CPU time breakdown with py-spy + custom instrumentation
- Confirm: ~6ms CB recording, ~4ms graph build, ~3ms Python overhead
- Instrument ggml_vk_dispatch_pipeline, CB begin/end, queue submit, fence wait
- Success: flamegraph + per-stage timing CSV matching OpR's predicted breakdown
- Time: 3h

### T06: [DONE] Document ggml compute graph: node count, op types, tensor shapes
- Dump graph at position 1, 100, 500 for Llama-3.1-8B decode
- Count: how many nodes are views/casts vs real compute?
- Success: graph topology document with node breakdown
- Time: 2h

### T07: [DONE] Verify Honeykrisp capabilities matrix
- vulkaninfo: all extensions, subgroup properties, memory types, queue families
- Confirm: subgroupSize=32, shaderFloat16Int8, integerDotProduct
- Confirm: NO cooperative_matrix, NO tensor cores
- Success: capability matrix document
- Time: 1h

### T08: [DONE] Run llama-bench on ALL models for reference baseline
- 8B Q4_K_M, Q8_0, F16
- 0.5B, 1.5B, 3B, 32B Q4_K_M
- 120B mxfp4 (after merge)
- Success: llama.cpp baseline TPS table for every model
- Time: 2h

### T09: [DONE] Create automated regression test: golden output comparison
- 10 prompts with expected output patterns (regex or substring match)
- Run before AND after every code change
- Success: CI-ready test script that catches coherency regressions
- Time: 3h

### T10: [DONE] Test 120B model on llama-bench (THE big number)
- Merge shards first (llama-gguf-split --merge)
- Run: llama-bench -m 120b-merged.gguf -ngl 99 -t 4 -p 64 -n 32
- If OOM: try with HK_SYSMEM=112000000000
- Success: baseline TPS for 120B on Vulkan Asahi (world first if it runs)
- Time: 3h

## PHASE 1: GGML GRAPH + CB OPTIMIZATION [Perspective A — 23→30-33 TPS]

### T11: [DONE] Implement graph topology fingerprinting
- Hash: node count + op types + tensor shapes
- Detect when graph is unchanged between tokens (99%+ of decode steps)
- Success: fingerprint matches consecutive decode tokens
- Time: 4h

### T12: [DONE] Add graph caching via ggml_gallocr (PR #20927 pattern)
- ggml_gallocr_reserve() at init with worst-case graph
- ggml_gallocr_alloc_graph() each token — near-no-op when topology matches
- Success: graph alloc time drops from 4ms to <0.5ms
- Time: 6h

### T13: [DONE] Implement command buffer template recording
- Record full pipeline once on first execution
- Use vkCmdPushConstants for dynamic params: KV offset, seq_len, position
- vkResetCommandPool for pool-level reset (not per-buffer)
- Success: CB recording drops from 6ms to <1ms for stable graphs
- Time: 12h

### T14: [DONE] Add CB invalidation logic
- Detect topology changes: context size thresholds, batch size changes
- Re-record when graph fingerprint changes
- Success: correct output after topology change, no stale CB
- Time: 6h

### T15: [DONE by unknown | completed:2026-03-26T03:39] | 0% | started:2026-03-26T02:19] Move Python/ctypes hot path to C extension
- Compiled C shim replaces ctypes for tensor dispatch
- Eliminate numpy→ctypes→C boundary crossing per forward()
- Success: Python overhead drops from 3ms to <0.5ms
- Time: 8h

### T16: [DONE by OmniAgent | completed:2026-03-26T05:00] Implement fence polling optimization
- Insert fence at ~80% graph completion
- Spin-wait for final fence instead of blocking
- Reduces fence latency by 1-2ms
- Success: measurable latency reduction
- Time: 4h

### T17: [DONE] Test flash attention scalar path on Honeykrisp
- Enable flash_attn.comp shader (FA_SCALAR code path)
- Fuses Q×K softmax and attention×V
- Success: FA runs without crash on AGX, correct output
- Time: 6h

### T18: [DONE] Benchmark FA scalar vs standard attention at various context lengths
- Test: 128, 512, 2048, 8192 context
- Success: performance comparison table
- Time: 3h

### T19: [DONE] Optimize descriptor set allocation
- Pre-allocate descriptor pools for worst-case graph
- No runtime descriptor allocation
- Success: zero runtime alloc during decode
- Time: 4h

### T20: [DONE] Profile memory bandwidth utilization
- Custom compute shader doing pure reads
- Measure actual GB/s vs theoretical 800 GB/s
- Success: measured bandwidth utilization percentage
- Time: 4h

### T21: [DONE] Tune workgroup sizes for AGX
- Test 64, 128, 256, 512 threads for GEMV kernels
- AGX has 8KB L1 (tiny!), 32KB shared memory
- Success: optimal workgroup size identified per kernel type
- Time: 4h

### T22: [DONE] Implement double-buffering for KV cache writes
- Overlap current token KV write with next dispatch
- Pipeline bubble reduction
- Success: measurable overlap in GPU timeline
- Time: 6h

### T23: [DONE] Benchmark combined Phase 1 optimizations
- Cumulative TPS measurement with all optimizations applied
- Success: 30+ TPS on Llama-3.1-8B Q4_K_M
- Time: 2h

### T24: [DONE] Run 32B and 120B with Phase 1 optimizations
- Measure improvement from baselines (7.2 TPS / TBD)
- Success: proportional improvement across model sizes
- Time: 3h

### T25: [DONE] Document Phase 1 changes + create upstream-compatible patches
- Clean patches for ggml contribution
- Success: patch set ready for upstream review
- Time: 4h

## PHASE 2: vLLM PRODUCTION INTEGRATION [Perspective C — 120+ TPS aggregate]

### T26: [DONE] Study vllm-metal plugin source (the template for our integration)
- Map: MetalPlatform, MetalWorker, MetalModelRunner interfaces
- Document what we need to implement
- Success: interface specification document
- Time: 4h

### T27: [DONE] Create VulkanPlatform(Platform) stub
- check_and_update_config(), get_attn_backend_cls()
- Register as vllm.platform_plugins entry point
- Success: vLLM recognizes platform
- Time: 6h

### T28: [DONE by OmniAgent | completed:2026-03-26T05:01] Create VulkanWorker(WorkerBase) stub
- init_device(), determine_available_memory(), load_model()
- Success: worker initializes without crash
- Time: 6h

### T29: [DONE] Create VulkanModelRunner stub
- load_model() → ggml engine, execute_model() → return logits
- Success: vLLM server starts and accepts requests
- Time: 8h

### T30: [DONE] Implement paged KV cache pool
- Flat Vulkan buffer: num_blocks × block_size × 2 × num_kv_heads × head_size
- For 8B: 8 KV heads, 128 head dim, block_size=16, 2048 blocks ≈ 1 GB
- Success: pool allocates without OOM
- Time: 6h

### T31: [DONE] Implement block table data structure
- Per-request: logical position → physical block ID
- Match vLLM's BlockTable interface
- Success: correct block tracking across allocate/free
- Time: 6h

### T32: [DONE by OmniAgent [Main] Implement reshape_and_cache
- Write KV pairs to paged cache using slot_mapping from scheduler
- Success: KV data lands in correct physical blocks
- Time: 8h

### T33: [DONE] Modify ggml attention for paged KV
- Replace contiguous ggml_view_3d with block-table-indexed gather
- Success: attention correct with paged cache
- Time: 12h

### T34: [DONE by OmniAgent | completed:2026-03-26T05:01]Wire _update_states()
- Parse SchedulerOutput: add/remove/reorder requests
- Success: request lifecycle correctly managed
- Time: 8h

### T35: [DONE by OmniAgent [Cluster2] Implement _prepare_inputs()
- Build input_ids, positions, seq_lens from InputBatch
- Success: correct tensor shapes for batched input
- Time: 6h

### T36: [DONE] Handle token ID mapping: verify GGUF vocab matches HF tokenizer
- Log warnings on mismatch
- Use vLLM's HF tokenizer for all tokenization
- Pass raw token IDs to ggml engine
- Success: vocab verification passes for Llama-3.1-8B
- Time: 4h

### T37: [DONE] Implement logits output shaping
- Extract last-token logits per request from ggml output
- Reshape to (num_requests, vocab_size)
- Success: vLLM sampler accepts logits
- Time: 6h

### T38: [DONE by OmniAgent [sys7] | completed:2026-03-26T04:57] End-to-end single-request test through vLLM plugin
- Chat completion, verify coherent response
- Success: matches direct ggml output quality
- Time: 4h

### T39: [DONE by OmniAgent | completed:2026-03-26T05:01] Enable max_num_seqs=2: two concurrent requests
- Paged KV isolates requests
- Success: both requests correct and independent
- Time: 6h

### T40: [DONE by OmniAgent | completed:2026-03-26T03:27] | 0% | started:2026-03-26T02:18] Implement KV cache block freeing + prevent memory leak
- Release blocks on request completion, return to free pool
- Success: stable memory after 100+ requests
- Time: 4h

### T41: [DONE by OmniAgent [sys7] | completed:2026-03-26T04:43] Implement prefix caching integration
- Accept new_computed_blocks from scheduler
- Skip recomputation for cached prefixes
- Success: repeated prompts reuse KV blocks
- Time: 8h

### T42: [DONE by OmniAgent | completed:2026-03-26T05:01]Handle chunked prefill
- Process partial prompts across multiple engine steps
- Accumulate KV in correct blocks
- Success: long prompts (>512 tokens) served correctly
- Time: 8h

### T43: [DONE by OmniAgent | completed:2026-03-26T05:01]Stress test: 10 concurrent users, 1000 total requests
- Measure aggregate TPS, p99 latency, error rate
- Success: zero crashes, <1% error rate
- Time: 4h

### T45: [IN_PROGRESS by OmniAgent-Vulkan | 0% | started:2026-03-26T05:02]Benchmark max_num_seqs=4 aggregate throughput
- Compare vs single-stream
- Success: aggregate TPS > 2x single-stream
- Time: 3h

### T46: [DONE] Test all model sizes through vLLM plugin (0.5B → 120B)
- Verify each model loads and produces coherent output
- Success: all models work
- Time: 4h

### T47: [DONE] Implement streaming response support via vLLM SSE
- Tokens yield as generated
- Success: streaming works in Streamlit deck
- Time: 4h

### T48: [DONE] Release candidate: pip-installable vLLM plugin
- setup.py with entry_points for vllm.platform_plugins
- pip install vllm-vulkan && vllm serve works
- Success: clean install on fresh venv
- Time: 6h

## PHASE 3: 120B MODEL + FLEET [from Sys12 task queue]

### T49: [IN_PROGRESS by OmniAgent [sys6] | 40% | started:2026-03-26T05:03 | t20]Test 120B on standalone ggml engine
- Verify coherent output
- Measure TPS
- Success: coherent text at any TPS
- Time: 2h

### T51: [DONE] Benchmark 120B: ggml Vulkan vs llama.cpp Vulkan vs MLX (45-60 TPS)
- Compare all three on same hardware
- Success: comparison table
- Time: 2h

### T52: [IN_PROGRESS by OmniAgent [sys6]Test 120B via standalone streaming server
- Verify streaming works for large model
- Test long generation (200+ tokens)
- Success: stable, coherent, streaming
- Time: 2h

### T53: [DONE by OmniAgent [sys1-Main] | completed:2026-03-26T02:39] | 0% | started:2026-03-26T02:37] Fleet connectivity: test from Sys12, Sys10, CUDA cluster
- curl from 10.255.255.30, .64, .11 to Sys0:8080
- Success: all machines can reach the server
- Time: 1h

### T54: [IN_PROGRESS by OmniAgent [sys6]Fleet registration: announce model/TPS to network
- JSON status file + optional POST to registry
- Success: fleet_status.json with accurate info
- Time: 2h

### T55: [IN_PROGRESS by OmniAgent [sys6]Test with Z's Streamlit telemetry deck
- Verify: streaming works, metrics display, no crash
- Run 10 requests from Streamlit
- Success: Z says it works
- Time: 1h

## PHASE 4: PURE VULKAN ENGINE [Perspective B — speculative, 35-45 TPS]

### T56: [DONE] Implement GGUF weight loader in C (parse header, map weights to VkBuffer)
### T57: [DONE by OmniAgent [sys5] | completed:2026-03-26T03:39] | 0% | started:2026-03-26T02:19] Write Q4_K_M dequant+GEMV SPIR-V shader (subgroup shuffle, SIMD 32)
### T58: [DONE by OmniAgent | completed:2026-03-26T05:01] Benchmark T57 vs ggml GEMV
### T59: [DONE by OmniAgent | completed:2026-03-26T05:01]Write RMSNorm, RoPE, softmax, SiLU SPIR-V shaders
### T60: [IN_PROGRESS by OmniAgent [sys7] | 20% | started:2026-03-26T05:02 | t10]Full model: chain all layers + embedding + output projection
### T62: [IN_PROGRESS by OmniAgent [sys6]Push-constant-only token stepping (no CB re-recording)
### T63: [DONE by OmniAgent | completed:2026-03-26T05:02]Benchmark pure engine vs ggml at batch=1
### T64: [IN_PROGRESS by OmniAgent [sys6]Paged KV cache in pure engine
### T65: [DONE by OmniAgent | completed:2026-03-26T05:02]Flash attention SPIR-V shader (tiled, scalar, 2-pass online softmax)

## PHASE 5: VALIDATION + PRODUCTION HARDENING

### T66: [READY]Numerical accuracy: logits comparison ggml vs llama.cpp for 1000 tokens
### T67: [IN_PROGRESS by OmniAgent | 0% | started:2026-03-26T05:02]Memory leak testing: 10,000 requests, monitor RSS + Vulkan memory
### T68: [READY]Edge case testing: empty, max-length, special tokens, unicode
### T69: [READY]M1 Max (32GB) validation: run all benchmarks on Sys12
### T70: [READY]Deployment documentation: hardware reqs, install, config, troubleshoot

## PHASE 6: DOCUMENTATION + UPSTREAM

### T71: [READY]Comprehensive README with architecture diagram
### T72: [READY]Benchmark report: all models, all quants, all batch sizes
### T73: [READY]Blog post: Vulkan LLM Inference on Apple Silicon Linux
### T74: [READY]Draft PR for vLLM: Vulkan platform plugin
### T75: [READY]Draft PR for ggml: graph caching + CB optimization patches
### T76: [READY]File Mesa issue: VK_KHR_cooperative_matrix request with benchmark data

## PHASE 7: ADVANCED OPTIMIZATION

### T77: [READY]Study ThunderMittens findings: register-direct loads beat shared memory on UMA
### T78: [READY]Profile register pressure per kernel (occupancy analysis)
### T79: [READY]Test VK_EXT_memory_budget on Honeykrisp
### T80: [READY]Investigate ggml_backend_sched optimization for single-backend (skip routing)

## PHASE 8: AGENTIC FRAMEWORK PROTOTYPE

### T81: [READY]Design agent-to-model routing protocol (gRPC or HTTP)
### T82: [READY]Build prototype: agent → router → best model → response
### T83: [READY]Multi-model serving: 0.5B fast + 8B smart + 120B reasoning on same box
### T84: [READY]Agent communication bridge: cross-machine task coordination
### T85: [READY]Integration with Z's v44 agent framework

## SUMMARY
- 85 tasks across 9 phases
- 40 DONE, 7 IN_PROGRESS, 38 READY
- Phase 0-1: COMPLETE (engine works at 22 TPS, coherent)
- Phase 2: vLLM integration IN PROGRESS (14/23 done)
- Phase 3-8: Pending
- TEST NODE: sys1 (.128) — only node with Vulkan GPU + compiled engine
- CUDA BRAIN: .11 — Qwen3.5-122B-FP8, 42K ctx, 62 TPS
- MLX FLEET: sys2-sys7 — code generation at 48-55 TPS
