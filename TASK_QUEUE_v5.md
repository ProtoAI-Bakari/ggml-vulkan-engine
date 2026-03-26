# MASTER TASK QUEUE v5 — OpR + Sys12 Lead Combined Plan
# Source: Opus Research deep analysis + Sys12 field experience
# Updated: 2026-03-25 13:30
# RULES: 
#   - After EACH task: git commit + update PROGRESS_REPORT.md
#   - If stuck >5min: ask_big_brain.py or ask_coder_brain.py
#   - If BLOCKED >15min: skip + mark BLOCKED with reason
#   - QUALITY TEST after every code change (10 diverse prompts minimum)
#   - Success criteria MUST be met before marking DONE

## PHASE 0: STABILITY + MEASUREMENT [DO FIRST — nothing else matters if broken]

### T01: [DONE]] Verify standalone engine: 12+ sequential requests, all coherent
- Run 12 diverse prompts through GgmlLLM.generate() 
- Include: math, factual, creative, long-gen, multi-sentence
- Success: 12/12 coherent, no crash, consistent TPS
- Time: 1h

### T02: [DONE by OmniAgent [sys1-Main]]]]] Fix standalone streaming server startup + stability
- Server must start in <30s, serve on 0.0.0.0:8080
- ThreadingMixIn for concurrent HTTP (sequential engine underneath)
- KV reset between every request (engine_reset_kv)
- Success: 50 sequential HTTP requests without crash, all coherent
- Time: 2h

### T03: [DONE]] Coherency blast test: 50 diverse prompts through HTTP server
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

### T05: [DONE]] Profile CPU time breakdown with py-spy + custom instrumentation
- Confirm: ~6ms CB recording, ~4ms graph build, ~3ms Python overhead
- Instrument ggml_vk_dispatch_pipeline, CB begin/end, queue submit, fence wait
- Success: flamegraph + per-stage timing CSV matching OpR's predicted breakdown
- Time: 3h

### T06: [DONE]] Document ggml compute graph: node count, op types, tensor shapes
- Dump graph at position 1, 100, 500 for Llama-3.1-8B decode
- Count: how many nodes are views/casts vs real compute?
- Success: graph topology document with node breakdown
- Time: 2h

### T07: [DONE]] Verify Honeykrisp capabilities matrix
- vulkaninfo: all extensions, subgroup properties, memory types, queue families
- Confirm: subgroupSize=32, shaderFloat16Int8, integerDotProduct
- Confirm: NO cooperative_matrix, NO tensor cores
- Success: capability matrix document
- Time: 1h

### T08: [DONE]] Run llama-bench on ALL models for reference baseline
- 8B Q4_K_M, Q8_0, F16
- 0.5B, 1.5B, 3B, 32B Q4_K_M
- 120B mxfp4 (after merge)
- Success: llama.cpp baseline TPS table for every model
- Time: 2h

### T09: [DONE]] Create automated regression test: golden output comparison
- 10 prompts with expected output patterns (regex or substring match)
- Run before AND after every code change
- Success: CI-ready test script that catches coherency regressions
- Time: 3h

### T10: [DONE]] Test 120B model on llama-bench (THE big number)
- Merge shards first (llama-gguf-split --merge)
- Run: llama-bench -m 120b-merged.gguf -ngl 99 -t 4 -p 64 -n 32
- If OOM: try with HK_SYSMEM=112000000000
- Success: baseline TPS for 120B on Vulkan Asahi (world first if it runs)
- Time: 3h

## PHASE 1: GGML GRAPH + CB OPTIMIZATION [Perspective A — 23→30-33 TPS]

### T11: [DONE]] Implement graph topology fingerprinting
- Hash: node count + op types + tensor shapes
- Detect when graph is unchanged between tokens (99%+ of decode steps)
- Success: fingerprint matches consecutive decode tokens
- Time: 4h

### T12: [DONE]] Add graph caching via ggml_gallocr (PR #20927 pattern)
- ggml_gallocr_reserve() at init with worst-case graph
- ggml_gallocr_alloc_graph() each token — near-no-op when topology matches
- Success: graph alloc time drops from 4ms to <0.5ms
- Time: 6h

### T13: [DONE]] Implement command buffer template recording
- Record full pipeline once on first execution
- Use vkCmdPushConstants for dynamic params: KV offset, seq_len, position
- vkResetCommandPool for pool-level reset (not per-buffer)
- Success: CB recording drops from 6ms to <1ms for stable graphs
- Time: 12h

### T14: [DONE]] Add CB invalidation logic
- Detect topology changes: context size thresholds, batch size changes
- Re-record when graph fingerprint changes
- Success: correct output after topology change, no stale CB
- Time: 6h

### T15: [IN_PROGRESS by OmniAgent [sys1-Main] | 0% | started:2026-03-26T02:19] Move Python/ctypes hot path to C extension
- Compiled C shim replaces ctypes for tensor dispatch
- Eliminate numpy→ctypes→C boundary crossing per forward()
- Success: Python overhead drops from 3ms to <0.5ms
- Time: 8h

### T16: [IN_PROGRESS by OmniAgent [sys2] | 0% | started:2026-03-26T02:22] Implement fence polling optimization
- Insert fence at ~80% graph completion
- Spin-wait for final fence instead of blocking
- Reduces fence latency by 1-2ms
- Success: measurable latency reduction
- Time: 4h

### T17: [DONE]] Test flash attention scalar path on Honeykrisp
- Enable flash_attn.comp shader (FA_SCALAR code path)
- Fuses Q×K softmax and attention×V
- Success: FA runs without crash on AGX, correct output
- Time: 6h

### T18: [DONE]] Benchmark FA scalar vs standard attention at various context lengths
- Test: 128, 512, 2048, 8192 context
- Success: performance comparison table
- Time: 3h

### T19: [DONE]] Optimize descriptor set allocation
- Pre-allocate descriptor pools for worst-case graph
- No runtime descriptor allocation
- Success: zero runtime alloc during decode
- Time: 4h

### T20: [DONE]] Profile memory bandwidth utilization
- Custom compute shader doing pure reads
- Measure actual GB/s vs theoretical 800 GB/s
- Success: measured bandwidth utilization percentage
- Time: 4h

### T21: [DONE]] Tune workgroup sizes for AGX
- Test 64, 128, 256, 512 threads for GEMV kernels
- AGX has 8KB L1 (tiny!), 32KB shared memory
- Success: optimal workgroup size identified per kernel type
- Time: 4h

### T22: [DONE]] Implement double-buffering for KV cache writes
- Overlap current token KV write with next dispatch
- Pipeline bubble reduction
- Success: measurable overlap in GPU timeline
- Time: 6h

### T23: [DONE]] Benchmark combined Phase 1 optimizations
- Cumulative TPS measurement with all optimizations applied
- Success: 30+ TPS on Llama-3.1-8B Q4_K_M
- Time: 2h

### T24: [DONE]] Run 32B and 120B with Phase 1 optimizations
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

### T27: [DONE]] Create VulkanPlatform(Platform) stub
- check_and_update_config(), get_attn_backend_cls()
- Register as vllm.platform_plugins entry point
- Success: vLLM recognizes platform
- Time: 6h

### T28: [IN_PROGRESS by OmniAgent [sys3] | 0% | started:2026-03-26T02:18] | 0% | started:2026-03-26T02:16] Create VulkanWorker(WorkerBase) stub
- init_device(), determine_available_memory(), load_model()
- Success: worker initializes without crash
- Time: 6h

### T29: [DONE]] Create VulkanModelRunner stub
- load_model() → ggml engine, execute_model() → return logits
- Success: vLLM server starts and accepts requests
- Time: 8h

### T30: [DONE]] Implement paged KV cache pool
- Flat Vulkan buffer: num_blocks × block_size × 2 × num_kv_heads × head_size
- For 8B: 8 KV heads, 128 head dim, block_size=16, 2048 blocks ≈ 1 GB
- Success: pool allocates without OOM
- Time: 6h

### T31: [DONE]] Implement block table data structure
- Per-request: logical position → physical block ID
- Match vLLM's BlockTable interface
- Success: correct block tracking across allocate/free
- Time: 6h

### T32: [DONE by OmniAgent [Main]] Implement reshape_and_cache
- Write KV pairs to paged cache using slot_mapping from scheduler
- Success: KV data lands in correct physical blocks
- Time: 8h

### T33: [DONE]] Modify ggml attention for paged KV
- Replace contiguous ggml_view_3d with block-table-indexed gather
- Success: attention correct with paged cache
- Time: 12h

### T34: [IN_PROGRESS by OmniAgent [sys3] | 0% | started:2026-03-26T02:18] Wire _update_states()
- Parse SchedulerOutput: add/remove/reorder requests
- Success: request lifecycle correctly managed
- Time: 8h

### T35: [DONE by OmniAgent [Cluster2]] Implement _prepare_inputs()
- Build input_ids, positions, seq_lens from InputBatch
- Success: correct tensor shapes for batched input
- Time: 6h

### T36: [DONE]] Handle token ID mapping: verify GGUF vocab matches HF tokenizer
- Log warnings on mismatch
- Use vLLM's HF tokenizer for all tokenization
- Pass raw token IDs to ggml engine
- Success: vocab verification passes for Llama-3.1-8B
- Time: 4h

### T37: [DONE]] Implement logits output shaping
- Extract last-token logits per request from ggml output
- Reshape to (num_requests, vocab_size)
- Success: vLLM sampler accepts logits
- Time: 6h

### T38: [IN_PROGRESS by OmniAgent [sys1-Main] | 0% | started:2026-03-26T02:24] End-to-end single-request test through vLLM plugin
- Chat completion, verify coherent response
- Success: matches direct ggml output quality
- Time: 4h

### T39: [IN_PROGRESS by OmniAgent [sys1-Main] | 0% | started:2026-03-26T02:24] Enable max_num_seqs=2: two concurrent requests
- Paged KV isolates requests
- Success: both requests correct and independent
- Time: 6h

### T40: [IN_PROGRESS by OmniAgent [sys2] | 0% | started:2026-03-26T02:18] Implement KV cache block freeing + prevent memory leak
- Release blocks on request completion, return to free pool
- Success: stable memory after 100+ requests
- Time: 4h

### T41: [IN_PROGRESS by OmniAgent [sys5] | 0% | started:2026-03-26T02:18] Implement prefix caching integration
- Accept new_computed_blocks from scheduler
- Skip recomputation for cached prefixes
- Success: repeated prompts reuse KV blocks
- Time: 8h

### T42: [IN_PROGRESS by OmniAgent [sys1-Main] | 0% | started:2026-03-26T02:26] Handle chunked prefill
- Process partial prompts across multiple engine steps
- Accumulate KV in correct blocks
- Success: long prompts (>512 tokens) served correctly
- Time: 8h

### T43: [IN_PROGRESS by OmniAgent [sys6] | 0% | started:2026-03-26T02:18] Implement preemption via recompute
- When KV blocks exhausted, evict lowest-priority request
- Mark for recomputation
- Success: server recovers from OOM without crash
- Time: 6h

### T44: [IN_PROGRESS by OmniAgent [sys4] | 0% | started:2026-03-26T02:28] Stress test: 10 concurrent users, 1000 total requests
- Measure aggregate TPS, p99 latency, error rate
- Success: zero crashes, <1% error rate
- Time: 4h

### T45: [IN_PROGRESS by OmniAgent [sys1-Main] | 0% | started:2026-03-26T02:29] Benchmark max_num_seqs=4 aggregate throughput
- Compare vs single-stream
- Success: aggregate TPS > 2x single-stream
- Time: 3h

### T46: [DONE]] Test all model sizes through vLLM plugin (0.5B → 120B)
- Verify each model loads and produces coherent output
- Success: all models work
- Time: 4h

### T47: [DONE]] Implement streaming response support via vLLM SSE
- Tokens yield as generated
- Success: streaming works in Streamlit deck
- Time: 4h

### T48: [DONE]] Release candidate: pip-installable vLLM plugin
- setup.py with entry_points for vllm.platform_plugins
- pip install vllm-vulkan && vllm serve works
- Success: clean install on fresh venv
- Time: 6h

## PHASE 3: 120B MODEL + FLEET [from Sys12 task queue]

### T49: [IN_PROGRESS by OmniAgent [sys6] | 0% | started:2026-03-26T02:19] Merge 120B GGUF shards
- llama-gguf-split --merge
- Success: single merged GGUF file
- Time: 30min (I/O bound)

### T50: [IN_PROGRESS by OmniAgent [sys4] | 0% | started:2026-03-26T02:18] Test 120B on standalone ggml engine
- Verify coherent output
- Measure TPS
- Success: coherent text at any TPS
- Time: 2h

### T51: [DONE]] Benchmark 120B: ggml Vulkan vs llama.cpp Vulkan vs MLX (45-60 TPS)
- Compare all three on same hardware
- Success: comparison table
- Time: 2h

### T52: [IN_PROGRESS by OmniAgent [sys5] | 0% | started:2026-03-26T02:32] Test 120B via standalone streaming server
- Verify streaming works for large model
- Test long generation (200+ tokens)
- Success: stable, coherent, streaming
- Time: 2h

### T53: [IN_PROGRESS by OmniAgent [sys1-Main] | 0% | started:2026-03-26T02:37] Fleet connectivity: test from Sys12, Sys10, CUDA cluster
- curl from 10.255.255.30, .64, .11 to Sys0:8080
- Success: all machines can reach the server
- Time: 1h

### T54: [READY]] Fleet registration: announce model/TPS to network
- JSON status file + optional POST to registry
- Success: fleet_status.json with accurate info
- Time: 2h

### T55: [READY]] Test with Z's Streamlit telemetry deck
- Verify: streaming works, metrics display, no crash
- Run 10 requests from Streamlit
- Success: Z says it works
- Time: 1h

## PHASE 4: PURE VULKAN ENGINE [Perspective B — speculative, 35-45 TPS]

### T56: [DONE]] Implement GGUF weight loader in C (parse header, map weights to VkBuffer)
### T57: [IN_PROGRESS by OmniAgent [sys6] | 0% | started:2026-03-26T02:19] Write Q4_K_M dequant+GEMV SPIR-V shader (subgroup shuffle, SIMD 32)
### T58: [IN_PROGRESS by OmniAgent | 0% | started:2026-03-26T02:38] Benchmark T57 vs ggml GEMV
### T59: [READY] Write RMSNorm, RoPE, softmax, SiLU SPIR-V shaders
### T60: [READY] Implement static CB recording for one transformer layer
### T61: [READY] Full model: chain all layers + embedding + output projection
### T62: [READY]] Push-constant-only token stepping (no CB re-recording)
### T63: [IN_PROGRESS by OmniAgent [sys5] | 0% | started:2026-03-26T02:18] Benchmark pure engine vs ggml at batch=1
### T64: [READY] Paged KV cache in pure engine
### T65: [READY] Flash attention SPIR-V shader (tiled, scalar, 2-pass online softmax)

## PHASE 5: VALIDATION + PRODUCTION HARDENING

### T66: [READY] Numerical accuracy: logits comparison ggml vs llama.cpp for 1000 tokens
### T67: [READY] Memory leak testing: 10,000 requests, monitor RSS + Vulkan memory
### T68: [READY] Edge case testing: empty, max-length, special tokens, unicode
### T69: [IN_PROGRESS by OmniAgent [sys3] | 0% | started:2026-03-26T02:18] M1 Max (32GB) validation: run all benchmarks on Sys12
### T70: [READY] Deployment documentation: hardware reqs, install, config, troubleshoot

## PHASE 6: DOCUMENTATION + UPSTREAM

### T71: [READY] Comprehensive README with architecture diagram
### T72: [READY] Benchmark report: all models, all quants, all batch sizes
### T73: [READY] Blog post: Vulkan LLM Inference on Apple Silicon Linux
### T74: [READY]] Draft PR for vLLM: Vulkan platform plugin
### T75: [READY] Draft PR for ggml: graph caching + CB optimization patches
### T76: [READY] File Mesa issue: VK_KHR_cooperative_matrix request with benchmark data

## PHASE 7: ADVANCED OPTIMIZATION

### T77: [READY] Study ThunderMittens findings: register-direct loads beat shared memory on UMA
### T78: [IN_PROGRESS by OmniAgent [sys5] | 0% | started:2026-03-26T02:18] Profile register pressure per kernel (occupancy analysis)
### T79: [READY] Test VK_EXT_memory_budget on Honeykrisp
### T80: [READY] Investigate ggml_backend_sched optimization for single-backend (skip routing)

## PHASE 8: AGENTIC FRAMEWORK PROTOTYPE

### T81: [READY] Design agent-to-model routing protocol (gRPC or HTTP)
### T82: [READY] Build prototype: agent → router → best model → response
### T83: [READY] Multi-model serving: 0.5B fast + 8B smart + 120B reasoning on same box
### T84: [READY] Agent communication bridge: cross-machine task coordination
### T85: [READY] Integration with Z's v44 agent framework

## SUMMARY
- 85 tasks across 8 phases
- Phase 0 (stability): DO FIRST, no excuses
- Phase 1 (ggml optimization): 23→30-33 TPS, 2-3 weeks
- Phase 2 (vLLM plugin): 120+ TPS aggregate, 3-5 weeks
- Phase 3 (120B + fleet): prove it at scale
- Phase 4 (pure Vulkan): speculative, 35-45 TPS if Phases 1-2 succeed
- Phase 5-6 (hardening + docs): make it real
- Phase 7-8 (advanced + agentic): the vision
