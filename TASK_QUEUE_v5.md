# MASTER TASK QUEUE v5
# Updated: 2026-03-26 03:40
#
# ┌─────────────────────────────────────────┐
# │ DONE:  40 | IN_PROGRESS:  7 | READY: 58 │
# │ Total: 105 tasks across 11 phases      │
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
# PHASE 9: Production Hardening         [T86-T95] —  0/10 READY
# PHASE 10: Fleet Automation            [T96-T105]—  0/10 READY
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

### T45: [IN_PROGRESS by OmniAgent [sys5] | 40% | started:2026-03-26T10:11 | t20] Test all model sizes through vLLM plugin (0.5B → 120B)
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

### T49: [DONE by OmniAgent [sys1-Main] | completed:2026-03-26T10:10] Benchmark 120B: ggml Vulkan vs llama.cpp Vulkan vs MLX (45-60 TPS)
- Compare all three on same hardware
- Success: comparison table
- Time: 2h

### T52: [IN_PROGRESS by OmniAgent [sys1-Main] | 90% | started:2026-03-26T10:11 | t100]

### T56: [DONE] Implement GGUF weight loader in C (parse header, map weights to VkBuffer)
### T57: [DONE by OmniAgent [sys5] | completed:2026-03-26T03:39] | 0% | started:2026-03-26T02:19] Write Q4_K_M dequant+GEMV SPIR-V shader (subgroup shuffle, SIMD 32)
### T58: [DONE by OmniAgent | completed:2026-03-26T05:01] Benchmark T57 vs ggml GEMV
### T59: [DONE by OmniAgent | completed:2026-03-26T05:01]Write RMSNorm, RoPE, softmax, SiLU SPIR-V shaders
### T60: [IN_PROGRESS by OmniAgent [sys7] | 80% | started:2026-03-26T10:13 | t40]Benchmark pure engine vs ggml at batch=1
### T64: [DONE by OmniAgent [Main] | completed:2026-03-26T09:20] | t110] | t100] | t90] | t80] | t70] | t60] | t50] | 20% | started:2026-03-26T09:04 | t10] | 90% | started:2026-03-26T08:54 | t110]M1 Max (32GB) validation: run all benchmarks on Sys12
### T70: [DONE by OmniAgent [sys3] | completed:2026-03-26T05:18]Comprehensive README with architecture diagram
### T72: [IN_PROGRESS by OmniAgent [sys6] | 80% | started:2026-03-26T10:13 | t40] | completed:2026-03-26T09:21]Profile register pressure per kernel (occupancy analysis)
### T79: [DONE by OmniAgent [Main] | completed:2026-03-26T09:23]Design agent-to-model routing protocol (gRPC or HTTP)
### T82: [DONE by OmniAgent [Main] | completed:2026-03-26T09:23]Multi-model serving: 0.5B fast + 8B smart + 120B reasoning on same box
### T84: [DONE]Agent communication bridge: cross-machine task coordination
### T85: [READY] Integration with Z's v44 agent framework

## PHASE 9: PRODUCTION HARDENING [Stability under real-world conditions]

### T86: [READY] 24-hour continuous load soak test (8B model, 4 concurrent users)
- Run sustained traffic for 24h, monitor memory, TPS drift, error rate
- Success: zero crashes, <2% TPS degradation, no memory leak
- Time: 26h

### T87: [READY] OOM recovery: graceful degradation when memory exhausted
- Trigger OOM with large batch + long context, verify server recovers
- Success: server returns 503, recovers within 10s, no restart needed
- Time: 6h

### T88: [READY] Request timeout and cancellation handling
- Implement per-request timeout (configurable, default 120s)
- Cancel in-flight generation on client disconnect
- Success: stale requests freed, no resource leak
- Time: 6h

### T89: [READY] Crash recovery: auto-restart with state preservation
- Watchdog process restarts task_server and vLLM on crash
- Preserve in-flight task states across restart
- Success: service back within 30s of crash, no task loss
- Time: 8h

### T90: [READY] Prometheus metrics endpoint for TPS, latency, queue depth
- Expose /metrics in Prometheus format
- Counters: requests_total, tokens_generated, errors_total
- Histograms: request_latency_seconds, ttft_seconds
- Success: Grafana dashboard shows live metrics
- Time: 6h

### T91: [READY] Log aggregation: structured JSON logging with rotation
- All components emit JSON logs to ~/AGENT/LOGS/
- Implement log rotation (max 100MB per file, 5 rotations)
- Success: logs parseable by jq, no disk exhaustion
- Time: 4h

### T92: [READY] Adversarial input fuzzing: malformed tokens, huge payloads, unicode edge cases
- Generate 10K fuzzed requests (oversized, null bytes, invalid UTF-8)
- Success: zero crashes, all invalid inputs return 4xx
- Time: 6h

### T93: [READY] KV cache fragmentation test under sustained churn
- Simulate 1000 short-lived requests to fragment block pool
- Measure allocation latency over time
- Success: allocation latency <1ms at p99 after 1000 requests
- Time: 4h

### T94: [READY] Context length boundary tests (1, 512, 2048, 8192, max)
- Verify correct output at each context length boundary
- Test context overflow handling
- Success: correct output at all lengths, graceful error at overflow
- Time: 4h

### T95: [READY] Concurrent model loading stress test
- Attempt to load same model from multiple workers simultaneously
- Verify file locking prevents corruption
- Success: no corruption, clean error for contention
- Time: 4h

## PHASE 10: FLEET AUTOMATION [Zero-touch deployment and operations]

### T96: [READY] Auto-deploy script: one-command fleet-wide model push
- Script pulls model to sys1, rsyncs to sys2-sys7
- Verify checksums after transfer
- Success: single command deploys model to all nodes in <10min
- Time: 8h

### T97: [READY] Health webhook: POST to Slack/Discord on node failure
- Monitor /health endpoint on all fleet nodes every 30s
- Fire webhook on 3 consecutive failures
- Success: notification within 2 minutes of node death
- Time: 4h

### T98: [READY] Model hot-swap: replace running model without downtime
- Load new model in background, atomic swap when ready
- Drain in-flight requests before swap
- Success: zero dropped requests during swap
- Time: 8h

### T99: [READY] Fleet-wide rolling restart with zero downtime
- Restart nodes one-at-a-time, verify health before proceeding
- Automatic rollback if new version fails health check
- Success: full fleet restart with zero request failures
- Time: 6h

### T100: [READY] Automated backup: task queue + logs + configs to NAS
- Hourly incremental backup via rsync to NAS
- Daily full snapshot with 7-day retention
- Success: backup runs unattended, restore tested
- Time: 4h

### T101: [READY] Fleet inventory API: GET /fleet returns all nodes, models, status
- Aggregate /health from all nodes into single endpoint
- Include: hostname, model loaded, TPS, uptime, memory usage
- Success: single curl shows full fleet state
- Time: 6h

### T102: [READY] Auto-scaling: route requests to least-loaded node
- Load balancer queries fleet inventory, routes to lowest queue depth
- Sticky sessions for streaming requests
- Success: even load distribution across 4+ nodes
- Time: 8h

### T103: [READY] Model version registry: track which GGUF is deployed where
- Central registry of model name, quant, sha256, deployed nodes
- CLI: fleet-models list, fleet-models deploy <model> <nodes>
- Success: accurate inventory, deploy/rollback works
- Time: 6h

### T104: [READY] Automated performance regression gate
- Run benchmark suite after every deploy
- Block rollout if TPS drops >5% from baseline
- Success: bad deploy automatically rolled back
- Time: 6h

### T105: [READY] Fleet dashboard: web UI showing all nodes, tasks, metrics
- Single-page app aggregating fleet inventory + Prometheus metrics
- Show: node grid, active tasks, TPS sparklines, alerts
- Success: accessible at sys1:8080/dashboard
- Time: 8h

## SUMMARY
- 105 tasks across 11 phases
- 40 DONE, 7 IN_PROGRESS, 58 READY
- Phase 0-1: COMPLETE (engine works at 22 TPS, coherent)
- Phase 2: vLLM integration IN PROGRESS (14/23 done)
- Phase 3-8: Pending
- Phase 9-10: Production Hardening + Fleet Automation (20 new tasks)
- TEST NODE: sys1 (.128) — only node with Vulkan GPU + compiled engine
- CUDA BRAIN: .11 — Qwen3.5-122B-FP8, 42K ctx, 62 TPS
- MLX FLEET: sys2-sys7 — code generation at 48-55 TPS
