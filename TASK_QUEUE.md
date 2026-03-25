# TASK QUEUE v2 — 33 ITEMS — DO NOT STOP UNTIL ALL DONE
# After each task: update PROGRESS_REPORT.md, commit to git, continue next.
# Target: 100+ TPS aggregate. Every task moves us closer.

## PHASE 1: CORE ENGINE (DONE)
### T1: [DONE] ggml fused MLP per-layer → 5.1 TPS
### T2: [DONE] Full transformer single graph → 10.9 TPS
### T3: [DONE] KV cache → 10.7 TPS (helps at long seq)
### T4: [DONE] F16 weights → 12.5 TPS
### T5: [DONE] Q4_K_M weights → 21.7 TPS (88% of llama.cpp!)

## PHASE 2: VLLM INTEGRATION
### T6: [ACTIVE] Wire ggml engine into vLLM-compatible API
### T7: [READY] Add streaming token output to ggml backend
### T8: [READY] Add SamplingParams support (temperature, top_p, top_k, repetition_penalty)
### T9: [READY] Add stop tokens and EOS handling
### T10: [READY] Multi-prompt batch generation (generate multiple prompts at once)

## PHASE 3: PERFORMANCE OPTIMIZATION
### T11: [READY] Profile per-token breakdown: what % is matmul vs attention vs overhead
### T12: [READY] Test Q8_0 GGUF (should be ~22 TPS with better quality)
### T13: [READY] Test Q5_K_M GGUF (quality vs speed tradeoff)
### T14: [READY] Optimize graph reuse between tokens (avoid graph rebuild)
### T15: [READY] Benchmark long sequences (256, 512, 1024 tokens) — where does KV cache help?
### T16: [READY] Tune ggml thread count for optimal CPU/GPU balance
### T17: [READY] Test with HK_SYSMEM=112000000000 for max Vulkan heap

## PHASE 4: MULTI-USER / BATCHING
### T18: [READY] Continuous batching: multiple concurrent users in one graph
### T19: [READY] Dynamic batch scheduling (add/remove users mid-generation)
### T20: [READY] Benchmark aggregate TPS at batch=4, 8, 16, 32, 64
### T21: [READY] Prefill optimization: use GPU batch advantage (195x at high batch)

## PHASE 5: MODEL COVERAGE
### T22: [READY] Test Qwen2.5-3B on ggml Vulkan engine
### T23: [READY] Test Qwen2.5-1.5B on ggml Vulkan engine
### T24: [READY] Test Qwen2.5-0.5B on ggml Vulkan engine (target: 100+ TPS single user)
### T25: [READY] Download and test Qwen3.5-0.8B (GDN attention — different arch)
### T26: [READY] Test Llama-3.1-8B-Instruct chat template / system prompts

## PHASE 6: PRODUCTION HARDENING
### T27: [READY] Error handling: graceful fallback when Vulkan OOMs
### T28: [READY] Memory monitoring: track Vulkan VRAM usage per request
### T29: [READY] Logging: structured JSON logs for TPS, latency, memory
### T30: [READY] Benchmark suite script: one command tests all models + quants

## PHASE 7: ADVANCED OPTIMIZATIONS
### T31: [READY] Study llama.cpp flash attention shader — can we improve ours?
### T32: [READY] Investigate Vulkan descriptor indexing for weight switching
### T33: [READY] Write comprehensive technical report: architecture, benchmarks, future work

## RULES
1. After completing a task, IMMEDIATELY start the next READY task
2. GIT COMMIT after every 2-3 tasks (never lose more than 30 min of work)
3. Update PROGRESS_REPORT.md after each task
4. If stuck >5 min: python ~/AGENT/ask_big_brain.py
5. If stuck >15 min: mark BLOCKED, skip to next
6. DO NOT wait for user input between tasks
7. The target is 100+ TPS aggregate. 21.7 single-user is the START, not the end.
