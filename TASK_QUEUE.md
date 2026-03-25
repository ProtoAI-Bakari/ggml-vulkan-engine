# TASK QUEUE v3 — OPTIMIZATION + PRODUCTION PHASE
# 30 items. DO NOT STOP. Pick next READY task after each completion.

## PHASE 1-6: DONE (T1-T29) — 24.8 TPS 8B, 69.8 TPS 0.5B, 121 TPS batch

## PHASE 7: SHADER EXPERIMENTS [DONE]
### T30: [DONE] Custom tiled matmul shader — ggml already near-optimal
### T31: [DONE] Baseline GFLOPS profiling — 1675 peak at batch=64

## PHASE 8: CLOSE THE MEDIAN→BEST GAP (21.7 → 24.8 consistently)
### T32: [READY] Profile graph scheduling overhead — where is the 3ms gap between median and best?
### T33: [READY] Pre-warm the ggml graph (run 5 warmup tokens before timing)
### T34: [READY] Pin CPU threads to performance cores (taskset on Firestorm cores only)
### T35: [READY] Test ggml thread count sweep (1,2,4,8,16,20) for optimal CPU/GPU balance
### T36: [READY] Disable CPU backend in ggml scheduler — force ALL ops to Vulkan

## PHASE 9: PRODUCTION SERVER
### T37: [READY] Fix streaming server to handle concurrent requests (ThreadingMixIn done on Sys12)
### T38: [READY] Add /v1/chat/completions with proper Llama-3.1 chat template
### T39: [READY] Add token counting in usage field (prompt_tokens + completion_tokens)
### T40: [READY] Add request timeout and error recovery
### T41: [READY] Add model hot-swapping (load different GGUF via API)
### T42: [READY] Benchmark server: sustained TPS over 100 requests
### T43: [READY] Benchmark server: concurrent users (2, 4, 8 simultaneous)

## PHASE 10: MULTI-MODEL FLEET
### T44: [READY] Run 0.5B + 8B simultaneously (different ports, same GPU)
### T45: [READY] Router: small model for simple tasks, big model for complex
### T46: [READY] Test all available GGUFs: Llama, Qwen, any others in ~/models/gguf/

## PHASE 11: INTEGRATION WITH Z'S FLEET
### T47: [READY] Test connectivity from other machines on 10.255.255.0/24 network
### T48: [READY] Write fleet registration script (announce model/TPS to central registry)
### T49: [READY] Test from Z's Streamlit telemetry deck (streaming, metrics)
### T50: [READY] Load test from multiple machines simultaneously

## PHASE 12: DOCUMENTATION + UPSTREAM
### T51: [READY] Write comprehensive README for the ggml Vulkan engine
### T52: [READY] Package as pip-installable (setup.py + wheel)
### T53: [READY] Draft blog post: World-First Vulkan LLM Inference on Asahi Linux
### T54: [READY] Prepare upstream PR for vLLM (Vulkan backend using ggml)
### T55: [READY] File Mesa Honeykrisp issue: request VK_KHR_cooperative_matrix support

## PHASE 13: ASAHI KERNEL/DRIVER INVESTIGATION
### T56: [READY] Clone AsahiLinux/mesa (honeykrisp branch)
### T57: [READY] Study hk_physical_device.c — what Vulkan extensions are exposed?
### T58: [READY] Search for simdgroup_matrix / cooperative_matrix code paths
### T59: [READY] Study AGX command stream format — can we inject custom GPU commands?
### T60: [READY] Profile Vulkan command buffer overhead — is the driver adding latency?
### T61: [READY] Test VK_EXT_memory_budget — does it work on Asahi?

## RULES
1. After each task: git commit + update PROGRESS_REPORT.md
2. If stuck >5 min: ask_big_brain.py or ask_coder_brain.py
3. If BLOCKED >15 min: skip to next, mark BLOCKED
4. DO NOT wait for user input between tasks
5. The fleet depends on this engine. Make it bulletproof.
