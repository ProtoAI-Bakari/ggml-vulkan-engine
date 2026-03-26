# TASK QUEUE FOR OMNIAGENT V4 — READ AND EXECUTE IN ORDER
# After each task: commit to git, update KNOWLEDGE_BASE.md, continue next task
# Use ask_coder_brain for code help. Use ask_big_brain for architecture help.
# DO NOT STOP. DO NOT ASK HUMAN. EXECUTE.

## CURRENT PRIORITY: VULKAN GPU ENGINE OPTIMIZATION

### T06: [IN_PROGRESS by OmniAgent [Sys4]] Integrate graph caching into ggml_llama_gguf.c
**Goal:** Eliminate 3ms/token graph rebuild overhead
**Method:**
1. Read ~/AGENT/ggml_vulkan_engine_optimized.py for the graph caching pattern
2. Apply same pattern to ggml_llama_gguf.c engine_forward():
   - Use ggml_reset() instead of ggml_init()/ggml_free() per token (ALREADY DONE)
   - Pre-build and cache the ggml_cgraph for batch=1 decode
   - Only rebuild graph when batch size changes (prefill vs decode)
   - Use ggml_backend_sched_reserve() to pre-allocate Vulkan buffers
3. Rebuild: gcc -shared -O2 -fPIC -o libggml_llama_gguf.so ggml_llama_gguf.c -I ~/GITDEV/llama.cpp/ggml/include -L ~/GITDEV/llama.cpp/build-lib/bin -lggml -lggml-base -lggml-vulkan -lggml-cpu -lm -Wl,-rpath,/home/z/GITDEV/llama.cpp/build-lib/bin
4. Benchmark: python benchmark_all.py --tokens 100 --models llama-8b-q4
5. Target: 22 TPS → 28+ TPS
**Files:** ~/AGENT/ggml_llama_gguf.c, ~/AGENT/libggml_llama_gguf.so

### T07: [IN_PROGRESS by Main] Add MoE FFN support to ggml_llama_gguf.c for gpt-oss-120b
**Goal:** Load and run 120B MoE model through our engine
**Architecture (from GGUF analysis):**
- 36 layers, hidden=2880, 128 experts, 4 active per token
- Expert weights: ffn_gate_exps [2880,2880,128] MXFP4
- Expert weights: ffn_up_exps [2880,2880,128] MXFP4
- Expert weights: ffn_down_exps [2880,2880,128] MXFP4
- Router: ffn_gate_inp [2880,128] F32
- All have biases
**Method:**
1. Study ~/GITDEV/llama.cpp/src/models/openai-moe-iswa.cpp
2. Study ~/GITDEV/llama.cpp/src/llama-graph.cpp build_moe_ffn (line 1201)
3. Add to ggml_llama_gguf.c:
   - Detect expert_count from GGUF metadata (gpt-oss.expert_count)
   - Load expert weight tensors (3D: [in, out, n_experts])
   - In forward pass: replace dense FFN with:
     a. Router: ggml_mul_mat(gate_inp, cur) → [128, n_tokens]
     b. Softmax: ggml_soft_max → expert probabilities
     c. TopK: ggml_argsort_top_k(probs, 4) → selected_experts [4, n_tokens]
     d. Expert FFN: ggml_mul_mat_id(gate_exps, cur, selected_experts)
     e. SwiGLU: gate * silu(up)
     f. Down proj: ggml_mul_mat_id(down_exps, act, selected_experts)
     g. Weighted sum: multiply by expert weights, reduce
4. Add 'gpt-oss' to architecture prefix list
5. Test: python -c "from ggml_vllm_backend import GgmlLLM; llm = GgmlLLM('~/models/gguf/gpt-oss-120b-mxfp4.gguf'); print(llm.generate('Hello', max_tokens=10))"
**Reference:** ~/GITDEV/llama.cpp/src/models/openai-moe-iswa.cpp
**Model:** ~/models/gguf/gpt-oss-120b-mxfp4.gguf (60GB, merged)

### T08: [DONE] Fix tokenizer discovery for all model families
**Goal:** ggml_vllm_backend.py _find_tokenizer() only finds Llama-3.1-8B
**Method:**
1. Make _find_tokenizer() search for ANY matching HF model in cache
2. Match by model family name (qwen, llama, gpt-oss, etc.)
3. If no local tokenizer, try downloading from HF
**Test:** All models in ~/models/gguf/ should auto-find tokenizers

### T09: [BLOCKED by C engine crash] Benchmark 32B Qwen with tokenizer fix
**Goal:** Verify Qwen2.5-32B produces coherent output at 7+ TPS
**Test:** python -c "from ggml_vllm_backend import GgmlLLM, SamplingParams; llm = GgmlLLM('~/models/gguf/Qwen2.5-32B-Instruct-Q4_K_M.gguf'); r = llm.generate('What is quantum computing?', params=SamplingParams(temperature=0, max_tokens=30)); print(f'{r.tps:.0f} TPS: {r.text}')"

### T10: [DONE by OmniAgent [Main]]] Run 50-request stress test on standalone server
**Goal:** Verify server handles 50 diverse sequential requests without crash
**Method:** Start server, run curl loop with diverse prompts, verify all responses coherent
**Test:** All 50 responses must be non-empty and relevant

## RULES FOR AGENT:
1. Execute tasks IN ORDER (T06 → T07 → T08 → ...)
2. After each task: git add + commit with descriptive message
3. If compilation fails: fix the error and retry
4. If stuck >10 minutes on a task: ask_coder_brain for help
5. DO NOT modify files owned by other agents
6. Update this file to mark tasks as [DONE] when complete
