#!/usr/bin/env python3
"""Apply T05 timing instrumentation to ggml_llama_gguf.c using targeted edits"""

with open('ggml_llama_gguf.c', 'r') as f:
    content = f.read()

# 1. Add timing fields to engine_t struct
content = content.replace(
    '    float rms_eps, rope_theta;\n\n    /* Pre-allocated compute buffer',
    '    float rms_eps, rope_theta;\n\n    /* T05: Timing instrumentation */\n    double t_graph_build_us;\n    double t_backend_compute_us;\n    int64_t token_count;\n\n    /* Pre-allocated compute buffer'
)

# 2. Add timing start at beginning of engine_forward
content = content.replace(
    'int engine_forward(engine_t *e, int n_tokens,\n                   const int32_t *tokens, const int32_t *positions,\n                   float *logits_out) {\n    int H = e->hidden_dim, I = e->intermediate;\n    int NH = e->n_heads, NKV = e->n_kv_heads, HD = e->head_dim;',
    'int engine_forward(engine_t *e, int n_tokens,\n                   const int32_t *tokens, const int32_t *positions,\n                   float *logits_out) {\n    int H = e->hidden_dim, I = e->intermediate;\n    int NH = e->n_heads, NKV = e->n_kv_heads, HD = e->head_dim;\n    double t_start = ggml_time_us();\n    e->token_count++;'
)

# 3. Add graph build timing start
content = content.replace(
    '    struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);',
    '    double t_graph_start = ggml_time_us();\n    struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);'
)

# 4. Add graph build timing end after alloc_graph check
content = content.replace(
    '    if (!ggml_backend_sched_alloc_graph(e->sched, graph)) {\n        return -2;\n    }',
    '    if (!ggml_backend_sched_alloc_graph(e->sched, graph)) {\n        e->t_graph_build_us = ggml_time_us() - t_graph_start;\n        e->t_backend_compute_us = 0;\n        return -2;\n    }\n    e->t_graph_build_us = ggml_time_us() - t_graph_start;'
)

# 5. Add compute timing around graph_compute
content = content.replace(
    '    enum ggml_status status = ggml_backend_sched_graph_compute(e->sched, graph);\n    if (status != GGML_STATUS_SUCCESS) {',
    '    double t_compute_start = ggml_time_us();\n    enum ggml_status status = ggml_backend_sched_graph_compute(e->sched, graph);\n    e->t_backend_compute_us = ggml_time_us() - t_compute_start;\n    double t_total = ggml_time_us() - t_start;\n    \n    /* Print timing every 10 tokens */\n    if (e->token_count % 10 == 0) {\n        fprintf(stderr, "[Timing @ %ld tokens] Graph: %.1f\\u03bcs, Compute: %.1f\\u03bcs, Total: %.1f\\u03bcs (%.1f TPS)\\n",\n                (long)e->token_count,\n                e->t_graph_build_us,\n                e->t_backend_compute_us,\n                t_total,\n                1e6 / t_total);\n    }\n    \n    if (status != GGML_STATUS_SUCCESS) {'
)

with open('ggml_llama_gguf.c', 'w') as f:
    f.write(content)

print("T05 timing instrumentation applied successfully")
