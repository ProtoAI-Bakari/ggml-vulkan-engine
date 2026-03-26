#!/usr/bin/env python3
"""Apply T37 logits output shaping patch to ggml_llama_gguf.c"""
import re

with open('ggml_llama_gguf.c', 'r') as f:
    content = f.read()

# 1. Add n_requests field after vocab_size (around line 50)
content = re.sub(
    r'(    int vocab_size, n_ctx;\n)',
    r'\1    int n_requests;  /* For batched forward: number of concurrent requests */\n',
    content
)

# 2. Add batch_* fields after compute_buf_size (around line 58)
content = re.sub(
    r'(    size_t compute_buf_size;\n)',
    r'\1    int32_t *batch_tokens;  /* [total_tokens] flattened tokens from all requests */\n    int32_t *batch_positions;  /* [total_tokens] flattened positions */\n    int32_t *batch_seq_lens;  /* [n_requests] sequence length per request */\n    int32_t *batch_block_tables;  /* [n_requests * max_blocks] paged KV block tables */\n',
    content
)

# 3. Add batched forward declaration after regular forward declaration
content = re.sub(
    r'(int engine_forward\(engine_t \*e, int n_tokens, const int32_t \*tokens, const int32_t \*positions, float \*logits_out\);\n)',
    r'\1\n/* Batched forward pass for vLLM integration (T37) */\nint engine_forward_batch(engine_t *e, int n_tokens, const int32_t *tokens, const int32_t *positions,\n                         const int32_t *seq_lens, const int32_t *block_tables, float *logits_out);\n',
    content
)

# 4. Add batch buffer allocation in engine_load_gguf (after compute_buf allocation)
# Find the compute_buf allocation and add batch buffer allocation after it
compute_buf_alloc = r'(    e->compute_buf_size = \(size_t\)max_nodes \* ggml_tensor_overhead\(\)\n                            \+ ggml_graph_overhead_custom\(GRAPH_SIZE, false\);\n    e->compute_buf = malloc\(e->compute_buf_size\);\n    if \(!e->compute_buf\) \{\n        fprintf\(stderr, "\[gguf\] ERROR: Failed to allocate compute buffer\\n\);\n        engine_free\(e\);\n        return NULL;\n    \})'

batch_alloc = r'''\1

    /* Allocate batch buffers for vLLM integration (T37) */
    e->batch_tokens = (int32_t*)malloc(8192 * sizeof(int32_t));  /* Max 8K tokens per batch */
    e->batch_positions = (int32_t*)malloc(8192 * sizeof(int32_t));
    e->batch_seq_lens = (int32_t*)malloc(16 * sizeof(int32_t));  /* Max 16 concurrent requests */
    e->batch_block_tables = (int32_t*)malloc(16 * 64 * sizeof(int32_t));  /* 16 reqs × 64 blocks */
    if (!e->batch_tokens || !e->batch_positions || !e->batch_seq_lens || !e->batch_block_tables) {
        fprintf(stderr, "[gguf] ERROR: Failed to allocate batch buffers\n");
        engine_free(e);
        return NULL;
    }'''

content = re.sub(compute_buf_alloc, batch_alloc, content)

# 5. Add batch buffer freeing in engine_free
content = re.sub(
    r'(    if \(e->compute_buf\) free\(e->compute_buf\);)',
    r'    if (e->compute_buf) free(e->compute_buf);\n    if (e->batch_tokens) free(e->batch_tokens);\n    if (e->batch_positions) free(e->batch_positions);\n    if (e->batch_seq_lens) free(e->batch_seq_lens);\n    if (e->batch_block_tables) free(e->batch_block_tables);',
    content
)

with open('ggml_llama_gguf.c', 'w') as f:
    f.write(content)

print("T37 patch applied successfully")
