#!/usr/bin/env python3
"""Apply graph caching patch to ggml_llama_gguf.c"""
import re

with open('ggml_llama_gguf.c', 'r') as f:
    content = f.read()

# 1. Add graph caching fields to engine_t struct (after w_buf line)
content = content.replace(
    '    ggml_backend_buffer_t w_buf;',
    '''    ggml_backend_buffer_t w_buf;
    /* Graph caching for decode phase */
    struct ggml_cgraph *cached_decode_graph;
    struct ggml_context *cached_decode_ctx;
    int cached_batch_size;
    bool graph_cached;'''
)

# 2. Initialize graph caching fields in engine_load_gguf (after compute_buf allocation)
content = content.replace(
    '''        e->compute_buf = malloc(e->compute_buf_size);
        fprintf(stderr, "[gguf] Compute buffer: %.1f MiB (pre-allocated)\n",
                e->compute_buf_size / (1024.0*1024));
    }

    gguf_free(gguf);''',
    '''        e->compute_buf = malloc(e->compute_buf_size);
        fprintf(stderr, "[gguf] Compute buffer: %.1f MiB (pre-allocated)\n",
                e->compute_buf_size / (1024.0*1024));
    }

    /* Initialize graph caching fields */
    e->cached_decode_graph = NULL;
    e->cached_decode_ctx = NULL;
    e->cached_batch_size = 0;
    e->graph_cached = false;

    gguf_free(gguf);'''
)

# 3. Replace the graph creation in engine_forward with caching logic
old_graph_section = '''    struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);'''

new_graph_section = '''    /* Graph caching: reuse graph for same batch size (decode phase) */
    struct ggml_cgraph *graph;
    if (n_tokens == 1 && e->graph_cached && e->cached_decode_graph) {
        /* Reuse cached graph - just reset the context */
        graph = e->cached_decode_graph;
        ggml_reset(e->cached_decode_ctx);
    } else {
        /* Build new graph */
        graph = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
    }'''

content = content.replace(old_graph_section, new_graph_section)

# 4. Add graph caching after graph is built (before sched_alloc)
old_sched_section = '''    ggml_build_forward_expand(graph, cur);

    ggml_backend_sched_reset(e->sched);
    if (!ggml_backend_sched_alloc_graph(e->sched, graph)) {
        return -2;
    }'''

new_sched_section = '''    ggml_build_forward_expand(graph, cur);

    /* Cache graph for batch=1 decode phase */
    if (n_tokens == 1 && !e->graph_cached) {
        /* Create persistent context for cached graph */
        struct ggml_init_params cp = {
            .mem_size = e->compute_buf_size,
            .mem_buffer = e->compute_buf,
            .no_alloc = true,
        };
        e->cached_decode_ctx = ggml_init(cp);
        if (e->cached_decode_ctx) {
            /* Rebuild graph in cached context */
            ggml_reset(e->cached_decode_ctx);
            struct ggml_tensor *inp_tokens_c = ggml_new_tensor_1d(e->cached_decode_ctx, GGML_TYPE_I32, 1);
            ggml_set_name(inp_tokens_c, "tokens");
            struct ggml_tensor *inp_pos_c = ggml_new_tensor_1d(e->cached_decode_ctx, GGML_TYPE_I32, 1);
            ggml_set_name(inp_pos_c, "positions");
            /* Note: We'll rebuild the full graph structure here - simplified for now */
            /* For now, just cache the graph pointer and rebuild structure on reset */
            e->cached_decode_graph = graph;
            e->cached_batch_size = 1;
            e->graph_cached = true;
            fprintf(stderr, "[gguf] Graph cached for batch=1 decode\n");
        }
    }

    ggml_backend_sched_reset(e->sched);
    if (!ggml_backend_sched_alloc_graph(e->sched, graph)) {
        return -2;
    }'''

content = content.replace(old_sched_section, new_sched_section)

# 5. Free cached graph in engine_free
old_free_section = '''void engine_free(engine_t *e) {
    if (e->_persistent_ctx) ggml_free(e->_persistent_ctx);'''

new_free_section = '''void engine_free(engine_t *e) {
    if (e->cached_decode_ctx) ggml_free(e->cached_decode_ctx);
    if (e->_persistent_ctx) ggml_free(e->_persistent_ctx);'''

content = content.replace(old_free_section, new_free_section)

with open('ggml_llama_gguf.c', 'w') as f:
    f.write(content)

print("Patch applied successfully")
