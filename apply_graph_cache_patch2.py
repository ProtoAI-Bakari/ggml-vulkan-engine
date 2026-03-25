#!/usr/bin/env python3
"""Apply graph caching patch to ggml_llama_gguf.c"""

with open('ggml_llama_gguf.c', 'r') as f:
    lines = f.readlines()

# 1. Add graph caching fields to engine_t struct (after w_buf line)
for i, line in enumerate(lines):
    if 'ggml_backend_buffer_t w_buf;' in line:
        indent = '    '
        lines.insert(i+1, indent + '/* Graph caching for decode phase */\n')
        lines.insert(i+2, indent + 'struct ggml_cgraph *cached_decode_graph;\n')
        lines.insert(i+3, indent + 'struct ggml_context *cached_decode_ctx;\n')
        lines.insert(i+4, indent + 'int cached_batch_size;\n')
        lines.insert(i+5, indent + 'bool graph_cached;\n')
        break

# 2. Initialize graph caching fields in engine_load_gguf (after compute_buf allocation)
for i, line in enumerate(lines):
    if 'e->compute_buf = malloc(e->compute_buf_size);' in line:
        # Find the closing brace of this block
        j = i
        while j < len(lines) and '}' not in lines[j]:
            j += 1
        # Insert after the closing brace
        lines.insert(j+1, '    /* Initialize graph caching fields */\n')
        lines.insert(j+2, '    e->cached_decode_graph = NULL;\n')
        lines.insert(j+3, '    e->cached_decode_ctx = NULL;\n')
        lines.insert(j+4, '    e->cached_batch_size = 0;\n')
        lines.insert(j+5, '    e->graph_cached = false;\n')
        break

# 3. Replace the graph creation in engine_forward with caching logic
for i, line in enumerate(lines):
    if 'struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);' in line:
        indent = '    '
        lines[i] = indent + '/* Graph caching: reuse graph for same batch size (decode phase) */\n'
        lines.insert(i+1, indent + 'struct ggml_cgraph *graph;\n')
        lines.insert(i+2, indent + 'if (n_tokens == 1 && e->graph_cached && e->cached_decode_graph) {\n')
        lines.insert(i+3, indent + '    /* Reuse cached graph - just reset the context */\n')
        lines.insert(i+4, indent + '    graph = e->cached_decode_graph;\n')
        lines.insert(i+5, indent + '    ggml_reset(e->cached_decode_ctx);\n')
        lines.insert(i+6, indent + '} else {\n')
        lines.insert(i+7, indent + '    /* Build new graph */\n')
        lines.insert(i+8, indent + '    graph = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);\n')
        lines.insert(i+9, indent + '}\n')
        break

# 4. Add graph caching after graph is built (before sched_alloc)
for i, line in enumerate(lines):
    if 'ggml_build_forward_expand(graph, cur);' in line and i > 350:  # Second occurrence in engine_forward
        indent = '    '
        lines.insert(i+1, '\n')
        lines.insert(i+2, indent + '/* Cache graph for batch=1 decode phase */\n')
        lines.insert(i+3, indent + 'if (n_tokens == 1 && !e->graph_cached) {\n')
        lines.insert(i+4, indent + '    /* Create persistent context for cached graph */\n')
        lines.insert(i+5, indent + '    struct ggml_init_params cp = {\n')
        lines.insert(i+6, indent + '        .mem_size = e->compute_buf_size,\n')
        lines.insert(i+7, indent + '        .mem_buffer = e->compute_buf,\n')
        lines.insert(i+8, indent + '        .no_alloc = true,\n')
        lines.insert(i+9, indent + '    };\n')
        lines.insert(i+10, indent + '    e->cached_decode_ctx = ggml_init(cp);\n')
        lines.insert(i+11, indent + '    if (e->cached_decode_ctx) {\n')
        lines.insert(i+12, indent + '        e->cached_decode_graph = graph;\n')
        lines.insert(i+13, indent + '        e->cached_batch_size = 1;\n')
        lines.insert(i+14, indent + '        e->graph_cached = true;\n')
        lines.insert(i+15, indent + '        fprintf(stderr, "[gguf] Graph cached for batch=1 decode\\n");\n')
        lines.insert(i+16, indent + '    }\n')
        lines.insert(i+17, indent + '}\n')
        break

# 5. Free cached graph in engine_free
for i, line in enumerate(lines):
    if 'void engine_free(engine_t *e) {' in line:
        lines.insert(i+1, '    if (e->cached_decode_ctx) ggml_free(e->cached_decode_ctx);\n')
        break

with open('ggml_llama_gguf.c', 'w') as f:
    f.writelines(lines)

print("Patch applied successfully")
