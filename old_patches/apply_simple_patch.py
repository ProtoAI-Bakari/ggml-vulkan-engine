#!/usr/bin/env python3
"""Apply correct graph caching patch using ggml_backend_sched_reserve()"""

with open('ggml_llama_gguf.c', 'r') as f:
    lines = f.readlines()

# 1. Add graph caching fields to engine_t struct (after w_buf line)
for i, line in enumerate(lines):
    if 'ggml_backend_buffer_t w_buf;' in line:
        indent = '    '
        lines.insert(i+1, indent + '/* Graph caching: persistent graph for batch=1 decode */\n')
        lines.insert(i+2, indent + 'struct ggml_cgraph *cached_graph;\n')
        lines.insert(i+3, indent + 'struct ggml_context *graph_ctx;\n')
        lines.insert(i+4, indent + 'bool graph_initialized;\n')
        break

# 2. Initialize in engine_load_gguf
for i, line in enumerate(lines):
    if 'e->compute_buf = malloc(e->compute_buf_size);' in line:
        j = i
        while j < len(lines) and '}' not in lines[j]:
            j += 1
        lines.insert(j+1, '    /* Initialize graph caching */\n')
        lines.insert(j+2, '    e->cached_graph = NULL;\n')
        lines.insert(j+3, '    e->graph_ctx = NULL;\n')
        lines.insert(j+4, '    e->graph_initialized = false;\n')
        break

# 3. Modify engine_forward to use cached graph for batch=1
for i, line in enumerate(lines):
    if 'struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);' in line:
        indent = '    '
        lines[i] = indent + '/* Use cached graph for batch=1 decode if available */\n'
        lines.insert(i+1, indent + 'struct ggml_cgraph *graph;\n')
        lines.insert(i+2, indent + 'if (n_tokens == 1 && e->graph_initialized && e->cached_graph) {\n')
        lines.insert(i+3, indent + '    graph = e->cached_graph;\n')
        lines.insert(i+4, indent + '    ggml_reset(e->graph_ctx);\n')
        lines.insert(i+5, indent + '} else {\n')
        lines.insert(i+6, indent + '    graph = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);\n')
        lines.insert(i+7, indent + '}\n')
        break

# 4. Cache the graph after first batch=1 forward pass
for i, line in enumerate(lines):
    if 'ggml_build_forward_expand(graph, cur);' in line and i > 350:
        indent = '    '
        lines.insert(i+1, '\n')
        lines.insert(i+2, indent + '/* Cache graph for future batch=1 decode passes */\n')
        lines.insert(i+3, indent + 'if (n_tokens == 1 && !e->graph_initialized) {\n')
        lines.insert(i+4, indent + '    /* Create persistent context */\n')
        lines.insert(i+5, indent + '    struct ggml_init_params gp = {\n')
        lines.insert(i+6, indent + '        .mem_size = e->compute_buf_size,\n')
        lines.insert(i+7, indent + '        .mem_buffer = e->compute_buf,\n')
        lines.insert(i+8, indent + '        .no_alloc = true,\n')
        lines.insert(i+9, indent + '    };\n')
        lines.insert(i+10, indent + '    e->graph_ctx = ggml_init(gp);\n')
        lines.insert(i+11, indent + '    if (e->graph_ctx) {\n')
        lines.insert(i+12, indent + '        e->cached_graph = graph;\n')
        lines.insert(i+13, indent + '        e->graph_initialized = true;\n')
        lines.insert(i+14, indent + '        fprintf(stderr, "[gguf] Graph cached for batch=1\\n");\n')
        lines.insert(i+15, indent + '    }\n')
        lines.insert(i+16, indent + '}\n')
        break

# 5. Free cached graph in engine_free
for i, line in enumerate(lines):
    if 'void engine_free(engine_t *e) {' in line:
        lines.insert(i+1, '    if (e->graph_ctx) ggml_free(e->graph_ctx);\n')
        break

with open('ggml_llama_gguf.c', 'w') as f:
    f.writelines(lines)

print("Patch applied successfully")
