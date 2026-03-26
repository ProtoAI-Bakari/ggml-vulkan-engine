#!/usr/bin/env python3
"""T12: Add ggml_gallocr graph caching to ggml_llama_gguf.c"""

with open('ggml_llama_gguf.c', 'r') as f:
    lines = f.readlines()

# 1. Add galloc field to engine_t struct (after line 72 which has "int return_hidden;")
for i, line in enumerate(lines):
    if 'int return_hidden;' in line:
        lines.insert(i+1, '    ggml_gallocr_t galloc;  /* T12: Graph allocator for caching */\n')
        break

# 2. Add galloc initialization in engine_warmup (before the return 0;)
for i, line in enumerate(lines):
    if 'e->graph_built = 0;' in line:
        # Insert after this line, before return 0
        j = i + 1
        while j < len(lines) and 'return 0;' not in lines[j]:
            j += 1
        if j < len(lines):
            lines.insert(j, '    \n')
            lines.insert(j+1, '    /* T12: Initialize ggml_gallocr for graph caching */\n')
            lines.insert(j+2, '    e->galloc = ggml_gallocr_new(ggml_backend_vk_buffer_type(0));\n')
            lines.insert(j+3, '    if (!e->galloc) {\n')
            lines.insert(j+4, '        fprintf(stderr, "[gguf] ERROR: Failed to create ggml_gallocr\\n");\n')
            lines.insert(j+5, '        return -1;\n')
            lines.insert(j+6, '    }\n')
            lines.insert(j+7, '    fprintf(stderr, "[gguf] Graph allocator created for T12 caching\\n");\n')
        break

# 3. Add galloc free in engine_free
for i, line in enumerate(lines):
    if 'if (e->sched) ggml_backend_sched_free(e->sched);' in line:
        lines.insert(i+1, '    if (e->galloc) ggml_gallocr_free(e->galloc);\n')
        break

# 4. Modify engine_forward to use gallocr (replace graph creation)
for i, line in enumerate(lines):
    if 'struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);' in line:
        lines[i] = '''    struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
    
    /* T12: With gallocr created, future work will use ggml_gallocr_alloc_graph() */
    /* Full integration requires: (1) reserve worst-case graph in init, (2) fingerprint topology (T11) */
    /* For now, allocator is ready for T11 fingerprinting integration */\n'''
        break

with open('ggml_llama_gguf.c', 'w') as f:
    f.writelines(lines)

print("T12 patch applied: galloc field added, initialized in warmup, freed in engine_free")
