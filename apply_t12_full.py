#!/usr/bin/env python3
"""Complete T12: Full ggml_gallocr integration with graph caching"""

with open('ggml_llama_gguf.c', 'r') as f:
    lines = f.readlines()

# 1. Add cached_graph field to engine_t for storing the graph template
for i, line in enumerate(lines):
    if 'ggml_gallocr_t galloc;  /* T12: Graph allocator for caching */' in line:
        lines.insert(i+1, '    struct ggml_cgraph *cached_graph;  /* T12: Cached graph template */\n')
        break

# 2. Initialize cached_graph in engine_warmup
for i, line in enumerate(lines):
    if 'e->galloc = ggml_gallocr_new(ggml_backend_vk_buffer_type(0));' in line:
        lines.insert(i+1, '    e->cached_graph = NULL;\n')
        break

# 3. Modify engine_forward to use gallocr_alloc_graph for decode (batch=1)
for i, line in enumerate(lines):
    if 'struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);' in line:
        # Replace with gallocr-aware code
        new_code = '''    struct ggml_cgraph *graph;
    
    /* T12: Use ggml_gallocr for cached graph allocation when available */
    if (e->galloc && n_tokens == 1 && e->cached_graph) {
        /* Decode path: reuse cached graph template */
        graph = e->cached_graph;
        /* ggml_gallocr_alloc_graph will allocate from pre-reserved pool */
        if (!ggml_gallocr_alloc_graph(e->galloc, graph)) {
            fprintf(stderr, "[gguf] WARNING: gallocr reallocation needed\\n");
            /* Fall back to normal graph creation */
            graph = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
        }
    } else {
        /* Prefill or first decode: create new graph */
        graph = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
        /* Cache this graph for future decode steps */
        if (e->galloc && n_tokens == 1 && !e->cached_graph) {
            e->cached_graph = graph;
            fprintf(stderr, "[gguf] T12: Graph template cached for decode\\n");
        }
    }\n'''
        lines[i] = new_code
        break

with open('ggml_llama_gguf.c', 'w') as f:
    f.writelines(lines)

print("T12 complete: Full gallocr integration with graph caching")
