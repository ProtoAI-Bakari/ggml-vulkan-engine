#!/usr/bin/env python3
"""Apply T06 graph caching patch to ggml_llama_gguf.c"""
import re

with open('ggml_llama_gguf.c', 'r') as f:
    lines = f.readlines()

# Find line numbers for key locations
forward_start = None
ggml_new_graph_line = None
backend_sched_reset_line = None

for i, line in enumerate(lines):
    if 'int engine_forward(engine_t *e, int n_tokens,' in line:
        forward_start = i
    if forward_start and 'struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);' in line:
        ggml_new_graph_line = i
    if ggml_new_graph_line and 'ggml_backend_sched_reset(e->sched);' in line:
        backend_sched_reset_line = i
        break

print(f"Found engine_forward at line {forward_start}")
print(f"Found ggml_new_graph_custom at line {ggml_new_graph_line}")
print(f"Found ggml_backend_sched_reset at line {backend_sched_reset_line}")

if not all([forward_start, ggml_new_graph_line, backend_sched_reset_line]):
    print("ERROR: Could not find all required lines")
    exit(1)

# Insert reuse_cached_graph check after function signature
insert_pos = ggml_new_graph_line
lines.insert(insert_pos, "    /* T06: Graph caching for decode - reuse graph structure on single-token decode */\n")
lines.insert(insert_pos + 1, "    bool reuse_cached_graph = (n_tokens == 1 && e->cached_decode_graph != NULL);\n")
lines.insert(insert_pos + 2, "\n")

# Replace the graph initialization line
new_graph_lines = [
    "    struct ggml_cgraph *graph;\n",
    "    \n",
    "    if (reuse_cached_graph) {\n",
    "        /* Reuse cached decode graph - skip rebuilding */\n",
    "        graph = e->cached_decode_graph;\n",
    "    } else {\n",
    "        /* Build new graph (prefill or first decode) */\n",
    "        graph = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);\n",
    "    }\n",
]

# Remove old line and insert new ones
lines[ggml_new_graph_line + 3] = ''  # Clear the old line
for i, new_line in enumerate(reversed(new_graph_lines)):
    lines.insert(ggml_new_graph_line + 3, new_line)

# Find the ggml_build_forward_expand line after backend_sched_reset
build_line = None
for i in range(backend_sched_reset_line, min(backend_sched_reset_line + 5, len(lines))):
    if 'ggml_build_forward_expand(graph, cur);' in lines[i]:
        build_line = i
        break

print(f"Found ggml_build_forward_expand at line {build_line}")

if build_line:
    # Replace the build line with caching logic
    caching_logic = [
        "    \n",
        "    /* T06: Cache the graph if this is first decode pass */\n",
        "    if (!reuse_cached_graph && n_tokens == 1 && e->cached_decode_graph == NULL) {\n",
        "        /* Build graph and cache it for future decode passes */\n",
        "        ggml_build_forward_expand(graph, cur);\n",
        "        \n",
        "        /* Duplicate graph into persistent context for caching */\n",
        "        size_t graph_size = ggml_graph_overhead_custom(graph->size, false);\n",
        "        e->cached_graph_ctx = ggml_init((struct ggml_init_params){\n",
        "            .mem_size   = graph_size,\n",
        "            .mem_buffer = NULL,\n",
        "            .no_alloc   = false\n",
        "        });\n",
        "        e->cached_decode_graph = ggml_graph_dup(e->cached_graph_ctx, graph);\n",
        "    } else if (!reuse_cached_graph) {\n",
        "        /* Non-cached path (prefill or batch > 1) */\n",
        "        ggml_build_forward_expand(graph, cur);\n",
        "    }\n",
        "    \n",
    ]
    
    # Remove old build line and insert new logic
    lines[build_line] = ''
    for i, new_line in enumerate(reversed(caching_logic)):
        lines.insert(build_line, new_line)

# Write the modified file
with open('ggml_llama_gguf.c', 'w') as f:
    f.writelines(lines)

print("Patch applied successfully!")
