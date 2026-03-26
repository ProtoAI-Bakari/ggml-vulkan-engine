#!/usr/bin/env python3
"""Add graph topology logging to ggml_llama_gguf.c for T06"""

with open('ggml_llama_gguf.c', 'r') as f:
    lines = f.readlines()

# Find the line with ggml_backend_tensor_set(inp_pos, positions...
insert_pos = None
for i, line in enumerate(lines):
    if 'ggml_backend_tensor_set(inp_pos, positions, 0, n_tokens * sizeof(int32_t));' in line:
        insert_pos = i + 1
        break

if not insert_pos:
    print("ERROR: Could not find insertion point")
    exit(1)

# Insert graph topology logging
logging_code = [
    "    /* T06: Print graph topology for documentation */\n",
    "    if (n_tokens == 1 && positions[0] < 2) {\n",
    "        fprintf(stderr, \"=== GGML Graph Topology (pos=%d, n_tokens=%d) ===\\n\", positions[0], n_tokens);\n",
    "        fprintf(stderr, \"Graph nodes: %d\\n\", ggml_graph_n_nodes(graph));\n",
    "        int views=0, casts=0, compute=0;\n",
    "        for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {\n",
    "            struct ggml_tensor *node = ggml_graph_node(graph, i);\n",
    "            if (node->op == GGML_OP_CPY || node->op == GGML_OP_VIEW) views++;\n",
    "            else if (node->op == GGML_OP_CAST) casts++;\n",
    "            else compute++;\n",
    "        }\n",
    "        fprintf(stderr, \"Breakdown: views=%d, casts=%d, compute=%d\\n\", views, casts, compute);\n",
    "    }\n",
    "\n",
]

for i, new_line in enumerate(reversed(logging_code)):
    lines.insert(insert_pos, new_line)

with open('ggml_llama_gguf.c', 'w') as f:
    f.writelines(lines)

print("Graph topology logging added successfully!")
