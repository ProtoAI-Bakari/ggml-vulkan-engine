#!/usr/bin/env python3
"""Fix the CAST op issue in ggml_llama_gguf.c"""

with open('ggml_llama_gguf.c', 'r') as f:
    content = f.read()

# Replace CAST-related code
content = content.replace(
    'else if (node->op == GGML_OP_CAST) casts++;',
    '/* CAST ops not tracked in this ggml version */'
)
content = content.replace(
    'int views=0, casts=0, compute=0;',
    'int views=0, compute=0;'
)
content = content.replace(
    'fprintf(stderr, "Breakdown: views=%d, casts=%d, compute=%d\\n", views, casts, compute);',
    'fprintf(stderr, "Breakdown: views=%d, compute=%d\\n", views, compute);'
)

with open('ggml_llama_gguf.c', 'w') as f:
    f.write(content)

print("Fixed CAST op issue!")
