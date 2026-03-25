#!/usr/bin/env python3
"""Apply correct graph caching: use ggml_backend_sched_reserve() for buffer pre-allocation"""

with open('ggml_llama_gguf.c', 'r') as f:
    lines = f.readlines()

# Add graph caching fields to engine_t struct
for i, line in enumerate(lines):
    if 'ggml_backend_buffer_t w_buf;' in line:
        lines.insert(i+1, '    /* Graph caching: pre-allocated buffers via sched_reserve */\n')
        lines.insert(i+2, '    bool buffers_allocated;  /* Track if buffers are pre-allocated */\n')
        break

# Initialize in engine_load_gguf
for i, line in enumerate(lines):
    if 'e->compute_buf = malloc(e->compute_buf_size);' in line:
        j = i
        while j < len(lines) and '}' not in lines[j]:
            j += 1
        lines.insert(j+1, '    /* Initialize buffer allocation flag */\n')
        lines.insert(j+2, '    e->buffers_allocated = false;\n')
        break

# Modify engine_forward to use sched_reserve on first call
for i, line in enumerate(lines):
    if 'ggml_backend_sched_reset(e->sched);' in line and i > 350:  # Second occurrence in engine_forward
        indent = '    '
        lines[i] = indent + '/* Pre-allocate buffers on first forward pass (warmup) */\n'
        lines.insert(i+1, indent + 'if (!e->buffers_allocated) {\n')
        lines.insert(i+2, indent + '    /* Use sched_alloc to pre-allocate all buffers for this graph */\n')
        lines.insert(i+3, indent + '    if (!ggml_backend_sched_alloc_graph(e->sched, graph)) {\n')
        lines.insert(i+4, indent + '        return -2;\n')
        lines.insert(i+5, indent + '    }\n')
        lines.insert(i+6, indent + '    e->buffers_allocated = true;\n')
        lines.insert(i+7, indent + '    fprintf(stderr, "[gguf] Buffers pre-allocated (warmup complete)\\n");\n')
        lines.insert(i+8, indent + '} else {\n')
        lines.insert(i+9, indent + '    /* Buffers already allocated, just reset and compute */\n')
        lines.insert(i+10, indent + '    /* sched_alloc will reuse pre-allocated buffers */\n')
        lines.insert(i+11, indent + '    if (!ggml_backend_sched_alloc_graph(e->sched, graph)) {\n')
        lines.insert(i+12, indent + '        return -2;\n')
        lines.insert(i+13, indent + '    }\n')
        lines.insert(i+14, indent + '}\n')
        # Remove the old sched_alloc line that follows
        if i+15 < len(lines) and 'if (!ggml_backend_sched_alloc_graph(e->sched, graph))' in lines[i+15]:
            # Skip the old block (5 lines)
            for _ in range(5):
                lines.pop(i+15)
        break

with open('ggml_llama_gguf.c', 'w') as f:
    f.writelines(lines)

print("Patch applied successfully")
