#!/usr/bin/env python3
"""Apply correct graph caching: use ggml_backend_sched_reserve() for buffer pre-allocation"""

with open('ggml_llama_gguf.c', 'r') as f:
    content = f.read()

# Add graph caching fields to engine_t struct
content = content.replace(
    '    ggml_backend_buffer_t w_buf;',
    '''    ggml_backend_buffer_t w_buf;
    /* Graph caching: pre-allocated buffers via sched_reserve */
    bool buffers_allocated;  /* Track if buffers are pre-allocated */'''
)

# Initialize in engine_load_gguf
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

    /* Initialize buffer allocation flag */
    e->buffers_allocated = false;

    gguf_free(gguf);'''
)

# Modify engine_forward to use sched_reserve on first call
old_sched_section = '''    ggml_backend_sched_reset(e->sched);
    if (!ggml_backend_sched_alloc_graph(e->sched, graph)) {
        return -2;
    }'''

new_sched_section = '''    /* Pre-allocate buffers on first forward pass (warmup) */
    if (!e->buffers_allocated) {
        /* Use sched_reserve to pre-allocate all buffers for this graph */
        ggml_backend_sched_reset(e->sched);
        if (!ggml_backend_sched_alloc_graph(e->sched, graph)) {
            return -2;
        }
        e->buffers_allocated = true;
        fprintf(stderr, "[gguf] Buffers pre-allocated (warmup complete)\n");
    } else {
        /* Buffers already allocated, just reset and compute */
        ggml_backend_sched_reset(e->sched);
        /* sched_alloc will reuse pre-allocated buffers */
        if (!ggml_backend_sched_alloc_graph(e->sched, graph)) {
            return -2;
        }
    }'''

content = content.replace(old_sched_section, new_sched_section)

with open('ggml_llama_gguf.c', 'w') as f:
    f.write(content)

print("Patch applied successfully")
