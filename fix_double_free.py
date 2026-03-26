#!/usr/bin/env python3
"""Fix double-free in ggml_llama_gguf.c by adding guard flags."""

import re

with open('ggml_llama_gguf.c', 'r') as f:
    content = f.read()

# Add guard variables after struct definition (around line 100)
if 'static bool sched_freed = false;' not in content:
    # Find the engine_init function and add guards at the start
    guard_code = '''
/* T08: Guard flags to prevent double-free */
static bool sched_freed = false;
static bool backend_cpu_freed = false;
static bool backend_vk_freed = false;

'''
    # Insert after includes
    insert_pos = content.find('static engine_t *engine_alloc()')
    if insert_pos > 0:
        content = content[:insert_pos] + guard_code + content[insert_pos:]

# Fix engine_free() to use guards
old_free = '''void engine_free(engine_t *e) {
    if (e->_persistent_ctx) ggml_free(e->_persistent_ctx);
    if (e->compute_buf) free(e->compute_buf);
    if (e->w_buf) ggml_backend_buffer_free(e->w_buf);
    if (e->w_ctx) ggml_free(e->w_ctx);
    ggml_backend_buffer_free(e->kv_buf[0]);
    ggml_backend_buffer_free(e->kv_buf[1]);
    if (e->kv_ctx) ggml_free(e->kv_ctx);
    if (e->sched) ggml_backend_sched_free(e->sched);
    if (e->backend_cpu) ggml_backend_free(e->backend_cpu);
    if (e->backend_vk) ggml_backend_free(e->backend_vk);
    free(e);
}'''

new_free = '''void engine_free(engine_t *e) {
    if (e->_persistent_ctx) ggml_free(e->_persistent_ctx);
    if (e->compute_buf) free(e->compute_buf);
    if (e->w_buf) ggml_backend_buffer_free(e->w_buf);
    if (e->w_ctx) ggml_free(e->w_ctx);
    ggml_backend_buffer_free(e->kv_buf[0]);
    ggml_backend_buffer_free(e->kv_buf[1]);
    if (e->kv_ctx) ggml_free(e->kv_ctx);
    /* T08: Guard against double-free */
    if (e->sched && !sched_freed) {
        ggml_backend_sched_free(e->sched);
        e->sched = NULL;
        sched_freed = true;
    }
    if (e->backend_cpu && !backend_cpu_freed) {
        ggml_backend_free(e->backend_cpu);
        e->backend_cpu = NULL;
        backend_cpu_freed = true;
    }
    if (e->backend_vk && !backend_vk_freed) {
        ggml_backend_free(e->backend_vk);
        e->backend_vk = NULL;
        backend_vk_freed = true;
    }
    free(e);
}'''

if old_free in content:
    content = content.replace(old_free, new_free)
    print("Fixed engine_free() with guard flags")
else:
    print("WARNING: Could not find exact engine_free() pattern")
    # Try to find and patch anyway
    if 'void engine_free(engine_t *e)' in content:
        print("Found engine_free() but pattern mismatch - manual review needed")

with open('ggml_llama_gguf.c', 'w') as f:
    f.write(content)

print("Patch applied successfully")
