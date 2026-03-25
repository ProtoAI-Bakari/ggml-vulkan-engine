#!/usr/bin/env python3
"""
Step 3: Optimized ggml Vulkan matmul engine with cached backend.
Reuses Vulkan backend, pre-allocates weight buffers, minimal per-call overhead.
"""
import ctypes
import numpy as np
import os
import time

# ---- Load shared libraries ----
LIB_DIR = os.path.expanduser("~/GITDEV/llama.cpp/build-lib/bin")
libggml_base = ctypes.CDLL(os.path.join(LIB_DIR, "libggml-base.so"), mode=ctypes.RTLD_GLOBAL)
libggml_cpu = ctypes.CDLL(os.path.join(LIB_DIR, "libggml-cpu.so"), mode=ctypes.RTLD_GLOBAL)
libggml_vk = ctypes.CDLL(os.path.join(LIB_DIR, "libggml-vulkan.so"), mode=ctypes.RTLD_GLOBAL)
libggml = ctypes.CDLL(os.path.join(LIB_DIR, "libggml.so"), mode=ctypes.RTLD_GLOBAL)

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_DEFAULT_GRAPH_SIZE = 2048
GGML_STATUS_SUCCESS = 0

class ggml_init_params(ctypes.Structure):
    _fields_ = [
        ("mem_size", ctypes.c_size_t),
        ("mem_buffer", ctypes.c_void_p),
        ("no_alloc", ctypes.c_bool),
    ]

# Function signatures
libggml_base.ggml_init.argtypes = [ggml_init_params]; libggml_base.ggml_init.restype = ctypes.c_void_p
libggml_base.ggml_free.argtypes = [ctypes.c_void_p]; libggml_base.ggml_free.restype = None
libggml_base.ggml_new_tensor_2d.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_int64]
libggml_base.ggml_new_tensor_2d.restype = ctypes.c_void_p
libggml_base.ggml_mul_mat.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
libggml_base.ggml_mul_mat.restype = ctypes.c_void_p
libggml_base.ggml_new_graph.argtypes = [ctypes.c_void_p]; libggml_base.ggml_new_graph.restype = ctypes.c_void_p
libggml_base.ggml_build_forward_expand.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libggml_base.ggml_build_forward_expand.restype = None
libggml_base.ggml_set_name.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
libggml_base.ggml_set_name.restype = ctypes.c_void_p
libggml_base.ggml_nbytes.argtypes = [ctypes.c_void_p]; libggml_base.ggml_nbytes.restype = ctypes.c_size_t
libggml.ggml_backend_vk_init.argtypes = [ctypes.c_size_t]; libggml.ggml_backend_vk_init.restype = ctypes.c_void_p
libggml.ggml_backend_cpu_init.argtypes = []; libggml.ggml_backend_cpu_init.restype = ctypes.c_void_p
libggml.ggml_backend_free.argtypes = [ctypes.c_void_p]; libggml.ggml_backend_free.restype = None
libggml.ggml_backend_alloc_ctx_tensors.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libggml.ggml_backend_alloc_ctx_tensors.restype = ctypes.c_void_p
libggml.ggml_backend_tensor_set.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
libggml.ggml_backend_tensor_set.restype = None
libggml.ggml_backend_tensor_get.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
libggml.ggml_backend_tensor_get.restype = None
libggml.ggml_backend_graph_compute.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libggml.ggml_backend_graph_compute.restype = ctypes.c_int
libggml.ggml_backend_buffer_free.argtypes = [ctypes.c_void_p]; libggml.ggml_backend_buffer_free.restype = None
libggml.ggml_backend_sched_new.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_bool, ctypes.c_bool]
libggml.ggml_backend_sched_new.restype = ctypes.c_void_p
libggml.ggml_backend_sched_alloc_graph.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libggml.ggml_backend_sched_alloc_graph.restype = ctypes.c_bool
libggml.ggml_backend_sched_graph_compute.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libggml.ggml_backend_sched_graph_compute.restype = ctypes.c_int
libggml.ggml_backend_sched_free.argtypes = [ctypes.c_void_p]; libggml.ggml_backend_sched_free.restype = None
libggml.ggml_backend_sched_reset.argtypes = [ctypes.c_void_p]; libggml.ggml_backend_sched_reset.restype = None


class GgmlVulkanEngine:
    """Persistent ggml Vulkan compute engine. Init once, call matmul many times."""

    def __init__(self):
        self.backend_vk = libggml.ggml_backend_vk_init(0)
        assert self.backend_vk, "Failed to init Vulkan backend"
        self.backend_cpu = libggml.ggml_backend_cpu_init()
        assert self.backend_cpu, "Failed to init CPU backend"

        backends = (ctypes.c_void_p * 2)(self.backend_vk, self.backend_cpu)
        self.sched = libggml.ggml_backend_sched_new(backends, None, 2, GGML_DEFAULT_GRAPH_SIZE, False, False)
        assert self.sched, "Failed to create scheduler"

        self._weight_cache = {}  # (N, K) -> (ctx, buffer, tensor_ptr)

    def cache_weight(self, name, weight_np, dtype='f32'):
        """Pre-load a weight matrix onto Vulkan.
        weight_np: (N, K) float32 numpy array.
        dtype: 'f32' or 'f16' — storage type on GPU.
        """
        N, K = weight_np.shape
        weight_np = np.ascontiguousarray(weight_np, dtype=np.float32)

        params = ggml_init_params()
        params.mem_size = 64 * 1024 * 1024
        params.mem_buffer = None
        params.no_alloc = True
        ctx = libggml_base.ggml_init(params)

        if dtype == 'f16':
            ggml_type = GGML_TYPE_F16
        else:
            ggml_type = GGML_TYPE_F32

        t_w = libggml_base.ggml_new_tensor_2d(ctx, ggml_type, K, N)
        libggml_base.ggml_set_name(t_w, name.encode())

        buf = libggml.ggml_backend_alloc_ctx_tensors(ctx, self.backend_vk)
        assert buf, f"Failed to alloc weight {name} on Vulkan"

        if dtype == 'f16':
            # Convert F32 -> ggml F16 and upload
            n_elements = N * K
            f16_buf = (ctypes.c_uint16 * n_elements)()
            libggml_base.ggml_fp32_to_fp16_row.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
            libggml_base.ggml_fp32_to_fp16_row(weight_np.ctypes.data, f16_buf, n_elements)
            libggml.ggml_backend_tensor_set(t_w, f16_buf, 0, n_elements * 2)
        else:
            libggml.ggml_backend_tensor_set(t_w, weight_np.ctypes.data, 0, weight_np.nbytes)

        self._weight_cache[name] = (ctx, buf, t_w, N, K)

    def matmul(self, name, input_np):
        """Compute input @ weight.T using cached weight on Vulkan.
        input_np: (M, K) float32. Returns (M, N) float32."""
        ctx_w, buf_w, t_w, N, K = self._weight_cache[name]
        M = input_np.shape[0]
        assert input_np.shape[1] == K

        input_np = np.ascontiguousarray(input_np, dtype=np.float32)

        # Create compute context (lightweight — just metadata)
        params = ggml_init_params()
        params.mem_size = 64 * 1024 * 1024
        params.mem_buffer = None
        params.no_alloc = True
        ctx_c = libggml_base.ggml_init(params)

        t_in = libggml_base.ggml_new_tensor_2d(ctx_c, GGML_TYPE_F32, K, M)
        libggml_base.ggml_set_name(t_in, b"input")

        t_out = libggml_base.ggml_mul_mat(ctx_c, t_w, t_in)
        libggml_base.ggml_set_name(t_out, b"output")

        graph = libggml_base.ggml_new_graph(ctx_c)
        libggml_base.ggml_build_forward_expand(graph, t_out)

        libggml.ggml_backend_sched_reset(self.sched)
        ok = libggml.ggml_backend_sched_alloc_graph(self.sched, graph)
        assert ok, "Failed to alloc graph"

        libggml.ggml_backend_tensor_set(t_in, input_np.ctypes.data, 0, input_np.nbytes)

        status = libggml.ggml_backend_sched_graph_compute(self.sched, graph)
        assert status == GGML_STATUS_SUCCESS, f"Compute failed: {status}"

        result = np.empty((M, N), dtype=np.float32)
        libggml.ggml_backend_tensor_get(t_out, result.ctypes.data, 0, result.nbytes)

        libggml_base.ggml_free(ctx_c)
        return result

    def close(self):
        for name, (ctx, buf, _, _, _) in self._weight_cache.items():
            libggml.ggml_backend_buffer_free(buf)
            libggml_base.ggml_free(ctx)
        self._weight_cache.clear()
        libggml.ggml_backend_sched_free(self.sched)
        libggml.ggml_backend_free(self.backend_cpu)
        libggml.ggml_backend_free(self.backend_vk)


def benchmark():
    print("=" * 70)
    print("ggml Vulkan Engine — cached backend benchmark")
    print("=" * 70)

    engine = GgmlVulkanEngine()

    tests = [
        ("qkv", 2048, 1536, [1, 4, 16, 64, 256]),
        ("mlp_gate", 8960, 1536, [1, 4, 16, 64, 256]),
        ("mlp_down", 1536, 8960, [1, 4, 16, 64, 256]),
        ("8b_gate", 14336, 4096, [1, 4, 16, 64, 256]),
    ]

    for name, N, K, batches in tests:
        W = np.random.randn(N, K).astype(np.float32)
        engine.cache_weight(name, W)

        print(f"\n{name} ({K}x{N}):")
        for M in batches:
            X = np.random.randn(M, K).astype(np.float32)

            # Warmup
            engine.matmul(name, X)

            # Benchmark
            n_iters = max(1, min(20, int(100 / max(M, 1))))
            times = []
            for _ in range(n_iters):
                t0 = time.perf_counter()
                result = engine.matmul(name, X)
                times.append(time.perf_counter() - t0)

            avg_ms = np.median(times) * 1000
            flops = 2 * M * K * N
            gflops = flops / (np.median(times) * 1e9)

            # Correctness
            expected = X @ W.T
            cos = np.dot(expected.flatten(), result.flatten()) / (
                np.linalg.norm(expected.flatten()) * np.linalg.norm(result.flatten()) + 1e-10)

            # CPU reference
            t0 = time.perf_counter()
            for _ in range(n_iters):
                _ = X @ W.T
            cpu_ms = (time.perf_counter() - t0) / n_iters * 1000

            print(f"  batch={M:5d} | ggml: {avg_ms:7.2f}ms | CPU: {cpu_ms:7.2f}ms | "
                  f"{cpu_ms/avg_ms:5.1f}x | {gflops:8.1f} GFLOPS | cos={cos:.6f}")

    engine.close()


if __name__ == "__main__":
    # Quick smoke test
    engine = GgmlVulkanEngine()
    W = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)  # 3x2
    engine.cache_weight("test", W)
    X = np.array([[1, 0], [0, 1]], dtype=np.float32)  # 2x2
    result = engine.matmul("test", X)  # should be [[1,3,5],[2,4,6]]
    expected = X @ W.T
    print(f"Smoke test: expected=\n{expected}\ngot=\n{result}\nmatch={np.allclose(expected, result)}")
    engine.close()
    print()
    benchmark()
