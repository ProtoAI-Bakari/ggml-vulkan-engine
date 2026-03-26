#!/usr/bin/env python3
"""
T05: Optimized Engine with Graph Caching
Key insight: The 1.48ms median is good, but 78ms spikes indicate graph allocation issues.
Solution: Cache the graph structure and only update input tensor.
"""
import ctypes
import numpy as np
import os
import time

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


class CachedGraph:
    """Cached graph structure for a specific (M, N, K) shape"""
    def __init__(self, ctx_c, t_in, t_out, graph, sched, N, K, max_M):
        self.ctx_c = ctx_c
        self.t_in = t_in
        self.t_out = t_out
        self.graph = graph
        self.sched = sched
        self.N = N
        self.K = K
        self.max_M = max_M
        self.allocated = True


class GgmlVulkanEngineOptimized:
    """Optimized engine with graph caching for repeated shapes"""

    def __init__(self, max_graphs=10):
        self.backend_vk = libggml.ggml_backend_vk_init(0)
        assert self.backend_vk, "Failed to init Vulkan backend"
        self.backend_cpu = libggml.ggml_backend_cpu_init()
        assert self.backend_cpu, "Failed to init CPU backend"

        backends = (ctypes.c_void_p * 2)(self.backend_vk, self.backend_cpu)
        self.sched = libggml.ggml_backend_sched_new(backends, None, 2, GGML_DEFAULT_GRAPH_SIZE, False, False)
        assert self.sched, "Failed to create scheduler"

        self._weight_cache = {}  # name -> (ctx, buf, t_w, N, K)
        self._graph_cache = {}   # (name, M) -> CachedGraph
        self.max_graphs = max_graphs

    def cache_weight(self, name, weight_np, dtype='f32'):
        """Pre-load a weight matrix onto Vulkan."""
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

        t_W = libggml_base.ggml_new_tensor_2d(ctx, ggml_type, K, N)
        libggml_base.ggml_set_name(t_W, name.encode())

        buf = libggml.ggml_backend_alloc_ctx_tensors(ctx, self.backend_vk)
        assert buf, f"Failed to alloc weight {name} on Vulkan"

        if dtype == 'f16':
            n_elements = N * K
            f16_buf = (ctypes.c_uint16 * n_elements)()
            libggml_base.ggml_fp32_to_fp16_row.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
            libggml_base.ggml_fp32_to_fp16_row(weight_np.ctypes.data, f16_buf, n_elements)
            libggml.ggml_backend_tensor_set(t_W, f16_buf, 0, n_elements * 2)
        else:
            libggml.ggml_backend_tensor_set(t_W, weight_np.ctypes.data, 0, weight_np.nbytes)

        self._weight_cache[name] = (ctx, buf, t_W, N, K)

    def _get_or_create_graph(self, name, M):
        """Get cached graph or create new one (LRU eviction if needed)"""
        cache_key = (name, M)
        
        if cache_key in self._graph_cache:
            return self._graph_cache[cache_key]
        
        # Evict oldest if at capacity
        if len(self._graph_cache) >= self.max_graphs:
            oldest_key = next(iter(self._graph_cache))
            old_graph = self._graph_cache.pop(oldest_key)
            libggml_base.ggml_free(old_graph.ctx_c)
        
        ctx_w, buf_w, t_W, N, K = self._weight_cache[name]
        
        # Create new graph context
        params = ggml_init_params()
        params.mem_size = 64 * 1024 * 1024
        params.mem_buffer = None
        params.no_alloc = True
        ctx_c = libggml_base.ggml_init(params)

        t_in = libggml_base.ggml_new_tensor_2d(ctx_c, GGML_TYPE_F32, K, M)
        libggml_base.ggml_set_name(t_in, b"input")

        t_out = libggml_base.ggml_mul_mat(ctx_c, t_W, t_in)
        libggml_base.ggml_set_name(t_out, b"output")

        graph = libggml_base.ggml_new_graph(ctx_c)
        libggml_base.ggml_build_forward_expand(graph, t_out)

        # Allocate graph ONCE and keep it
        libggml.ggml_backend_sched_reset(self.sched)
        ok = libggml.ggml_backend_sched_alloc_graph(self.sched, graph)
        assert ok, "Failed to alloc graph"

        cached_graph = CachedGraph(ctx_c, t_in, t_out, graph, self.sched, N, K, M)
        self._graph_cache[cache_key] = cached_graph
        
        return cached_graph

    def matmul(self, name, input_np):
        """Compute input @ weight.T using cached graph when possible."""
        M = input_np.shape[0]
        input_np = np.ascontiguousarray(input_np, dtype=np.float32)
        
        # Get or create cached graph
        cached_graph = self._get_or_create_graph(name, M)
        
        # Just update input tensor data (fast!)
        libggml.ggml_backend_tensor_set(cached_graph.t_in, input_np.ctypes.data, 0, input_np.nbytes)

        # Compute (graph already allocated)
        status = libggml.ggml_backend_sched_graph_compute(self.sched, cached_graph.graph)
        assert status == GGML_STATUS_SUCCESS, f"Compute failed: {status}"

        N = cached_graph.N
        result = np.empty((M, N), dtype=np.float32)
        libggml.ggml_backend_tensor_get(cached_graph.t_out, result.ctypes.data, 0, result.nbytes)

        return result

    def close(self):
        for name, (ctx, buf, _, _, _) in self._weight_cache.items():
            libggml.ggml_backend_buffer_free(buf)
            libggml_base.ggml_free(ctx)
        
        for key, graph in self._graph_cache.items():
            libggml_base.ggml_free(graph.ctx_c)
        
        self._weight_cache.clear()
        self._graph_cache.clear()
        libggml.ggml_backend_sched_free(self.sched)
        libggml.ggml_backend_free(self.backend_cpu)
        libggml.ggml_backend_free(self.backend_vk)


def benchmark_optimized():
    print("="*80)
    print("OPTIMIZED ENGINE WITH GRAPH CACHING - BENCHMARK")
    print("="*80)
    
    engine = GgmlVulkanEngineOptimized(max_graphs=5)
    
    tests = [
        ("small", 256, 1536),
        ("medium", 1536, 4096),
        ("large", 4096, 14336),
    ]
    
    for name, N, K in tests:
        W = np.random.randn(N, K).astype(np.float32)
        engine.cache_weight(name, W)
        
        print(f"\n{name} ({K}x{N}):")
        
        # Test different batch sizes
        for M in [1, 4, 16, 64]:
            X = np.random.randn(M, K).astype(np.float32)
            
            # Warmup (creates graph)
            engine.matmul(name, X)
            
            # Benchmark (uses cached graph)
            n_iters = 50
            times = []
            for _ in range(n_iters):
                X = np.random.randn(M, K).astype(np.float32)
                t0 = time.perf_counter()
                result = engine.matmul(name, X)
                times.append((time.perf_counter() - t0) * 1000)
            
            avg_ms = np.mean(times)
            median_ms = np.median(times)
            p99_ms = np.percentile(times, 99)
            min_ms = np.min(times)
            max_ms = np.max(times)
            
            # Correctness
            expected = X @ W.T
            cos = np.dot(expected.flatten(), result.flatten()) / (
                np.linalg.norm(expected.flatten()) * np.linalg.norm(result.flatten()) + 1e-10)
            
            print(f"  M={M:3d} | avg={avg_ms:6.2f}ms  median={median_ms:6.2f}ms  p99={p99_ms:6.2f}ms  "
                  f"min={min_ms:6.2f}ms  max={max_ms:6.2f}ms  cos={cos:.6f}")
    
    engine.close()


if __name__ == "__main__":
    benchmark_optimized()
