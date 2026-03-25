#!/usr/bin/env python3
"""
Step 2: Python ctypes bindings for ggml Vulkan matmul.
Calls ggml_mul_mat via libggml.so's Vulkan backend.
"""
import ctypes
import ctypes.util
import numpy as np
import os
import time

# ---- Load shared libraries ----
LIB_DIR = os.path.expanduser("~/GITDEV/llama.cpp/build-lib/bin")

# Load in dependency order
libggml_base = ctypes.CDLL(os.path.join(LIB_DIR, "libggml-base.so"), mode=ctypes.RTLD_GLOBAL)
libggml_cpu = ctypes.CDLL(os.path.join(LIB_DIR, "libggml-cpu.so"), mode=ctypes.RTLD_GLOBAL)
libggml_vk = ctypes.CDLL(os.path.join(LIB_DIR, "libggml-vulkan.so"), mode=ctypes.RTLD_GLOBAL)
libggml = ctypes.CDLL(os.path.join(LIB_DIR, "libggml.so"), mode=ctypes.RTLD_GLOBAL)

# ---- Constants ----
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_K = 12
GGML_DEFAULT_GRAPH_SIZE = 2048
GGML_STATUS_SUCCESS = 0

# ---- Struct definitions ----
class ggml_init_params(ctypes.Structure):
    _fields_ = [
        ("mem_size", ctypes.c_size_t),
        ("mem_buffer", ctypes.c_void_p),
        ("no_alloc", ctypes.c_bool),
    ]

# ---- Function signatures ----
# ggml core
libggml_base.ggml_init.argtypes = [ggml_init_params]
libggml_base.ggml_init.restype = ctypes.c_void_p  # ggml_context*

libggml_base.ggml_free.argtypes = [ctypes.c_void_p]
libggml_base.ggml_free.restype = None

libggml_base.ggml_new_tensor_2d.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_int64]
libggml_base.ggml_new_tensor_2d.restype = ctypes.c_void_p  # ggml_tensor*

libggml_base.ggml_mul_mat.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
libggml_base.ggml_mul_mat.restype = ctypes.c_void_p  # ggml_tensor*

libggml_base.ggml_new_graph.argtypes = [ctypes.c_void_p]
libggml_base.ggml_new_graph.restype = ctypes.c_void_p  # ggml_cgraph*

libggml_base.ggml_build_forward_expand.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libggml_base.ggml_build_forward_expand.restype = None

libggml_base.ggml_set_name.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
libggml_base.ggml_set_name.restype = ctypes.c_void_p

libggml_base.ggml_nbytes.argtypes = [ctypes.c_void_p]
libggml_base.ggml_nbytes.restype = ctypes.c_size_t

# ggml backend
libggml.ggml_backend_vk_init.argtypes = [ctypes.c_size_t]
libggml.ggml_backend_vk_init.restype = ctypes.c_void_p  # ggml_backend_t

libggml.ggml_backend_free.argtypes = [ctypes.c_void_p]
libggml.ggml_backend_free.restype = None

libggml.ggml_backend_get_default_buffer_type.argtypes = [ctypes.c_void_p]
libggml.ggml_backend_get_default_buffer_type.restype = ctypes.c_void_p

libggml.ggml_backend_alloc_ctx_tensors.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libggml.ggml_backend_alloc_ctx_tensors.restype = ctypes.c_void_p  # ggml_backend_buffer_t

libggml.ggml_backend_tensor_set.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
libggml.ggml_backend_tensor_set.restype = None

libggml.ggml_backend_tensor_get.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
libggml.ggml_backend_tensor_get.restype = None

libggml.ggml_backend_graph_compute.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libggml.ggml_backend_graph_compute.restype = ctypes.c_int  # ggml_status

libggml.ggml_backend_buffer_free.argtypes = [ctypes.c_void_p]
libggml.ggml_backend_buffer_free.restype = None

# ggml backend scheduler
libggml.ggml_backend_sched_new.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),  # backends array
    ctypes.c_void_p,                   # bufts (NULL)
    ctypes.c_int,                      # n_backends
    ctypes.c_size_t,                   # graph_size
    ctypes.c_bool,                     # parallel
    ctypes.c_bool,                     # op_offload
]
libggml.ggml_backend_sched_new.restype = ctypes.c_void_p

libggml.ggml_backend_sched_alloc_graph.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libggml.ggml_backend_sched_alloc_graph.restype = ctypes.c_bool

libggml.ggml_backend_sched_graph_compute.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libggml.ggml_backend_sched_graph_compute.restype = ctypes.c_int

libggml.ggml_backend_sched_free.argtypes = [ctypes.c_void_p]
libggml.ggml_backend_sched_free.restype = None

# CPU backend (needed as fallback for scheduler)
libggml.ggml_backend_cpu_init.argtypes = []
libggml.ggml_backend_cpu_init.restype = ctypes.c_void_p


def ggml_matmul_vulkan(A_np, B_np):
    """
    Compute A @ B.T using ggml Vulkan backend.

    ggml_mul_mat(a, b) computes: b @ a.T (result shape: [b.rows, a.rows])
    So to get A @ B.T we pass: ggml_mul_mat(B_tensor, A_tensor)

    A: (M, K) float32 numpy array
    B: (N, K) float32 numpy array  (weights, stored as [out_features, in_features])
    Result: (M, N) float32 numpy array = A @ B.T
    """
    M, K = A_np.shape
    N = B_np.shape[0]
    assert B_np.shape[1] == K, f"Shape mismatch: A({M},{K}) B({N},{B_np.shape[1]})"

    A_np = np.ascontiguousarray(A_np, dtype=np.float32)
    B_np = np.ascontiguousarray(B_np, dtype=np.float32)

    # Init Vulkan backend
    backend_vk = libggml.ggml_backend_vk_init(0)
    assert backend_vk, "Failed to init Vulkan backend"

    # Create context for weight tensor (B) — allocated on Vulkan
    params_w = ggml_init_params()
    params_w.mem_size = 256 * 1024 * 1024  # 256MB for tensor metadata
    params_w.mem_buffer = None
    params_w.no_alloc = True
    ctx_w = libggml_base.ggml_init(params_w)

    # Create weight tensor B (N, K) on Vulkan
    # ggml is row-major: tensor_2d(type, ne0=K, ne1=N) gives shape [N, K]
    t_B = libggml_base.ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, K, N)
    libggml_base.ggml_set_name(t_B, b"weight")

    # Allocate B on Vulkan backend
    buf_w = libggml.ggml_backend_alloc_ctx_tensors(ctx_w, backend_vk)
    assert buf_w, "Failed to allocate weight buffer on Vulkan"

    # Copy weight data to Vulkan
    libggml.ggml_backend_tensor_set(t_B, B_np.ctypes.data, 0, B_np.nbytes)

    # Create context for compute graph (A input + result)
    params_c = ggml_init_params()
    params_c.mem_size = 256 * 1024 * 1024
    params_c.mem_buffer = None
    params_c.no_alloc = True
    ctx_c = libggml_base.ggml_init(params_c)

    # Create input tensor A (M, K)
    t_A = libggml_base.ggml_new_tensor_2d(ctx_c, GGML_TYPE_F32, K, M)
    libggml_base.ggml_set_name(t_A, b"input")

    # Create matmul: result = B @ A.T ... wait, ggml_mul_mat(a,b) = b @ a.T
    # We want A @ B.T = result(M, N)
    # So: ggml_mul_mat(t_B, t_A) = t_A @ t_B.T = (M,K) @ (K,N) = (M,N) ✓
    t_result = libggml_base.ggml_mul_mat(ctx_c, t_B, t_A)
    libggml_base.ggml_set_name(t_result, b"result")

    # Build compute graph
    graph = libggml_base.ggml_new_graph(ctx_c)
    libggml_base.ggml_build_forward_expand(graph, t_result)

    # Use scheduler — needs Vulkan + CPU (CPU as fallback, must be last)
    backend_cpu = libggml.ggml_backend_cpu_init()
    assert backend_cpu, "Failed to init CPU backend"
    backends = (ctypes.c_void_p * 2)(backend_vk, backend_cpu)
    sched = libggml.ggml_backend_sched_new(backends, None, 2, GGML_DEFAULT_GRAPH_SIZE, False, False)
    assert sched, "Failed to create scheduler"

    ok = libggml.ggml_backend_sched_alloc_graph(sched, graph)
    assert ok, "Failed to allocate graph"

    # Copy input data
    libggml.ggml_backend_tensor_set(t_A, A_np.ctypes.data, 0, A_np.nbytes)

    # Compute!
    status = libggml.ggml_backend_sched_graph_compute(sched, graph)
    assert status == GGML_STATUS_SUCCESS, f"Compute failed with status {status}"

    # Read result
    result_size = M * N * 4  # float32
    result_np = np.empty((M, N), dtype=np.float32)
    libggml.ggml_backend_tensor_get(t_result, result_np.ctypes.data, 0, result_size)

    # Cleanup
    libggml.ggml_backend_sched_free(sched)
    libggml.ggml_backend_buffer_free(buf_w)
    libggml_base.ggml_free(ctx_c)
    libggml_base.ggml_free(ctx_w)
    libggml.ggml_backend_free(backend_cpu)
    libggml.ggml_backend_free(backend_vk)

    return result_np


def benchmark():
    print("=" * 60)
    print("ggml Vulkan matmul benchmark")
    print("=" * 60)

    test_cases = [
        ("QKV 1536x2048", 1, 1536, 2048),
        ("MLP gate 1536x8960", 1, 1536, 8960),
        ("MLP down 8960x1536", 1, 8960, 1536),
        ("8B gate 4096x14336", 1, 4096, 14336),
        ("QKV batch=16", 16, 1536, 2048),
        ("MLP gate batch=16", 16, 1536, 8960),
        ("8B gate batch=16", 16, 4096, 14336),
        ("8B gate batch=64", 64, 4096, 14336),
    ]

    for name, M, K, N in test_cases:
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(N, K).astype(np.float32)

        # Correctness check
        expected = A @ B.T

        try:
            # Warmup
            result = ggml_matmul_vulkan(A, B)

            # Check correctness
            cos_sim = np.dot(expected.flatten(), result.flatten()) / (
                np.linalg.norm(expected.flatten()) * np.linalg.norm(result.flatten()) + 1e-10
            )

            # Benchmark (3 runs)
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                result = ggml_matmul_vulkan(A, B)
                t1 = time.perf_counter()
                times.append(t1 - t0)

            avg_ms = np.mean(times) * 1000
            flops = 2 * M * K * N
            gflops = flops / (np.mean(times) * 1e9)

            # CPU reference
            t0 = time.perf_counter()
            _ = A @ B.T
            cpu_ms = (time.perf_counter() - t0) * 1000

            speedup = cpu_ms / avg_ms if avg_ms > 0 else 0

            print(f"  {name:30s} | ggml: {avg_ms:7.2f}ms | CPU: {cpu_ms:7.2f}ms | "
                  f"{speedup:5.1f}x | {gflops:7.1f} GFLOPS | cos={cos_sim:.6f}")
        except Exception as e:
            print(f"  {name:30s} | FAILED: {e}")


if __name__ == "__main__":
    # Quick smoke test
    print("Smoke test: 4x4 matmul...")
    A = np.array([[1, 2], [3, 4]], dtype=np.float32)
    B = np.array([[5, 6], [7, 8]], dtype=np.float32)
    expected = A @ B.T
    result = ggml_matmul_vulkan(A, B)
    print(f"  Expected:\n{expected}")
    print(f"  Got:\n{result}")
    print(f"  Match: {np.allclose(expected, result, atol=0.01)}")
    print()

    benchmark()
