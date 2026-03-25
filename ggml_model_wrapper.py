"""
GgmlModelWrapper: Drop-in replacement for vLLM's model forward pass.
Uses ggml Vulkan engine for the transformer compute, returns logits to vLLM.

vLLM handles: API, scheduling, batching, streaming, sampling
ggml handles: full transformer forward (QKV, attention, MLP, norms) on Vulkan GPU

Integration point: gpu_model_runner.py line 3191
  Before: return self.model(input_ids=..., positions=...)
  After:  return self.ggml_wrapper(input_ids=..., positions=...)
"""
import ctypes
import numpy as np
import os
import sys
import torch

# Load ggml engine
_LIB_PATH = os.path.expanduser("~/AGENT/libggml_llama_gguf.so")
_lib = ctypes.CDLL(_LIB_PATH)
_lib.engine_load_gguf.argtypes = [ctypes.c_char_p, ctypes.c_int]
_lib.engine_load_gguf.restype = ctypes.c_void_p
_lib.engine_forward.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
_lib.engine_forward.restype = ctypes.c_int
_lib.engine_reset_kv.argtypes = [ctypes.c_void_p]
_lib.engine_warmup.argtypes = [ctypes.c_void_p]
_lib.engine_warmup.restype = ctypes.c_int
_lib.engine_free.argtypes = [ctypes.c_void_p]
_lib.engine_set_return_hidden.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.engine_set_return_hidden.restype = None


class GgmlModelWrapper(torch.nn.Module):
    """
    Wraps ggml Vulkan engine as a PyTorch Module for vLLM integration.

    Takes input_ids + positions, returns hidden_states (logits from last layer).
    ggml handles the ENTIRE forward pass including attention and KV cache.
    """

    def __init__(self, gguf_path, n_ctx=2048, vocab_size=128256, hidden_dim=4096, return_hidden=False):
        super().__init__()
        self.gguf_path = os.path.expanduser(gguf_path)
        self.n_ctx = n_ctx
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.return_hidden = return_hidden

        print(f"[GgmlModelWrapper] Loading {self.gguf_path}...", file=sys.stderr)
        self._engine = _lib.engine_load_gguf(self.gguf_path.encode(), n_ctx)
        if not self._engine:
            raise RuntimeError(f"Failed to load ggml model: {self.gguf_path}")
        if return_hidden:
            _lib.engine_set_return_hidden(self._engine, 1)
        _lib.engine_warmup(self._engine)
        print(f"[GgmlModelWrapper] Ready. hidden={hidden_dim}, return_hidden={return_hidden}", file=sys.stderr)

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **kwargs):
        """
        Forward pass compatible with vLLM's model runner.

        Args:
            input_ids: (n_tokens,) int64 tensor
            positions: (n_tokens,) int64 tensor
            intermediate_tensors: ignored (we handle full model)
            inputs_embeds: ignored (we do our own embedding)

        Returns:
            hidden_states: (n_tokens, vocab_size) float32 tensor (logits)
        """
        n_tokens = input_ids.shape[0]

        # Convert to numpy int32 (ggml expects int32)
        tokens_np = input_ids.cpu().numpy().astype(np.int32)
        positions_np = positions.cpu().numpy().astype(np.int32)

        # Allocate output — hidden_states (4096) or logits (vocab_size)
        out_dim = self.hidden_dim if self.return_hidden else self.vocab_size
        logits_np = np.empty((n_tokens, out_dim), dtype=np.float32)

        # Call ggml engine
        ret = _lib.engine_forward(
            self._engine, n_tokens,
            tokens_np.ctypes.data,
            positions_np.ctypes.data,
            logits_np.ctypes.data
        )

        if ret != 0:
            raise RuntimeError(f"ggml forward failed: {ret}")

        # Convert back to torch tensor on CPU (vLLM Vulkan path expects CPU tensors)
        return torch.from_numpy(logits_np)

    def reset_kv(self):
        """Reset KV cache for new sequence."""
        _lib.engine_reset_kv(self._engine)

    def __del__(self):
        if hasattr(self, '_engine') and self._engine:
            _lib.engine_free(self._engine)


def patch_vllm_model_runner(runner, gguf_path, n_ctx=2048):
    """
    Monkey-patch a vLLM GPUModelRunner to use ggml for forward pass.

    The ggml engine returns LOGITS directly (after lm_head).
    vLLM expects HIDDEN STATES and applies lm_head itself.
    We patch both _model_forward AND the logits_processor to handle this.
    """
    vocab_size = 128256
    try:
        vocab_size = runner.vllm_config.model_config.hf_config.vocab_size
    except:
        pass

    hidden_dim = 4096
    try:
        hidden_dim = runner.vllm_config.model_config.hf_config.hidden_size
    except:
        pass

    # Return LOGITS directly — we'll bypass vLLM's compute_logits
    wrapper = GgmlModelWrapper(gguf_path, n_ctx=n_ctx, vocab_size=vocab_size,
                               hidden_dim=hidden_dim, return_hidden=False)

    # The ggml engine returns logits. We need to:
    # 1. Return logits from _model_forward
    # 2. Skip vLLM's lm_head application (logits_processor)

    # ggml returns LOGITS directly. We patch execute_model to skip compute_logits.
    def ggml_model_forward(input_ids=None, positions=None, **kwargs):
        return wrapper(input_ids, positions, **kwargs)

    runner._model_forward = ggml_model_forward
    runner._ggml_wrapper = wrapper
    runner._ggml_returns_logits = True

    # Monkey-patch model.compute_logits to pass through when ggml provides logits
    original_compute_logits = runner.model.compute_logits
    def ggml_compute_logits(hidden_states):
        # If hidden_states has vocab_size dim, it's already logits from ggml
        if hidden_states.shape[-1] == wrapper.vocab_size:
            return hidden_states
        return original_compute_logits(hidden_states)
    runner.model.compute_logits = ggml_compute_logits

    print(f"[GgmlModelWrapper] Patched vLLM model runner with ggml backend", file=sys.stderr)
