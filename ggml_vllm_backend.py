#!/usr/bin/env python3
"""
T6: ggml Vulkan backend for vLLM.
Wraps our C engine (libggml_llama_gguf.so) behind vLLM's LLM-compatible API.
Handles tokenization, sampling, and generation loop in Python.
The C engine handles the full transformer forward pass on Vulkan GPU.

Usage:
    from ggml_vllm_backend import GgmlLLM
    llm = GgmlLLM("~/models/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
    text = llm.generate("The capital of France is", max_tokens=100, temperature=0)
    print(text)
"""
import ctypes
import numpy as np
import os
import time
from dataclasses import dataclass

# Load the C engine
_LIB_PATH = os.path.expanduser("~/AGENT/libggml_llama_gguf.so")
_lib = ctypes.CDLL(_LIB_PATH)
_lib.engine_load_gguf.argtypes = [ctypes.c_char_p, ctypes.c_int]
_lib.engine_load_gguf.restype = ctypes.c_void_p
_lib.engine_forward.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
_lib.engine_forward.restype = ctypes.c_int
_lib.engine_reset_kv.argtypes = [ctypes.c_void_p]
_lib.engine_free.argtypes = [ctypes.c_void_p]


@dataclass
class GenerationResult:
    text: str
    token_ids: list
    tps: float
    prefill_tps: float
    total_time: float


class GgmlLLM:
    """vLLM-compatible LLM interface backed by ggml Vulkan engine."""

    def __init__(self, model_path, n_ctx=2048, tokenizer_path=None):
        self.model_path = os.path.expanduser(model_path)
        self.n_ctx = n_ctx

        # Load tokenizer
        tok_path = tokenizer_path or self._find_tokenizer()
        if tok_path:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
        else:
            self.tokenizer = None

        # Load model via C engine
        print(f"[GgmlLLM] Loading {self.model_path}...")
        t0 = time.time()
        self._engine = _lib.engine_load_gguf(self.model_path.encode(), n_ctx)
        if not self._engine:
            raise RuntimeError(f"Failed to load model: {self.model_path}")
        print(f"[GgmlLLM] Loaded in {time.time()-t0:.1f}s")

        # Get vocab size from first forward
        self.vocab_size = 128256  # Llama-3.1 default

    def _find_tokenizer(self):
        """Find HF tokenizer for the model."""
        # Check if there's a matching HF model in cache
        base = os.path.basename(self.model_path).lower()
        cache = os.path.expanduser("~/.cache/huggingface/hub")
        if "llama-3.1-8b" in base:
            candidates = [
                os.path.join(cache, d, "snapshots")
                for d in os.listdir(cache)
                if "llama-3.1-8b-instruct" in d.lower()
            ] if os.path.exists(cache) else []
            for c in candidates:
                if os.path.exists(c):
                    snaps = os.listdir(c)
                    if snaps:
                        return os.path.join(c, snaps[0])
        return None

    def generate(self, prompt, max_tokens=100, temperature=0.0, top_k=1, stop_tokens=None):
        """Generate text from prompt. Returns GenerationResult."""
        if stop_tokens is None:
            stop_tokens = {128001, 128008, 128009}  # Llama-3.1 stop tokens

        # Tokenize
        if self.tokenizer:
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        else:
            input_ids = [128000]  # BOS

        _lib.engine_reset_kv(self._engine)
        generated_ids = []
        logits_buf = np.empty((max(len(input_ids), 1), self.vocab_size), dtype=np.float32)

        # Prefill
        tokens = np.array(input_ids, dtype=np.int32)
        positions = np.arange(len(input_ids), dtype=np.int32)
        logits_buf = np.empty((len(input_ids), self.vocab_size), dtype=np.float32)

        t_start = time.perf_counter()
        ret = _lib.engine_forward(self._engine, len(input_ids),
                                  tokens.ctypes.data, positions.ctypes.data,
                                  logits_buf.ctypes.data)
        t_prefill = time.perf_counter() - t_start

        if ret != 0:
            return GenerationResult(f"[PREFILL FAILED: {ret}]", [], 0, 0, 0)

        prefill_tps = len(input_ids) / t_prefill

        # Sample first token
        next_token = self._sample(logits_buf[-1], temperature, top_k)
        generated_ids.append(next_token)

        # Decode loop
        decode_times = []
        logits_1 = np.empty((1, self.vocab_size), dtype=np.float32)

        for i in range(max_tokens - 1):
            if next_token in stop_tokens:
                break

            tok = np.array([next_token], dtype=np.int32)
            pos = np.array([len(input_ids) + i], dtype=np.int32)

            t0 = time.perf_counter()
            ret = _lib.engine_forward(self._engine, 1,
                                      tok.ctypes.data, pos.ctypes.data,
                                      logits_1.ctypes.data)
            decode_times.append(time.perf_counter() - t0)

            if ret != 0:
                break

            next_token = self._sample(logits_1[0], temperature, top_k)
            generated_ids.append(next_token)

        t_total = time.perf_counter() - t_start

        # Decode token IDs to text
        if self.tokenizer:
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        else:
            text = f"[{len(generated_ids)} tokens]"

        decode_tps = len(decode_times) / sum(decode_times) if decode_times else 0

        return GenerationResult(
            text=text,
            token_ids=generated_ids,
            tps=decode_tps,
            prefill_tps=prefill_tps,
            total_time=t_total,
        )

    def _sample(self, logits, temperature=0.0, top_k=1):
        """Sample next token from logits."""
        if temperature == 0 or top_k == 1:
            return int(np.argmax(logits))

        # Temperature scaling
        logits = logits / temperature

        # Top-k
        if top_k > 0:
            indices = np.argsort(logits)[-top_k:]
            mask = np.full_like(logits, -np.inf)
            mask[indices] = logits[indices]
            logits = mask

        # Softmax
        logits = logits - np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return int(np.random.choice(len(probs), p=probs))

    def batch_generate(self, prompts, **kwargs):
        """Generate for multiple prompts sequentially."""
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, **kwargs))
        return results

    def __del__(self):
        if hasattr(self, '_engine') and self._engine:
            _lib.engine_free(self._engine)


if __name__ == "__main__":
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/models/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")

    llm = GgmlLLM(model)

    prompts = [
        "The capital of France is",
        "Explain quantum computing in one paragraph:",
        "Write a haiku about programming:",
        "2 + 2 =",
    ]

    print("\n" + "=" * 60)
    print("GgmlLLM Benchmark — Q4_K_M on Vulkan")
    print("=" * 60)

    for prompt in prompts:
        result = llm.generate(prompt, max_tokens=100, temperature=0)
        print(f"\nPrompt: {prompt}")
        print(f"Output: {result.text[:150]}")
        print(f"Decode: {result.tps:.1f} TPS | Prefill: {result.prefill_tps:.1f} TPS | "
              f"Tokens: {len(result.token_ids)} | Time: {result.total_time:.1f}s")
