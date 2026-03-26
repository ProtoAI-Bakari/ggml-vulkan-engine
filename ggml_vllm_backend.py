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
import json
import logging
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("ggml_vllm")

# Load the C engine
_LIB_PATH = os.path.expanduser("~/AGENT/libggml_llama_gguf.so")
_lib = ctypes.CDLL(_LIB_PATH)
_lib.engine_load_gguf.argtypes = [ctypes.c_char_p, ctypes.c_int]
_lib.engine_load_gguf.restype = ctypes.c_void_p
_lib.engine_forward.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
_lib.engine_forward.restype = ctypes.c_int
_lib.engine_reset_kv.argtypes = [ctypes.c_void_p]
_lib.engine_warmup.argtypes = [ctypes.c_void_p]
_lib.engine_warmup.restype = ctypes.c_int
_lib.engine_get_vocab_size.argtypes = [ctypes.c_void_p]
_lib.engine_get_vocab_size.restype = ctypes.c_int
_lib.engine_free.argtypes = [ctypes.c_void_p]


@dataclass
class SamplingParams:
    max_tokens: int = 100
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 40
    stop: list = None
    stop_token_ids: set = None
    seed: int = None
    repetition_penalty: float = 1.0
    repetition_penalty: float = 1.0

    def __post_init__(self):
        if self.stop is None:
            self.stop = []
        if self.stop_token_ids is None:
            self.stop_token_ids = set()


@dataclass
class GenerationResult:
    text: str
    token_ids: list
    tps: float
    prefill_tps: float
    total_time: float
    finish_reason: str = "length"  # "length", "stop", "error"
    prefill_tokens: int = 0
    decode_tokens: int = 0

    def to_json(self):
        return json.dumps({
            "text": self.text[:200],
            "tps": round(self.tps, 1),
            "prefill_tps": round(self.prefill_tps, 1),
            "total_time": round(self.total_time, 2),
            "finish_reason": self.finish_reason,
            "tokens_generated": len(self.token_ids),
        })


class GgmlLLM:
    """vLLM-compatible LLM interface backed by ggml Vulkan engine."""

    def __init__(self, model_path, n_ctx=8192, tokenizer_path=None):
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
        logger.info(f"Loading {self.model_path}")
        t0 = time.time()
        try:
            self._engine = _lib.engine_load_gguf(self.model_path.encode(), n_ctx)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {self.model_path}: {e}")
        if not self._engine:
            raise RuntimeError(f"Failed to load model (null engine): {self.model_path}")

        # Warmup to prime scheduler buffers
        _lib.engine_warmup(self._engine)
        load_time = time.time() - t0
        logger.info(f"Loaded in {load_time:.1f}s")
        print(f"[GgmlLLM] Loaded in {load_time:.1f}s", flush=True)

        # T36: Get actual vocab size from C engine
        self.vocab_size = _lib.engine_get_vocab_size(self._engine)
        print(f"[INFO] Vocab size from engine: {self.vocab_size}", flush=True)
        
        # T36: Verify vocab size matches HF tokenizer
        if self.tokenizer:
            hf_vocab = len(self.tokenizer.get_vocab())
            if self.vocab_size != hf_vocab:
                print(f"[WARNING] Vocab mismatch: GGUF={self.vocab_size}, HF={hf_vocab}", flush=True)
            else:
                print(f"[INFO] Vocab sizes match: {self.vocab_size}", flush=True)
        # Stats tracking
        self._total_tokens = 0
        self._total_time = 0

    def _find_tokenizer(self):
        """Find HF tokenizer for the model (T08: support all model families)."""
        base = os.path.basename(self.model_path).lower()
        cache = os.path.expanduser("~/.cache/huggingface/hub")
        
        if not os.path.exists(cache):
            return None
        
        # Model family mappings: GGUF name pattern -> HF model name patterns
        family_patterns = {
            "llama": ["llama-3", "llama-3.1", "llama-3.2", "llama-3-instruct", "meta-llama"],
            "qwen": ["qwen2.5", "qwen-2.5", "qwen2", "qwen-2", "qwen"],
            "gpt-oss": ["gpt-oss", "openai-gpt"],
            "mistral": ["mistral", "mixtral", "codestral"],
            "gemma": ["gemma", "gemma2", "gemma-2"],
            "phi": ["phi-3", "phi-3.5", "phi3", "microsoft-phi"],
            "deepseek": ["deepseek", "deepseek-coder"],
            "yi": ["yi-34b", "yi-6b", "01-ai-yi"],
        }
        
        # Detect model family from filename
        detected_family = None
        for family, patterns in family_patterns.items():
            for pattern in patterns:
                if pattern in base:
                    detected_family = family
                    break
            if detected_family:
                break
        
        # If no family detected, try generic search
        if not detected_family:
            # Look for any HF model snapshot
            for d in os.listdir(cache):
                if d.startswith("models--"):
                    snapshot_dir = os.path.join(cache, d, "snapshots")
                    if os.path.exists(snapshot_dir):
                        snaps = os.listdir(snapshot_dir)
                        if snaps and os.path.exists(os.path.join(snapshot_dir, snaps[0], "tokenizer.json")):
                            return os.path.join(snapshot_dir, snaps[0])
            return None
        
        # Search for matching model family in cache
        patterns = family_patterns[detected_family]
        for d in os.listdir(cache):
            if d.startswith("models--"):
                # Decode HF model name (e.g., models--meta-llama--Llama-3.1-8B-Instruct)
                parts = d.replace("models--", "").split("--")
                if len(parts) >= 2:
                    model_name = "--".join(parts[1:]).lower()
                    for pattern in patterns:
                        if pattern in model_name:
                            snapshot_dir = os.path.join(cache, d, "snapshots")
                            if os.path.exists(snapshot_dir):
                                snaps = os.listdir(snapshot_dir)
                                if snaps:
                                    snap_path = os.path.join(snapshot_dir, snaps[0])
                                    # Verify it has tokenizer files
                                    if (os.path.exists(os.path.join(snap_path, "tokenizer.json")) or
                                        os.path.exists(os.path.join(snap_path, "tokenizer.model"))):
                                        print(f"[tokenizer] Found {detected_family} tokenizer at {snap_path}")
                                        return snap_path
        
        # Fallback: try to download from HF with vocab validation
        print(f"[tokenizer] No cached tokenizer found for {detected_family}, attempting download...")
        try:
            from transformers import AutoTokenizer
            # Map detected family to common HF model IDs
            hf_model_map = {
                "llama": ["meta-llama/Meta-Llama-3.1-8B-Instruct"],
                "qwen": ["Qwen/Qwen2.5-32B", "Qwen/Qwen2.5-32B-Instruct", "Qwen/Qwen2.5-32B-Base"],
                "mistral": ["mistralai/Mistral-7B-Instruct-v0.3"],
                "gemma": ["google/gemma-2-27b-it"],
                "phi": ["microsoft/Phi-3.5-mini-instruct"],
                "deepseek": ["deepseek-ai/deepseek-coder-33b-instruct"],
                "gpt-oss": ["openai/gpt-oss-120b"],
            }
            model_ids = hf_model_map.get(detected_family, ["meta-llama/Meta-Llama-3.1-8B-Instruct"])
            
            # Try each model ID until we find one with matching vocab
            for model_id in model_ids:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                    tok_vocab = len(tokenizer.get_vocab())
                    print(f"[tokenizer] {model_id} has vocab={tok_vocab}")
                    
                    # For Qwen, we need vocab=128256 to match GGUF
                    
                    # Save tokenizer to a temp location
                    import tempfile
                    temp_dir = tempfile.mkdtemp(prefix="ggml_tokenizer_")
                    tokenizer.save_pretrained(temp_dir)
                    print(f"[tokenizer] Downloaded {model_id} tokenizer to {temp_dir}")
                    return temp_dir
                except Exception as e:
                    print(f"[tokenizer] Failed {model_id}: {e}")
                    continue
            
            print(f"[tokenizer] No compatible tokenizer found for {detected_family}")
            return None
        except Exception as e:
            print(f"[tokenizer] Failed to download tokenizer: {e}")
            return None

    def generate(self, prompt, params=None, stream_callback=None, **kwargs):
        """Generate text from prompt.

        Args:
            prompt: str or list[int] (token IDs)
            params: SamplingParams object, or pass kwargs directly
            stream_callback: callable(token_text, token_id) called per token for streaming
        Returns: GenerationResult
        """
        if params is None:
            params = SamplingParams(**kwargs) if kwargs else SamplingParams()

        # Default stop tokens for Llama-3.1
        stop_ids = params.stop_token_ids | {128001, 128008, 128009}

        if params.seed is not None:
            np.random.seed(params.seed)

        # Tokenize
        if isinstance(prompt, list):
            input_ids = prompt
        elif self.tokenizer:
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        else:
            input_ids = [128000]

        _lib.engine_reset_kv(self._engine)
        generated_ids = []

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
            return GenerationResult(f"[PREFILL FAILED: {ret}]", [], 0, 0, 0, "error")

        prefill_tps = len(input_ids) / t_prefill

        # Sample first token
        next_token = self._sample(logits_buf[-1], params, generated_ids)
        generated_ids.append(next_token)
        if stream_callback and self.tokenizer:
            stream_callback(self.tokenizer.decode([next_token]), next_token)

        # Decode loop
        decode_times = []
        logits_1 = np.empty((1, self.vocab_size), dtype=np.float32)
        finish_reason = "length"

        for i in range(params.max_tokens - 1):
            if next_token in stop_ids:
                finish_reason = "stop"
                break

            tok = np.array([next_token], dtype=np.int32)
            pos = np.array([len(input_ids) + i], dtype=np.int32)

            t0 = time.perf_counter()
            ret = _lib.engine_forward(self._engine, 1,
                                      tok.ctypes.data, pos.ctypes.data,
                                      logits_1.ctypes.data)
            decode_times.append(time.perf_counter() - t0)

            if ret != 0:
                finish_reason = "error"
                break

            next_token = self._sample(logits_1[0], params, generated_ids)
            generated_ids.append(next_token)

            if stream_callback and self.tokenizer:
                stream_callback(self.tokenizer.decode([next_token]), next_token)

            # Check stop strings
            if params.stop and self.tokenizer:
                text_so_far = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                for s in params.stop:
                    if text_so_far.endswith(s):
                        finish_reason = "stop"
                        break
                if finish_reason == "stop":
                    break

        t_total = time.perf_counter() - t_start

        if self.tokenizer:
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        else:
            text = f"[{len(generated_ids)} tokens]"

        decode_tps = len(decode_times) / sum(decode_times) if decode_times else 0

        # Track stats
        self._total_tokens += len(generated_ids)
        self._total_time += t_total

        result = GenerationResult(
            text=text, token_ids=generated_ids, tps=decode_tps,
            prefill_tps=prefill_tps, total_time=t_total, finish_reason=finish_reason,
            prefill_tokens=len(input_ids), decode_tokens=len(generated_ids),
        )
        logger.info(result.to_json())
        return result

    def _sample(self, logits, params, prev_tokens=None):
        """Sample next token with full sampling params."""
        logits = logits.copy()

        # Repetition penalty
        if params.repetition_penalty != 1.0 and prev_tokens:
            for tid in set(prev_tokens):
                if logits[tid] > 0:
                    logits[tid] /= params.repetition_penalty
                else:
                    logits[tid] *= params.repetition_penalty

        # Greedy
        if params.temperature == 0:
            return int(np.argmax(logits))

        # Temperature
        logits = logits / params.temperature

        # Top-k
        if params.top_k > 0:
            indices = np.argsort(logits)[-params.top_k:]
            mask = np.full_like(logits, -np.inf)
            mask[indices] = logits[indices]
            logits = mask

        # Softmax
        logits = logits - np.max(logits)
        probs = np.exp(logits) / (np.sum(np.exp(logits)) + 1e-10)

        # Top-p (nucleus)
        if params.top_p < 1.0:
            sorted_idx = np.argsort(probs)[::-1]
            cumsum = np.cumsum(probs[sorted_idx])
            cutoff = np.searchsorted(cumsum, params.top_p) + 1
            mask = np.zeros_like(probs)
            mask[sorted_idx[:cutoff]] = probs[sorted_idx[:cutoff]]
            probs = mask / (mask.sum() + 1e-10)

        # Min-p
        if params.min_p > 0:
            max_p = probs.max()
            probs[probs < params.min_p * max_p] = 0
            probs = probs / (probs.sum() + 1e-10)

        return int(np.random.choice(len(probs), p=probs))


    def _prepare_inputs(self, batch):
        """Prepare input tensors from InputBatch for batched processing (T35).
        
        Args:
            batch: InputBatch or dict containing:
                - request_ids: list of request identifiers
                - requests: dict mapping request_id -> {tokens, seq_len}
                - block_tables: dict mapping request_id -> list of block IDs
        
        Returns:
            dict with keys:
                - input_ids: np.ndarray [total_tokens] int32
                - positions: np.ndarray [total_tokens] int32  
                - seq_lens: list[int] [batch_size]
                - block_tables: np.ndarray [batch_size, max_blocks] int32
        """
        input_ids_list = []
        positions_list = []
        seq_lens = []
        block_tables_list = []
        
        max_blocks_per_seq = 0
        
        # Handle both InputBatch objects and raw dicts
        if hasattr(batch, 'request_ids'):
            # InputBatch object
            request_ids = batch.request_ids
            requests = batch.requests
            block_tables = batch.block_tables
        else:
            # Raw dict
            request_ids = batch.get('request_ids', [])
            requests = batch.get('requests', {})
            block_tables = batch.get('block_tables', {})
        
        for req_id in request_ids:
            req_data = requests.get(req_id, {})
            
            # Get tokens for this request
            tokens = req_data.get('tokens', [])
            seq_len = len(tokens)
            seq_lens.append(seq_len)
            
            # Extend input_ids and positions
            input_ids_list.extend(tokens)
            positions_list.extend(range(seq_len))
            
            # Get block table for this request
            bt = block_tables.get(req_id, [])
            block_tables_list.append(bt)
            max_blocks_per_seq = max(max_blocks_per_seq, len(bt))
        
        # Pad block tables to uniform width
        padded_block_tables = []
        for bt in block_tables_list:
            padded_bt = bt + [0] * (max_blocks_per_seq - len(bt))
            padded_block_tables.append(padded_bt)
        
        # Convert to numpy arrays (compatible with ctypes)
        input_ids = np.array(input_ids_list, dtype=np.int32)
        positions = np.array(positions_list, dtype=np.int32)
        block_tables = np.array(padded_block_tables, dtype=np.int32) if padded_block_tables else np.empty((0, 0), dtype=np.int32)
        
        return {
            "input_ids": input_ids,
            "positions": positions,
            "seq_lens": seq_lens,
            "block_tables": block_tables,
            "num_seqs": len(request_ids),
            "total_tokens": len(input_ids_list),
        }

    def stats(self):
        """Return engine statistics."""
        avg_tps = self._total_tokens / self._total_time if self._total_time > 0 else 0
        return {
            "total_tokens": self._total_tokens,
            "total_time": round(self._total_time, 2),
            "avg_tps": round(avg_tps, 1),
            "model": os.path.basename(self.model_path),
            "vocab_size": self.vocab_size,
            "n_ctx": self.n_ctx,
        }

    def batch_generate(self, prompts, **kwargs):
        """Generate for multiple prompts sequentially."""
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, **kwargs))
        return results

    def close(self):
        """Explicitly free engine resources."""
        if hasattr(self, "_engine") and self._engine and not getattr(self, "_closed", False):
            # _lib.engine_free(self._engine)  # Skip to avoid double-free
            self._engine = None
            self._closed = True

    def __del__(self):
        self.close()

if __name__ == "__main__":
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/models/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")

    llm = GgmlLLM(model)

    print("\n" + "=" * 60)
    print("GgmlLLM Benchmark — Q4_K_M on Vulkan")
    print("=" * 60)

    # T6: Basic generation
    print("\n--- Greedy (temperature=0) ---")
    for prompt in ["The capital of France is", "2 + 2 ="]:
        r = llm.generate(prompt, params=SamplingParams(temperature=0, max_tokens=50))
        print(f"  [{r.tps:.1f} TPS] {prompt} → {r.text[:80]}  [{r.finish_reason}]")

    # T7: Streaming
    print("\n--- Streaming ---")
    print("  Streaming: ", end="", flush=True)
    r = llm.generate("Once upon a time",
                     params=SamplingParams(temperature=0.7, max_tokens=30, top_k=40),
                     stream_callback=lambda text, _: print(text, end="", flush=True))
    print(f"  [{r.tps:.1f} TPS]")

    # T8: Sampling params
    print("\n--- Temperature=0.8 top_p=0.9 rep_penalty=1.1 ---")
    r = llm.generate("Write a creative story about a robot:",
                     params=SamplingParams(temperature=0.8, top_p=0.9,
                                           repetition_penalty=1.1, max_tokens=80))
    print(f"  [{r.tps:.1f} TPS] {r.text[:150]}")

    # T9: Stop tokens/strings
    print("\n--- Stop on newline ---")
    r = llm.generate("List three colors:\n1.",
                     params=SamplingParams(temperature=0, max_tokens=50, stop=["\n3."]))
    print(f"  [{r.tps:.1f} TPS] {r.text[:100]}  [{r.finish_reason}]")

    # T10: Multi-prompt batch
    print("\n--- Batch generation (4 prompts) ---")
    t0 = time.time()
    results = llm.batch_generate(
        ["Hello!", "Goodbye!", "What is AI?", "Why is the sky blue?"],
        params=SamplingParams(temperature=0, max_tokens=30),
    )
    elapsed = time.time() - t0
    total_tok = sum(len(r.token_ids) for r in results)
    print(f"  {total_tok} tokens in {elapsed:.1f}s = {total_tok/elapsed:.1f} aggregate TPS")
    for r in results:
        print(f"    [{r.tps:.1f} TPS] {r.text[:60]}")
