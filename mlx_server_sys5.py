"""
MLX Server — Sys5 (M1 Ultra 128GB)
Model: GLM-4.5-4bit — General reasoning brain
  /v1/chat/completions  — OpenAI-compatible (stream or not)
  /health               — liveness probe
"""

import asyncio
import json
import time
import uuid
import threading
from contextlib import asynccontextmanager
from threading import Thread
from typing import List, Optional
from datetime import datetime

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from pydantic import BaseModel

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID        = "mlx-community/GLM-4.7-Flash-8bit"
HOST            = "0.0.0.0"
PORT            = 8000
MAX_TOKENS      = 32000
MAX_KV_SIZE     = 32000
MAX_CONCURRENT  = 4
TEMPERATURE     = 0.7
TOP_P           = 0.9
REPETITION_PEN  = 1.05
HOSTNAME        = "sys5"

# ── Model ──────────────────────────────────────────────────────────────────────
print(f"[{HOSTNAME}] Loading {MODEL_ID}...")
model, tokenizer = load(MODEL_ID)
sampler           = make_sampler(temp=TEMPERATURE, top_p=TOP_P)
logits_processors = make_logits_processors(repetition_penalty=REPETITION_PEN)
_gpu_lock = threading.Lock()
_active_count = 0
_active_lock = threading.Lock()
print(f"[{HOSTNAME}] Model ready. GPU lock enabled. Max concurrent: {MAX_CONCURRENT}")

# ── Inference bridge ───────────────────────────────────────────────────────────
def _inference_thread(history, q: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    with _gpu_lock:
        prompt = tokenizer.apply_chat_template(history, add_generation_prompt=True)
        for chunk in stream_generate(
            model, tokenizer, prompt=prompt,
            max_tokens=MAX_TOKENS, max_kv_size=MAX_KV_SIZE,
            sampler=sampler, logits_processors=logits_processors,
        ):
            text = chunk.text if hasattr(chunk, "text") else chunk
            asyncio.run_coroutine_threadsafe(q.put(text), loop)
        asyncio.run_coroutine_threadsafe(q.put(None), loop)

async def _run_generation(history):
    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    t = Thread(target=_inference_thread, args=(history, q, loop), daemon=True)
    t.start()
    while True:
        tok = await q.get()
        if tok is None:
            break
        yield tok
    t.join()

# ── Schemas ────────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    stream: bool = False
    max_tokens: int = MAX_TOKENS
    temperature: float = TEMPERATURE
    model: Optional[str] = None

# ── App ────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title=f"MLX Server ({HOSTNAME})", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

SYSTEM_PROMPT = f"You are an AI assistant running on {HOSTNAME} (Apple M1 Ultra 128GB). You are part of a multi-agent swarm working on Vulkan GPU inference optimization for vLLM on Asahi Linux. Be concise, technical, and helpful."

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID, "host": HOSTNAME, "active": _active_count}

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [
        {"id": MODEL_ID, "object": "model", "owned_by": "mlx-community"}
    ]}

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, authorization: Optional[str] = Header(None)):
    global _active_count
    with _active_lock:
        if _active_count >= MAX_CONCURRENT:
            raise HTTPException(status_code=429, detail=f"Max {MAX_CONCURRENT} concurrent requests")
        _active_count += 1

    try:
        history = [{"role": m.role, "content": m.content} for m in req.messages]
        if not any(m["role"] == "system" for m in history):
            history.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if req.stream:
            async def sse_logged():
                global _active_count
                _start = time.perf_counter()
                _first = None
                _tok_count = 0
                try:
                    async for tok in _run_generation(history):
                        if _first is None:
                            _first = time.perf_counter()
                        _tok_count += 1
                        chunk = {
                            "id": rid, "object": "chat.completion.chunk", "created": created,
                            "model": MODEL_ID,
                            "choices": [{"index": 0, "delta": {"content": tok}, "finish_reason": None}],
                        }
                        yield "data: " + json.dumps(chunk) + "\n\n"
                    _end = time.perf_counter()
                    _pp = (_first - _start) if _first else 0
                    _tg = _end - (_first or _end)
                    _tps = _tok_count / _tg if _tg > 0 else 0
                    _itl = _tg / _tok_count * 1000 if _tok_count else 0
                    _q = req.messages[-1].content[:50].replace("\n", " ")
                    _ts = datetime.now().strftime("%H:%M:%S")
                    print(f"[{_ts}] [{HOSTNAME}] STREAM [{_q:50s}] PP:{_pp:.2f}s TG:{_tg:.2f}s TTFT:{_pp:.2f}s TPS:{_tps:.1f} ITL:{_itl:.1f}ms Tok:{_tok_count} Total:{(_end-_start):.2f}s", flush=True)
                    done = {"id": rid, "object": "chat.completion.chunk", "created": created,
                            "model": MODEL_ID, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                    yield "data: " + json.dumps(done) + "\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    with _active_lock:
                        _active_count -= 1
            return StreamingResponse(sse_logged(), media_type="text/event-stream")

        # non-streaming
        start = time.perf_counter()
        first_tok_t = None
        tokens = 0
        full = ""
        async for tok in _run_generation(history):
            if first_tok_t is None:
                first_tok_t = time.perf_counter()
            full += tok
            tokens += 1
        end = time.perf_counter()

        gen_t = end - (first_tok_t or end)
        _pp = (first_tok_t - start) if first_tok_t else 0
        _tps = tokens / gen_t if gen_t > 0 else 0
        _q = req.messages[-1].content[:50].replace(chr(10), " ")
        _ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{_ts}] [{HOSTNAME}] [{_q:50s}] PP:{_pp:.2f}s TG:{gen_t:.2f}s TTFT:{_pp:.2f}s TPS:{_tps:5.1f} Tok:{tokens:4d} Total:{(end-start):.2f}s", flush=True)

        return {
            "id": rid, "object": "chat.completion", "created": created, "model": MODEL_ID,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": full}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": tokens, "total_tokens": tokens},
            "x_metrics": {
                "ttft": round(_pp, 3), "tps": round(_tps, 2),
                "itl_ms": round(gen_t / tokens * 1000, 2) if tokens else 0,
                "elapsed": round(end - start, 3),
            },
        }
    finally:
        if not req.stream:
            with _active_lock:
                _active_count -= 1

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
