#!/usr/bin/env python3
"""
OpenAI-compatible streaming inference server for ggml Vulkan engine.
Serves /v1/completions and /v1/chat/completions endpoints.
Designed for Z's Streamlit telemetry deck and fleet integration.

Usage:
  python ggml_server.py --model ~/models/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --port 8080
  curl http://localhost:8080/v1/chat/completions -d '{"model":"llama-8b","messages":[{"role":"user","content":"Hello"}],"stream":true}'
"""
import argparse
import json
import os
import sys
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from threading import Lock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ggml_vllm_backend import GgmlLLM, SamplingParams

# Chat templates
LLAMA_TEMPLATE = {
    "system_start": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
    "system_end": "<|eot_id|>",
    "user_start": "<|start_header_id|>user<|end_header_id|>\n\n",
    "user_end": "<|eot_id|>",
    "assistant_start": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "assistant_end": "<|eot_id|>",
}

QWEN_TEMPLATE = {
    "system_start": "<|im_start|>system\n",
    "system_end": "<|im_end|>\n",
    "user_start": "<|im_start|>user\n",
    "user_end": "<|im_end|>\n",
    "assistant_start": "<|im_start|>assistant\n",
    "assistant_end": "<|im_end|>\n",
}


def format_chat(messages, template):
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += template["system_start"] + content + template["system_end"]
        elif role == "user":
            prompt += template["user_start"] + content + template["user_end"]
        elif role == "assistant":
            prompt += template["assistant_start"] + content + template["assistant_end"]
    prompt += template["assistant_start"]
    return prompt


class InferenceHandler(BaseHTTPRequestHandler):
    llm = None
    model_name = "unknown"
    template = LLAMA_TEMPLATE
    gen_lock = Lock()
    request_count = 0
    total_tokens = 0
    start_time = None

    def log_message(self, format, *args):
        pass  # Suppress default logging

    def _send_json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, data):
        chunk = f"data: {json.dumps(data)}\n\n".encode()
        self.wfile.write(chunk)
        self.wfile.flush()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "model": self.model_name})
        elif self.path == "/v1/models":
            self._send_json(200, {"data": [{"id": self.model_name, "object": "model"}]})
        elif self.path == "/metrics":
            uptime = time.time() - (self.start_time or time.time())
            self._send_json(200, {
                "requests": self.request_count,
                "total_tokens": self.total_tokens,
                "uptime_s": round(uptime, 1),
                "avg_tps": round(self.total_tokens / uptime, 1) if uptime > 0 else 0,
                **self.llm.stats(),
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length > 0 else {}

        if self.path == "/v1/completions":
            self._handle_completions(body)
        elif self.path == "/v1/chat/completions":
            self._handle_chat(body)
        else:
            self._send_json(404, {"error": "not found"})

    def _handle_chat(self, body):
        messages = body.get("messages", [])
        prompt = format_chat(messages, self.template)
        body["_prompt"] = prompt
        self._handle_completions(body, is_chat=True)

    def _handle_completions(self, body, is_chat=False):
        prompt = body.get("_prompt", body.get("prompt", ""))
        stream = body.get("stream", False)
        req_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        params = SamplingParams(
            temperature=body.get("temperature", 0.0),
            top_p=body.get("top_p", 1.0),
            top_k=body.get("top_k", -1),
            max_tokens=body.get("max_tokens", 256),
            repetition_penalty=body.get("repetition_penalty", 1.0),
        )

        InferenceHandler.request_count += 1

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            tokens_gen = [0]
            def on_token(text, token_id):
                tokens_gen[0] += 1
                delta = {"role": "assistant", "content": text} if is_chat else {"text": text}
                chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk" if is_chat else "text_completion",
                    "created": int(time.time()),
                    "model": self.model_name,
                    "choices": [{"index": 0, "delta" if is_chat else "text": delta, "finish_reason": None}],
                }
                try:
                    self._send_sse(chunk)
                except BrokenPipeError:
                    pass

            with self.gen_lock:
                result = self.llm.generate(prompt, params=params, stream_callback=on_token)

            # Final chunk
            final = {
                "id": req_id,
                "object": "chat.completion.chunk" if is_chat else "text_completion",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [{"index": 0, "delta" if is_chat else "text": {} if is_chat else "", "finish_reason": result.finish_reason}],
                "usage": {"prompt_tokens": result.prefill_tokens, "completion_tokens": result.decode_tokens, "total_tokens": result.prefill_tokens + result.decode_tokens},
            }
            self._send_sse(final)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            InferenceHandler.total_tokens += result.decode_tokens

        else:
            with self.gen_lock:
                result = self.llm.generate(prompt, params=params)

            resp = {
                "id": req_id,
                "object": "chat.completion" if is_chat else "text_completion",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [{
                    "index": 0,
                    "message" if is_chat else "text": {"role": "assistant", "content": result.text} if is_chat else result.text,
                    "finish_reason": result.finish_reason,
                }],
                "usage": {
                    "prompt_tokens": result.prefill_tokens,
                    "completion_tokens": result.decode_tokens,
                    "total_tokens": result.prefill_tokens + result.decode_tokens,
                },
                "_tps": result.tps,
            }
            self._send_json(200, resp)
            InferenceHandler.total_tokens += result.decode_tokens


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(description="ggml Vulkan inference server")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--n-ctx", type=int, default=2048)
    args = parser.parse_args()

    model_base = os.path.basename(args.model).lower()
    InferenceHandler.model_name = model_base.replace(".gguf", "")
    InferenceHandler.template = QWEN_TEMPLATE if "qwen" in model_base else LLAMA_TEMPLATE

    print(f"Loading {args.model}...")
    InferenceHandler.llm = GgmlLLM(args.model, n_ctx=args.n_ctx)
    InferenceHandler.start_time = time.time()

    server = ThreadedHTTPServer((args.host, args.port), InferenceHandler)
    print(f"Server running on http://{args.host}:{args.port}")
    print(f"  Model: {InferenceHandler.model_name}")
    print(f"  Endpoints: /v1/completions, /v1/chat/completions, /health, /metrics, /v1/models")
    print(f"  Stream: curl {args.host}:{args.port}/v1/chat/completions -d '{{\"messages\":[{{\"role\":\"user\",\"content\":\"Hello\"}}],\"stream\":true}}'")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
