#!/usr/bin/env python3
"""
ask_big_brain.py — Query the Qwen3.5-122B on CUDA cluster for hard problems.
Senior engineer on speed dial for Vulkan/ggml/shader questions.

Usage:
  python ask_big_brain.py "How do I chain 32 ggml_mul_mat ops into one compute graph?"
  python ask_big_brain.py --file problem.txt
  python ask_big_brain.py --code file.c "What's wrong with this ggml code?"

The model has 133K context — you can send entire files.
"""

import argparse
import json
import os
import sys
import urllib.request

ENDPOINT = "http://10.255.255.11:8000/v1/completions"
MODEL = "/vmrepo/models/Qwen3.5-122B-A10B-FP8"
MAX_TOKENS = 16384

SYSTEM_CONTEXT = """You are a senior GPU systems engineer specializing in:
- Vulkan compute shaders (GLSL/SPIR-V)
- Apple AGX GPU architecture (M1/M2 Ultra, Asahi Linux)
- ggml library internals (tensor ops, compute graphs, Vulkan backend)
- LLM inference optimization (vLLM, llama.cpp, memory bandwidth)
- Matrix multiplication optimization (tiling, shared memory, cooperative matrix)

Give precise, actionable answers. Include code when relevant. No fluff."""


LOG_FILE = os.path.expanduser("~/AGENT/BRAIN_CONVERSATIONS.md")

def _log_brain(role, content):
    import datetime
    with open(LOG_FILE, "a") as f:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n## [{ts}] {role} -> 122B\n{content[:2000]}\n")

def ask(question, code_context=None, file_context=None, max_tokens=MAX_TOKENS, temperature=0):
    prompt = SYSTEM_CONTEXT + "\n\n"

    if file_context:
        prompt += f"=== FILE CONTEXT ===\n{file_context}\n=== END FILE ===\n\n"

    if code_context:
        prompt += f"=== CODE ===\n{code_context}\n=== END CODE ===\n\n"

    prompt += f"QUESTION: {question}\n\nANSWER:"

    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": ["QUESTION:", "=== FILE"],
    }).encode()

    req = urllib.request.Request(
        ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            text = result["choices"][0]["text"].strip()
            usage = result.get("usage", {})
            tokens = usage.get("completion_tokens", 0)
            _log_brain("QUESTION", question[:500])
            print(text)
            _log_brain("ANSWER", text[:2000])
            print(f"\n--- {tokens} tokens | {MODEL.split('/')[-1]} ---", file=sys.stderr)
            return text
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="Ask the 122B model a question")
    parser.add_argument("question", nargs="*", help="The question to ask")
    parser.add_argument("--file", help="Include file contents as context")
    parser.add_argument("--code", help="Include code file as context")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=0)
    args = parser.parse_args()

    question = " ".join(args.question) if args.question else None
    file_context = None
    code_context = None

    if args.file:
        with open(args.file) as f:
            file_context = f.read()

    if args.code:
        with open(args.code) as f:
            code_context = f.read()

    if not question and not file_context:
        question = sys.stdin.read()

    if not question and file_context:
        question = "Analyze this file and provide insights."

    ask(question, code_context=code_context, file_context=file_context,
        max_tokens=args.max_tokens, temperature=args.temperature)


if __name__ == "__main__":
    main()
