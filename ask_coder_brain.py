#!/usr/bin/env python3
"""
ask_coder_brain.py — Qwen3 Coder on Sys4 (10.255.255.4:8000)
Uses chat completions. Logs all conversations.
"""
import argparse, json, sys, urllib.request, os, datetime

ENDPOINT = "http://10.255.255.4:8000/v1/chat/completions"
MODEL = "mlx-community/Qwen3-Coder-Next-8bit"
LOG_FILE = os.path.expanduser("~/AGENT/BRAIN_CONVERSATIONS.md")

SYSTEM = "You are an expert low-level systems programmer. Specialize in C/C++ GPU compute (Vulkan, Metal, CUDA), ggml internals, Vulkan compute shaders (GLSL/SPIR-V), LLM inference optimization. Write working code. Be precise."

def log(role, content):
    with open(LOG_FILE, "a") as f:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n## [{ts}] {role} → {MODEL}\n{content}\n")

def ask(question, code_context=None, max_tokens=16384, temperature=0):
    msgs = [{"role": "system", "content": SYSTEM}]
    if code_context:
        msgs.append({"role": "user", "content": f"CODE:\n```\n{code_context}\n```\n\n{question}"})
    else:
        msgs.append({"role": "user", "content": question})

    log("QUESTION", question[:500])
    payload = json.dumps({"model": MODEL, "messages": msgs, "max_tokens": max_tokens, "temperature": temperature}).encode()
    req = urllib.request.Request(ENDPOINT, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            text = result["choices"][0]["message"]["content"]
            tokens = result.get("usage", {}).get("total_tokens", 0)
            print(text)
            log("ANSWER", text[:2000])
            print(f"\n--- {tokens} tokens | {MODEL} (coder) ---", file=sys.stderr)
            return text
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*")
    parser.add_argument("--code", help="Code file")
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    question = " ".join(args.question) if args.question else sys.stdin.read()
    code_ctx = open(args.code).read() if args.code else None
    ask(question, code_context=code_ctx, max_tokens=args.max_tokens)

if __name__ == "__main__": main()
