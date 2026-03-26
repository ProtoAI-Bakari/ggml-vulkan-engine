#!/usr/bin/env python3
"""
auto_model_tester.py - Automated benchmark for MLX brain servers and CUDA cluster.

Tests all endpoints with standardized prompts, measures TTFT/TPS/latency via
streaming SSE, and outputs a rich comparison table + JSON log.
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ENDPOINTS = [
    {"url": "http://10.255.255.2:8000",  "name": "ARCHITECT",  "model": "Qwen3-235B-Thinking"},
    {"url": "http://10.255.255.3:8000",  "name": "ENGINEER",   "model": "Qwen3.5-122B"},
    {"url": "http://10.255.255.4:8000",  "name": "CODER",      "model": "Qwen3-Coder-Next-8bit"},
    {"url": "http://10.255.255.5:8000",  "name": "DESIGNER",   "model": "GLM-4.7-Flash"},
    {"url": "http://10.255.255.6:8000",  "name": "REVIEWER",   "model": "Qwen3.5-122B"},
    {"url": "http://10.255.255.7:8000",  "name": "FAST-CODER", "model": "Qwen3-Coder-Next-4bit"},
    {"url": "http://10.255.255.11:8000", "name": "CUDA",       "model": "Qwen3.5-122B-FP8"},
]

ALL_PROMPTS = [
    "Write a Python function to reverse a linked list",
    "Write a C function that performs matrix multiplication with SIMD intrinsics",
    "Review this code for bugs: `int* p = malloc(10); free(p); *p = 5;`",
    "Design the architecture for a distributed KV cache across 4 GPU nodes",
    "Write a bash script that monitors GPU memory usage every 5 seconds",
]

LOGS_DIR = Path("/home/z/AGENT/LOGS")
CONNECT_TIMEOUT = 10   # seconds
STREAM_TIMEOUT  = 120  # seconds per prompt

# Rough token estimator: split on whitespace + punctuation
def rough_token_count(text: str) -> int:
    return max(1, len(re.findall(r"\S+", text)))

def contains_code(text: str) -> bool:
    """Return True if the response has a fenced code block or obvious code patterns."""
    if "```" in text:
        return True
    # Bare-minimum heuristics: indented blocks, function/def/fn/void keywords
    code_patterns = [
        r"\bdef\s+\w+\s*\(",
        r"\bvoid\s+\w+\s*\(",
        r"\bint\s+\w+\s*\(",
        r"\bfunction\b",
        r"#!/",
        r"\bfor\s*\(",
        r"\bwhile\s*\(",
        r"\$\(",
    ]
    for pat in code_patterns:
        if re.search(pat, text):
            return True
    return False

# ---------------------------------------------------------------------------
# Streaming inference
# ---------------------------------------------------------------------------

def stream_completion(endpoint_url: str, prompt: str) -> dict:
    """
    Send a chat-completion streaming request and measure timing metrics.
    Returns a dict with: ttft, total_time, tps, token_count, has_code,
    response_text, error.
    """
    url = endpoint_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": "default",          # servers typically expose one model
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": 1024,
        "temperature": 0.2,
    }
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

    result = {
        "ttft": None,
        "total_time": None,
        "tps": None,
        "token_count": 0,
        "has_code": False,
        "response_text": "",
        "error": None,
    }

    t_start = time.perf_counter()
    t_first_token = None
    chunks = []

    try:
        with requests.post(
            url,
            json=payload,
            headers=headers,
            stream=True,
            timeout=(CONNECT_TIMEOUT, STREAM_TIMEOUT),
        ) as resp:
            if resp.status_code != 200:
                result["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
                return result

            for raw_line in resp.iter_lines(chunk_size=None, decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        if t_first_token is None:
                            t_first_token = time.perf_counter()
                        chunks.append(content)

    except requests.exceptions.ConnectTimeout:
        result["error"] = "Connection timeout"
        return result
    except requests.exceptions.ReadTimeout:
        result["error"] = "Read timeout"
        return result
    except requests.exceptions.ConnectionError as e:
        result["error"] = f"Connection error: {e}"
        return result
    except Exception as e:
        result["error"] = f"Unexpected: {e}"
        return result

    t_end = time.perf_counter()

    full_text = "".join(chunks)
    result["response_text"] = full_text
    result["has_code"] = contains_code(full_text)
    result["total_time"] = round(t_end - t_start, 3)

    if t_first_token is not None:
        result["ttft"] = round(t_first_token - t_start, 3)

    token_count = rough_token_count(full_text)
    result["token_count"] = token_count

    generation_time = t_end - (t_first_token if t_first_token else t_start)
    if generation_time > 0 and token_count > 0:
        result["tps"] = round(token_count / generation_time, 1)

    return result

# ---------------------------------------------------------------------------
# Rich table printing (using only stdlib)
# ---------------------------------------------------------------------------

def truncate(s, n):
    if s is None:
        return "N/A"
    s = str(s)
    return s if len(s) <= n else s[:n-1] + "…"

def fmt_float(v, decimals=2):
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"

def print_table(all_results, prompts):
    """Print a plain-text table summarizing results."""
    COL_NAME  = 14
    COL_MODEL = 22
    COL_TTFT  = 8
    COL_TOT   = 8
    COL_TPS   = 7
    COL_TOK   = 6
    COL_CODE  = 5
    COL_ERR   = 24

    sep_char = "-"
    total_w = COL_NAME + COL_MODEL + COL_TTFT + COL_TOT + COL_TPS + COL_TOK + COL_CODE + COL_ERR + 7*3 + 2

    def row(*cols, widths):
        parts = [c.ljust(w) for c, w in zip(cols, widths)]
        return "| " + " | ".join(parts) + " |"

    widths = [COL_NAME, COL_MODEL, COL_TTFT, COL_TOT, COL_TPS, COL_TOK, COL_CODE, COL_ERR]
    header_labels = ["ENDPOINT", "MODEL", "TTFT(s)", "TOT(s)", "TPS", "TOKS", "CODE", "ERROR"]

    print()
    print("=" * total_w)
    print("  MODEL BENCHMARK RESULTS")
    print("=" * total_w)

    for pidx, prompt in enumerate(prompts):
        print(f"\n  Prompt {pidx+1}: {prompt[:80]}")
        print(sep_char * total_w)
        print(row(*header_labels, widths=widths))
        print(sep_char * total_w)

        for ep_results in all_results:
            ep = ep_results["endpoint"]
            res = ep_results["prompt_results"][pidx]
            print(row(
                truncate(ep["name"], COL_NAME),
                truncate(ep["model"], COL_MODEL),
                fmt_float(res.get("ttft")),
                fmt_float(res.get("total_time")),
                fmt_float(res.get("tps"), 1),
                str(res.get("token_count", 0)),
                "YES" if res.get("has_code") else "no",
                truncate(res.get("error") or "", COL_ERR),
                widths=widths,
            ))
        print(sep_char * total_w)

    # Summary averages
    print(f"\n{'='*total_w}")
    print("  SUMMARY AVERAGES (across all tested prompts, excluding errors)")
    print(sep_char * total_w)
    print(row(*header_labels, widths=widths))
    print(sep_char * total_w)

    for ep_results in all_results:
        ep = ep_results["endpoint"]
        prs = ep_results["prompt_results"]
        ok = [r for r in prs if r.get("error") is None]
        avg_ttft  = (sum(r["ttft"] for r in ok if r["ttft"] is not None) / len(ok)) if ok else None
        avg_tot   = (sum(r["total_time"] for r in ok if r["total_time"] is not None) / len(ok)) if ok else None
        avg_tps   = (sum(r["tps"] for r in ok if r["tps"] is not None) / len(ok)) if ok else None
        avg_tok   = (sum(r["token_count"] for r in ok) / len(ok)) if ok else 0
        code_pct  = (sum(1 for r in ok if r.get("has_code")) / len(ok) * 100) if ok else 0
        err_count = len(prs) - len(ok)
        print(row(
            truncate(ep["name"], COL_NAME),
            truncate(ep["model"], COL_MODEL),
            fmt_float(avg_ttft),
            fmt_float(avg_tot),
            fmt_float(avg_tps, 1),
            f"{avg_tok:.0f}" if avg_tok else "0",
            f"{code_pct:.0f}%",
            f"{err_count} errors" if err_count else "",
            widths=widths,
        ))
    print(sep_char * total_w)
    print()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Automated benchmark for MLX brain servers and CUDA cluster."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only run the first 2 prompts (faster testing).",
    )
    parser.add_argument(
        "--endpoints",
        nargs="*",
        metavar="NAME",
        help="Limit to specific endpoint names, e.g. --endpoints CODER CUDA",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving the JSON results file.",
    )
    args = parser.parse_args()

    prompts = ALL_PROMPTS[:2] if args.quick else ALL_PROMPTS

    endpoints = ENDPOINTS
    if args.endpoints:
        names_upper = {n.upper() for n in args.endpoints}
        endpoints = [e for e in ENDPOINTS if e["name"].upper() in names_upper]
        if not endpoints:
            print(f"No matching endpoints for: {args.endpoints}")
            sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = LOGS_DIR / f"model_benchmark_{timestamp}.json"

    mode_label = "QUICK (2 prompts)" if args.quick else f"FULL ({len(prompts)} prompts)"
    print(f"\nStarting benchmark — {mode_label} — {len(endpoints)} endpoints")
    print(f"Timestamp: {timestamp}")

    all_results = []

    for ep in endpoints:
        ep_entry = {
            "endpoint": ep,
            "prompt_results": [],
        }
        print(f"\n  [{ep['name']}] {ep['model']} @ {ep['url']}")

        for pidx, prompt in enumerate(prompts):
            print(f"    Prompt {pidx+1}/{len(prompts)}: {prompt[:60]}...", end="", flush=True)
            res = stream_completion(ep["url"], prompt)
            if res["error"]:
                print(f"  ERROR: {res['error']}")
            else:
                print(f"  TTFT={res['ttft']}s  TPS={res['tps']}  tokens={res['token_count']}  code={res['has_code']}")
            ep_entry["prompt_results"].append(res)

        all_results.append(ep_entry)

    print_table(all_results, prompts)

    if not args.no_save:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        output = {
            "timestamp": timestamp,
            "mode": "quick" if args.quick else "full",
            "prompts": prompts,
            "results": [
                {
                    "endpoint": ep_r["endpoint"],
                    "prompt_results": [
                        {k: v for k, v in r.items() if k != "response_text"}
                        for r in ep_r["prompt_results"]
                    ],
                }
                for ep_r in all_results
            ],
        }
        with open(log_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to: {log_path}")


if __name__ == "__main__":
    main()
