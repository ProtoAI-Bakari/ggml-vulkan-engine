#!/usr/bin/env python3
"""
brain_bridge.py — Real-time bidirectional conversation bridge to local AI brains.

Supports multi-turn conversations, multi-brain orchestration, chaining,
streaming SSE display, and full conversation trace logging.

Brains:
  brain  — Qwen3.5-122B on 10.255.255.11:8000 (reasoning/architecture)
  coder  — Qwen3-Coder-Next on 10.255.255.4:8000 (code generation)
  mini   — MiniMaxM2 on 192.168.1.164:8765 (fast/light tasks)

Usage:
  python brain_bridge.py "how do I tile a matmul in Vulkan compute?"
  python brain_bridge.py --brain coder "write a GLSL shader for softmax"
  python brain_bridge.py --brain brain,coder --discuss "best MoE routing strategy"
  python brain_bridge.py --chain brain,coder "design a fused attention kernel"
  python brain_bridge.py --interactive --brain coder
"""

import argparse
import datetime
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    import urllib.request
    import urllib.error

# ── ANSI color codes ────────────────────────────────────────────────────────
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    # Brain colors
    BRAIN   = "\033[38;5;214m"  # orange — 122B reasoning
    CODER   = "\033[38;5;120m"  # green  — coder
    MINI    = "\033[38;5;183m"  # purple — mini
    CLAUDE  = "\033[38;5;75m"   # blue   — Claude (us)
    USER    = "\033[38;5;255m"  # white  — user/question
    SYSTEM  = "\033[38;5;245m"  # gray   — system messages
    ERR     = "\033[38;5;196m"  # red    — errors
    HEAD    = "\033[38;5;39m"   # cyan   — headers
    STREAM  = "\033[38;5;252m"  # light  — streaming tokens
    SEP     = "\033[38;5;240m"  # dark gray — separators

BRAIN_COLORS = {
    "brain": C.BRAIN,
    "coder": C.CODER,
    "mini":  C.MINI,
    "claude": "\033[38;5;141m",  # Purple for Claude
    "fast":   "\033[38;5;208m",  # Orange for fast rewrite
}

BRAIN_LABELS = {
    "brain": "122B-Reasoning",
    "coder": "Qwen3-Coder",
    "mini":  "MiniMaxM2",
    "claude": "Claude-Opus-4.6",
    "fast":   "Qwen3B-Fast",
}

# ── Brain endpoint configuration ────────────────────────────────────────────
BRAINS = {
    "brain": {
        "endpoint": "http://10.255.255.11:8000/v1/chat/completions",
        "model": "/vmrepo/models/Qwen3.5-122B-A10B-FP8",
        "system": (
            "You are a senior GPU systems engineer. Expert in Vulkan compute shaders "
            "(GLSL/SPIR-V), Apple AGX architecture (M1/M2 Ultra, Asahi Linux), ggml "
            "internals (tensor ops, compute graphs, Vulkan backend), LLM inference "
            "optimization. Give precise, actionable answers. Include code when relevant."
        ),
        "max_tokens": 4096,
    },
    "coder": {
        "endpoint": "http://10.255.255.4:8000/v1/chat/completions",
        "model": "mlx-community/Qwen3-Coder-Next-8bit",
        "system": (
            "You are an expert low-level systems programmer. Specialize in C/C++ GPU "
            "compute (Vulkan, Metal, CUDA), ggml internals, Vulkan compute shaders "
            "(GLSL/SPIR-V), LLM inference optimization. Write working code. Be precise."
        ),
        "max_tokens": 4096,
    },
    "mini": {
        "endpoint": "http://192.168.1.164:8765/v1/chat/completions",
        "model": "MiniMaxM2",
        "system": "You are a helpful AI assistant. Be concise and precise.",
        "max_tokens": 2048,
    },
    "fast": {
        "endpoint": "http://localhost:8081/v1/chat/completions",
        "model": "qwen2.5-3b-instruct-q4_k_m",
        "system": (
            "You are a fast file editor. When given a file and instructions, "
            "output ONLY the changed lines with their line numbers. "
            "Format: LINE_NUMBER: new_content. Nothing else."
        ),
        "max_tokens": 4096,
    },
    "claude": {
        "endpoint": "cli://claude",  # Special: uses 'claude -p' subprocess
        "model": "claude-opus-4-6",
        "system": (
            "You are Claude, assisting with a Vulkan GPU inference engine project "
            "on Apple M1 Ultra (Asahi Linux). The project uses ggml library with "
            "Vulkan backend. Current engine achieves 22 TPS on 8B Llama Q4. "
            "Be direct, give code when asked, no fluff."
        ),
        "max_tokens": 16384,
    },
}

LOG_FILE = os.path.expanduser("~/AGENT/BRAIN_CONVERSATIONS.md")


# ── Utilities ────────────────────────────────────────────────────────────────

def _ts():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _separator(char="─", width=80):
    print(f"{C.SEP}{char * width}{C.RESET}")

def _header(text):
    _separator("━")
    print(f"{C.HEAD}{C.BOLD}  {text}{C.RESET}")
    _separator("━")

def _log(brain_name, role, content, session_id=None):
    """Append to the conversation log file."""
    label = BRAIN_LABELS.get(brain_name, brain_name)
    try:
        with open(LOG_FILE, "a") as f:
            sid = f" [session:{session_id}]" if session_id else ""
            f.write(f"\n## [{_ts()}] {role} -> {label}{sid}\n{content}\n")
    except Exception:
        pass  # never let logging break anything

def _print_question(brain_name, question):
    color = BRAIN_COLORS.get(brain_name, C.USER)
    label = BRAIN_LABELS.get(brain_name, brain_name)
    print(f"\n{C.DIM}[{_ts()}]{C.RESET} {C.CLAUDE}{C.BOLD}Claude{C.RESET} -> {color}{label}{C.RESET}")
    print(f"{C.USER}{question}{C.RESET}")
    print()

def _print_response_header(brain_name):
    color = BRAIN_COLORS.get(brain_name, C.USER)
    label = BRAIN_LABELS.get(brain_name, brain_name)
    print(f"{C.DIM}[{_ts()}]{C.RESET} {color}{C.BOLD}{label}{C.RESET} -> {C.CLAUDE}Claude{C.RESET}")

def _print_stats(brain_name, tokens=0, elapsed=0):
    color = BRAIN_COLORS.get(brain_name, C.SYSTEM)
    label = BRAIN_LABELS.get(brain_name, brain_name)
    tps = tokens / elapsed if elapsed > 0 else 0
    print(f"\n{C.DIM}--- {label}: {tokens} tokens, {elapsed:.1f}s, {tps:.1f} t/s ---{C.RESET}")


# ── HTTP transport layer ─────────────────────────────────────────────────────

def _post_json(url, payload, timeout=120):
    """POST JSON, return parsed response. Works with or without requests."""
    if HAS_REQUESTS:
        resp = requests.post(
            url,
            json=payload,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()
    else:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

def _post_stream(url, payload, timeout=120):
    """POST with streaming SSE. Yields text chunks as they arrive."""
    if HAS_REQUESTS:
        resp = requests.post(
            url,
            json=payload,
            timeout=timeout,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            stream=True,
        )
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data = line[6:]
                if data.strip() == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    text = delta.get("content", "")
                    if text:
                        yield text
                except json.JSONDecodeError:
                    continue
    else:
        # Fallback: no streaming without requests
        result = _post_json(url, {k: v for k, v in payload.items() if k != "stream"}, timeout)
        text = result["choices"][0]["message"]["content"]
        yield text


# ── BrainBridge class ────────────────────────────────────────────────────────

class BrainBridge:
    """Real-time bidirectional conversation bridge to local AI brains."""

    def __init__(self, default_brain="brain"):
        self.default_brain = default_brain
        self.brains = dict(BRAINS)
        self._sessions = {}  # brain_name -> list of messages (for multi-turn)
        self._session_counter = 0

        # Probe which brains are reachable
        self.available = {}
        for name in self.brains:
            self.available[name] = self._probe(name)

        online = [f"{BRAIN_LABELS[n]}" for n, ok in self.available.items() if ok]
        offline = [f"{BRAIN_LABELS[n]}" for n, ok in self.available.items() if not ok]

        _header("BrainBridge initialized")
        if online:
            print(f"  {C.CODER}ONLINE:{C.RESET}  {', '.join(online)}")
        if offline:
            print(f"  {C.DIM}OFFLINE: {', '.join(offline)}{C.RESET}")
        _separator()

    def _probe(self, brain_name):
        """Quick check if a brain is reachable."""
        cfg = self.brains[brain_name]
        base = cfg["endpoint"].rsplit("/v1/", 1)[0]
        try:
            if HAS_REQUESTS:
                r = requests.get(f"{base}/v1/models", timeout=5)
                return r.status_code == 200
            else:
                req = urllib.request.Request(f"{base}/v1/models")
                with urllib.request.urlopen(req, timeout=5):
                    return True
        except Exception:
            return False

    def _build_messages(self, brain_name, question, context=None, system_prompt=None, history=None):
        """Build the messages array for chat completions."""
        cfg = self.brains[brain_name]
        system = system_prompt or cfg["system"]
        msgs = [{"role": "system", "content": system}]

        if history:
            msgs.extend(history)

        user_content = question
        if context:
            user_content = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

        msgs.append({"role": "user", "content": user_content})
        return msgs

    # ── Core: ask one brain ──────────────────────────────────────────────

    def ask(self, brain=None, question="", context=None, system_prompt=None,
            stream=True, max_tokens=None, temperature=0, session_id=None):
        """
        Ask a single question to one brain. Streams response to terminal.
        Returns the full response text.
        """
        brain = brain or self.default_brain
        if brain not in self.brains:
            print(f"{C.ERR}Unknown brain: {brain}. Available: {', '.join(self.brains)}{C.RESET}")
            return ""

        cfg = self.brains[brain]
        msgs = self._build_messages(brain, question, context, system_prompt,
                                     history=self._sessions.get(session_id))

        _print_question(brain, question)
        _log(brain, "QUESTION", question, session_id)

        payload = {
            "model": cfg["model"],
            "messages": msgs,
            "max_tokens": max_tokens or cfg["max_tokens"],
            "temperature": temperature,
            "stream": stream and HAS_REQUESTS,
        }

        _print_response_header(brain)
        color = BRAIN_COLORS.get(brain, C.STREAM)
        full_text = ""
        t0 = time.time()

        try:
            # Special handling for Claude (subprocess, not HTTP)
            if cfg["endpoint"].startswith("cli://"):
                import subprocess
                prompt = question
                if context:
                    prompt = f"Context:\n{context}\n\nQuestion: {question}"
                result = subprocess.run(
                    ["claude", "-p", prompt],
                    capture_output=True, text=True, timeout=180,
                    cwd=os.path.expanduser("~/AGENT")
                )
                full_text = result.stdout.strip()
                print(f"{color}{full_text}{C.RESET}")
            elif payload["stream"]:
                for chunk in _post_stream(cfg["endpoint"], payload, timeout=120):
                    full_text += chunk
                    sys.stdout.write(f"{color}{chunk}{C.RESET}")
                    sys.stdout.flush()
            else:
                result = _post_json(cfg["endpoint"], payload, timeout=120)
                full_text = result["choices"][0]["message"]["content"]
                print(f"{color}{full_text}{C.RESET}")

            elapsed = time.time() - t0
            # Rough token estimate: ~0.75 tokens per word
            est_tokens = len(full_text.split()) * 4 // 3
            _print_stats(brain, est_tokens, elapsed)

        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n{C.ERR}ERROR [{brain}]: {e}{C.RESET}")
            _log(brain, "ERROR", str(e), session_id)
            return ""

        # Log response
        _log(brain, "ANSWER", full_text[:4000], session_id)

        # Track in session if multi-turn
        if session_id is not None:
            if session_id not in self._sessions:
                self._sessions[session_id] = []
            self._sessions[session_id].append({"role": "user", "content": question})
            self._sessions[session_id].append({"role": "assistant", "content": full_text})

        return full_text

    # ── Discuss: same question to multiple brains ────────────────────────

    def discuss(self, question, brains=None, context=None):
        """
        Ask the same question to multiple brains simultaneously.
        Returns dict of {brain_name: response}.
        """
        brains = brains or ["brain", "coder"]
        results = {}

        _header(f"DISCUSS: asking {len(brains)} brains")
        print(f"{C.USER}{question}{C.RESET}\n")

        # Log the discussion start
        _log("discuss", "QUESTION", f"[multi-brain: {','.join(brains)}] {question}")

        def _ask_one(brain_name):
            return brain_name, self.ask(brain_name, question, context=context, stream=False)

        with ThreadPoolExecutor(max_workers=len(brains)) as pool:
            futures = {pool.submit(_ask_one, b): b for b in brains}
            for future in as_completed(futures):
                brain_name, response = future.result()
                results[brain_name] = response

        # Summary
        _header("DISCUSSION SUMMARY")
        for b, resp in results.items():
            label = BRAIN_LABELS.get(b, b)
            color = BRAIN_COLORS.get(b, C.USER)
            preview = resp[:200].replace("\n", " ") if resp else "(no response)"
            print(f"  {color}{label}:{C.RESET} {preview}...")
        _separator()

        return results

    # ── Chain: sequential multi-brain pipeline ───────────────────────────

    def chain(self, question, steps=None):
        """
        Chain conversation across brains. Each step gets the previous answer as context.

        steps: list of (brain_name, instruction) tuples.
        If None, defaults to brain -> coder chain.
        """
        if steps is None:
            steps = [
                ("brain", "Analyze and design a solution for this:"),
                ("coder", "Based on the analysis above, write the implementation code:"),
            ]

        _header(f"CHAIN: {len(steps)} steps")
        print(f"{C.USER}Initial question: {question}{C.RESET}\n")

        _log("chain", "START", f"[{len(steps)} steps] {question}")

        results = []
        accumulated_context = question

        for i, (brain, instruction) in enumerate(steps):
            step_label = f"Step {i+1}/{len(steps)}"
            _header(f"CHAIN {step_label}: {BRAIN_LABELS.get(brain, brain)}")
            print(f"{C.DIM}Instruction: {instruction}{C.RESET}\n")

            full_question = f"{instruction}\n\n{accumulated_context}"
            response = self.ask(brain, full_question, stream=True)
            results.append({"brain": brain, "instruction": instruction, "response": response})
            accumulated_context = f"Previous analysis:\n{response}\n\nOriginal question:\n{question}"

        _header("CHAIN COMPLETE")
        for i, r in enumerate(results):
            label = BRAIN_LABELS.get(r["brain"], r["brain"])
            color = BRAIN_COLORS.get(r["brain"], C.USER)
            preview = r["response"][:150].replace("\n", " ") if r["response"] else "(empty)"
            print(f"  {color}Step {i+1} ({label}):{C.RESET} {preview}...")
        _separator()

        return results

    # ── Conversation: multi-turn interactive session ─────────────────────

    def conversation(self, brain=None, system_prompt=None):
        """
        Generator-based multi-turn conversation with one brain.
        Yields each response. Send next question via .send().

        Usage:
            conv = bridge.conversation("coder")
            resp = next(conv)  # greeting/ready message
            resp = conv.send("write a matmul kernel")
            resp = conv.send("now optimize it for M1 Ultra")
        """
        brain = brain or self.default_brain
        self._session_counter += 1
        session_id = f"conv-{self._session_counter}"

        self._sessions[session_id] = []

        _header(f"CONVERSATION with {BRAIN_LABELS.get(brain, brain)}")
        print(f"{C.DIM}Session: {session_id} | Type 'quit' to end{C.RESET}")
        _separator()

        _log(brain, "SESSION_START", f"Interactive conversation {session_id}", session_id)

        # Initial yield — caller sends first question
        question = yield f"[session {session_id} ready]"

        while question and question.strip().lower() not in ("quit", "exit", "q"):
            response = self.ask(brain, question, system_prompt=system_prompt,
                              stream=True, session_id=session_id)
            question = yield response

        _log(brain, "SESSION_END", f"Conversation {session_id} ended", session_id)
        _header(f"Session {session_id} ended")

    # ── Interactive CLI conversation ─────────────────────────────────────

    def interactive(self, brain=None, system_prompt=None):
        """
        Blocking interactive conversation loop. Reads from stdin.
        """
        brain = brain or self.default_brain
        self._session_counter += 1
        session_id = f"interactive-{self._session_counter}"
        self._sessions[session_id] = []

        _header(f"INTERACTIVE SESSION with {BRAIN_LABELS.get(brain, brain)}")
        print(f"{C.DIM}Session: {session_id}")
        print(f"Commands: /quit /clear /switch <brain> /brains{C.RESET}")
        _separator()

        _log(brain, "SESSION_START", f"Interactive {session_id}", session_id)

        current_brain = brain

        while True:
            try:
                color = BRAIN_COLORS.get(current_brain, C.USER)
                prompt = f"\n{C.CLAUDE}{C.BOLD}You{C.RESET} ({color}{BRAIN_LABELS.get(current_brain, current_brain)}{C.RESET})> "
                question = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not question:
                continue

            if question.startswith("/"):
                cmd = question.lower().split()
                if cmd[0] in ("/quit", "/q", "/exit"):
                    break
                elif cmd[0] == "/clear":
                    self._sessions[session_id] = []
                    print(f"{C.SYSTEM}History cleared.{C.RESET}")
                    continue
                elif cmd[0] == "/switch" and len(cmd) > 1:
                    new_brain = cmd[1]
                    if new_brain in self.brains:
                        current_brain = new_brain
                        print(f"{C.SYSTEM}Switched to {BRAIN_LABELS[new_brain]}{C.RESET}")
                    else:
                        print(f"{C.ERR}Unknown brain: {new_brain}{C.RESET}")
                    continue
                elif cmd[0] == "/brains":
                    for name, cfg in self.brains.items():
                        status = f"{C.CODER}ONLINE" if self.available.get(name) else f"{C.ERR}OFFLINE"
                        marker = " <--" if name == current_brain else ""
                        print(f"  {BRAIN_COLORS[name]}{BRAIN_LABELS[name]}{C.RESET} ({name}) {status}{C.RESET}{marker}")
                    continue
                elif cmd[0] == "/discuss":
                    q = " ".join(cmd[1:]) if len(cmd) > 1 else input("Question: ").strip()
                    if q:
                        self.discuss(q)
                    continue

            self.ask(current_brain, question, stream=True, session_id=session_id)

        _log(current_brain, "SESSION_END", f"Interactive {session_id} ended", session_id)
        _header(f"Session {session_id} ended")


# ── Programmatic interface for Claude agent use ──────────────────────────────

_bridge = None

def get_bridge():
    """Get or create the singleton BrainBridge."""
    global _bridge
    if _bridge is None:
        _bridge = BrainBridge()
    return _bridge

def ask_brain(question, brain="brain", **kwargs):
    """Quick function: ask a brain. Returns response text."""
    return get_bridge().ask(brain, question, **kwargs)

def ask_coder(question, **kwargs):
    """Quick function: ask the coder brain."""
    return get_bridge().ask("coder", question, **kwargs)

def ask_reasoning(question, **kwargs):
    """Quick function: ask the 122B reasoning brain."""
    return get_bridge().ask("brain", question, **kwargs)

def discuss(question, brains=None):
    """Quick function: ask multiple brains."""
    return get_bridge().discuss(question, brains)

def chain(question, steps=None):
    """Quick function: chain across brains."""
    return get_bridge().chain(question, steps)


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BrainBridge — Real-time AI brain conversation bridge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "explain Vulkan descriptor sets"
  %(prog)s --brain coder "write a GLSL shader for softmax"
  %(prog)s --discuss --brain brain,coder "best MoE routing strategy"
  %(prog)s --chain --brain brain,coder "design fused attention"
  %(prog)s --interactive --brain coder
        """,
    )
    parser.add_argument("question", nargs="*", help="Question to ask")
    parser.add_argument("--brain", "-b", default="brain",
                       help="Brain(s) to query. Comma-separated for discuss/chain. Default: brain")
    parser.add_argument("--discuss", "-d", action="store_true",
                       help="Ask same question to all specified brains")
    parser.add_argument("--chain", "-c", action="store_true",
                       help="Chain question through brains sequentially")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Interactive multi-turn conversation")
    parser.add_argument("--context", help="File to include as context")
    parser.add_argument("--system", help="Custom system prompt")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming")

    args = parser.parse_args()
    brains_list = [b.strip() for b in args.brain.split(",")]

    bridge = BrainBridge(default_brain=brains_list[0])

    # Read context file if provided
    context = None
    if args.context:
        with open(args.context) as f:
            context = f.read()

    if args.interactive:
        bridge.interactive(brain=brains_list[0], system_prompt=args.system)
        return

    question = " ".join(args.question) if args.question else None
    if not question:
        if not sys.stdin.isatty():
            question = sys.stdin.read().strip()
        else:
            print(f"{C.ERR}No question provided. Use --interactive for conversation mode.{C.RESET}")
            sys.exit(1)

    if args.discuss:
        bridge.discuss(question, brains=brains_list, context=context)
    elif args.chain:
        bridge.chain(question, steps=[(b, "Continue from the previous analysis:") for b in brains_list])
    else:
        bridge.ask(
            brains_list[0], question,
            context=context,
            system_prompt=args.system,
            stream=not args.no_stream,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )


if __name__ == "__main__":
    main()

# ── Module-level convenience: ask Claude ──────────────────────────────────
def ask_claude(question, **kwargs):
    """Quick: ask Claude Opus via CLI subprocess."""
    return get_bridge().ask(brain="claude", question=question, **kwargs)
