CHECKLIST_BANNER = """
---
### 🛠️ STRATEGIC CHECKLIST
- [ ] **Infrastructure:** Scaling/Latency verified?
- [ ] **Security:** Secrets/Auth patterns checked?
- [ ] **Observability:** Logs/Metrics defined?
---
"""
import os
import sys
import time
import json
import threading
import subprocess
import re
import requests
from ddgs import DDGS
from openai import OpenAI

# --- DYNAMIC CONFIGURATION ---
SERVER_IP   = "127.0.0.1" # Change to 10.255.255.11 if hitting the Ray cluster head node directly
SERVER_PORT = "8000"
BASE_URL    = f"http://{SERVER_IP}:{SERVER_PORT}/v1"

try:
    _r = requests.get(f"{BASE_URL}/models", timeout=2)
    MODEL_NAME = _r.json()["data"][0]["id"]
except:
    MODEL_NAME = "qwen3-coder-30b"

# Upgraded context limits for massive models
MAX_CONTEXT = 131072
CONTEXT_RESET_THRESHOLD = 125000

# Added 300s timeout to prevent freezing on massive context generations
client = OpenAI(base_url=BASE_URL, api_key="sk-4090", timeout=300.0)

# =====================================================
# CONTEXT MANAGEMENT (SLIDING WINDOW)
# =====================================================
def slide_window(history, max_tokens=CONTEXT_RESET_THRESHOLD):
    """Smartly drops the oldest messages while preserving Role structures."""
    while sum(len(str(m)) // 3 for m in history) > max_tokens and len(history) > 5:
        # Never touch the system prompt (index 0)
        # Safely remove messages from index 1. 
        # If we remove an assistant tool_call, we MUST remove the tool response that follows it.
        dropped = history.pop(1)
        
        # If we just orphaned a tool response, kill it too.
        while len(history) > 1 and history[1]["role"] == "tool":
            history.pop(1)
            
        # If the new oldest message is an orphaned assistant message without tool_calls, that's fine.
    return history

# =====================================================
# TOOLS (UNSHEATHED TO 250,000 CHARACTERS)
# =====================================================
def execute_bash(command: str) -> str:
    try:
        res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60, env=os.environ)
        out = res.stdout if res.stdout else res.stderr
        return out.strip()[:250000] if out else "Done."
    except Exception as e:
        return f"Error: {e}"

def search_web(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if not results:
                return "No results found."
            return "\n\n".join(
                f"Result {i+1}:\nTitle: {r.get('title', '')}\nSnippet: {r.get('body', '')}\nURL: {r.get('href', '')}"
                for i, r in enumerate(results)
            )
    except Exception as e:
        return f"Search failed: {e}"

def search_image(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=3))
            if results: return "\n".join([f"![Image]({r['image']})" for r in results])
    except: pass
    try:
        search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={query}&limit=1&format=json"
        search_res = requests.get(search_url).json()
        if len(search_res[1]) > 0:
            title = search_res[1][0]
            img_url = f"https://en.wikipedia.org/w/api.php?action=query&titles={title}&prop=pageimages&format=json&pithumbsize=800"
            img_res = requests.get(img_url).json()
            pages = img_res.get("query", {}).get("pages", {})
            for pid, pdata in pages.items():
                if "thumbnail" in pdata: return f"![Image]({pdata['thumbnail']['source']})"
        return f"Could not find exact images for '{query}'."
    except Exception as e:
        return f"Image search failed entirely: {str(e)}"

def read_file(path: str, offset: int = 0, limit: int = 250000) -> str:
    try:
        with open(os.path.expanduser(path), 'r') as f:
            f.seek(offset)
            return f.read(limit)
    except Exception as e: return f"Error: {e}"

def write_file(path: str, content: str) -> str:
    try:
        full_path = os.path.expanduser(path)
        with open(full_path, 'w') as f: f.write(content)
        return f"Successfully wrote {len(content)} chars to {full_path}"
    except Exception as e: return f"Error: {e}"

def restart_agent() -> str:
    print("\n⚠️  RESTARTING AGENT PROCESS...\n")
    try:
        os.utime(__file__, None)
        if "--web" not in sys.argv: os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e: return f"Restart failed: {e}"
    return "Restart triggered via file touch. Browser will reload automatically."

TOOL_DISPATCH = {
    "execute_bash": execute_bash, "search_web": search_web, "search_image": search_image,
    "read_file": read_file, "write_file": write_file, "restart_agent": restart_agent
}

TOOLS = [
    {"type": "function", "function": {"name": "execute_bash", "description": "Execute bash command. Returns stdout.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "search_web", "description": "Search internet text.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "search_image", "description": "Search for images. Returns working Markdown image URLs.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read local file. Supports offset and limit for pagination.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write text to file (overwrite). Use for code editing.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "restart_agent", "description": "Restart the agent process. Call this after editing your own source code.", "parameters": {"type": "object", "properties": {}, "required": []}}}
]

SYSTEM_PROMPT = f"""You are Z-Alpha, an autonomous local agent running on an RTX 4090.
Context Window: {MAX_CONTEXT} tokens.

CRITICAL SYSTEM KNOWLEDGE:
1. Model Storage: `/slowrepo/models` contains the AI models you currently run on.
2. Extra Storage: `/repo` contains other models and large files.
3. Directory Sizing: Always use `du -sh <target>`, NEVER use `ls -la`.
4. Images: MUST use the `search_image` tool to get real URLs. Output exactly `![alt](URL)`.

CRITICAL JSON RULE: NEVER use `<arguments":`. You MUST use strict standard JSON `"arguments":` inside tool calls.
DO NOT chat or politely describe what you are going to do. DO NOT output raw XML directly to the user.
If you need information, IMMEDIATELY execute a tool call. Trust your tools.

AVAILABLE TOOLS:
{json.dumps(TOOLS, indent=2)}

TO CALL A TOOL, YOU MUST OUTPUT EXACTLY THIS FORMAT:
<tool_call>
{{"name": "tool_name", "arguments": {{"arg1": "val"}}}}
</tool_call>
"""

def extract_qwen_tools(text):
    """Bulletproof regex to catch Qwen's XML, auto-fixing syntax hallucinations and multi-parameters."""
    text = text.replace('<arguments":', '"arguments":')
    tcs = []
    
    for m in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL):
        try:
            tc = json.loads(m.group(1).strip())
            tcs.append({"id": f"xml_{time.time()}", "name": tc.get("name"), "arguments": json.dumps(tc.get("arguments", {}))})
        except: pass
        
    for m in re.finditer(r'<function=([^>]+)>(.*?)</function>', text, re.DOTALL):
        fn_name = m.group(1).strip()
        params_str = m.group(2)
        args = {}
        for pm in re.finditer(r'<parameter=([^>]+)>(.*?)</parameter>', params_str, re.DOTALL):
            args[pm.group(1).strip()] = pm.group(2).strip()
            
        for k in ["offset", "limit"]:
            if k in args:
                try: args[k] = int(args[k])
                except: pass
                
        tcs.append({"id": f"xml_{time.time()}", "name": fn_name, "arguments": json.dumps(args)})
        
    return tcs

def get_multiline_input():
    print("\n" + "="*50)
    print("Architect Input (Paste multiline blocks freely).")
    print("Submit by pressing Ctrl+D, or type 'EOF' on a new line.")
    print("="*50)
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "EOF": break
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines).strip()

# =====================================================
# CLI MODE
# =====================================================
def run_cli():
    session_out_tokens = 0
    current_gpu_mem_gb = 0.0
    current_ctx_tokens = 0

    def _gpu_monitor():
        nonlocal current_gpu_mem_gb
        while True:
            try:
                out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip()
                current_gpu_mem_gb = sum(int(x) for x in out.split('\n')) / 1024.0
            except: pass
            time.sleep(2)

    threading.Thread(target=_gpu_monitor, daemon=True).start()
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    print(f"\n⚡ Z-Alpha CLI ({MAX_CONTEXT} tok) — Model: {MODEL_NAME}")

    while True:
        try:
            user_input = get_multiline_input()
            if user_input.lower() in ("exit", "quit"): break
            if not user_input: continue
            history.append({"role": "user", "content": user_input})

            # SLIDING WINDOW ENFORCEMENT
            history = slide_window(history)

            while True:
                start = time.perf_counter()
                first_token = None
                full_content = ""
                completion_toks = 0
                tool_calls_acc = {}

                print(f"\n[Z-Alpha]: ", end="", flush=True)
                safe_max_tokens = max(256, min(8192, MAX_CONTEXT - current_ctx_tokens - 256))
                stream = client.chat.completions.create(model=MODEL_NAME, messages=history, stream=True, max_tokens=safe_max_tokens)

                for chunk in stream:
                    if not chunk.choices: continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        if not first_token: first_token = time.perf_counter()
                        print(delta.content, end="", flush=True)
                        full_content += delta.content

                if full_content: print()
                
                full_content = full_content.replace('<arguments":', '"arguments":')
                tool_calls = extract_qwen_tools(full_content)
                
                if tool_calls:
                    full_content = re.sub(r'<tool_call>.*?</tool_call>', '', full_content, flags=re.DOTALL)
                    full_content = re.sub(r'<function=.*?</function>', '', full_content, flags=re.DOTALL).replace('</tool_call>', '').strip()

                ast_msg = {"role": "assistant", "content": full_content}
                
                if tool_calls:
                    safe_tcs = []
                    for t in tool_calls:
                        try: json.loads(t["arguments"]); s_args = t["arguments"]
                        except: s_args = "{}"
                        safe_tcs.append({"id": t["id"], "type": "function", "function": {"name": t["name"], "arguments": s_args}})
                    ast_msg["tool_calls"] = safe_tcs

                history.append(ast_msg)
                if not tool_calls: break

                for tc in tool_calls:
                    fn_name = tc["name"]
                    try: args = json.loads(tc["arguments"])
                    except: args = {}
                    
                    print(f"\n[🔧 Tool Executing] {fn_name}")
                    if args:
                        for k, v in args.items():
                            val_str = str(v)
                            print(f"  -> {k}: {val_str[:500] + '... [TRUNCATED]' if len(val_str) > 500 else val_str}")
                    
                    try:
                        res = TOOL_DISPATCH.get(fn_name, lambda **k: f"Unknown: {fn_name}")(**args)
                    except TypeError as e:
                        res = f"Tool Execution Failed: Missing/invalid arguments. Details: {str(e)}"
                    
                    out_str = str(res)
                    print(f"[📤 Output ({len(out_str)} chars)]:\n{out_str[:1500] + '... [TRUNCATED]' if len(out_str) > 1500 else out_str}\n" + "-"*40)
                    history.append({"role": "tool", "tool_call_id": tc["id"], "name": fn_name, "content": res})

        except KeyboardInterrupt: break

# =====================================================
# STREAMLIT MODE
# =====================================================
def run_web():
    import streamlit as st
    import pandas as pd
    import numpy as np

    st.set_page_config(page_title="Z-Alpha MLPerf", page_icon="⚡", layout="wide")
    st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .custom-metric-container { display: flex; flex-direction: row; justify-content: space-between; gap: 10px; width: 100%; }
    .custom-metric-box { background-color: #1E1E1E; border: 1px solid #333; border-radius: 5px; padding: 15px 10px; flex: 1; text-align: center; min-width: 0; }
    .metric-label { font-size: 0.85rem; color: #888; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #FAFAFA; white-space: normal; word-wrap: break-word; line-height: 1.2; }
    .stMetric { background-color: #1E1E1E; padding: 10px; border-radius: 5px; border: 1px solid #333; }
    .stChatMessage img { max-width: 100% !important; max-height: 400px !important; object-fit: contain !important; border-radius: 8px; }
    [data-testid="column"]:nth-child(2) { position: sticky !important; top: 2rem; height: 90vh; overflow-y: auto; }
</style>
""", unsafe_allow_html=True)

    submitted = False
    user_input = ""

    if "history" not in st.session_state: st.session_state.history = [{"role": "system", "content": SYSTEM_PROMPT}]
    if "display" not in st.session_state: st.session_state.display = []
    if "ctx" not in st.session_state: st.session_state.ctx = 0
    if "total_tokens" not in st.session_state: st.session_state.total_tokens = 0
    if "is_running" not in st.session_state: st.session_state.is_running = False
    if "last_run_stats" not in st.session_state: st.session_state.last_run_stats = None

    st.sidebar.title("⚡ Z-Alpha Web")
    st.sidebar.success(f"🟢 {MODEL_NAME}")

    if st.sidebar.button("🗑️ Clear Session", use_container_width=True):
        st.session_state.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.session_state.display = []
        st.session_state.is_running = False
        st.rerun()

    if st.sidebar.button("🔄 Restart Agent", use_container_width=True): restart_agent()

    col_chat, col_telemetry = st.columns([2, 1], gap="medium")

    with col_telemetry:
        st.subheader("📡 Telemetry Deck")
        if st.session_state.last_run_stats and not st.session_state.is_running:
            s = st.session_state.last_run_stats
            st.markdown(f'''<div class="custom-metric-container">
                <div class="custom-metric-box"><div class="metric-label">Words</div><div class="metric-value">{s['input_words']:,}</div></div>
                <div class="custom-metric-box"><div class="metric-label">Tokens</div><div class="metric-value">~{s['input_tokens']:,}</div></div>
            </div>''', unsafe_allow_html=True)
            st.line_chart(pd.DataFrame(s['token_data']), x="Token", y="Inst TPS", height=200)

    with col_chat:
        st.subheader("💬 Inference Stream")
        chat_box = st.container(height=650, border=False)
        with chat_box:
            for entry in st.session_state.display:
                if entry["role"] == "system_event":
                    st.info(entry["content"])
                    continue
                with st.chat_message(entry["role"]):
                    st.markdown(entry["content"])
                    tools = entry.get("tools", [])
                    if tools:
                        with st.expander(f"🛠️ {len(tools)} Tool Executions"):
                            for t in tools:
                                st.markdown(f"**`{t['name']}`** `({t['arg'][:100]})`")
                                st.code(t["output"], language="text")

        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area("Architect:", height=150, placeholder="Paste logs... (Enter to send, Shift+Enter for newline)", label_visibility="collapsed")
            submitted = st.form_submit_button("⚡ Send", use_container_width=True)
            
            import streamlit.components.v1 as components
            components.html('''
                <script>
                const doc = window.parent.document;
                doc.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        const textareas = doc.querySelectorAll('textarea');
                        if (textareas.length > 0 && doc.activeElement === textareas[0]) {
                            e.preventDefault();
                            const btns = doc.querySelectorAll('button[kind="formSubmit"]');
                            if (btns.length > 0) btns[0].click();
                        }
                    }
                });
                </script>
            ''', height=0, width=0)

    if submitted and user_input.strip() and not st.session_state.is_running:
        prompt = user_input.strip()
        st.session_state.is_running = True
        st.session_state.history.append({"role": "user", "content": prompt})
        st.session_state.display.append({"role": "user", "content": prompt, "tools": []})

        # SLIDING WINDOW ENFORCEMENT
        st.session_state.history = slide_window(st.session_state.history)

        with col_telemetry:
            m1, m2 = st.columns(2)
            ph_ttft = m1.empty()
            ph_tps  = m2.empty()
            chart_ph = st.empty()

        all_tools_this_turn = []
        while True:
            start = time.perf_counter()
            first_tok = None; full_content = ""; completion_toks = 0
            tool_calls_acc = {}; token_data = []; toks = 0; last_tok = start

            safe_max_tokens = max(256, min(8192, MAX_CONTEXT - st.session_state.ctx - 256))
            stream = client.chat.completions.create(model=MODEL_NAME, messages=st.session_state.history, stream=True, max_tokens=safe_max_tokens)

            with chat_box:
                st_chat = st.chat_message("assistant")
                resp_box = st_chat.empty()

            for chunk in stream:
                if not chunk.choices: continue
                delta = chunk.choices[0].delta
                if delta.content:
                    if not first_tok: first_tok = time.perf_counter()
                    full_content += delta.content; toks += 1
                    resp_box.markdown(full_content + "▌")
                    lat = time.perf_counter() - last_tok
                    last_tok = time.perf_counter()
                    if toks > 1:
                        inst_tps = min(1/lat if lat>0 else 0, 500)
                        token_data.append({"Token": toks, "Latency": lat, "Inst TPS": inst_tps})
                        if toks % 10 == 0: chart_ph.line_chart(pd.DataFrame(token_data), x="Token", y="Inst TPS", height=200)

            full_content = full_content.replace('<arguments":', '"arguments":')
            t_list = extract_qwen_tools(full_content)
            
            if t_list:
                full_content = re.sub(r'<tool_call>.*?</tool_call>', '', full_content, flags=re.DOTALL)
                full_content = re.sub(r'<function=.*?</function>', '', full_content, flags=re.DOTALL).replace('</tool_call>', '').strip()

            if full_content: resp_box.markdown(full_content)
            elif t_list: resp_box.markdown("⚙️ Processing Tools...")

            ast_msg = {"role": "assistant", "content": full_content}
            if t_list:
                safe_tcs = []
                for t in t_list:
                    try: json.loads(t["arguments"]); s_args = t["arguments"]
                    except: s_args = "{}"
                    safe_tcs.append({"id": t["id"], "type": "function", "function": {"name": t["name"], "arguments": s_args}})
                ast_msg["tool_calls"] = safe_tcs
            
            final_content = CHECKLIST_BANNER + "\n\n" + full_content
            st.session_state.history.append(ast_msg)

            if not t_list:
                st.session_state.display.append({"role": "assistant", "content": final_content, "tools": all_tools_this_turn})
                st.session_state.is_running = False
                st.rerun()

            for tc in t_list:
                fn = tc["name"]
                try: args = json.loads(tc["arguments"])
                except: args = {}
                arg_preview = list(args.values())[0] if args else ""
                with chat_box:
                    with st.spinner(f"Running `{fn}({arg_preview[:60]}...)`..."):
                        try: res = TOOL_DISPATCH.get(fn, lambda **k: "Error")(**args)
                        except TypeError as e: res = f"Execution Failed: {e}"
                all_tools_this_turn.append({"name": fn, "arg": arg_preview, "output": res})
                st.session_state.history.append({"role": "tool", "tool_call_id": tc["id"], "name": fn, "content": res})

if __name__ == "__main__":
    if "--web" in sys.argv: run_web()
    elif "--launch-web" in sys.argv: subprocess.run(["streamlit", "run", __file__, "--server.port", "8502", "--", "--web"])
    else: run_cli()
