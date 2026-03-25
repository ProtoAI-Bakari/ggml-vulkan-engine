"""
Claude <-> Agent Bridge
Shared file-based communication channel.

AGENT writes to:  ~/AGENT/bridge/agent_to_claude.jsonl  (append-only)
CLAUDE writes to: ~/AGENT/bridge/claude_to_agent.jsonl  (append-only)

Each line is a JSON message: {"ts": "...", "type": "error|question|status|answer|directive", "content": "..."}

Agent checks for new claude messages every turn.
Claude checks agent messages via: tail ~/AGENT/bridge/agent_to_claude.jsonl
"""
import os, json, time
from datetime import datetime

BRIDGE_DIR = os.path.expanduser("~/AGENT/bridge")
AGENT_TO_CLAUDE = os.path.join(BRIDGE_DIR, "agent_to_claude.jsonl")
CLAUDE_TO_AGENT = os.path.join(BRIDGE_DIR, "claude_to_agent.jsonl")

os.makedirs(BRIDGE_DIR, exist_ok=True)

def send_to_claude(msg_type, content):
    """Agent calls this to send a message to Claude."""
    msg = {"ts": datetime.now().isoformat(), "type": msg_type, "content": content}
    with open(AGENT_TO_CLAUDE, "a") as f:
        f.write(json.dumps(msg) + "\n")

def read_from_claude(last_pos=0):
    """Agent calls this to read new messages from Claude."""
    try:
        if not os.path.exists(CLAUDE_TO_AGENT):
            return [], last_pos
        with open(CLAUDE_TO_AGENT, "r") as f:
            f.seek(last_pos)
            lines = f.readlines()
            new_pos = f.tell()
        msgs = []
        for line in lines:
            line = line.strip()
            if line:
                try:
                    msgs.append(json.loads(line))
                except:
                    pass
        return msgs, new_pos
    except:
        return [], last_pos

def send_to_agent(msg_type, content):
    """Claude calls this (via bash) to send a message to the agent."""
    msg = {"ts": datetime.now().isoformat(), "type": msg_type, "content": content}
    with open(CLAUDE_TO_AGENT, "a") as f:
        f.write(json.dumps(msg) + "\n")
