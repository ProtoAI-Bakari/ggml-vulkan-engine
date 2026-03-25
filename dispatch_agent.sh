#!/bin/bash
# dispatch_agent.sh — Send a task to a remote agent brain
TARGET=${1:-brain}
TASK="$2"
[ -z "$TASK" ] && echo "Usage: $0 [coder|brain] 'task'" && exit 1

case $TARGET in
  coder) URL="http://10.255.255.4:8000/v1/chat/completions"; MODEL="mlx-community/Qwen3-Coder-Next-8bit";;
  brain) URL="http://10.255.255.11:8000/v1/chat/completions"; MODEL="/vmrepo/models/Qwen3.5-122B-A10B-FP8";;
  *) echo "Unknown: $TARGET"; exit 1;;
esac

curl -s -X POST "$URL" -H "Content-Type: application/json"   -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$TASK\"}],\"max_tokens":16384,\"temperature\":0.3}"   | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" 2>/dev/null
