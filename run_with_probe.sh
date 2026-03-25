#!/bin/bash
# Start server in background and run probe
~/vrun_0.17.1_v11_vulkan_surgical_final_v2.sh 2>&1 &
SERVER_PID=$!
sleep 30
python3 ~/AGENT/inference_probe.py
kill $SERVER_PID 2>/dev/null