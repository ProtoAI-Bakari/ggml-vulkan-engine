#!/bin/bash
TASK=$1; AGENT=${2:-unknown}; FILE=~/AGENT/TASK_QUEUE_v4.md
sed -i "/$TASK/s/\[IN_PROGRESS by [^]]*\]/[DONE by $AGENT]/" "$FILE"
echo "COMPLETED $TASK"
