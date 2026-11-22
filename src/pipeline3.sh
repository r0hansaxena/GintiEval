#!/bin/bash
cd "$(dirname "$0")" && . ../.venv/bin/activate
while read -r name repo; do
  echo "=== $name $(date +%H:%M:%S)"
  python run.py --backend mlx --batch-size 48 --model "$repo" --name "$name" 2>&1 | grep -E "items to run|Error|error|Traceback" | tail -4
done <<'LIST'
gemma-3-4b mlx-community/gemma-3-4b-it-qat-4bit
aya-expanse-8b mlx-community/aya-expanse-8b-4bit
qwen2.5-7b mlx-community/Qwen2.5-7B-Instruct-4bit
LIST
echo "ALLDONE $(date +%H:%M:%S)"
