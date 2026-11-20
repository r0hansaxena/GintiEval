#!/bin/bash
# Sequentially download + evaluate the local open-weight models.
cd "$(dirname "$0")" && . ../.venv/bin/activate
while read -r name repo; do
  [ -z "$name" ] && continue
  hf download "$repo" >/dev/null 2>&1 || { echo "DOWNLOAD FAILED $repo"; continue; }
  echo "=== $name $(date +%H:%M:%S)"
  python run.py --backend mlx --model "$repo" --name "$name" 2>&1 | grep -E "items to run|Error|error|Traceback" | tail -4
done <<'LIST'
llama-3.2-3b mlx-community/Llama-3.2-3B-Instruct-4bit
gemma-3-4b mlx-community/gemma-3-4b-it-qat-4bit
qwen2.5-7b mlx-community/Qwen2.5-7B-Instruct-4bit
llama-3.1-8b mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
aya-expanse-8b mlx-community/aya-expanse-8b-4bit
gemma-3-12b mlx-community/gemma-3-12b-it-qat-4bit
LIST
echo "PIPELINE DONE $(date +%H:%M:%S)"
