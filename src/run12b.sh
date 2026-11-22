#!/bin/bash
cd "$(dirname "$0")" && . ../.venv/bin/activate
while ! grep -q ALLDONE ../results/pipeline2.log; do sleep 10; done
echo "=== gemma-3-12b $(date +%H:%M:%S)" > ../results/pipeline4.log
for bs in 32 8; do
  python run.py --backend mlx --batch-size $bs --model mlx-community/gemma-3-12b-it-qat-4bit --name gemma-3-12b 2>&1 | grep -E "items to run|Error|error|Traceback" >> ../results/pipeline4.log
done
echo "FINISHED $(date +%H:%M:%S)" >> ../results/pipeline4.log
