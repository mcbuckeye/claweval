#!/bin/bash
# Run ClawEval one model at a time, saving results incrementally
cd /Users/kayleighbot/Projects/claweval
source .venv/bin/activate

MODELS=(
  "qwen/qwen3-coder-next"
  "qwen3.5-27b"
  "qwen/qwen3.5-35b-a3b"
  "nvidia/nemotron-3-nano"
  "nvidia/nemotron-3-super"
  "qwen/qwen3.5-9b"
)

for model in "${MODELS[@]}"; do
  echo "=== Running $model ===" | tee -a /tmp/claweval-full.log
  claweval run --model "$model" 2>&1 | tee -a /tmp/claweval-full.log
  echo "=== Done with $model ===" | tee -a /tmp/claweval-full.log
  echo "" | tee -a /tmp/claweval-full.log
done

echo "ALL MODELS COMPLETE" | tee -a /tmp/claweval-full.log
