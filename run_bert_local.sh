#!/usr/bin/env bash
# Fine-tuned BERT router (RouteLLM's router type) for every small model and question set,
# on the local workstation (two RTX 2080 Ti). Labels of the 7B model come from the stored
# feature files (their answer texts were regenerated and differ on 0.2% of the questions).
set -uo pipefail
cd "$(dirname "$0")/skipgate"
ROOT="$(cd ../.. && pwd)"
PY="$ROOT/build/venv/Scripts/python.exe"
B="$ROOT/build/models/bert-base-uncased"
L="$ROOT/build/local_logs"
R=../results
export PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false
run() { CUDA_VISIBLE_DEVICES=$1 "$PY" bert_router.py --answers "$2" --model "$B" "${@:3}" > "$L/bert_$(basename "$2" .answers.jsonl)_$(basename "$(dirname "$2")").log" 2>&1; }
(
  run 0 $R/stage4/qwen1.5b_gsm8k.answers.jsonl
  run 0 $R/stage4/qwen1.5b_mmlu.answers.jsonl
  run 0 $R/stage0/qwen1.5b_triviaqa.answers.jsonl
  run 0 $R/fresh/qwen1.5b_gsm8k.answers.jsonl
  run 0 $R/fresh/qwen1.5b_mmlu.answers.jsonl
) &
(
  run 1 $R/stage4/qwen7b_gsm8k.answers.jsonl --labels $R/stage4/qwen7b_gsm8k.labels.npy
  run 1 $R/stage4/qwen7b_mmlu.answers.jsonl --labels $R/stage4/qwen7b_mmlu.labels.npy
  run 1 $R/stage6/llama3b_gsm8k.answers.jsonl
  run 1 $R/stage6/llama3b_mmlu.answers.jsonl
) &
wait
echo BERT_DONE
