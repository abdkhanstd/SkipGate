#!/usr/bin/env bash
# Stage 2a: latency timing (GPU A) and sampled answers for the sampling baselines (GPU B).
# Usage: bash run_stage2_a.sh latency <gpu>   |   bash run_stage2_a.sh sample <gpu>
# NOTE: this stage was run with an answer budget of 320 tokens (now passed explicitly).
# run_stage4.sh later re-decoded every unfinished answer with 640 tokens.
set -euo pipefail
cd "$(dirname "$0")/skipgate"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate skipgate
export CUDA_VISIBLE_DEVICES="${2:-1}"
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
RES=../results/stage2
mkdir -p "$RES"
if [ "$1" = latency ]; then
  python latency_v1.py --model Qwen/Qwen2.5-1.5B-Instruct --out "$RES/latency_qwen1.5b.json"
  python latency_v1.py --model Qwen/Qwen2.5-7B-Instruct --out "$RES/latency_qwen7b.json"
  echo LATENCY_DONE
else
  for TASK in triviaqa gsm8k; do
    if [ ! -f "$RES/samples_qwen1.5b_${TASK}.npz" ]; then
      python sample.py --model Qwen/Qwen2.5-1.5B-Instruct --task "$TASK" --n 4000 --max_new_tokens 320 \
        --samples 10 --bs 16 --out "$RES/samples_qwen1.5b_${TASK}.npz"
    fi
  done
  echo SAMPLE_DONE
fi
