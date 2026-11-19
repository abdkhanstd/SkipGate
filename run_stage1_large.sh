#!/usr/bin/env bash
# Stage 1a: label the large model on the same questions (no hidden states needed).
# Usage: bash run_stage1_large.sh <gpu_index>
# NOTE: this stage was run with an answer budget of 320 tokens (now passed explicitly).
# run_stage4.sh later re-decoded every unfinished answer with 640 tokens.
set -euo pipefail
cd "$(dirname "$0")/skipgate"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate skipgate
export CUDA_VISIBLE_DEVICES="${1:-1}"
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
MODEL=Qwen/Qwen2.5-7B-Instruct
TAG=qwen7b
RES=../results/stage1
mkdir -p "$RES"
for TASK in triviaqa gsm8k; do
  if [ ! -f "$RES/${TAG}_${TASK}.npz" ]; then
    python extract.py --model "$MODEL" --task "$TASK" --n 4000 --max_new_tokens 320 --bs 24 --no_hidden \
      --out "$RES/${TAG}_${TASK}.npz"
  fi
done
echo STAGE1_LARGE_DONE
