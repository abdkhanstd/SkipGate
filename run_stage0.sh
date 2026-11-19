#!/usr/bin/env bash
# Stage 0: one small model, two tasks, probe AUROC only.
# Usage: bash run_stage0.sh <gpu_index>
# NOTE: this stage was run with an answer budget of 320 tokens (now passed explicitly).
# run_stage4.sh later re-decoded every unfinished answer with 640 tokens.
set -euo pipefail
cd "$(dirname "$0")/skipgate"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate skipgate
export CUDA_VISIBLE_DEVICES="${1:-1}"
export TOKENIZERS_PARALLELISM=false
# huggingface.co is not reachable from the server, the mirror is
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
MODEL=Qwen/Qwen2.5-1.5B-Instruct
TAG=qwen1.5b
RES=../results/stage0
mkdir -p "$RES"
for TASK in triviaqa gsm8k; do
  if [ ! -f "$RES/${TAG}_${TASK}.npz" ]; then
    python extract.py --model "$MODEL" --task "$TASK" --n 4000 --max_new_tokens 320 --bs 16 --out "$RES/${TAG}_${TASK}.npz"
  fi
  python probe_stage0.py --npz "$RES/${TAG}_${TASK}.npz" --out "$RES/${TAG}_${TASK}.probe.json"
done
echo STAGE0_DONE
