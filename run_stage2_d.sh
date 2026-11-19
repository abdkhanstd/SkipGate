#!/usr/bin/env bash
# Stage 2d: rerun of the 7B model on MMLU with hidden states. The first attempt in
# run_stage2_c.sh ran out of GPU memory at batch size 12 on the longest prompts.
# Waits for run_stage2_c.sh to finish, then uses one GPU with batch size 6.
# NOTE: this stage was run with an answer budget of 320 tokens (now passed explicitly).
# run_stage4.sh later re-decoded every unfinished answer with 640 tokens.
set -uo pipefail
cd "$(dirname "$0")/skipgate"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate skipgate
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export MMLU_PARQUET="${MMLU_PARQUET:-$HOME/skipgate/data/mmlu/test.parquet}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=../results
L=../../logs
while ! grep -q STAGE2C_DONE "$L/stage2c.log" 2>/dev/null; do sleep 30; done
if [ ! -f "$R/stage2/qwen7b_mmlu.npz" ]; then
  CUDA_VISIBLE_DEVICES="${1:-1}" python extract.py --model Qwen/Qwen2.5-7B-Instruct --task mmlu \
    --n 4000 --max_new_tokens 320 --bs 6 --out "$R/stage2/qwen7b_mmlu.npz" > "$L/ext_qwen7b_mmlu_retry.log" 2>&1
fi
echo STAGE2D_DONE
