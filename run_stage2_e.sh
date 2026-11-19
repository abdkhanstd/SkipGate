#!/usr/bin/env bash
# Stage 2e: reruns of two jobs that ran out of GPU memory in run_stage2_c.sh.
#   1. sampled answers of the 1.5B model on MMLU (GPU 0, batch size 8, OOM-safe code)
#   2. batched throughput of the 14B model (both GPUs, after run_stage2_d.sh has finished)
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
M14="${M14:-$HOME/skipgate/models/Qwen2.5-14B-Instruct}"
[ -f "$R/stage2/samples_qwen1.5b_mmlu.npz" ] || CUDA_VISIBLE_DEVICES=0 python sample.py \
  --model Qwen/Qwen2.5-1.5B-Instruct --task mmlu --n 4000 --max_new_tokens 320 --samples 10 --bs 8 \
  --out "$R/stage2/samples_qwen1.5b_mmlu.npz" > "$L/sample_mmlu_retry.log" 2>&1
echo SAMPLE_MMLU_DONE
while ! grep -q STAGE2D_DONE "$L/stage2d.log" 2>/dev/null; do sleep 30; done
for BS in 64 32; do
  CUDA_VISIBLE_DEVICES=0,1 python latency_v1.py --model "$M14" --multi_gpu \
    --n 256 --n_single 4 --bs $BS --out $R/stage2/throughput_qwen14B_bs$BS.json \
    > "$L/throughput14b_bs$BS.log" 2>&1 && break
done
echo STAGE2E_DONE
