#!/usr/bin/env bash
# Stage 6, second part, on the server. The local workstation (run_stage6_local.sh) turned out
# to be several times slower than planned, because Windows lets a full GPU spill into system
# memory. The runs that had not finished there were stopped and are done here:
#   fresh questions: Qwen2.5-7B on MMLU
#   second family:   Llama-3.2-3B on MMLU (with hidden states), Llama-3.1-8B on GSM8K and MMLU
# Finished locally and copied here: Qwen2.5-1.5B on fresh GSM8K and MMLU, Qwen2.5-7B on fresh
# GSM8K, Llama-3.2-3B on GSM8K. Then the 14B timing is repeated with a cap on the weights per
# card (the first attempt ran out of memory next to another process).
set -uo pipefail
cd "$(dirname "$0")/skipgate"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate skipgate
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export MMLU_PARQUET="${MMLU_PARQUET:-$HOME/skipgate/data/mmlu/test.parquet}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=../results
L=../../logs
MD="$HOME/skipgate/models"
mkdir -p $R/fresh $R/stage6
run() {  # gpu model out task batch [extra flags]
  local gpu=$1 model=$2 out=$3 task=$4 bs=$5; shift 5
  [ -f "$out" ] || CUDA_VISIBLE_DEVICES=$gpu python extract.py --model "$model" --task "$task" \
    --n 4000 --bs "$bs" --out "$out" "$@" \
    > "$L/$(basename "$(dirname "$out")")_$(basename "$out" .npz).log" 2>&1
}
(
  run 0 $MD/Llama-3.2-3B-Instruct $R/stage6/llama3b_mmlu.npz mmlu 12
  run 0 Qwen/Qwen2.5-7B-Instruct $R/fresh/qwen7b_mmlu.npz mmlu 12 --offset 4000 --no_hidden
) &
(
  run 1 $MD/Llama-3.1-8B-Instruct $R/stage6/llama8b_gsm8k.npz gsm8k 24 --no_hidden
  run 1 $MD/Llama-3.1-8B-Instruct $R/stage6/llama8b_mmlu.npz mmlu 24 --no_hidden
) &
wait
echo EXTRACT_DONE
CUDA_VISIBLE_DEVICES=0,1 python latency.py --model $MD/Qwen2.5-14B-Instruct --tag qwen14b --multi_gpu \
  --max_memory 12 17 --bs 16 64 128 --out_dir $R/stage5 > $L/timing_qwen14b.log 2>&1
echo STAGE6_SERVER_DONE
