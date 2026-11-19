#!/usr/bin/env bash
# Stage 2c: third dataset (MMLU with reasoning), 7B as a second small model, and the 14B
# large model. Two single-GPU queues run in parallel, then the 14B model uses both cards.
# NOTE: this stage was run with an answer budget of 320 tokens (now passed explicitly).
# run_stage4.sh later re-decoded every unfinished answer with 640 tokens.
set -uo pipefail
cd "$(dirname "$0")/skipgate"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate skipgate
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# models and the other two datasets are already cached, MMLU comes from a local parquet file
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export MMLU_PARQUET="${MMLU_PARQUET:-$HOME/skipgate/data/mmlu/test.parquet}"
R=../results
L=../../logs
mkdir -p $R/stage2

ext() {  # ext <gpu> <model> <tag> <task> <bs> [extra flags]
  local gpu=$1 model=$2 tag=$3 task=$4 bs=$5; shift 5
  [ -f "$R/stage2/${tag}_${task}.npz" ] && return 0
  CUDA_VISIBLE_DEVICES=$gpu python extract.py --model "$model" --task "$task" --n 4000 --max_new_tokens 320 \
    --bs "$bs" --out "$R/stage2/${tag}_${task}.npz" "$@" > "$L/ext_${tag}_${task}.log" 2>&1
}

queue0() {
  ext 0 Qwen/Qwen2.5-1.5B-Instruct qwen1.5b mmlu 16
  [ -f "$R/stage2/samples_qwen1.5b_mmlu.npz" ] || CUDA_VISIBLE_DEVICES=0 python sample.py \
    --model Qwen/Qwen2.5-1.5B-Instruct --task mmlu --n 4000 --max_new_tokens 320 --samples 10 --bs 16 \
    --out "$R/stage2/samples_qwen1.5b_mmlu.npz" > "$L/sample_mmlu.log" 2>&1
}
queue1() {
  ext 1 Qwen/Qwen2.5-7B-Instruct qwen7b mmlu 12
  ext 1 Qwen/Qwen2.5-7B-Instruct qwen7b gsm8k 12
}
queue0 & queue1 & wait
echo SINGLE_GPU_DONE

# wait for the 14B download (fetched with ModelScope, the HF mirror failed on the weights)
M14="${M14:-$HOME/skipgate/models/Qwen2.5-14B-Instruct}"
while pgrep -u "$USER" -f "[m]odelscope download" > /dev/null; do sleep 30; done
for TASK in gsm8k mmlu triviaqa; do
  ext 0,1 "$M14" qwen14b "$TASK" 16 --no_hidden --multi_gpu
done
CUDA_VISIBLE_DEVICES=0,1 python latency_v1.py --model "$M14" --multi_gpu \
  --n 64 --out $R/stage2/latency_qwen14b.json > "$L/latency14b.log" 2>&1
CUDA_VISIBLE_DEVICES=0,1 python latency_v1.py --model "$M14" --multi_gpu \
  --n 256 --n_single 4 --bs 128 --out $R/stage2/throughput_qwen14B_bs128.json >> "$L/latency14b.log" 2>&1
echo STAGE2C_DONE
