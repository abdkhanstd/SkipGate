#!/usr/bin/env bash
# Stage 4: remove the truncation artefact. In stages 0 to 2 the answer budget was 320
# tokens, and 7 to 19 percent of the GSM8K and MMLU answers hit it and were scored wrong.
# Here every unfinished greedy answer is decoded again with a budget of 640 tokens
# (finished answers are unchanged under greedy decoding), and the sampled answers of the
# sampling baselines are drawn again with the same budget. Then everything is calibrated.
# TriviaQA answers never reach the budget and are left as they are.
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
B=640
mkdir -p $R/stage4

redo() {  # gpu model tag task bs source [extra flags]
  local gpu=$1 model=$2 tag=$3 task=$4 bs=$5 src=$6; shift 6
  [ -f "$R/stage4/${tag}_${task}.npz" ] || CUDA_VISIBLE_DEVICES=$gpu python extract.py \
    --model "$model" --task "$task" --n 4000 --bs "$bs" --max_new_tokens $B --redo_from "$src" \
    --out "$R/stage4/${tag}_${task}.npz" "$@" > "$L/redo_${tag}_${task}.log" 2>&1
}
(
  redo 0 Qwen/Qwen2.5-1.5B-Instruct qwen1.5b gsm8k 16 $R/stage0/qwen1.5b_gsm8k.npz
  redo 0 Qwen/Qwen2.5-1.5B-Instruct qwen1.5b mmlu 16 $R/stage2/qwen1.5b_mmlu.npz
) &
(
  redo 1 Qwen/Qwen2.5-7B-Instruct qwen7b gsm8k 8 $R/stage2/qwen7b_gsm8k.npz
  redo 1 Qwen/Qwen2.5-7B-Instruct qwen7b mmlu 6 $R/stage2/qwen7b_mmlu.npz
) &
wait
echo REDO_SMALL_DONE
redo 0,1 "$M14" qwen14b gsm8k 16 $R/stage2/qwen14b_gsm8k.npz --no_hidden --multi_gpu
redo 0,1 "$M14" qwen14b mmlu 16 $R/stage2/qwen14b_mmlu.npz --no_hidden --multi_gpu
echo REDO_LARGE_DONE

samp() {  # gpu task bs
  [ -f "$R/stage4/samples_qwen1.5b_$2.npz" ] || CUDA_VISIBLE_DEVICES=$1 python sample.py \
    --model Qwen/Qwen2.5-1.5B-Instruct --task $2 --n 4000 --samples 10 --bs $3 --max_new_tokens $B \
    --out "$R/stage4/samples_qwen1.5b_$2.npz" > "$L/sample4_$2.log" 2>&1
}
samp 0 gsm8k 16 &
samp 1 mmlu 8 &
wait
echo SAMPLES_DONE
python run_all_calib.py --parallel 3 --jobs 12 --force
echo STAGE4_DONE
