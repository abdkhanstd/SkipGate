#!/usr/bin/env bash
# Stage 4b: the first stage-4 run crashed while writing the answer texts (a numpy integer
# in the JSON dump), after the features and labels were saved. This repeats the redo into
# stage4b so that the answer texts exist, and checks that the labels agree with stage 4.
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
mkdir -p $R/stage4b
redo() {  # gpu model tag task bs source [extra flags]; labels only, the features exist
  local gpu=$1 model=$2 tag=$3 task=$4 bs=$5 src=$6; shift 6
  CUDA_VISIBLE_DEVICES=$gpu python extract.py --model "$model" --task "$task" --n 4000 \
    --bs "$bs" --max_new_tokens 640 --redo_from "$src" --no_hidden \
    --out "$R/stage4b/${tag}_${task}.npz" "$@" > "$L/redo4b_${tag}_${task}.log" 2>&1
}
(
  redo 0 Qwen/Qwen2.5-1.5B-Instruct qwen1.5b gsm8k 16 $R/stage0/qwen1.5b_gsm8k.npz
  redo 0 Qwen/Qwen2.5-1.5B-Instruct qwen1.5b mmlu 16 $R/stage2/qwen1.5b_mmlu.npz
) &
(
  redo 1 Qwen/Qwen2.5-7B-Instruct qwen7b gsm8k 16 $R/stage2/qwen7b_gsm8k.npz
  redo 1 Qwen/Qwen2.5-7B-Instruct qwen7b mmlu 16 $R/stage2/qwen7b_mmlu.npz
) &
wait
redo 0,1 "$M14" qwen14b gsm8k 16 $R/stage2/qwen14b_gsm8k.npz --multi_gpu
redo 0,1 "$M14" qwen14b mmlu 16 $R/stage2/qwen14b_mmlu.npz --multi_gpu
python - <<'PY'
import numpy as np, shutil
for m in ("qwen1.5b", "qwen7b", "qwen14b"):
    for t in ("gsm8k", "mmlu"):
        a = np.load(f"../results/stage4/{m}_{t}.npz"); b = np.load(f"../results/stage4b/{m}_{t}.npz")
        same = float((a["label"] == b["label"]).mean()); ng = float((a["n_gen"] == b["n_gen"]).mean())
        print(m, t, "labels equal", same, "lengths equal", ng, flush=True)
        shutil.copy(f"../results/stage4b/{m}_{t}.answers.jsonl", f"../results/stage4/{m}_{t}.answers.jsonl")
PY
echo STAGE4B_DONE
