#!/usr/bin/env bash
# Stage 6, run on a local workstation (two RTX 2080 Ti with 22 GB each, Windows, Git Bash)
# because the shared server was occupied. The results of this stage (answers, labels and
# hidden states) do not depend on the speed of the hardware. All timing stays on the server
# (run_stage5_timing.sh).
#
# (a) Confirmation on fresh questions: the 4,000 questions per dataset that follow the main
#     4,000 in the shuffled order, with the pipeline frozen.
# (b) A second model family: Llama-3.2-3B-Instruct (small) and Llama-3.1-8B-Instruct (large)
#     on GSM8K and MMLU.
# See CONFIRMATION_PLAN.md, which was written before this script produced any output.
#
# Usage: bash run_stage6_local.sh [download|extract|calib ...]   (default: download extract)
#        The calibration needs the timing files of stage 5 and is started separately.
set -uo pipefail
cd "$(dirname "$0")/skipgate"
ROOT="$(cd ../.. && pwd)"
PY="$ROOT/build/venv/Scripts/python.exe"
MD="$ROOT/build/models"
L="$ROOT/build/local_logs"
R=../results
export HF_HOME="$ROOT/build/hf" TOKENIZERS_PARALLELISM=false PYTHONIOENCODING=utf-8
mkdir -p "$L" $R/fresh $R/stage6
STEPS="${*:-download extract}"

if [[ " $STEPS " == *" download "* ]]; then
  "$PY" download_models.py --dir "$MD" > "$L/download.log" 2>&1
  echo DOWNLOAD_DONE
fi

run() {  # gpu model out task batch [extra flags]
  local gpu=$1 model=$2 out=$3 task=$4 bs=$5; shift 5
  [ -f "$out" ] || CUDA_VISIBLE_DEVICES=$gpu "$PY" extract.py --model "$MD/$model" --task "$task" \
    --n 4000 --bs "$bs" --out "$out" "$@" \
    > "$L/$(basename "$(dirname "$out")")_$(basename "$out" .npz).log" 2>&1
}
# Batch sizes are larger than on the server: on this machine the time per decoding step is
# dominated by the CPU, so larger batches cost little. extract.py halves a batch that does
# not fit in memory.
if [[ " $STEPS " == *" extract "* ]]; then
  (
    for T in gsm8k mmlu; do run 0 qwen1.5b $R/fresh/qwen1.5b_$T.npz $T 32 --offset 4000; done
    for T in gsm8k mmlu; do run 0 llama3b $R/stage6/llama3b_$T.npz $T 24; done
  ) &
  (
    for T in gsm8k mmlu; do run 1 qwen7b $R/fresh/qwen7b_$T.npz $T 64 --offset 4000 --no_hidden; done
    for T in gsm8k mmlu; do run 1 llama8b $R/stage6/llama8b_$T.npz $T 48 --no_hidden; done
  ) &
  wait
  echo EXTRACT_DONE
fi

if [[ " $STEPS " == *" calib "* ]]; then
  export SKIPGATE_ENCODER="$MD/minilm" SKIPGATE_ENCODER2="$MD/bge" SKIPGATE_LOGS="$L"
  "$PY" run_all_calib.py --set fresh --parallel 2 --jobs 16 --force
  "$PY" run_all_calib.py --set llama --parallel 2 --jobs 16 --force
  echo CALIB_DONE
fi
echo LOCAL_DONE
