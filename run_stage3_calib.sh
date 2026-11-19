#!/usr/bin/env bash
# Stage 3: final calibration of every pair and dataset once all GPU runs have finished.
set -uo pipefail
cd "$(dirname "$0")/skipgate"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate skipgate
L=../../logs
while ! grep -q STAGE2E_DONE "$L/stage2e.log" 2>/dev/null; do sleep 30; done
while ! grep -q ALL_CALIB_DONE "$L/final_driver2.log" 2>/dev/null; do sleep 30; done
python run_all_calib.py --parallel 3 --jobs 12 --force --only \
  qwen1.5b-qwen7b_mmlu qwen1.5b-qwen14b_gsm8k qwen1.5b-qwen14b_mmlu qwen1.5b-qwen14b_triviaqa \
  qwen7b-qwen14b_gsm8k qwen7b-qwen14b_mmlu
echo STAGE3_DONE
