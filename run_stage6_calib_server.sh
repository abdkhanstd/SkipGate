#!/usr/bin/env bash
# Calibration of the Llama pair on MMLU on the server, where its hidden states were extracted.
# (The other stage-6 calibrations run on the local workstation.)
set -uo pipefail
cd "$(dirname "$0")/skipgate"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate skipgate
while [ ! -f ../results/stage6/llama8b_mmlu.npz ]; do sleep 30; done
sleep 20
python run_all_calib.py --set llama --parallel 1 --jobs 16 --force --only llama3b-llama8b_mmlu
echo CALIB_LLAMA_MMLU_DONE
