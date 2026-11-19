#!/usr/bin/env bash
# Stage 5: repeat of the cost measurements with fixed-length decoding and three repeats, on
# the server (RTX 3090), for all five models. The models are timed one after another so that
# they do not compete for the CPU. Starts when both cards have been idle for three minutes,
# because other jobs share this server. GPU memory held by other processes is logged.
set -uo pipefail
cd "$(dirname "$0")/skipgate"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate skipgate
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=../results/stage5
L=../../logs
MD="$HOME/skipgate/models"
mkdir -p $R
idle=0
while [ $idle -lt 6 ]; do
  busy=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '$1>5' | wc -l)
  if [ "$busy" -eq 0 ]; then idle=$((idle+1)); else idle=0; fi
  sleep 30
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > $R/gpu_state_before.csv
uptime >> $R/gpu_state_before.csv
t() { CUDA_VISIBLE_DEVICES=$1 python latency.py --model "$2" --tag $3 --bs 16 64 128 \
        --out_dir $R "${@:4}" > $L/timing_$3.log 2>&1; }
t 1 Qwen/Qwen2.5-1.5B-Instruct qwen1.5b
t 1 Qwen/Qwen2.5-7B-Instruct qwen7b
t 1 $MD/Llama-3.2-3B-Instruct llama3b
t 1 $MD/Llama-3.1-8B-Instruct llama8b
t 0,1 $MD/Qwen2.5-14B-Instruct qwen14b --multi_gpu
echo STAGE5_DONE
