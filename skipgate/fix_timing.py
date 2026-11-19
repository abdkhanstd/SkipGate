"""Decode-only batched rates for the stage-5 timing files.

latency.py (stage 5) timed each batched call from the prompt on, so its tokens per second
include the prompt pass, which the cost model also charges through the prompt price w. This
script removes the prompt pass. For every batch the prompt pass was timed separately (one
generated token), so the decode time of a run is its total time minus the prompt time, which
is the measured time per prompt token times the number of prompt tokens. The prompt-token
count is exact: it is recomputed with the tokenizer from the same 256 prompts.

Adds to each results/stage5/throughput_*.json and latency_*.json:
  n_prompt_tokens_batched        prompt tokens in the 256 prompts (without padding)
  batched_decode_tokens_per_s    generated tokens per second, prompt pass removed
  batched_decode_runs            the same per repeat
  batched_prompt_weight_decode   prompt price w relative to a decoded token at this batch size
Usage: python fix_timing.py --models_dir ../../build/models   (no GPU needed)
"""
import argparse
import glob
import json
import os

import numpy as np
from transformers import AutoTokenizer

from data import load_task, system_prompt

TOK = {"qwen1.5b": "qwen1.5b", "qwen7b": "qwen7b", "llama3b": "llama3b", "llama8b": "llama8b"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--dir", default="../results/stage5")
    args = ap.parse_args()
    ntok_cache = {}
    for p in sorted(glob.glob(os.path.join(args.dir, "*.json"))):
        name = os.path.basename(p)
        if not (name.startswith("throughput_") or name.startswith("latency_")):
            continue
        d = json.load(open(p))
        if "batched_prefill_s_per_prompt_token" not in d or d.get("decode_only"):
            continue                     # newer latency.py output is already decode-only
        tag = name.split("_")[1].replace(".json", "")
        if tag not in ntok_cache:
            tok = AutoTokenizer.from_pretrained(os.path.join(args.models_dir, TOK[tag]))
            items = load_task(d["task"], 4000, 0)[:d["n"]]
            prompts = [tok.apply_chat_template(
                [{"role": "system", "content": system_prompt(d["task"])},
                 {"role": "user", "content": it["question"]}],
                tokenize=False, add_generation_prompt=True) for it in items]
            ntok_cache[tag] = int(sum(len(tok(x).input_ids) for x in prompts))
        ntok = ntok_cache[tag]
        K, n = d["new_tokens"], d["n"]
        tp = d["batched_prefill_s_per_prompt_token"] * ntok       # prompt time of one repeat
        dec = []
        for r in d["batched_runs"]:
            tb = n * K / r                                         # time of the K-token calls
            dec.append(n * (K - 1) / (tb - tp))                    # K - 1 tokens after the first
        d["n_prompt_tokens_batched"] = ntok
        d["batched_decode_runs"] = [float(x) for x in dec]
        d["batched_decode_tokens_per_s"] = float(np.mean(dec))
        d["batched_decode_tokens_per_s_sd"] = float(np.std(dec, ddof=1)) if len(dec) > 1 else 0.0
        d["batched_prompt_weight_decode"] = float(d["batched_prefill_s_per_prompt_token"] * np.mean(dec))
        json.dump(d, open(p, "w"), indent=2)
        print(f"{name:34s} bs {d['batched_bs']:3d}  with prompt {d['batched_gen_tokens_per_s']:7.0f}  "
              f"decode only {d['batched_decode_tokens_per_s']:7.0f}  w {d['batched_prompt_weight']:.3f} -> "
              f"{d['batched_prompt_weight_decode']:.3f}")


if __name__ == "__main__":
    main()
