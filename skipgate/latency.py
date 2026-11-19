"""Measure wall-clock cost per prompt token and per generated token for one model.

Every timed call decodes exactly `--new_tokens` tokens (end-of-sequence is suppressed), so
the per-token numbers do not depend on how long the answers of a model happen to be.
Single-stream (batch 1) timings give the latency one user sees. Batched timings give the
throughput of a loaded server. Each batched measurement is repeated `--repeats` times over
the same prompts, and the mean and standard deviation over the repeats are stored.

Outputs (same schema as the earlier version, plus spread fields)
  <out_dir>/latency_<tag>.json            single-stream numbers and the first batch size
  <out_dir>/throughput_<tag>_bs<B>.json   one file per batch size
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load_task, system_prompt


def sync_time():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)
    return time.perf_counter()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="short model name used in the file names")
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--n", type=int, default=256, help="prompts per batched measurement")
    ap.add_argument("--n_single", type=int, default=32, help="prompts timed one at a time")
    ap.add_argument("--bs", type=int, nargs="+", default=[16, 64, 128])
    ap.add_argument("--new_tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--multi_gpu", action="store_true")
    ap.add_argument("--max_memory", type=float, nargs="+", default=None,
                    help="with --multi_gpu: GiB of weights per card, to leave room for activations "
                         "when other processes hold memory on the cards")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if args.multi_gpu:
        mm = {i: f"{g}GiB" for i, g in enumerate(args.max_memory)} if args.max_memory else None
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map="auto", max_memory=mm).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).cuda().eval()
    items = load_task(args.task, 4000, 0)[:args.n]
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": system_prompt(args.task)},
         {"role": "user", "content": it["question"]}],
        tokenize=False, add_generation_prompt=True) for it in items]
    K = args.new_tokens

    def gen(enc, n_new):
        # fixed length: no end-of-sequence, so every sequence decodes n_new tokens
        with torch.no_grad():
            return model.generate(**enc, max_new_tokens=n_new, min_new_tokens=n_new,
                                  do_sample=False, temperature=None, top_p=None, top_k=None,
                                  eos_token_id=None, pad_token_id=tok.pad_token_id)

    enc0 = tok(prompts[0], return_tensors="pt").to("cuda")
    for _ in range(3):
        gen(enc0, 8)                                   # warm-up

    t_first, t_full, n_prompt = [], [], []
    for p in prompts[:args.n_single]:
        enc = tok(p, return_tensors="pt").to("cuda")
        t0 = sync_time(); gen(enc, 1); t1 = sync_time()
        out = gen(enc, K); t2 = sync_time()
        assert out.shape[1] - enc.input_ids.shape[1] == K
        t_first.append(t1 - t0)
        t_full.append(t2 - t1)
        n_prompt.append(enc.input_ids.shape[1])
    t_first, t_full, n_prompt = np.array(t_first), np.array(t_full), np.array(n_prompt)
    decode = (t_full - t_first) / (K - 1)
    base = {"model": args.model, "task": args.task, "new_tokens": K,
            "n_single": int(len(t_first)),
            "single_prefill_s": float(t_first.mean()), "single_prefill_s_sd": float(t_first.std(ddof=1)),
            "single_decode_s_per_token": float(decode.mean()),
            "single_decode_s_per_token_sd": float(decode.std(ddof=1)),
            "mean_prompt_tokens": float(n_prompt.mean()),
            "gpu_mem_other_gb": None}

    first = True
    for bs in args.bs:
        try:
            enc = tok(prompts[:bs], return_tensors="pt", padding=True).to("cuda")
            gen(enc, 8)                                # warm-up at this batch size
            rates, pre = [], []
            for _ in range(args.repeats):
                tb, tp, ntok = 0.0, 0.0, 0
                for b in range(0, len(prompts), bs):
                    enc = tok(prompts[b:b + bs], return_tensors="pt", padding=True).to("cuda")
                    t0 = sync_time(); gen(enc, 1); t1 = sync_time(); gen(enc, K); t2 = sync_time()
                    tp += t1 - t0
                    tb += t2 - t1
                    ntok += int(enc.attention_mask.sum())
                # tb includes the prompt pass of every call: remove it, so that the rate is
                # decode-only (the prompt is charged separately through the prompt price)
                rates.append(len(prompts) * (K - 1) / (tb - tp))
                pre.append(tp / ntok)          # seconds per prompt token in the batched pass
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"batch size {bs}: out of memory, skipped", flush=True)
            continue
        res = dict(base, n=len(prompts), batched_bs=bs, repeats=args.repeats,
                   batched_gen_tokens_per_s=float(np.mean(rates)),
                   batched_gen_tokens_per_s_sd=float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0,
                   batched_runs=[float(r) for r in rates],
                   batched_prefill_s_per_prompt_token=float(np.mean(pre)),
                   # price of a prompt token in generated tokens at this batch size
                   batched_prompt_weight=float(np.mean(pre) * np.mean(rates)),
                   decode_only=True)
        with open(os.path.join(args.out_dir, f"throughput_{args.tag}_bs{bs}.json"), "w") as f:
            json.dump(res, f, indent=2)
        if first:
            with open(os.path.join(args.out_dir, f"latency_{args.tag}.json"), "w") as f:
                json.dump(res, f, indent=2)
            first = False
        print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
