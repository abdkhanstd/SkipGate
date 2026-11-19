"""First version of the timing script, kept for the record.

It produced the timing files in results/stage2 (single measurements, greedy decoding to the
end of each answer). Its batched tokens per second depend on the answer lengths of the model,
because a batch runs until its longest answer ends. latency.py replaces it with fixed-length
decoding and repeats.

Measure wall-clock cost per prompt token and per generated token for one model.
Single-stream (batch 1) timings give the latency a user sees. A batched run gives
throughput.
"""
import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load_task, system_prompt


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--n_single", type=int, default=64,
                    help="questions timed one at a time (the batched run uses all --n)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--multi_gpu", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if args.multi_gpu:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map="auto").eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).cuda().eval()
    items = load_task(args.task, 4000, 0)[:args.n]
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": system_prompt(args.task)},
         {"role": "user", "content": it["question"]}],
        tokenize=False, add_generation_prompt=True) for it in items]
    mnt = 320          # the answer budget at the time of these measurements
    kw = dict(do_sample=False, temperature=None, top_p=None, top_k=None,
              pad_token_id=tok.pad_token_id)

    def gen(enc, n_new):
        with torch.no_grad():
            return model.generate(**enc, max_new_tokens=n_new, **kw)

    enc0 = tok(prompts[0], return_tensors="pt").to("cuda")
    for _ in range(3):
        gen(enc0, 8)                                   # warm-up

    t_first, t_full, n_prompt, n_gen = [], [], [], []
    for p in prompts[:args.n_single]:
        enc = tok(p, return_tensors="pt").to("cuda")
        t0 = sync_time(); gen(enc, 1); t1 = sync_time()
        out = gen(enc, mnt); t2 = sync_time()
        t_first.append(t1 - t0)
        t_full.append(t2 - t1)
        n_prompt.append(enc.input_ids.shape[1])
        n_gen.append(out.shape[1] - enc.input_ids.shape[1])
    t_first, t_full = np.array(t_first), np.array(t_full)
    n_prompt, n_gen = np.array(n_prompt), np.array(n_gen)
    decode = (t_full - t_first) / np.maximum(n_gen - 1, 1)

    tb, tokens = 0.0, 0
    for b in range(0, len(prompts), args.bs):
        enc = tok(prompts[b:b + args.bs], return_tensors="pt", padding=True).to("cuda")
        t0 = sync_time(); out = gen(enc, mnt); tb += sync_time() - t0
        new = out[:, enc.input_ids.shape[1]:]
        tokens += int((new != tok.pad_token_id).sum())

    res = {"model": args.model, "task": args.task, "n": len(prompts),
           "single_prefill_s": float(t_first.mean()),
           "single_decode_s_per_token": float(decode.mean()),
           "single_total_s_per_query": float(t_full.mean()),
           "mean_prompt_tokens": float(n_prompt.mean()), "mean_gen_tokens": float(n_gen.mean()),
           "batched_bs": args.bs, "batched_s_per_query": tb / len(prompts),
           "batched_gen_tokens_per_s": tokens / tb}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
