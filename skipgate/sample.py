"""Draw N sampled answers per question from one model.

Used by the sampling-based baselines (Conformal Cascade style set-size deferral and a
self-consistency agreement gate). Questions and their order match extract.py.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import (extract_letter, extract_number, load_task, max_new_tokens, normalize_text,
                  score, system_prompt)


def answer_key(task, text):
    """Canonical form used to count agreeing samples."""
    if task == "gsm8k":
        v = extract_number(text)
        return "none" if v is None else f"{v:.4f}"
    if task == "mmlu":
        return extract_letter(text) or "none"
    return normalize_text(text.split("\n")[0]) or "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=["triviaqa", "gsm8k", "mmlu"])
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).cuda().eval()
    eos_ids = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|eot_id|>"):
        i = tok.convert_tokens_to_ids(t)
        if i is not None and i != tok.unk_token_id:
            eos_ids.add(i)
    eos_list = sorted(eos_ids)

    items = load_task(args.task, args.n, args.seed, args.offset)
    sysmsg, N = system_prompt(args.task), args.samples
    # the override applies to GSM8K and MMLU. TriviaQA answers are short and keep 32 tokens
    mnt = max_new_tokens(args.task)
    if args.max_new_tokens and args.task != "triviaqa":
        mnt = args.max_new_tokens
    order = sorted(range(len(items)), key=lambda j: len(items[j]["question"]))
    keys = [None] * len(items)
    labels = np.zeros((len(items), N), dtype=np.int8)
    n_gen = np.zeros((len(items), N), dtype=np.int32)
    n_prompt = np.zeros(len(items), dtype=np.int32)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    def process(idx):
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": sysmsg},
             {"role": "user", "content": items[j]["question"]}],
            tokenize=False, add_generation_prompt=True) for j in idx]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=mnt, do_sample=True, temperature=args.temperature,
                top_p=1.0, top_k=None, num_return_sequences=N, eos_token_id=eos_list,
                pad_token_id=tok.pad_token_id)
        gen = out[:, enc.input_ids.shape[1]:].view(len(idx), N, -1)
        is_eos = torch.zeros_like(gen, dtype=torch.bool)
        for e in eos_list:
            is_eos |= gen == e
        for r, j in enumerate(idx):
            ks = []
            for s in range(N):
                pos = torch.nonzero(is_eos[r, s])
                ng = int(pos[0]) if len(pos) else gen.shape[-1]
                text = tok.decode(gen[r, s, :ng], skip_special_tokens=True).strip()
                ks.append(answer_key(args.task, text))
                labels[j, s] = score(args.task, text, items[j]["gold"])
                n_gen[j, s] = ng
            keys[j] = ks
            n_prompt[j] = int(enc.attention_mask[r].sum())

    def safe(idx):
        """Run a batch, and split it in two when the GPU runs out of memory."""
        try:
            process(idx)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(idx) == 1:
                raise
            safe(idx[:len(idx) // 2])
            safe(idx[len(idx) // 2:])

    for b in tqdm(range(0, len(order), args.bs)):
        safe(order[b:b + args.bs])

    meta = {"model": args.model, "task": args.task, "n": len(items), "samples": N,
            "temperature": args.temperature, "max_new_tokens": mnt,
            "finished_rate": float((n_gen < mnt).mean()), "mean_sample_accuracy": float(labels.mean()),
            "mean_gen_tokens": float(n_gen.mean()), "runtime_s": time.time() - t0,
            "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
    np.savez_compressed(args.out, labels=labels, n_gen=n_gen, n_prompt=n_prompt,
                        ids=np.asarray([it["id"] for it in items]))
    with open(args.out.replace(".npz", ".keys.jsonl"), "w", encoding="utf-8") as f:
        for it, ks in zip(items, keys):
            f.write(json.dumps({"id": it["id"], "keys": ks}, ensure_ascii=False) + "\n")
    with open(args.out.replace(".npz", ".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
