"""Run a frozen LLM once per question and save everything the gates need.

Per question we store
  h_pre   hidden state of the last prompt token, all layers (before any decoding)
  h_post  hidden state at the step that emits end-of-sequence, all layers
  h_mean  mean hidden state over the generated tokens, all layers
  token log-probability statistics of the greedy answer
  the answer text, the 0/1 correctness label and token counts
All features come from the single greedy pass. No extra forward pass is used.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load_task, max_new_tokens, score, system_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=["triviaqa", "gsm8k", "mmlu"])
    ap.add_argument("--multi_gpu", action="store_true",
                    help="spread the model over all visible GPUs (for models above 24 GB)")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip this many questions of the shuffled dataset (fresh questions)")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_hidden", action="store_true",
                    help="skip hidden states (used for the large model, which needs labels only)")
    ap.add_argument("--max_new_tokens", type=int, default=None,
                    help="override the per-dataset answer budget")
    ap.add_argument("--redo_from", default=None,
                    help="npz of an earlier run: only the answers that hit its token budget are "
                         "decoded again (greedy decoding leaves the finished ones unchanged)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if args.multi_gpu:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map="auto").eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).cuda().eval()
    eos_ids = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|eot_id|>"):
        i = tok.convert_tokens_to_ids(t)
        if i is not None and i != tok.unk_token_id:
            eos_ids.add(i)
    eos_list = sorted(eos_ids)

    items = load_task(args.task, args.n, args.seed, args.offset)
    sysmsg = system_prompt(args.task)
    # the override applies to GSM8K and MMLU. TriviaQA answers are short and keep 32 tokens
    mnt = max_new_tokens(args.task)
    if args.max_new_tokens and args.task != "triviaqa":
        mnt = args.max_new_tokens
    # sort by length so batches pad little, restore order at the end
    order = sorted(range(len(items)), key=lambda i: len(items[i]["question"]))

    rec = {k: [None] * len(items) for k in
           ("h_pre", "h_post", "h_mean", "lp_mean", "lp_min", "lp_sum", "p_first",
            "n_prompt", "n_gen", "label", "finished")}
    texts = [None] * len(items)
    prev_meta = {}
    if args.redo_from:
        prev = np.load(args.redo_from)
        assert list(prev["ids"]) == [it["id"] for it in items], "question order differs"
        for k in rec:
            if k in prev.files:
                rec[k] = list(prev[k])
        with open(args.redo_from.replace(".npz", ".answers.jsonl"), encoding="utf-8") as f:
            texts = [json.loads(line)["pred"] for line in f]
        prev_meta = json.load(open(args.redo_from.replace(".npz", ".meta.json")))
        order = [i for i in order if not prev["finished"][i]]
        print(f"redo of {len(order)} unfinished answers with a budget of {mnt} tokens", flush=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    def process(idx):
        prompts = [
            tok.apply_chat_template(
                [{"role": "system", "content": sysmsg},
                 {"role": "user", "content": items[i]["question"]}],
                tokenize=False, add_generation_prompt=True)
            for i in idx
        ]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=mnt, do_sample=False, temperature=None, top_p=None,
                top_k=None, eos_token_id=eos_list, pad_token_id=tok.pad_token_id,
                output_hidden_states=not args.no_hidden, output_scores=not args.no_hidden,
                return_dict_in_generate=True)
        gen = out.sequences[:, enc.input_ids.shape[1]:]            # [B, T]
        T = gen.shape[1]
        # hs[t] is a tuple over layers of [B, seq, D]; the last position is the newest token
        hs = None if args.no_hidden else torch.stack(
            [torch.stack([l[:, -1, :] for l in step], dim=1) for step in out.hidden_states],
            dim=1)                                                   # [B, T, L, D]
        # with --no_hidden the per-step logits are not kept (they fill the GPU on long
        # answers), so the log-probability fields are NaN for that run
        tok_lp = torch.full(gen.shape, float("nan"), device=gen.device) if args.no_hidden \
            else torch.stack(
                [torch.log_softmax(s.float(), -1).gather(-1, gen[:, t:t + 1]).squeeze(-1)
                 for t, s in enumerate(out.scores)], dim=1)         # [B, T]
        is_eos = torch.zeros_like(gen, dtype=torch.bool)
        for e in eos_list:
            is_eos |= gen == e
        for r, i in enumerate(idx):
            pos = torch.nonzero(is_eos[r])
            fin = len(pos) > 0
            t_eos = int(pos[0]) if fin else T - 1    # step whose state emits EOS
            n_gen = t_eos if fin else T              # content tokens before EOS
            n_use = max(n_gen, 1)
            lp = tok_lp[r, :n_use]
            text = tok.decode(gen[r, :n_gen], skip_special_tokens=True).strip()
            texts[i] = text
            if hs is not None:
                rec["h_pre"][i] = hs[r, 0].float().cpu().numpy().astype(np.float16)
                rec["h_post"][i] = hs[r, t_eos].float().cpu().numpy().astype(np.float16)
                rec["h_mean"][i] = hs[r, :n_use].float().mean(0).cpu().numpy().astype(np.float16)
            rec["lp_mean"][i] = float(lp.mean())
            rec["lp_min"][i] = float(lp.min())
            rec["lp_sum"][i] = float(lp.sum())
            rec["p_first"][i] = float(tok_lp[r, 0].exp())
            rec["n_prompt"][i] = int(enc.attention_mask[r].sum())
            rec["n_gen"][i] = n_gen
            rec["finished"][i] = int(fin)
            rec["label"][i] = score(args.task, text, items[i]["gold"])
        del out, hs, tok_lp

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

    elapsed = time.time() - t0
    np.savez_compressed(
        args.out,
        **{k: np.asarray(v) for k, v in rec.items() if v[0] is not None},
        ids=np.asarray([it["id"] for it in items]))
    meta = {
        "model": args.model, "task": args.task, "n": len(items), "offset": args.offset,
        "accuracy": float(np.mean(rec["label"])),
        "finished_rate": float(np.mean(rec["finished"])),
        "mean_gen_tokens": float(np.mean(rec["n_gen"])),
        "mean_prompt_tokens": float(np.mean(rec["n_prompt"])),
        "max_new_tokens": mnt,
        "runtime_s": elapsed + prev_meta.get("runtime_s", 0.0),
        "peak_gpu_mem_gb": max(prev_meta.get("peak_gpu_mem_gb", 0.0),
                               sum(torch.cuda.max_memory_allocated(i)
                                   for i in range(torch.cuda.device_count())) / 2**30),
    }
    if args.redo_from:
        meta["redo_runtime_s"], meta["redo_n"] = elapsed, len(order)
    with open(args.out.replace(".npz", ".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(args.out.replace(".npz", ".answers.jsonl"), "w", encoding="utf-8") as f:
        for it, t, l in zip(items, texts, rec["label"]):
            f.write(json.dumps({"id": it["id"], "question": it["question"], "gold": it["gold"],
                                "pred": t, "label": int(l)}, ensure_ascii=False) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
