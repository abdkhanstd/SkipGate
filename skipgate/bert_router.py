"""Fine-tuned BERT router, the router type of RouteLLM (Ong et al.), as a stronger baseline.

RouteLLM's BERT router is a BERT classifier on the query text that predicts whether the small
model's answer is good enough. Here the target is the correctness of the small model's answer
(the label used by every other gate), and the training data are the probe-fitting questions
of each seed, so the router sees exactly the data the probes see. The router is trained once
per seed with the same split as calibrate.py (fit half: 80% train, 20% validation for the
epoch), then scores every question. calibrate.py reads the scores with --bert.

Output: <answers file without .answers.jsonl>.bert.npz with
  ids      question ids in the order of the answers file
  scores   [seeds, n] predicted probability that the small model is right
  val_auc  [seeds] validation AUROC of the chosen epoch
Usage: python bert_router.py --answers ../results/stage4/qwen1.5b_gsm8k.answers.jsonl \
           --model ../../build/models/bert-base-uncased [--labels labels.npy]
"""
import argparse
import json
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--labels", default=None,
                    help="npy of 0/1 labels in the same order (overrides the labels in the answers file)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.answers, encoding="utf-8")]
    ids = np.asarray([r["id"] for r in rows])
    qs = [r["question"] for r in rows]
    y = np.load(args.labels).astype(int) if args.labels else np.asarray([int(r["label"]) for r in rows])
    n = len(y)
    tok = AutoTokenizer.from_pretrained(args.model)
    # tokenised once, padded per batch to the longest question of the batch (much faster than
    # padding every question to max_len)
    toks = [tok(q, truncation=True, max_length=args.max_len)["input_ids"] for q in qs]
    lens = np.array([len(t) for t in toks])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    scores = np.zeros((args.seeds, n), dtype=np.float32)
    val_auc = np.zeros(args.seeds)

    def batch(j):
        m = int(lens[j].max())
        ids_ = torch.full((len(j), m), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(j), m), dtype=torch.long)
        for r, k in enumerate(j):
            ids_[r, :lens[k]] = torch.as_tensor(toks[k])
            att[r, :lens[k]] = 1
        return ids_.to(dev), att.to(dev)

    def predict(model, idx):
        model.eval()
        out = []
        idx = np.asarray(idx)
        order = idx[np.argsort(lens[idx])]
        res = {}
        with torch.no_grad():
            for b in range(0, len(order), 128):
                j = order[b:b + 128]
                a_, m_ = batch(j)
                with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                    logits = model(input_ids=a_, attention_mask=m_).logits
                for k, v in zip(j, torch.softmax(logits.float(), -1)[:, 1].cpu().numpy()):
                    res[k] = v
        return np.array([res[k] for k in idx])

    t0 = time.time()
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)            # same split as calibrate.py
        perm = rng.permutation(n)
        fit = perm[:n // 2]
        tr, va = fit[:int(.8 * len(fit))], fit[int(.8 * len(fit)):]
        torch.manual_seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        steps = args.epochs * int(np.ceil(len(tr) / args.bs))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))
        best = (-1.0, None)
        g = torch.Generator().manual_seed(seed)
        for ep in range(args.epochs):
            model.train()
            order = tr[torch.randperm(len(tr), generator=g).numpy()]
            for b in range(0, len(order), args.bs):
                j = order[b:b + args.bs]
                a_, m_ = batch(j)
                with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                    loss = model(input_ids=a_, attention_mask=m_, labels=torch.as_tensor(y[j]).to(dev)).loss
                loss.backward()
                opt.step(); sched.step(); opt.zero_grad()
            auc = roc_auc_score(y[va], predict(model, va))
            if auc > best[0]:
                best = (auc, predict(model, np.arange(n)))
            print(f"seed {seed} epoch {ep} val AUROC {auc:.3f}", flush=True)
        val_auc[seed], scores[seed] = best
        del model, opt
        torch.cuda.empty_cache()
    out = args.answers.replace(".answers.jsonl", ".bert.npz")
    np.savez(out, ids=ids, scores=scores, val_auc=val_auc)
    print(json.dumps({"out": out, "val_auc": val_auc.round(3).tolist(), "runtime_s": time.time() - t0}))


if __name__ == "__main__":
    main()
