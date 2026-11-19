"""Risk-controlled calibration of the two gates, the bound check, baselines and ablations.

Policy   skip gate:   s_pre(x)  <  t1  -> send to the large model, the small model never decodes
         accept gate: s_post(x,y) >= t2 -> keep the small answer, otherwise escalate
Risk     R(t1,t2) = P(small answer kept and wrong)
Goal     P( R(t1,t2) <= alpha ) >= 1 - delta   over the draw of the calibration set

Calibration is Learn-then-Test with fixed-sequence testing. For a fixed t1 the accept
threshold is lowered one calibration item at a time, starting from "accept nothing"
(risk 0), and each step is tested with the exact binomial tail p-value. The sequence stops
at the first failure. The t1 grid is handled by a Bonferroni split of delta. Among all
certified pairs the one with the lowest calibration cost is used.

Data use per seed: 50% of questions fit the probes (inner 80/20 for layer and C). The other
50% is split at random `reps` times into calibration and test halves.

Cost models (all are evaluated in one run, the probes are fitted once)
  c<r>     compute: generated tokens weighted by a per-token large/small ratio r. A prompt
           token costs w generated tokens of the same model, where w is the measured ratio
           of prefill time per prompt token to decoding time per token. With batched timing
           files (--throughput) w is the batched value (0.2 to 0.4 at batch size 128),
           otherwise the single-stream value (about 0.01). wsingle_<r> uses the latter.
           Sampling baselines pay for every sampled token.
  w<v>_<r>  the same at the measured ratio with w set to v in {0, 0.05, 0.1, 0.25, 1}
           (sensitivity to the price of a prompt token).
  latency  measured single-stream seconds (from latency.py). Sampling baselines are charged
           only for their longest sample, as if all samples decode in one parallel batch.
           This is optimistic for those baselines.

Methods
  skipgate            two probe gates, joint calibration (ours)
  accept_only         accept gate only (ablation: no skip gate)
  route_only          skip gate only, every non-skipped answer is kept (ablation: no accept gate)
  twogate_noprobe     same two-gate calibration, but fed with a question-only router score and
                      the token-probability score instead of hidden states. This separates the
                      gain from better input information from the gain of the method.
  token_gate          token-probability accept gate with the same calibration
  query_router        question-only router (sentence embedding + logistic regression), skip only.
                      It does not read the small model, so a skipped query pays no small-model
                      prompt pass. The cost of the sentence encoder is not charged.
  best_single         accept gate and skip gate each calibrated at delta/2, the cheaper one on
                      the calibration set is used (ablation: is a joint grid needed)
  skipgate_m<k>       SkipGate with a skip grid of k values instead of 10
  skipgate_ws         SkipGate with a weighted split of delta: a quarter for the pure accept
                      gate, a quarter for the pure skip gate, half for the interior grid pairs
  skipgate_wf         skipgate_ws plus the candidate "always use the large model" (risk 0, the
                      small model is not run), chosen when no certified pair is cheaper on the
                      calibration set. Its cost on the calibration set never exceeds 1.
  query_router_big    the query-only router with a larger sentence encoder (--encoder2)
  bert_router         fine-tuned BERT classifier on the question (RouteLLM's router type, --bert)
  feature_gate        learned deferral gate: logistic regression on the token-probability
                      statistics and the answer length, same calibration (accept only)
  random_cert         random routing made safe: the kept share uses a Clopper-Pearson upper
                      bound on the small model's error rate, so the risk bound holds
  sc_gate_n<k>        self-consistency gate that uses only the first k samples
  always_large, always_small   references on the same test split
  random_route        reference: keeps a random share of answers so that the expected risk
                      equals alpha, using the calibration error rate (no guarantee)
  oracle              privileged reference: keeps exactly the correct small answers and
                      skips the others before any small-model work
  length_gate         answer-length accept gate (control for length as a confound)
  accept_uncorrected  accept gate with an empirical threshold and no multiplicity correction
  conformal_cascade   our reimplementation of set-size deferral: split conformal prediction
                      over answer frequencies, accept when the prediction set is a singleton.
                      The published method is multiple choice only. Open answers are grouped
                      by their canonical form here, so this is not an official reproduction.
  sc_gate             agreement of the majority answer as the accept score, same calibration
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
from scipy.stats import beta, binom
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from probe_stage0 import ece, probe, scalar_baseline

# skip-gate grid as quantiles of s_pre. It reaches 0.9 so that heavy skipping (the best
# choice when the two models cost about the same) is available to the calibration.
T1_QUANTILES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# grids of other sizes for the ablation of m
T1_ABLATION = {2: (0.0, 0.5), 5: (0.0, 0.2, 0.4, 0.6, 0.8),
               20: tuple(round(0.05 * i, 2) for i in range(20))}


def fixed_sequence(score, wrong, eligible, n, alpha, delta):
    """Lowest certified accept threshold. Accept rule: eligible and score >= t."""
    idx = np.where(eligible)[0]
    idx = idx[np.argsort(-score[idx])]
    if len(idx) == 0:
        return np.inf
    s = score[idx]
    k = np.cumsum(wrong[idx])
    # tied scores are accepted together, so only the end of each tie block is a threshold
    ends = np.nonzero(np.r_[s[1:] != s[:-1], True])[0]
    p = binom.cdf(k[ends], n, alpha)
    fail = np.nonzero(p > delta)[0]
    if len(fail) == 0:
        return float(s[-1])
    return np.inf if fail[0] == 0 else float(s[ends[fail[0] - 1]])


def empirical_threshold(score, wrong, n, alpha):
    """Uncorrected baseline: largest accept set whose calibration risk is <= alpha."""
    idx = np.argsort(-score)
    s = score[idx]
    k = np.cumsum(wrong[idx])
    ends = np.nonzero(np.r_[s[1:] != s[:-1], True])[0]
    ok = np.nonzero(k[ends] / n <= alpha)[0]
    return np.inf if len(ok) == 0 else float(s[ends[ok[-1]]])


def gates(s_pre, s_post, t1, t2):
    skip = s_pre < t1
    return (~skip) & (s_post >= t2), skip


def cost_of(accept, skip, dec, c, pre_on_skip=True):
    """Mean cost relative to always using the large model. Methods that read hidden states
    pay the small-model prompt pass on every query. A method that decides from the question
    text alone (pre_on_skip=False) pays it only for the queries it does not skip."""
    pre = c["pre_s"] if pre_on_skip else (~skip) * c["pre_s"]
    return float(np.mean(pre + (~skip) * dec + (~accept) * c["large"]) / np.mean(c["large"]))


def quality(accept, skip, y_kept, y_l):
    return {"risk": float(np.mean(accept & (y_kept == 0))),
            "acc": float(np.mean(np.where(accept, y_kept, y_l))),
            "accept": float(np.mean(accept)), "skip": float(np.mean(skip))}


def weighted_candidates(s_pre, s_post, wrong, grid, alpha, delta):
    """Certified pairs under a weighted split of delta (weights fixed in advance):
    delta/4 for the pure accept gate, delta/4 for the pure skip gate (one fixed-sequence walk
    over the skip score with every decoded answer kept), delta/2 shared by the interior pairs."""
    n = len(s_pre)
    ones = np.ones(n, bool)
    inner = [t for t in grid if np.isfinite(t)]
    out = [(-np.inf, fixed_sequence(s_post, wrong, ones, n, alpha, delta / 4)),
           (fixed_sequence(s_pre, wrong, ones, n, alpha, delta / 4), -np.inf)]
    out += [(t1, fixed_sequence(s_post, wrong, s_pre >= t1, n, alpha, delta / (2 * len(inner))))
            for t1 in inner]
    return out


def random_row(q_keep, wrong, y, y_l, c):
    """Random routing that keeps a share q_keep, in expectation over its coin flips.
    Queries that are not kept go to the large model without any small-model work."""
    return {"risk": q_keep * float(np.mean(wrong)),
            "acc": q_keep * float(y.mean()) + (1 - q_keep) * float(y_l.mean()),
            "accept": q_keep, "skip": 1 - q_keep,
            "cost": float(np.mean(q_keep * (c["pre_s"] + c["dec_s"]) + (1 - q_keep) * c["large"])
                          / np.mean(c["large"]))}


def take(d, i):
    return {k: v[i] for k, v in d.items()}


def skip_grid(s_pre_val, quantiles=T1_QUANTILES):
    """Skip thresholds fixed before calibration: quantiles of s_pre on the probe validation
    split, which is disjoint from the calibration and test data."""
    return [-np.inf if q == 0 else float(np.quantile(s_pre_val, q)) for q in quantiles]


def two_gate_candidates(s_pre, s_post, wrong, grid, alpha, delta):
    """Certified (t1, t2) pairs, one per skip-grid value. All hold jointly w.p. 1 - delta."""
    n = len(s_pre)
    return [(t1, fixed_sequence(s_post, wrong, s_pre >= t1, n, alpha, delta / len(grid)))
            for t1 in grid]


def cheapest(cands, s_pre, s_post, c, pre_on_skip=True):
    """Pair with the lowest calibration cost (the first one on ties). Returns t1, t2, cost."""
    best = None
    for t1, t2 in cands:
        a, s = gates(s_pre, s_post, t1, t2)
        v = cost_of(a, s, c["dec_s"], c, pre_on_skip)
        if best is None or v < best[0]:
            best = (v, t1, t2)
    return best[1], best[2], best[0]


def load_samples(npz_path, ids, n_use=None):
    """Per-question statistics of the sampled answers (the first n_use of them)."""
    z = np.load(npz_path)
    assert (z["ids"] == ids).all(), "samples must cover the same questions in the same order"
    N = n_use or z["labels"].shape[1]
    labels, n_gen = z["labels"][:, :N], z["n_gen"][:, :N]
    freqs = np.zeros((len(ids), N))            # answer-group frequencies, sorted descending
    maj_ok = np.zeros(len(ids), dtype=int)
    with open(npz_path.replace(".npz", ".keys.jsonl"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            keys = json.loads(line)["keys"][:N]
            cnt = Counter(keys).most_common()
            freqs[i, :len(cnt)] = [c / N for _, c in cnt]
            maj_ok[i] = int(labels[i, keys.index(cnt[0][0])])
    return {"freqs": freqs, "maj_ok": maj_ok, "true_freq": labels.mean(1),
            "ng_sum": n_gen.sum(1).astype(float), "ng_max": n_gen.max(1).astype(float)}


def conformal_cascade_accept(true_freq_cal, freqs_test, alpha):
    """Split conformal over the score 1 - frequency. Accept when the set is a singleton."""
    s = np.sort(1.0 - true_freq_cal)
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.zeros(len(freqs_test), bool)
    q = s[k - 1]
    if q >= 1.0:                  # answers that were never sampled are in the set too
        return np.zeros(len(freqs_test), bool)
    in_set = (freqs_test > 0) & (1.0 - freqs_test <= q + 1e-12)
    return in_set.sum(1) == 1


def question_embeddings(small_npz, ids, encoder, suffix=".qemb.npy", pooling="mean"):
    """Sentence embeddings of the question text (mean pooling, or the normalised first-token
    state for encoders trained that way), cached next to the features."""
    cache = small_npz.replace(".npz", suffix)
    if os.path.exists(cache):
        return np.load(cache)
    import torch
    from transformers import AutoModel, AutoTokenizer
    qs = []
    with open(small_npz.replace(".npz", ".answers.jsonl"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            assert str(r["id"]) == str(ids[i])
            qs.append(r["question"])
    tok, enc = AutoTokenizer.from_pretrained(encoder), AutoModel.from_pretrained(encoder).eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(qs), 64):
            x = tok(qs[b:b + 64], padding=True, truncation=True, max_length=256, return_tensors="pt")
            h = enc(**x).last_hidden_state
            if pooling == "cls":
                out.append(torch.nn.functional.normalize(h[:, 0], dim=-1).numpy())
                continue
            m = x["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1)).numpy())
    emb = np.concatenate(out).astype(np.float32)
    np.save(cache, emb)
    return emb


def fit_router(E, y, tr, va, te):
    """Question-only router. Returns scores on `te` and on the validation split."""
    sc = StandardScaler().fit(E[tr])
    best = (-1, None)
    for C in (1e-3, 1e-2, 1e-1):
        clf = LogisticRegression(C=C, max_iter=1000).fit(sc.transform(E[tr]), y[tr])
        auc = roc_auc_score(y[va], clf.predict_proba(sc.transform(E[va]))[:, 1])
        if auc > best[0]:
            best = (auc, clf)
    f = lambda i: best[1].predict_proba(sc.transform(E[i]))[:, 1]  # noqa
    return f(te), f(va)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", required=True)
    ap.add_argument("--large", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", default=None, help="npz from sample.py for sampling baselines")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.02, 0.05, 0.10, 0.20])
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--popcheck", type=int, default=400,
                    help="calibration draws per seed for the noise-free check of the guarantee")
    ap.add_argument("--cost_ratios", type=float, nargs="+", default=[4.94],
                    help="compute models: per-token cost of large over small")
    ap.add_argument("--latency", nargs=2, default=None, metavar=("SMALL_JSON", "LARGE_JSON"),
                    help="add the latency cost model from these latency.py outputs")
    ap.add_argument("--throughput", nargs=2, default=None, metavar=("SMALL_JSON", "LARGE_JSON"),
                    help="batched timing files: the compute models take the price of a prompt "
                         "token from their batched measurement instead of the single-stream one")
    ap.add_argument("--sens_ratio", type=float, default=None,
                    help="compute ratio at which the prompt weight is also set to 1 and to 0")
    ap.add_argument("--shift", action="store_true",
                    help="MMLU only: also calibrate on one half of the subjects and test on the other")
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="sentence encoder of the question-only router")
    ap.add_argument("--encoder2", default=None,
                    help="larger sentence encoder for a second query-only router (first-token pooling)")
    ap.add_argument("--bert", default=None,
                    help="npz from bert_router.py: scores of a fine-tuned BERT router per seed")
    ap.add_argument("--param_ratio", type=float, default=None,
                    help="parameter ratio: also evaluated with a prompt token priced like a generated one")
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()

    S, L = np.load(args.small), np.load(args.large)
    assert (S["ids"] == L["ids"]).all(), "small and large runs must cover the same questions"
    y, y_l = S["label"].astype(int), L["label"].astype(int)
    n = len(y)
    np_s, ng_s = S["n_prompt"].astype(float), S["n_gen"].astype(float)
    np_l, ng_l = L["n_prompt"].astype(float), L["n_gen"].astype(float)
    samp = load_samples(args.samples, S["ids"]) if args.samples else None
    samp_n = {k: load_samples(args.samples, S["ids"], k) for k in (3, 5)} if args.samples else {}
    E = question_embeddings(args.small, S["ids"], args.encoder)
    E2 = question_embeddings(args.small, S["ids"], args.encoder2, ".qemb2.npy", "cls") if args.encoder2 else None
    BERT = None
    if args.bert:
        BERT = np.load(args.bert)
        assert (BERT["ids"].astype(str) == S["ids"].astype(str)).all(), "BERT scores in another question order"

    # prompt weight: cost of a prompt token in units of a generated token of the same model
    w_s = w_l = 1.0
    if args.latency:
        ls, ll = (json.load(open(p)) for p in args.latency)
        ps, pl = (x["single_prefill_s"] / x["mean_prompt_tokens"] for x in (ls, ll))
        ds, dl = ls["single_decode_s_per_token"], ll["single_decode_s_per_token"]
        w_s, w_l = ps / ds, pl / dl
    w_single = (w_s, w_l)
    if args.throughput:
        ts_, tl_ = (json.load(open(p)) for p in args.throughput)
        # batched serving: the prompt pass is no longer one cheap step. The decode-only value
        # (fix_timing.py) is relative to a decoded token without the prompt pass.
        key = "batched_prompt_weight_decode" if "batched_prompt_weight_decode" in ts_ else "batched_prompt_weight"
        if key in ts_ and key in tl_:
            w_s, w_l = ts_[key], tl_[key]

    def compute_model(r, ws, wl):
        c = {"pre_s": ws * np_s, "dec_s": ng_s, "large": r * (wl * np_l + ng_l),
             "dec_samp": samp["ng_sum"] if samp else ng_s}
        for k, sm in samp_n.items():
            c[f"dec_samp{k}"] = sm["ng_sum"]
        return c

    costs = {f"c{r:g}": compute_model(r, w_s, w_l) for r in args.cost_ratios}
    if args.sens_ratio:
        for v in (0.0, 0.05, 0.1, 0.25, 1.0):
            costs[f"w{v:g}_{args.sens_ratio:g}"] = compute_model(args.sens_ratio, v, v)
        costs[f"wsingle_{args.sens_ratio:g}"] = compute_model(args.sens_ratio, *w_single)
    if args.param_ratio:
        # counting arithmetic work, a prompt token costs as much as a generated token
        costs[f"w1_{args.param_ratio:g}"] = compute_model(args.param_ratio, 1.0, 1.0)
    if args.latency:
        costs["latency"] = {"pre_s": ps * np_s, "dec_s": ds * ng_s, "large": pl * np_l + dl * ng_l,
                            "dec_samp": ds * samp["ng_max"] if samp else ds * ng_s}
        for k, sm in samp_n.items():
            costs["latency"][f"dec_samp{k}"] = ds * sm["ng_max"]

    scal = {k: S[k] for k in ("lp_mean", "lp_min", "lp_sum", "p_first")}
    out = {"n": n, "acc_small": float(y.mean()), "acc_large": float(y_l.mean()),
           "cost_small_only": {k: float(np.mean(c["pre_s"] + c["dec_s"]) / np.mean(c["large"]))
                               for k, c in costs.items()},
           "prompt_weight": {"small": w_s, "large": w_l, "small_single": w_single[0],
                             "large_single": w_single[1]},
           "delta": args.delta, "runs": [], "probe_quality": [], "popcheck": [], "shift": []}
    if samp:
        out["acc_majority"] = float(samp["maj_ok"].mean())
        # share of questions for which none of the N samples is correct
        out["frac_never_correct"] = float((samp["true_freq"] == 0).mean())

    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        fit, pool = perm[:n // 2], perm[n // 2:]
        tr, va = fit[:int(.8 * len(fit))], fit[int(.8 * len(fit)):]
        sc, info = {}, {}
        val = {}
        for name in ("h_pre", "h_post", "h_mean"):
            sc[name], info[name], val[name] = probe(S[name], y, tr, va, pool, args.jobs, True)
        post = max(("h_post", "h_mean"), key=lambda k: info[k]["val_auroc"])
        tokname = max(scal, key=lambda k: roc_auc_score(y[va], scalar_baseline(scal[k], y, tr, va)))
        sc["token"] = scalar_baseline(scal[tokname], y, fit, pool)
        sc["length"] = scalar_baseline(S["n_gen"].astype(float), y, fit, pool)
        # learned deferral gate on all token statistics and the answer length
        F = np.column_stack([scal[k] for k in sorted(scal)] + [S["n_gen"].astype(float)])
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=-1e4)
        fsc = StandardScaler().fit(F[fit])
        fclf = LogisticRegression(C=1.0, max_iter=2000).fit(fsc.transform(F[fit]), y[fit])
        sc["feat"] = fclf.predict_proba(fsc.transform(F[pool]))[:, 1]
        sc["query"], val["query"] = fit_router(E, y, tr, va, pool)
        if E2 is not None:
            sc["query2"], val["query2"] = fit_router(E2, y, tr, va, pool)
        if BERT is not None:
            # fine-tuned on the train part of this seed's fit half (bert_router.py, same split)
            sc["bert"] = BERT["scores"][seed][pool].astype(float)
        grids = {"h_pre": skip_grid(val["h_pre"]), "query": skip_grid(val["query"])}
        grids_m = {k: skip_grid(val["h_pre"], qs) for k, qs in T1_ABLATION.items()}
        yp = y[pool]
        q = {"seed": seed, "post_probe": post, "token_baseline": tokname,
             "pre_layer": info["h_pre"]["layer"], "post_layer": info[post]["layer"],
             "val_auroc_by_layer": {k: info[k]["val_auroc_by_layer"] for k in info}}
        for nm, key in (("pre", "h_pre"), ("post", post), ("token", "token"),
                        ("length", "length"), ("query", "query"), ("feat", "feat"),
                        ("query2", "query2"), ("bert", "bert")):
            if key not in sc:
                continue
            q[nm] = {"auroc": float(roc_auc_score(yp, sc[key])), "ece": ece(sc[key], yp),
                     "brier": float(brier_score_loss(yp, sc[key]))}
        if samp:
            q["sc_agreement"] = {"auroc": float(
                roc_auc_score(samp["maj_ok"][pool], samp["freqs"][pool, 0]))}
        out["probe_quality"].append(q)
        print({k: (round(v["auroc"], 3) if isinstance(v, dict) and "auroc" in v else v)
               for k, v in q.items() if k != "val_auroc_by_layer"}, flush=True)

        ypool, ylpool = y[pool], y_l[pool]
        cpool = {k: take(c, pool) for k, c in costs.items()}
        if samp:
            fpool, majpool = samp["freqs"][pool], samp["maj_ok"][pool]
            tfpool = samp["true_freq"][pool]
            fpool_n = {k: (sm["freqs"][pool], sm["maj_ok"][pool]) for k, sm in samp_n.items()}
        m = len(pool)
        for rep in range(args.reps):
            p2 = rng.permutation(m)
            ca, te = p2[:m // 2], p2[m // 2:]
            wrong_c, nc = (ypool[ca] == 0).astype(int), len(ca)
            ones, noskip = np.ones(nc, bool), np.zeros(len(te), bool)
            cc = {k: take(c, ca) for k, c in cpool.items()}
            ct = {k: take(c, te) for k, c in cpool.items()}
            err_cal = float(wrong_c.mean())
            all_c, none_c = np.ones(nc, bool), np.zeros(nc, bool)
            all_t, none_t = np.ones(len(te), bool), np.zeros(len(te), bool)
            for alpha in args.alphas:
                fs = lambda s, d=args.delta: fixed_sequence(s[ca], wrong_c, ones, nc, alpha, d)  # noqa
                # single-gate methods: thresholds do not depend on the cost model.
                # entries: skip score, accept score, t1, t2, small prompt pass paid on skips
                single = {
                    "accept_only": ("h_pre", post, -np.inf, fs(sc[post]), True),
                    "token_gate": ("h_pre", "token", -np.inf, fs(sc["token"]), True),
                    "length_gate": ("h_pre", "length", -np.inf, fs(sc["length"]), True),
                    "feature_gate": ("h_pre", "feat", -np.inf, fs(sc["feat"]), True),
                    "route_only": ("h_pre", post, fs(sc["h_pre"]), -np.inf, True),
                    "query_router": ("query", post, fs(sc["query"]), -np.inf, False),
                    **({"query_router_big": ("query2", post, fs(sc["query2"]), -np.inf, False)}
                       if "query2" in sc else {}),
                    **({"bert_router": ("bert", post, fs(sc["bert"]), -np.inf, False)}
                       if "bert" in sc else {}),
                    "accept_uncorrected": ("h_pre", post, -np.inf, empirical_threshold(
                        sc[post][ca], wrong_c, nc, alpha), True),
                }
                # rows: accept/skip on test, labels of kept answers, decoding-cost key,
                #       prompt pass paid on skips, accept/skip on calibration (for its cost)
                res = {}
                for meth, (a, b, t1, t2, pos) in single.items():
                    acc, skp = gates(sc[a][te], sc[b][te], t1, t2)
                    res[meth] = (acc, skp, ypool[te], "dec_s", pos, gates(sc[a][ca], sc[b][ca], t1, t2))
                res["always_large"] = (none_t, all_t, ypool[te], "dec_s", False, (none_c, all_c))
                res["always_small"] = (all_t, none_t, ypool[te], "dec_s", True, (all_c, none_c))
                res["oracle"] = (ypool[te] == 1, ypool[te] == 0, ypool[te], "dec_s", False,
                                 (ypool[ca] == 1, ypool[ca] == 0))
                if samp:
                    acc = conformal_cascade_accept(tfpool[ca], fpool[te], alpha)
                    acc_c = conformal_cascade_accept(tfpool[ca], fpool[ca], alpha)
                    res["conformal_cascade"] = (acc, noskip, majpool[te], "dec_samp", True, (acc_c, none_c))
                    t = fixed_sequence(fpool[ca, 0], (majpool[ca] == 0).astype(int), ones, nc,
                                       alpha, args.delta)
                    res["sc_gate"] = (fpool[te, 0] >= t, noskip, majpool[te], "dec_samp", True,
                                      (fpool[ca, 0] >= t, none_c))
                    for k, (fk, mk) in fpool_n.items():
                        t = fixed_sequence(fk[ca, 0], (mk[ca] == 0).astype(int), ones, nc,
                                           alpha, args.delta)
                        res[f"sc_gate_n{k}"] = (fk[te, 0] >= t, noskip, mk[te], f"dec_samp{k}", True,
                                                (fk[ca, 0] >= t, none_c))
                # two-gate methods: skip score, accept score, grid, prompt pass paid on skips
                two = {"skipgate": ("h_pre", post, grids["h_pre"], True),
                       "twogate_noprobe": ("query", "token", grids["query"], False)}
                for k, g in grids_m.items():
                    two[f"skipgate_m{k}"] = ("h_pre", post, g, True)
                cands = {meth: two_gate_candidates(sc[a][ca], sc[b][ca], wrong_c, g, alpha, args.delta)
                         for meth, (a, b, g, _) in two.items()}
                two["skipgate_ws"] = ("h_pre", post, grids["h_pre"], True)
                cands["skipgate_ws"] = weighted_candidates(sc["h_pre"][ca], sc[post][ca], wrong_c,
                                                           grids["h_pre"], alpha, args.delta)
                # best single gate: each calibrated at delta / 2, the cheaper one is used
                half = [(-np.inf, fs(sc[post], args.delta / 2)), (fs(sc["h_pre"], args.delta / 2), -np.inf)]
                q_keep = min(1.0, alpha / max(err_cal, 1e-9))
                # certified variant: Clopper-Pearson upper bound on the error rate at level delta
                k_err = int(wrong_c.sum())
                e_up = 1.0 if k_err == nc else float(beta.ppf(1 - args.delta, k_err + 1, nc - k_err))
                q_cert = min(1.0, alpha / max(e_up, 1e-9))
                for cm in costs:
                    rows = dict(res)
                    for meth, (a, b, g, pos) in two.items():
                        t1, t2, _ = cheapest(cands[meth], sc[a][ca], sc[b][ca], cc[cm], pos)
                        rows[meth] = (*gates(sc[a][te], sc[b][te], t1, t2), ypool[te], "dec_s", pos,
                                      gates(sc[a][ca], sc[b][ca], t1, t2))
                    # weighted split plus the free candidate "always large" (cost exactly 1)
                    t1, t2, v = cheapest(cands["skipgate_ws"], sc["h_pre"][ca], sc[post][ca], cc[cm])
                    rows["skipgate_wf"] = res["always_large"] if v >= 1.0 else rows["skipgate_ws"]
                    t1, t2, _ = cheapest(half, sc["h_pre"][ca], sc[post][ca], cc[cm])
                    rows["best_single"] = (*gates(sc["h_pre"][te], sc[post][te], t1, t2), ypool[te],
                                           "dec_s", True, gates(sc["h_pre"][ca], sc[post][ca], t1, t2))
                    for meth, (acc, skp, yk, deck, pos, (acc_c, skp_c)) in rows.items():
                        out["runs"].append({
                            "seed": seed, "rep": rep, "alpha": alpha, "cm": cm, "method": meth,
                            **quality(acc, skp, yk, ylpool[te]),
                            "cost": cost_of(acc, skp, ct[cm][deck], ct[cm], pos),
                            "cost_cal": cost_of(acc_c, skp_c, cc[cm][deck], cc[cm], pos)})
                    # random routing at the same target, in expectation over its coin flips
                    for meth, q in (("random_route", q_keep), ("random_cert", q_cert)):
                        out["runs"].append({
                            "seed": seed, "rep": rep, "alpha": alpha, "cm": cm, "method": meth,
                            **random_row(q, ypool[te] == 0, ypool[te], ylpool[te], ct[cm]),
                            "cost_cal": random_row(q, wrong_c, ypool[ca], ylpool[ca], cc[cm])["cost"]})

        # Population check of the guarantee without test noise. The held-out pool is treated
        # as the population: calibration sets are drawn from it i.i.d. (with replacement) and
        # the risk of the calibrated thresholds is computed exactly on the whole pool.
        wrong_pool = ypool == 0
        for b in range(args.popcheck):
            ix = rng.integers(0, m, m // 2)
            wrong_c, nc = wrong_pool[ix].astype(int), len(ix)
            ones = np.ones(nc, bool)
            for alpha in args.alphas:
                fs = lambda s, d=args.delta: fixed_sequence(s[ix], wrong_c, ones, nc, alpha, d)  # noqa
                single = {
                    "accept_only": ("h_pre", post, -np.inf, fs(sc[post])),
                    "token_gate": ("h_pre", "token", -np.inf, fs(sc["token"])),
                    "feature_gate": ("h_pre", "feat", -np.inf, fs(sc["feat"])),
                    "route_only": ("h_pre", post, fs(sc["h_pre"]), -np.inf),
                    "accept_uncorrected": ("h_pre", post, -np.inf, empirical_threshold(
                        sc[post][ix], wrong_c, nc, alpha)),
                }
                for meth, (a, bb, t1, t2) in single.items():
                    acc, _ = gates(sc[a], sc[bb], t1, t2)
                    out["popcheck"].append({"cm": "-", "alpha": alpha, "method": meth,
                                            "risk": float(np.mean(acc & wrong_pool))})
                k_err = int(wrong_c.sum())
                e_up = 1.0 if k_err == nc else float(beta.ppf(1 - args.delta, k_err + 1, nc - k_err))
                out["popcheck"].append({"cm": "-", "alpha": alpha, "method": "random_cert",
                                        "risk": min(1.0, alpha / max(e_up, 1e-9)) * float(wrong_pool.mean())})
                pc_cands = {
                    "skipgate": two_gate_candidates(sc["h_pre"][ix], sc[post][ix], wrong_c,
                                                    grids["h_pre"], alpha, args.delta),
                    "skipgate_ws": weighted_candidates(sc["h_pre"][ix], sc[post][ix], wrong_c,
                                                       grids["h_pre"], alpha, args.delta),
                    "best_single": [(-np.inf, fs(sc[post], args.delta / 2)),
                                    (fs(sc["h_pre"], args.delta / 2), -np.inf)]}
                for cm in costs:
                    cix = take(cpool[cm], ix)
                    for meth, cd in pc_cands.items():
                        t1, t2, v = cheapest(cd, sc["h_pre"][ix], sc[post][ix], cix)
                        acc, _ = gates(sc["h_pre"], sc[post], t1, t2)
                        risk = float(np.mean(acc & wrong_pool))
                        out["popcheck"].append({"cm": cm, "alpha": alpha, "method": meth, "risk": risk})
                        if meth == "skipgate_ws":
                            out["popcheck"].append({"cm": cm, "alpha": alpha, "method": "skipgate_wf",
                                                    "risk": 0.0 if v >= 1.0 else risk})

        # Shift check (MMLU): calibrate on one half of the subjects, test on the other half.
        # The subjects of calibration and test questions are disjoint, so exchangeability
        # does not hold by construction.
        if args.shift:
            subj = np.array([str(i).rsplit("-", 1)[0] for i in S["ids"]])[pool]
            us = np.unique(subj)
            for rep in range(args.reps):
                half_s = set(us[rng.permutation(len(us))[:len(us) // 2]])
                in_a = np.array([u in half_s for u in subj])
                ca, te = np.where(in_a)[0], np.where(~in_a)[0]
                wrong_c, nc = (ypool[ca] == 0).astype(int), len(ca)
                ones = np.ones(nc, bool)
                for alpha in args.alphas:
                    t2 = fixed_sequence(sc[post][ca], wrong_c, ones, nc, alpha, args.delta)
                    acc, skp = gates(sc["h_pre"][te], sc[post][te], -np.inf, t2)
                    out["shift"].append({"seed": seed, "rep": rep, "alpha": alpha, "cm": "-",
                                         "method": "accept_only", "n_cal": nc,
                                         "risk": float(np.mean(acc & (ypool[te] == 0)))})
                    cands = two_gate_candidates(sc["h_pre"][ca], sc[post][ca], wrong_c,
                                                grids["h_pre"], alpha, args.delta)
                    for cm in costs:
                        t1, t2, _ = cheapest(cands, sc["h_pre"][ca], sc[post][ca], take(cpool[cm], ca))
                        acc, skp = gates(sc["h_pre"][te], sc[post][te], t1, t2)
                        out["shift"].append({"seed": seed, "rep": rep, "alpha": alpha, "cm": cm,
                                             "method": "skipgate", "n_cal": nc,
                                             "risk": float(np.mean(acc & (ypool[te] == 0)))})

    sh = {}
    for r in out["shift"]:
        sh.setdefault((r["cm"], r["alpha"], r["method"]), []).append(r["risk"])
    out["shift_summary"] = {f"{cm}/a={a}/{m}": {"violation_rate": float(np.mean(np.array(v) > a)),
                                                 "risk": float(np.mean(v)), "n": len(v)}
                            for (cm, a, m), v in sorted(sh.items())}
    del out["shift"]

    # population check summary: share of calibration draws whose exact risk exceeds alpha
    pc = {}
    for r in out["popcheck"]:
        pc.setdefault((r["cm"], r["alpha"], r["method"]), []).append(r["risk"])
    out["popcheck_summary"] = {f"{cm}/a={a}/{m}": {"violation_rate": float(np.mean(np.array(v) > a)),
                                                    "risk": float(np.mean(v)), "n": len(v)}
                               for (cm, a, m), v in sorted(pc.items())}
    del out["popcheck"]

    groups = {}
    for r in out["runs"]:
        groups.setdefault((r["cm"], r["alpha"], r["method"]), []).append(r)
    for k, v in out["popcheck_summary"].items():
        if "/a=0.05/" in k and (k.startswith("-") or k.startswith(list(costs)[0] + "/")):
            print(f"popcheck {k:40s} viol {v['violation_rate']:.3f} risk {v['risk']:.4f}")
    summ = {}
    for (cm, alpha, meth), rs in sorted(groups.items()):
        risk = np.array([r["risk"] for r in rs])
        summ[f"{cm}/a={alpha}/{meth}"] = {
            "risk": float(risk.mean()), "violation_rate": float((risk > alpha).mean()),
            **{k: float(np.mean([r[k] for r in rs])) for k in ("cost", "acc", "accept", "skip")}}
    out["summary"] = summ
    # store the per-run rows column-wise to keep the file small
    cols = list(out["runs"][0])
    out["runs"] = {c: [round(r[c], 5) if isinstance(r[c], float) else r[c] for r in out["runs"]]
                   for c in cols}
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"acc small {out['acc_small']:.3f} large {out['acc_large']:.3f} "
          f"small-only cost {out['cost_small_only']}"
          + (f"  majority-vote acc {out['acc_majority']:.3f}" if samp else ""))
    for k, v in summ.items():
        print(f"{k:44s} risk {v['risk']:.4f} viol {v['violation_rate']:.3f} cost {v['cost']:.3f} "
              f"acc {v['acc']:.3f} accept {v['accept']:.3f} skip {v['skip']:.3f}")


if __name__ == "__main__":
    main()
