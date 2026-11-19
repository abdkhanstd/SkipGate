"""Stage 0: do hidden-state probes predict correctness better than token probabilities?

For each seed the questions are split 60/20/20 into train/val/test. A logistic probe is
fitted per layer on train, the layer and regularisation are chosen on val, and all numbers
are reported on test. Token-probability baselines are calibrated on train+val by a
one-dimensional logistic fit, so they get the same data as the probes.
"""
import argparse
import json

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

CS = (1e-3, 1e-2, 1e-1)


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def fit_layer(X, y, tr, va, layer):
    sc = StandardScaler().fit(X[tr, layer])
    xtr, xva = sc.transform(X[tr, layer]), sc.transform(X[va, layer])
    best = (-1, None, None)
    for C in CS:
        clf = LogisticRegression(C=C, max_iter=1000).fit(xtr, y[tr])
        auc = roc_auc_score(y[va], clf.predict_proba(xva)[:, 1])
        if auc > best[0]:
            best = (auc, C, clf)
    return layer, best[0], best[1], sc, best[2]


def probe(X, y, tr, va, te, jobs, return_val=False):
    X = X.astype(np.float32)
    res = Parallel(n_jobs=jobs)(delayed(fit_layer)(X, y, tr, va, l) for l in range(X.shape[1]))
    layer, vauc, C, sc, clf = max(res, key=lambda r: r[1])
    p = clf.predict_proba(sc.transform(X[te, layer]))[:, 1]
    info = {"layer": int(layer), "C": C, "val_auroc": float(vauc),
            "val_auroc_by_layer": [float(r[1]) for r in sorted(res, key=lambda r: r[0])]}
    if return_val:   # scores on the validation split, used to place the skip-gate grid
        return p, info, clf.predict_proba(sc.transform(X[va, layer]))[:, 1]
    return p, info


def scalar_baseline(s, y, trva, te):
    s = s.reshape(-1, 1).astype(np.float64)
    sc = StandardScaler().fit(s[trva])
    clf = LogisticRegression(C=1e3, max_iter=1000).fit(sc.transform(s[trva]), y[trva])
    return clf.predict_proba(sc.transform(s[te]))[:, 1]


def metrics(p, y):
    return {"auroc": float(roc_auc_score(y, p)), "ece": ece(p, y),
            "brier": float(brier_score_loss(y, p))}


def paired_boot(pa, pb, y, rng, B=2000):
    """Bootstrap CI and two-sided p-value for AUROC(a) - AUROC(b) on the same test items."""
    n, d = len(y), []
    while len(d) < B:
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    d = np.asarray(d)
    pval = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"diff": float(roc_auc_score(y, pa) - roc_auc_score(y, pb)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p": float(min(1.0, max(pval, 1.0 / B)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()

    d = np.load(args.npz)
    y = d["label"].astype(int)
    n = len(y)
    scal = {"lp_mean": d["lp_mean"], "lp_min": d["lp_min"], "lp_sum": d["lp_sum"],
            "p_first": d["p_first"]}
    out = {"n": n, "accuracy": float(y.mean()), "seeds": []}
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        tr, va, te = perm[:int(.6 * n)], perm[int(.6 * n):int(.8 * n)], perm[int(.8 * n):]
        trva = np.concatenate([tr, va])
        r, preds = {}, {}
        for name in ("h_pre", "h_post", "h_mean"):
            preds[name], info = probe(d[name], y, tr, va, te, args.jobs)
            r[name] = {**metrics(preds[name], y[te]), **info}
        for name, s in scal.items():
            preds[name] = scalar_baseline(s, y, trva, te)
            r[name] = metrics(preds[name], y[te])
        best_tok = max(scal, key=lambda k: roc_auc_score(y[va], scalar_baseline(scal[k], y, tr, va)))
        best_post = max(("h_post", "h_mean"), key=lambda k: r[k]["val_auroc"])
        r["best_token_baseline"] = best_tok
        r["best_post_probe"] = best_post
        r["post_vs_token"] = paired_boot(preds[best_post], preds[best_tok], y[te], rng)
        r["pre_vs_token"] = paired_boot(preds["h_pre"], preds[best_tok], y[te], rng)
        out["seeds"].append(r)
        print(seed, {k: round(v["auroc"], 4) for k, v in r.items()
                     if isinstance(v, dict) and "auroc" in v}, flush=True)

    summ = {}
    for k in ("h_pre", "h_post", "h_mean", *scal):
        for m in ("auroc", "ece", "brier"):
            v = np.array([s[k][m] for s in out["seeds"]])
            summ[f"{k}/{m}"] = [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0]
    out["summary_mean_std"] = summ
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
