"""Run calibrate.py for every model pair and dataset that has its inputs on disk.

Each run evaluates a sweep of compute cost ratios, the measured throughput ratio, the
parameter ratio, the measured single-stream latency model and a sensitivity sweep over the
price of a prompt token.

Usage
  python run_all_calib.py                          main experiments -> results/final
  python run_all_calib.py --set fresh              confirmation run on fresh questions
                                                   (features in results/fresh) -> results/final_fresh
  python run_all_calib.py --set llama              second model family -> results/final
"""
import argparse
import json
import os
import subprocess
import sys

R = "../results"
PARAMS = {"qwen1.5b": 1.54, "qwen7b": 7.62, "qwen14b": 14.77,       # billions
          "llama3b": 3.21, "llama8b": 8.03}
SWEEP = [1, 1.5, 2, 3, 5, 7.5, 10, 15]
# larger sentence encoder of the second query-only router (local copy of BAAI/bge-base-en-v1.5)
ENCODER2 = os.environ.get("SKIPGATE_ENCODER2", os.path.expanduser("~/skipgate/models/bge-base-en-v1.5"))
# optional local copy of the small sentence encoder, and the folder for the calibration logs
ENCODER = os.environ.get("SKIPGATE_ENCODER")
LOGS = os.environ.get("SKIPGATE_LOGS", "../../logs")
SETS = {
    # name: (pairs, tasks, feature folders in order of preference, output folder)
    "main": ((("qwen1.5b", "qwen7b"), ("qwen1.5b", "qwen14b"), ("qwen7b", "qwen14b")),
             ("gsm8k", "mmlu", "triviaqa"), ("stage4", "stage2", "stage0", "stage1"), "final"),
    "fresh": ((("qwen1.5b", "qwen7b"),), ("gsm8k", "mmlu"), ("fresh",), "final_fresh"),
    "llama": ((("llama3b", "llama8b"),), ("gsm8k", "mmlu"), ("stage6",), "final"),
}


def timing(name, first=False):
    """Timing file: the repeated fixed-length measurement (stage 5) if present, else the
    first measurement (stage 2, which used upper-case B in throughput file names).
    first=True forces the first measurement. It is used for both models of a pair with the
    14B model, whose repeated timing ran out of GPU memory next to another process."""
    new = f"{R}/stage5/{name}"
    if os.path.exists(new) and not first:
        return new
    return f"{R}/stage2/" + name.replace("b_bs", "B_bs")


def feat(model, task, folders):
    """Path of the run of `model` on `task` (hidden states are needed only for small models)."""
    cands = [f"{R}/{st}/{model}_{task}.npz" for st in folders]
    return next((p for p in cands if os.path.exists(p)), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="main", choices=sorted(SETS))
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    pairs, tasks, folders, out_dir = SETS[args.set]
    os.makedirs(f"{R}/{out_dir}", exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               HF_HUB_OFFLINE="1")
    cmds = []
    for small, large in pairs:
        for task in tasks:
            name = f"{small}-{large}_{task}"
            if args.only and name not in args.only:
                continue
            s, l = feat(small, task, folders), feat(large, task, folders)
            out = f"{R}/{out_dir}/{name}.json"
            if not s or not l or (os.path.exists(out) and not args.force):
                print("skip", name, "(missing input or already done)")
                continue
            if "h_pre" not in __import__("numpy").load(s).files:
                print("skip", name, "(no hidden states for the small model)")
                continue
            ratios = set(SWEEP) | {round(PARAMS[large] / PARAMS[small], 2)}
            sens = None
            thr_files = None
            first = "qwen14b" in (small, large)
            # measured throughput ratio at the largest batch size that both models could run
            for bs in (128, 64, 32):
                ts, tl = (timing(f"throughput_{m}_bs{bs}.json", first) for m in (small, large))
                if os.path.exists(ts) and os.path.exists(tl):
                    # decode-only rate (fix_timing.py) where it exists: the prompt pass is
                    # charged separately through the prompt price
                    a, b = (json.load(open(p)).get("batched_decode_tokens_per_s",
                                                   json.load(open(p))["batched_gen_tokens_per_s"])
                            for p in (ts, tl))
                    ratios.add(round(a / b, 2))
                    sens = round(a / b, 2)
                    thr_files = [ts, tl]
                    break
            cmd = [sys.executable, "calibrate.py", "--small", s, "--large", l, "--out", out,
                   "--jobs", str(args.jobs), "--cost_ratios", *[str(r) for r in sorted(ratios)]]
            if sens:
                cmd += ["--sens_ratio", str(sens)]
            if thr_files:
                cmd += ["--throughput", *thr_files]
            cmd += ["--param_ratio", str(round(PARAMS[large] / PARAMS[small], 2))]
            if os.path.isdir(ENCODER2):
                cmd += ["--encoder2", ENCODER2]
            if ENCODER:
                cmd += ["--encoder", ENCODER]
            bert = s.replace(".npz", ".bert.npz")
            if os.path.exists(bert):
                cmd += ["--bert", bert]
            if task == "mmlu":
                cmd += ["--shift"]
            ls, ll = (timing(f"latency_{m}.json", first) for m in (small, large))
            if os.path.exists(ls) and os.path.exists(ll):
                cmd += ["--latency", ls, ll]
            samp = next((q for q in (f"{R}/{st}/samples_{small}_{task}.npz" for st in folders)
                         if os.path.exists(q)), None)
            if samp:
                cmd += ["--samples", samp]
            cmds.append((name, cmd))
    running = []
    for name, cmd in cmds:
        print("start", name, flush=True)
        running.append((name, subprocess.Popen(
            cmd, env=env, stdout=open(f"{LOGS}/{out_dir}_{name}.log", "w"),
            stderr=subprocess.STDOUT)))
        if len(running) >= args.parallel:
            n0, p0 = running.pop(0)
            print("done", n0, p0.wait(), flush=True)
    for n0, p0 in running:
        print("done", n0, p0.wait(), flush=True)
    print("ALL_CALIB_DONE")


if __name__ == "__main__":
    main()
