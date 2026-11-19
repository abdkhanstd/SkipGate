"""Dataset loading, prompts and correctness scoring for SkipGate."""
import os
import random
import re
import string

from datasets import load_dataset

TRIVIA_SYS = "Answer the question with a short phrase only. Do not explain."
GSM_SYS = (
    "Solve the math problem step by step. "
    "Finish with a final line of the form 'The answer is <number>'."
)


def normalize_text(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def load_triviaqa(n, seed=0):
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    seen, items = set(), []
    for ex in ds:
        key = normalize_text(ex["question"])
        if key in seen:
            continue
        seen.add(key)
        aliases = list(ex["answer"]["normalized_aliases"]) + [
            normalize_text(ex["answer"]["value"])
        ]
        aliases = sorted({a for a in aliases if a})
        items.append({"id": ex["question_id"], "question": ex["question"], "gold": aliases})
    random.Random(seed).shuffle(items)
    return items[:n]


def _gsm_gold(ans):
    return ans.split("####")[-1].strip().replace(",", "")


def load_gsm8k(n, seed=0):
    items, seen = [], set()
    for split in ("train", "test"):
        ds = load_dataset("openai/gsm8k", "main", split=split)
        for i, ex in enumerate(ds):
            key = normalize_text(ex["question"])
            if key in seen:
                continue
            seen.add(key)
            items.append(
                {"id": f"{split}-{i}", "question": ex["question"], "gold": _gsm_gold(ex["answer"])}
            )
    random.Random(seed).shuffle(items)
    return items[:n]


MMLU_SYS = (
    "Answer the multiple choice question. Reason briefly step by step. "
    "Finish with a final line of the form 'The answer is (X)' where X is A, B, C or D."
)


def load_mmlu(n, seed=0):
    # MMLU_PARQUET points at a local copy of cais/mmlu all/test-00000-of-00001.parquet,
    # for machines where the Hub cannot be reached through the datasets library
    local = os.environ.get("MMLU_PARQUET")
    if local:
        ds = load_dataset("parquet", data_files=local, split="train")
    else:
        ds = load_dataset("cais/mmlu", "all", split="test")
    items, seen = [], set()
    for i, ex in enumerate(ds):
        key = normalize_text(ex["question"] + " " + " ".join(ex["choices"]))
        if key in seen:
            continue
        seen.add(key)
        q = ex["question"].strip() + "\n" + "\n".join(
            f"({l}) {c}" for l, c in zip("ABCD", ex["choices"]))
        items.append({"id": f"{ex['subject']}-{i}", "question": q, "gold": "ABCD"[ex["answer"]]})
    random.Random(seed).shuffle(items)
    return items[:n]


def extract_letter(pred):
    m = re.findall(r"answer is[^A-Da-d]{0,5}\(?([A-D])\)?(?![a-z])", pred)
    if m:
        return m[-1]
    m = re.findall(r"\(([A-D])\)", pred)
    return m[-1] if m else None


def score_mmlu(pred, gold):
    return int(extract_letter(pred) == gold)


def system_prompt(task):
    return {"triviaqa": TRIVIA_SYS, "gsm8k": GSM_SYS, "mmlu": MMLU_SYS}[task]


def max_new_tokens(task):
    return 32 if task == "triviaqa" else 640


def score_triviaqa(pred, gold_aliases):
    p = normalize_text(pred.split("\n")[0])
    if not p:
        return 0
    for a in gold_aliases:
        if p == a or re.search(r"\b" + re.escape(a) + r"\b", p):
            return 1
    return 0


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_number(pred):
    m = re.findall(r"answer is[^0-9\-]*(-?\d[\d,]*\.?\d*)", pred, flags=re.I)
    cand = m[-1] if m else None
    if cand is None:
        nums = _NUM.findall(pred)
        cand = nums[-1] if nums else None
    if cand is None:
        return None
    cand = cand.replace(",", "").rstrip(".")
    try:
        return float(cand)
    except ValueError:
        return None


def score_gsm8k(pred, gold):
    v = extract_number(pred)
    if v is None:
        return 0
    try:
        return int(abs(v - float(gold)) < 1e-4)
    except ValueError:
        return 0


def load_task(task, n, seed=0, offset=0):
    """The n questions that follow the first `offset` questions of the shuffled dataset.
    offset=0 gives the main question set, offset=4000 a disjoint set of fresh questions."""
    fn = {"triviaqa": load_triviaqa, "gsm8k": load_gsm8k, "mmlu": load_mmlu}[task]
    return fn(n + offset, seed)[offset:]


def score(task, pred, gold):
    return {"triviaqa": score_triviaqa, "gsm8k": score_gsm8k, "mmlu": score_mmlu}[task](pred, gold)
