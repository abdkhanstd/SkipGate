# SkipGate

Source code and raw experiment outputs accompanying the SkipGate paper.

## What it does

SkipGate routes each question either to a cheap language model or to an expensive one.
Neither model is trained or modified. Instead, two logistic-regression probes look at
activations that the cheap model produces anyway during its one greedy generation:

| Probe | Input activation | Decision |
|-------|------------------|----------|
| skip probe | hidden vector at the final token of the prompt | hand the question straight to the expensive model, before any generation |
| accept probe | hidden vectors of the generated response | return the cheap model's response, or escalate |

The two cut-offs are picked together by a Learn-then-Test search: a fixed-sequence test on
an exact binomial p-value, with a Bonferroni correction across a preset list of skip
cut-offs. The procedure promises, with confidence `1 - delta` over the calibration draw,
that P(response returned by the cheap model AND incorrect) <= `alpha`. Note that this is a
joint probability over all questions, not an error rate conditional on acceptance. The
promise assumes calibration and deployment questions are exchangeable (i.i.d.). In our
cross-subject MMLU test it did not hold, so recalibrate for every new domain.

**Models.** Qwen2.5-Instruct at 1.5B, 7B and 14B parameters, plus a Llama pair
(Llama-3.2-3B-Instruct as the cheap model, Llama-3.1-8B-Instruct as the expensive one).

**Benchmarks.** GSM8K, MMLU and TriviaQA. For GSM8K we merge the official train and test
splits into one pool, since no model here is fine-tuned on it. GSM8K numbers are therefore
not test-split-only numbers.

## Repository map

```
skipgate/
  data.py            loading of the benchmarks, prompt templates, answer checking
  extract.py         greedy generation per question; stores response, correctness, activations
                     and token statistics (accept-probe activation = the step emitting EOS,
                     or the final step when the token limit is hit)
  sample.py          multiple stochastic responses per question (sampling-based baselines)
  latency.py         timing with a fixed number of new tokens, single-stream and batched, repeated
  latency_v1.py      earlier timing script (results/stage2); the 14B pairs rely on it
  fix_timing.py      strips prompt processing from the stage-5 batched measurements
  bert_router.py     fine-tuned BERT router (RouteLLM-style); its scores feed calibrate.py --bert
  download_models.py pulls model weights onto an additional machine
  probe_stage0.py    probe training, with layer and L2 strength chosen on held-out data
  calibrate.py       threshold search, every baseline and ablation, validity checks
  run_all_calib.py   loops calibrate.py over all model pairs and benchmarks
run_stage*.sh        shell launchers for the individual experiment stages
results/
  final/             gzipped JSON per (pair, benchmark) covering every calibration/test draw
  final_fresh/       same format, for the newly drawn questions
  stage0 .. stage6/, fresh/  per-question responses (jsonl), run metadata, timing logs
```

## Installation

```bash
conda create -n skipgate python=3.11
conda activate skipgate
pip install -r requirements.txt
```

Hardware: a server with two 24 GB RTX 3090s. The 1.5B and 7B models fit on a single GPU,
while the 14B model is sharded across both (`--multi_gpu`). Part of the stage-6 extraction
ran on a workstation with two RTX 2080 Ti GPUs (`run_stage6_local.sh`).

## Running the pipeline

```bash
cd skipgate
# step 1: activations of the cheap model and responses of the expensive model
python extract.py --model Qwen/Qwen2.5-1.5B-Instruct --task gsm8k --out ../results/stage4/qwen1.5b_gsm8k.npz
python extract.py --model Qwen/Qwen2.5-7B-Instruct --task gsm8k --no_hidden --out ../results/stage4/qwen7b_gsm8k.npz
# step 2 (optional): stochastic responses for the sampling-based baselines
python sample.py --model Qwen/Qwen2.5-1.5B-Instruct --task gsm8k --out ../results/stage4/samples_qwen1.5b_gsm8k.npz
# step 3 (optional): timing, from which the cost ratios are derived
python latency.py --model Qwen/Qwen2.5-1.5B-Instruct --tag qwen1.5b --out_dir ../results/stage5
# step 4: calibration plus all comparisons, for every pair and benchmark
python run_all_calib.py
```

Valid `--task` values: `gsm8k`, `mmlu`, `triviaqa`. Without access to the Hugging Face Hub,
point the `MMLU_PARQUET` environment variable at a local copy of MMLU `all/test`.

## Output format

Each `results/final/<small>-<large>_<task>.json` (gzip-compressed, restore with `gunzip -k`)
contains:

- `runs`: a column-oriented table, one row per combination of probe seed (5),
  calibration/test draw (100), `alpha`, cost model `cm` and `method`. Columns include the
  test `risk` (returned and incorrect), `acc`, the fraction returned by the cheap model
  (`accept`), the fraction skipped (`skip`), the test `cost` normalised by the
  always-expensive cost, and `cost_cal` measured on the calibration draw.
- `probe_quality`: per-seed AUROC, ECE and Brier score of every confidence signal.
- `popcheck_summary`: guarantee check against an exactly known population risk.
- `shift_summary` (MMLU only): calibration and test taken from non-overlapping subjects.

Cost models are named `c<r>` (per-token cost ratio `r`, prompt tokens at their measured
price), `w<v>_<r>` (a prompt token priced as `v` generated tokens, v in 0, 0.05, 0.1, 0.25, 1)
and `latency` (measured single-stream seconds).

Method keys: `skipgate` (proposed), `accept_only`, `route_only` (skip probe alone),
`best_single`, `skipgate_m2/m5/m20`, `skipgate_ws`, `twogate_noprobe`, `feature_gate`,
`random_cert`, `token_gate`, `query_router`, `length_gate`, `sc_gate`, `sc_gate_n3/n5`,
`conformal_cascade`, `accept_uncorrected`, with `always_large`, `always_small`,
`random_route` and `oracle` as reference points. See the docstrings in `calibrate.py`.
`conformal_cascade` re-implements only the acceptance rule of Conformal Cascade on our
own models and prompts, so treat it as an approximation of that method.

**7B labels.** The response texts in `results/stage4/qwen7b_*.answers.jsonl` come from a
separate generation pass and differ from the correctness labels used in calibration for
0.2% to 0.3% of questions. The labels themselves are in `results/stage4/qwen7b_*.labels.npy`
(same row order as the answer file), and the affected questions (7 for GSM8K, 11 for MMLU)
are listed in `results/stage4/qwen7b_*.label_mismatch.jsonl`.

## Not in this repository

- Activation files (`*.npz` from `extract.py`, 1 to 2.3 GB per model and benchmark) and
  query embeddings, because of their size. Rerun `extract.py` to rebuild them.
- Code for typesetting the paper's tables and figures.

## License

MIT (see `LICENSE`).
