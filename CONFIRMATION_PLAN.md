# Plan for the confirmation runs (stage 6)

Written on 2026-11-11, before `run_stage6.sh` produced any output. At this time no model had
been run on the fresh questions and no Llama model had been run on any question.

## Why

All choices of the pipeline (skip grid, token budget, cost accounting, the weighted split of
delta) were made while looking at results on the same 4,000 questions per dataset. Stage 6
tests the frozen pipeline on data that played no part in those choices.

## What is run

1. **Fresh questions.** Questions 4,001 to 8,000 of the shuffled GSM8K and MMLU sets
   (`--offset 4000`), disjoint from the main 4,000. Small model Qwen2.5-1.5B, large model
   Qwen2.5-7B. Probes are refitted on the fresh questions by the same recipe.
2. **Second model family.** Llama-3.2-3B-Instruct (small) and Llama-3.1-8B-Instruct (large) on
   the main 4,000 questions of GSM8K and MMLU.

Everything else is unchanged: prompts, 640-token budget, scoring, 5 seeds, 100 splits,
delta = 0.1, the same code (`calibrate.py`), cost ratios from the repeated timing (stage 5).
Sampling-based baselines are not rerun.

## Methods fixed in advance

- **SkipGate** as reported so far: uniform split of delta over a grid of 10 skip thresholds.
- **SkipGate-W** (`skipgate_wf` in the code): weighted split of delta (a quarter for the pure
  accept gate, a quarter for the pure skip gate, half over the nine interior skip thresholds)
  plus the free candidate "always use the large model". This variant was designed after
  seeing the main results. Stage 6 is its first test on data it was not designed on, and it is
  the variant we expect to be the more robust one.

## Questions and what counts as confirmation

Primary setting: alpha = 5%, each pair's measured throughput ratio.

- **Q1 (validity).** Held-out violation rate of both variants at most delta = 10% for every
  alpha, and at most 10% in the population check.
- **Q2 (gain over trivial policies, GSM8K).** Both variants cheaper than random routing and
  than the query-only router, with a positive paired difference in at least 4 of 5 seeds.
- **Q3 (joint calibration, GSM8K).** SkipGate cheaper than the better single gate
  (`best_single`) in at least 4 of 5 seeds. The main experiments showed about 0.02.
- **Q4 (robustness of SkipGate-W).** Over all cost models and risk targets of the stage-6
  runs, SkipGate-W within 0.02 of the better of the two single gates in at least 95% of the
  settings, and never above the cost of the large model by more than 0.01.
- **Q5 (MMLU).** We expect no gain over the skip gate alone and at most a small gain over
  random routing, as in the main experiments. A clear gain would be a surprise.

Every answer is reported, whichever way it comes out. If the Llama small model is too weak or
too strong on a dataset for any gate to matter (error rate far above or below the risk
targets), this is reported as such and not counted for or against Q2 and Q3.

## Outcome (added on 2026-11-12, after the runs; updated after the timing correction)

The text above is unchanged. Outcomes at alpha = 5% and the measured throughput ratio, with
the corrected (decode-only) timing:

- Q1: met for SkipGate (held-out violation rate at most 8.6%, population check at most 1.8%).
  NOT met for SkipGate-W: its held-out rate exceeded 10% in 5 of 176 settings, all at
  alpha = 20%, maximum 11.0%. Its population check is met (at most 3.5%).
- Q2: met in both settings (5 of 5 seeds, except SkipGate-W against the router with Llama,
  4 of 5).
- Q3: met in both settings, 5 of 5 seeds (fresh GSM8K 0.035, Llama GSM8K 0.024 cheaper than
  the better single gate).
- Q4: met (within 0.02 of the better full-delta single gate in 97% of 176 settings, largest
  cost 1.00), mostly because of the always-large fallback.
- Q5: weaker than expected: SkipGate is 0.027 (fresh MMLU) and 0.060 (Llama MMLU) more costly
  than random routing, in every seed.

Deviations from the plan:
- After the plan was written, and before any confirmation calibration ran, the compute models
  switched from the single-stream to the batched prompt price. At the single-stream price Q5
  would have come out as expected.
- After review the batched rates were corrected for a double-counted prompt pass
  (fix_timing.py) and every calibration was rerun.
- The plan names run_stage6.sh. The runs used run_stage6_local.sh and run_stage6_server.sh.
- The extraction runs were split between the server and a local workstation (two RTX 2080 Ti)
  because the server was occupied. The 14B timing could not be repeated, so the pairs with the
  14B model keep the first timing.
