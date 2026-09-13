# Results and Ablations

**CS4.406 Information Retrieval and Extraction — Assignment 2, Learning from Click-Logs**

This document records every measured figure together with the command that produced it, the date
of measurement and the team member who ran it. A figure reported without its originating command
is not treated as a result.

**Statistical convention.** Headline metrics carry a bootstrap 95% confidence interval obtained by
resampling impressions. A claimed improvement is reported only with a **paired** bootstrap 95% CI
on the per-impression difference; the term "beats" is used only where that interval excludes zero.
Where a later measurement contradicts an earlier one, both are retained and the correction is
stated explicitly.

**A1 results** are preserved at the A1 submission commit: `git show be15ee6:RESULTS.md`. The A1
facts that A2 depends on are summarised in `CONTEXT.md` §3.

---

## P0 · Setup verification

### Kaggle GPU check on Aayush's account — 2026-09-12, Aayush Pandey

Command: `.venv/bin/kaggle kernels push -p scripts/kaggle/gpu_check --accelerator NvidiaTeslaT4`,
then `.venv/bin/kaggle kernels output aayushpandey18602/a2-gpu-check -p <dir>`. Kernel ran
23:29–23:30 IST, status COMPLETE, `RESULT PASS`.

| Fact | Value |
|---|---|
| Kaggle image | torch 2.10.0+cu128, CUDA 12.8, Python 3.12 |
| GPUs seen | `device_count 2`; both `Tesla T4, 15360 MiB` |
| 4096² matmul, gpu0 | fp32 50.08 ms · fp16 17.59 ms (2.8×) |
| 4096² matmul, gpu1 | fp32 39.86 ms · fp16 4.35 ms (9.2×; gpu0's fp16 number likely includes cuBLAS warm-up) |
| autocast fp16 + `GradScaler` backward | finite gradients, loss 0.332 |
| Weekly GPU quota (`kaggle quota`) | **30 h**, 0 used, refreshes **2026-09-19 00:00** |

The 30 h/week is the budget for all of P3.1 (NRMS on both datasets) plus the ablation runs, and
it refreshes one day before the deadline. Anurag's account carries the P5 inference (C-019).

### NRMS smoke test on EB-NeRD demo, Kaggle T4 — 2026-09-13, Aayush Pandey (P0 step 7)

Command: `.venv/bin/kaggle kernels push -p scripts/kaggle/nrms_smoke --accelerator NvidiaTeslaT4`;
log via `.venv/bin/kaggle kernels logs aayushpandey18602/a2-nrms-smoke`. Benchmark commit
`5164e2c`, demo pulled from the official S3 bucket inside the kernel, `src/baselines/ebrec_compat`
patched in (C-021). Recipe: train ∪ validation of demo, npratio 4, history 20, title length 30,
xlm-roberta-**base** word embeddings (the published recipe uses -large), batch 32, **1 epoch**,
seed 123 / model seed 42, last day (2023-05-31) held out. Three runs; v1 and v2 failed inside the
benchmark's polars helpers and are recorded in C-021.

| Fact (run v3, `RESULT PASS`) | Value |
|---|---|
| Kaggle image | TF 2.20.0, Keras 3.13.2, polars 1.35.2, numpy 2.0.2, transformers 5.0.0; 2× T4 visible to TF |
| Impressions after wu2019 sampling | train 45,614 · held-out day 4,779 |
| Parameters | 193,563,936 (192.0M is the 250,002 × 768 embedding table) |
| Precision | `mixed_float16` **fails** in `SelfAttention.call()` (dtype mismatch) → **float32** (C-022) |
| One epoch, one T4 | **273 s**, 1,426 steps, 192 ms/step; wall time of the whole kernel 441 s |
| Keras in-training metrics | train AUC 0.633, loss 1.527 · val AUC 0.583, val loss 1.594 |
| Benchmark `MetricEvaluator` on the held-out day, sampled 5-candidate slates | **AUC 0.5592 · MRR 0.5329 · nDCG@5 0.6469 · nDCG@10 0.6469** (equal because slates have 5 items) |
| Same seed, run v2 (which reached the end of training) | val AUC 0.5907 vs 0.5827 in v3: **±0.01 run-to-run on GPU with fixed seeds** |
| GPU quota consumed by P0 (3 smoke runs + GPU check) | 0.27 h of 30 h |

What this settles: the framework (TF/Keras), that the code runs on the shared Kaggle image with
the shim, the float32 cost per epoch on demo, and that the exit-gate item "NRMS runs end to end
on demo on Kaggle" is met. What it does not: these are **not** reproduction numbers (1 epoch,
-base embeddings, sampled slates rather than full validation slates). P3.1 must (a) evaluate on
the full validation slates, (b) fix determinism (`TF_DETERMINISTIC_OPS=1`) or report over seeds,
and (c) run the published recipe on `ebnerd_small`.

---

## Q1 · Behavioural features

**No feature-quality number exists yet.** There is no AUC, no ablation and no feature importance,
because nothing has been trained on these features. That arrives with the Q2 reranker. What
follows are (a) the data facts the feature definitions rest on and (b) the verification of the
implementations.

### Data facts behind the Phase 1 definitions — 2026-09-11, Anurag Kaushal

Command: `make check-data` (runs `scripts/check_phase1_data.py`; laptop, about 5 min;
per-candidate click rates use the first 50,000 impressions of each split, and everything else is
the full split).

| Fact | Value | Used by |
|---|---|---|
| MIND split starts (first impression) | 2019-11-09 00:00:19 · 2019-11-15 00:00:01 · 2019-11-16 00:00:05 (train / dev / test) | C-008 `untimed_ts` |
| MIND history varies within a split | for 0 of 33,617 / 14,826 / 484,059 repeat users | C-008 |
| MIND history identical in small train and dev | 5,943 of 5,943 shared users | C-008 |
| Mean candidates per impression | MIND 37.2 / 37.5 / 39.3; EB-NeRD small validation 12.0 | C-009 cost model |
| EB-NeRD columns absent from the test file | `article_id`, `article_ids_clicked`, `next_read_time`, `next_scroll_percentage` | C-013 |
| EB-NeRD inview lists sorted by id | 0.2% (ascending or descending), in train, validation and test alike | C-013 `cand_position` |
| Click rate by list-position quintile, q0 → q4 | EB-NeRD train 0.0942 · 0.0869 · 0.0910 · 0.0914 · 0.0903; val 0.0870 · 0.0830 · 0.0830 · 0.0824 · 0.0801 | C-013 |
| | MIND train 0.0449 · 0.0394 · 0.0430 · 0.0390 · 0.0362; dev 0.0440 · 0.0387 · 0.0433 · 0.0390 · 0.0373 | C-013 |
| Largest "session" when keyed by `session_id` alone (test) | 200,000 impressions: the `is_beyond_accuracy` rows, all `session_id` 0 and `impression_id` 0 | C-013 session key |
| Keyed by (user_id, session_id), test | 6,766,532 sessions, max 118, p99 9, 47,258,574 self-join pairs, 195 with tied timestamps | C-013 |
| Keyed by (user_id, session_id), small train | 120,587 sessions, max 24, p99 8, 731,211 pairs, 5 with tied timestamps | C-013 |
| EB-NeRD impressions with no click (small train) | 0; mean 1.006 clicks per impression | C-013 (`n_prior_clicks_in_session` ≈ `session_pos` − 1) |
| `scroll_percentage` null in behaviors | 70.3% train, 71.6% test | C-013 |
| History clicks with null read time / scroll (small train) | 0 / 10.5% of 2,426,247 | C-013 dwell means |

### Verification of the implementations

| Check | Result | Command |
|---|---|---|
| Full suite, project venv | **279 passed** at the Phase 1.2 commit; **310 passed** after Phase 2 (lambdarank, rerank modules) | `make test` |
| Full suite, a *fresh* venv built from the pinned `requirements.txt` | **279 passed** with 64 pins (C-011); **310 passed** with 65 pins, lightgbm 4.7.0 added (C-016); `pip check` clean and `pip freeze` equal to the pins both times | see C-011, C-016 |
| Full suite, **clean clone on Aayush's machine** (P0 step 2), 2026-09-12 | **318 passed**, 0 failed, 6.48 s; `pip check` clean; `pip freeze` equal to the 65 pins. Python 3.12.3, 15 GB RAM, no GPU | `make test` (318 passed again after `make` was installed; the first run used the recipe directly, `PYTHONPATH=. .venv/bin/pytest tests/ -v`) |
| Planted bugs, each run alone in an otherwise-correct implementation | every one fails its dedicated test (per-bug tables in `SPEC.md` §11.1–§11.8) | scratch scripts, logged in `AI_USAGE.md` |

### Half-life grid (P1-D2) — EB-NeRD validation, 2026-09-11, Anurag Kaushal

Command: `PYTHONPATH=. .venv/bin/python -u scripts/tune_half_life_ebnerd.py` (log
`data/processed/halflife_ebnerd.log`, JSON `data/processed/halflife_ebnerd.json`).

- **Sample:** a seeded random 50,000 of the 244,647 validation impressions (598,197 candidates).
- **Scoring:** each h ranks candidates by `recency_weighted_profile` alone.
- **History:** 2,204,173 clicks, all before the earliest sampled *t*; 0 missing categories.

| h | AUC [95% CI] | nDCG@10 [95% CI] | paired ΔAUC vs ∞ | paired ΔnDCG@10 vs ∞ |
|---|---|---|---|---|
| 6 h | 0.5544 [0.5516, 0.5569] | 0.4636 [0.4613, 0.4660] | −0.0063 [−0.0082, −0.0042] | −0.0066 [−0.0082, −0.0049] |
| 24 h | 0.5590 [0.5564, 0.5614] | 0.4678 [0.4655, 0.4702] | −0.0016 [−0.0031, −0.0000] | −0.0024 [−0.0037, −0.0011] |
| 72 h | 0.5610 [0.5583, 0.5635] | 0.4699 [0.4674, 0.4721] | +0.0003 [−0.0007, +0.0014] | −0.0003 [−0.0012, +0.0006] |
| **∞ (no decay)** | 0.5606 [0.5579, 0.5631] | 0.4702 [0.4677, 0.4724] | — | — |

**Reading:**

- **Short half-lives are significantly worse**, and 72 h is indistinguishable from ∞.
- **Chosen: h = ∞** (CONTEXT.md C-014). It is confirmed inside the full GBDT below: h = 72 h is
  worse there by 0.0035 AUC, CI [−0.0052, −0.0020].
- **`category_match` gives numbers identical to these as a single-feature ranker.** Within one
  impression ‖P‖ is constant, so the cosine is a monotone rescaling of P(c) and orders candidates
  identically. Its extra signal can only act across impressions.

### Freshness coverage (§11.8), measured in the reranker runs

- **EB-NeRD validation:** `published_time` ≥ *t* for 23 of 1,197,444 candidate rows (0.002%),
  which become NaN.
- **MIND dev:** 2,438 of 2,740,998 candidates (0.09%) were never listed in an impression before *t*
  and are not in any history, so they are NaN.

## Q2 · Two-stage reranker — before vs after (2026-09-11, Anurag Kaushal)

The protocol is `SPEC.md` §12: fit on the train period, evaluate on the next period; in-impression
reranking (PLAN D1 framing (a)). Every metric carries a bootstrap 95% CI over impressions, and
every Δ is a **paired** bootstrap over the same impressions.

**Two runs.** The first measured version used A1's pointwise HistGBDT. The second added LightGBM
`lambdarank` (D2, C-016) and the conditional ablation. The second run **reproduced every pointwise
number exactly** (everything is seeded), so both sets below come from one command per dataset.

**Read the absolute values with care.** Both evaluation periods sit right after training. A1
measured its dev-to-leaderboard offset at −0.03 to −0.09 AUC (§7), so these are optimistic
estimates of test performance. Differences between rows share one protocol and are the reliable
part.

### EB-NeRD — fit on 100,000 train-week impressions, evaluated on 100,000 validation-week impressions (1,197,444 candidates)

Command: `PYTHONPATH=. .venv/bin/python -u scripts/rerank_ebnerd_a2.py` (log
`data/processed/rerank_ebnerd_a2.log`, JSON `data/processed/rerank_ebnerd_a2.json`, about 9 min).

| Ranker | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| stage 1: BM25 only | 0.5036 [0.5020, 0.5054] | 0.3201 [0.3183, 0.3219] | 0.3521 [0.3500, 0.3543] | 0.4354 [0.4337, 0.4371] |
| stage 1: word2vec only | 0.5059 [0.5038, 0.5079] | 0.3214 [0.3196, 0.3232] | 0.3526 [0.3504, 0.3546] | 0.4379 [0.4362, 0.4395] |
| pointwise GBDT: A1 base (7) | 0.6359 [0.6340, 0.6377] | 0.4039 [0.4019, 0.4059] | 0.4571 [0.4548, 0.4592] | 0.5194 [0.5176, 0.5210] |
| pointwise GBDT: A2 (13), h = ∞ | 0.6368 [0.6349, 0.6387] | 0.4016 [0.3997, 0.4037] | 0.4550 [0.4528, 0.4573] | 0.5176 [0.5160, 0.5195] |
| pointwise GBDT: A2, h = 72 h | 0.6333 [0.6314, 0.6351] | 0.3959 [0.3939, 0.3979] | 0.4492 [0.4469, 0.4515] | 0.5140 [0.5122, 0.5157] |
| **lambdarank: A1 base (7)** | **0.6702 [0.6685, 0.6721]** | **0.4345 [0.4326, 0.4365]** | **0.4953 [0.4932, 0.4974]** | **0.5495 [0.5478, 0.5512]** |
| lambdarank: A2 (13) | 0.6585 [0.6567, 0.6603] | 0.4241 [0.4220, 0.4260] | 0.4830 [0.4809, 0.4851] | 0.5389 [0.5371, 0.5406] |

| Paired Δ (EB-NeRD) | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| lambdarank − pointwise, base features | +0.0343 [+0.0328, +0.0359] | +0.0306 [+0.0288, +0.0324] | +0.0382 [+0.0364, +0.0400] | +0.0301 [+0.0287, +0.0316] |
| lambdarank − pointwise, A2 features | +0.0217 [+0.0200, +0.0233] | +0.0225 [+0.0204, +0.0243] | +0.0280 [+0.0260, +0.0299] | +0.0213 [+0.0197, +0.0228] |
| lambdarank A2 − BM25 (the two-stage gain) | +0.1549 [+0.1523, +0.1570] | +0.1040 [+0.1016, +0.1061] | +0.1309 [+0.1283, +0.1331] | +0.1035 [+0.1013, +0.1053] |
| pointwise: A2 − base | +0.0009 [−0.0009, +0.0026] | −0.0023 [−0.0042, −0.0005] | −0.0021 [−0.0040, −0.0003] | −0.0017 [−0.0033, −0.0002] |
| **lambdarank: A2 − base** | **−0.0118 [−0.0132, −0.0105]** | **−0.0105 [−0.0122, −0.0089]** | **−0.0123 [−0.0140, −0.0106]** | **−0.0106 [−0.0120, −0.0092]** |
| pointwise: A2 h = 72 h − A2 h = ∞ | −0.0035 [−0.0052, −0.0020] | −0.0057 [−0.0076, −0.0037] | −0.0057 [−0.0077, −0.0038] | −0.0036 [−0.0052, −0.0021] |

**Leave-one-family-out, lambdarank, EB-NeRD.** Triggered: every metric of A2 − base had a CI below
0. Each row is a refit without that family. "vs full" = (without − with): **positive means the
family was costing performance**.

| Family removed | AUC vs full | MRR vs full | nDCG@5 vs full | nDCG@10 vs full | AUC vs base |
|---|---|---|---|---|---|
| category profile (`recency_weighted_profile`, `category_match`) | −0.0091 [−0.0104, −0.0079] | −0.0112 [−0.0127, −0.0097] | −0.0119 [−0.0134, −0.0105] | −0.0076 [−0.0088, −0.0063] | −0.0209 [−0.0220, −0.0197] |
| list position (`cand_position`) | +0.0019 [+0.0009, +0.0030] | −0.0027 [−0.0039, −0.0013] | −0.0007 [−0.0019, +0.0007] | +0.0012 [+0.0002, +0.0024] | −0.0098 [−0.0114, −0.0084] |
| session (`session_pos`) | +0.0056 [+0.0045, +0.0068] | +0.0002 [−0.0011, +0.0016] | +0.0013 [−0.0001, +0.0028] | −0.0003 [−0.0015, +0.0009] | −0.0061 [−0.0075, −0.0048] |
| **dwell (`hist_read_time_mean`, `hist_scroll_mean`)** | **+0.0144 [+0.0132, +0.0154]** | **+0.0135 [+0.0121, +0.0147]** | **+0.0158 [+0.0144, +0.0170]** | **+0.0120 [+0.0109, +0.0130]** | **+0.0026 [+0.0014, +0.0038]** |

A2 without dwell, compared with the base, over all four metrics: AUC +0.0026 [+0.0014, +0.0038],
MRR +0.0030 [+0.0016, +0.0044], nDCG@5 +0.0035 [+0.0020, +0.0049], nDCG@10 +0.0014
[+0.0003, +0.0026].

### MIND — fit on 80,000 MINDsmall_train impressions, evaluated on all 73,152 MINDsmall_dev impressions (2,740,998 candidates)

Command: `PYTHONPATH=. .venv/bin/python -u scripts/rerank_mind_a2.py` (log
`data/processed/rerank_mind_a2.log`, JSON `data/processed/rerank_mind_a2.json`, about 8 min).

| Ranker | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| stage 1: BM25 only | 0.5451 [0.5428, 0.5471] | 0.2538 [0.2516, 0.2559] | 0.2676 [0.2651, 0.2700] | 0.3289 [0.3265, 0.3311] |
| stage 1: MiniLM only | 0.6353 [0.6330, 0.6374] | 0.3084 [0.3062, 0.3107] | 0.3371 [0.3346, 0.3397] | 0.3960 [0.3937, 0.3986] |
| **pointwise GBDT: A1 v4 base (7)** | **0.6747 [0.6725, 0.6768]** | **0.3295 [0.3272, 0.3321]** | **0.3630 [0.3603, 0.3657]** | **0.4217 [0.4193, 0.4243]** |
| pointwise GBDT: A2 (10) | 0.6727 [0.6705, 0.6747] | 0.3258 [0.3235, 0.3284] | 0.3574 [0.3547, 0.3602] | 0.4171 [0.4146, 0.4198] |
| lambdarank: A1 v4 base (7) | 0.6663 [0.6642, 0.6683] | 0.3239 [0.3217, 0.3263] | 0.3580 [0.3555, 0.3608] | 0.4169 [0.4145, 0.4195] |
| lambdarank: A2 (10) | 0.6676 [0.6654, 0.6696] | 0.3222 [0.3199, 0.3246] | 0.3560 [0.3534, 0.3587] | 0.4151 [0.4127, 0.4177] |

| Paired Δ (MIND) | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| **lambdarank − pointwise, base features** | **−0.0084 [−0.0095, −0.0073]** | **−0.0056 [−0.0065, −0.0045]** | **−0.0050 [−0.0060, −0.0039]** | **−0.0048 [−0.0056, −0.0038]** |
| lambdarank − pointwise, A2 features | −0.0051 [−0.0064, −0.0038] | −0.0035 [−0.0047, −0.0023] | −0.0014 [−0.0025, −0.0002] | −0.0020 [−0.0030, −0.0009] |
| lambdarank A2 − MiniLM | +0.0323 [+0.0305, +0.0341] | +0.0138 [+0.0120, +0.0155] | +0.0189 [+0.0169, +0.0210] | +0.0191 [+0.0173, +0.0208] |
| pointwise: A2 − base | −0.0021 [−0.0031, −0.0009] | −0.0037 [−0.0048, −0.0026] | −0.0056 [−0.0068, −0.0044] | −0.0045 [−0.0056, −0.0035] |
| lambdarank: A2 − base | +0.0013 [+0.0002, +0.0023] | −0.0017 [−0.0028, −0.0007] | −0.0020 [−0.0032, −0.0009] | −0.0018 [−0.0028, −0.0008] |

**Leave-one-family-out, lambdarank, MIND.** Triggered: the MRR, nDCG@5 and nDCG@10 CIs of
A2 − base were below 0. Same reading as above.

| Family removed | AUC vs full | MRR vs full | nDCG@5 vs full | nDCG@10 vs full | AUC vs base |
|---|---|---|---|---|---|
| category match (`category_match`) | −0.0008 [−0.0016, +0.0001] | −0.0014 [−0.0023, −0.0005] | −0.0015 [−0.0024, −0.0005] | −0.0009 [−0.0017, −0.0001] | +0.0005 [−0.0005, +0.0016] |
| list position (`cand_position`) | −0.0032 [−0.0040, −0.0022] | −0.0005 [−0.0014, +0.0004] | −0.0008 [−0.0018, +0.0002] | −0.0004 [−0.0012, +0.0004] | −0.0019 [−0.0029, −0.0008] |
| **freshness (`freshness_hours`, first-seen)** | −0.0015 [−0.0025, −0.0004] | **+0.0027 [+0.0016, +0.0038]** | **+0.0025 [+0.0014, +0.0038]** | **+0.0021 [+0.0011, +0.0032]** | −0.0002 [−0.0011, +0.0007] |

**Cross-implementation check (MIND).** `recency_weighted_profile` equals A1's `cat_affinity` to a
maximum absolute difference of 1.11e-16 over 2,658,091 dev rows with history.

### What these numbers say

1. **The two-stage reranker works on both datasets.** The best model beats the best stage-1
   generator by +0.155 AUC on EB-NeRD (lambdarank A2 vs BM25) and +0.039 on MIND (pointwise base
   vs MiniLM). All CIs are far from 0.
2. **The listwise objective is a large win on EB-NeRD and a loss on MIND.**
   - EB-NeRD: lambdarank beats pointwise by +0.034 AUC on the same base features.
   - MIND: lambdarank loses by −0.008 AUC.
   - So D2 is not settled by one number (C-016).
   - Untested hypothesis: MIND slates are long (mean 37, max 299, against EB-NeRD's 11 and 100),
     and lambdarank's default truncation level of 30 limits which positions produce gradients.
3. **Lambdarank did not rescue the Phase 1 features as a block; the ablation found the reason.**
   - On EB-NeRD the whole block costs −0.012 AUC, but that loss is the **dwell family**. Without
     dwell, A2 **beats the base on all four metrics**.
   - The category profile is the most useful Phase 1 family: removing it costs 0.009 AUC.
   - On MIND, first-seen freshness costs 0.002–0.003 of MRR/nDCG while helping AUC slightly;
     category match and list position help.
4. **Selection bias, stated.** The ablation was read on the same validation impressions it would
   be used to choose features from. "A2 without dwell beats base" is therefore a *finding to
   confirm* on held-out impressions (another seeded sample, or the gap-aware protocol) before it
   becomes the shipped feature set.
5. **The half-life stays at ∞.** Pointwise permutation importance still ranks `n_candidates`
   first, which is meaningless for within-impression order. The ablation above is the importance
   evidence to use.

### Follow-up 1 · MIND: does a truncation level covering the longest slate rescue lambdarank? (C-016 option (c), C-017)

Command: `PYTHONPATH=. .venv/bin/python -u scripts/check_truncation_mind.py` (log
`data/processed/truncation_mind.log`, JSON `data/processed/truncation_mind.json`).

- **Setup:** the same 80,000 train impressions and all 73,152 dev impressions. The longest slate
  is 299, so the truncation level is **300**; LightGBM's documented default is 30.
- **The parameter takes effect:** a test shows truncation 1 and truncation 5 train different
  models.
- **Guarding against selection on dev:** dev is also split at its median impression time
  (09:47:21) into two halves of 36,576 impressions each.

| Ranker, all dev | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| pointwise GBDT: A1 v4 base | **0.6747 [0.6725, 0.6768]** | **0.3295** | **0.3630** | **0.4217** |
| lambdarank, truncation 30 (default): base | 0.6663 [0.6642, 0.6683] | 0.3239 | 0.3580 | 0.4169 |
| lambdarank, truncation 300: base | 0.6661 [0.6639, 0.6681] | 0.3246 | 0.3586 | 0.4176 |
| lambdarank, truncation 30: A2 | 0.6676 [0.6654, 0.6696] | 0.3222 | 0.3560 | 0.4151 |
| lambdarank, truncation 300: A2 | 0.6686 [0.6666, 0.6707] | 0.3234 | 0.3570 | 0.4166 |

| Paired Δ | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| truncation 300 − 30, base, all dev | −0.0002 [−0.0010, +0.0006] | +0.0006 [−0.0002, +0.0014] | +0.0005 [−0.0003, +0.0014] | +0.0007 [−0.0000, +0.0015] |
| truncation 300 − 30, base, earlier half | −0.0021 [−0.0032, −0.0010] | −0.0000 | −0.0000 | −0.0000 |
| truncation 300 − 30, base, later half | +0.0016 [+0.0004, +0.0029] | +0.0013 | +0.0011 | +0.0014 |
| truncation 300 − 30, A2, all dev | +0.0011 [+0.0003, +0.0019] | +0.0011 [+0.0003, +0.0019] | +0.0010 [+0.0000, +0.0019] | +0.0015 [+0.0007, +0.0022] |
| **lambdarank (trunc 300) − pointwise, base, all dev** | **−0.0087 [−0.0097, −0.0074]** | **−0.0050 [−0.0059, −0.0038]** | **−0.0044 [−0.0054, −0.0033]** | **−0.0041 [−0.0049, −0.0031]** |
| same, earlier half | −0.0134 [−0.0153, −0.0116] | −0.0078 | −0.0068 | −0.0068 |
| same, later half | −0.0040 [−0.0052, −0.0026] | −0.0021 | −0.0021 | −0.0014 [−0.0027, +0.0000] |

**Reading:**

- **Truncation does not rescue lambdarank on MIND.** Its effect on the base features is nil over
  all of dev, and it changes sign between the halves.
- **Pointwise beats lambdarank on MIND in both halves**, and on every metric except one borderline
  cell.
- **Side finding:** every model is much weaker in the later half of the day (pointwise AUC 0.7016 →
  0.6479). That is consistent with the staleness A1 measured (§7): counts and history stop at the
  split boundary, and the gap grows through the day.

### Follow-up 2 · EB-NeRD: does dropping the dwell family hold on impressions it was not chosen on? (C-017)

Command: `PYTHONPATH=. .venv/bin/python -u scripts/holdout_dwell_ebnerd.py` (log
`data/processed/holdout_dwell_ebnerd.log`, JSON `data/processed/holdout_dwell_ebnerd.json`).

- **Models:** three lambdarank models refitted on the identical 100,000 train-week impressions.
- **Selection** = the 100,000 validation impressions the ablation was read on.
- **Holdout** = the other **144,647** validation-week impressions (1,731,498 candidates), disjoint
  and never scored before.

| Ranker | Selection AUC | Holdout AUC | Holdout MRR | Holdout nDCG@5 | Holdout nDCG@10 |
|---|---|---|---|---|---|
| lambdarank: A1 base (7) | 0.6702 [0.6685, 0.6721] | 0.6709 [0.6694, 0.6725] | 0.4344 [0.4327, 0.4360] | 0.4951 [0.4934, 0.4968] | 0.5495 [0.5481, 0.5509] |
| lambdarank: A2 (13) | 0.6585 [0.6567, 0.6603] | 0.6587 [0.6572, 0.6603] | 0.4227 [0.4211, 0.4243] | 0.4817 [0.4799, 0.4834] | 0.5380 [0.5366, 0.5395] |
| **lambdarank: A2 without dwell (11)** | **0.6728 [0.6710, 0.6745]** | **0.6738 [0.6723, 0.6754]** | **0.4375 [0.4360, 0.4393]** | **0.4983 [0.4966, 0.5001]** | **0.5509 [0.5495, 0.5525]** |

| Paired Δ, holdout | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| A2 − base | −0.0122 [−0.0134, −0.0110] | −0.0117 [−0.0130, −0.0103] | −0.0134 [−0.0147, −0.0121] | −0.0116 [−0.0127, −0.0104] |
| A2 without dwell − A2 | +0.0151 [+0.0142, +0.0160] | +0.0148 [+0.0137, +0.0159] | +0.0166 [+0.0156, +0.0177] | +0.0129 [+0.0120, +0.0138] |
| **A2 without dwell − base** | **+0.0029 [+0.0018, +0.0039]** | **+0.0032 [+0.0019, +0.0044]** | **+0.0032 [+0.0021, +0.0045]** | **+0.0014 [+0.0004, +0.0023]** |

**Reading:**

- **The selection numbers reproduce the ablation exactly,** so the refactor of `build()` to take an
  explicit sample changed nothing.
- **On the holdout every effect replicates,** and in size too. **Dropping dwell generalises beyond
  the sample it was chosen on**, and A2 without dwell beats the base on all four metrics.
- **The limit of this test:** the holdout is the same week. It rules out fitting to the particular
  sample; it does not test drift over time. That remains the gap-aware protocol's job.

### Locked configuration — `src/rerank/config.FINAL` (C-018), verified 2026-09-11

Re-measured by the committed scripts, `scripts/rerank_ebnerd_a2.py` and
`scripts/rerank_mind_a2.py`, which fit the `FINAL` model from the config. That rerun reproduced
every row above exactly.

| Dataset | Final model | AUC | MRR | nDCG@5 | nDCG@10 | Δ vs best stage-1 generator |
|---|---|---|---|---|---|---|
| EB-NeRD (100k validation impressions) | lambdarank, 11 features (A2 without dwell) | **0.6728 [0.6710, 0.6745]** | 0.4376 | 0.4988 | 0.5509 | +0.1692 AUC over BM25 [+0.1666, +0.1714] |
| MIND (73,152 dev impressions) | pointwise, 7 features (A1 v4) | **0.6747 [0.6725, 0.6768]** | 0.3295 | 0.3630 | 0.4217 | +0.0394 AUC over MiniLM [+0.0376, +0.0411] |

- EB-NeRD final − lambdarank base: AUC +0.0026 [+0.0014, +0.0038], MRR +0.0030, nDCG@5 +0.0035,
  nDCG@10 +0.0014 [+0.0003, +0.0026]. On the 144,647-impression holdout, AUC is 0.6738 (follow-up 2).
- MIND's final model is asserted to give predictions bit-identical to the measured pointwise base
  fit.
- **Tests:** 318 passed, including `tests/test_rerank_config.py`.

**Test-file checks (no metric).** The full feature pipeline ran on the first 5,000 impressions of
each unlabelled test file, scored by the **final** models (before C-018, by the lambdarank A2
model):

- EB-NeRD: 56,292 rows; MIND: 199,107 rows;
- no label column, and no click count on EB-NeRD;
- every prediction is finite.

## Q3 · Baseline reproduced, then beaten

**Every Kaggle run — successes and failures, with commit, mode, metrics, timings and log path — is
one row in [`scripts/kaggle/RUN_LEDGER.md`](scripts/kaggle/RUN_LEDGER.md), appended by
`scripts/kaggle/ledger.py <kernel> <version>` after each run.** The tables below cite ledger rows
by kernel and version; a number that is not in the ledger was not run.

### Q3.1 · NRMS reproduced on both datasets — 2026-09-14, Aayush Pandey (P3.1, SPEC.md §13, CONTEXT.md C-021–C-025)

Both runs: one Kaggle T4, float32, op determinism on, seeds fixed; every validation impression
scored on its full slate; metrics from **our** `src/eval/metrics` (the code behind the Q2 numbers)
with a 1,000-resample bootstrap CI over impressions, and required by the kernel to equal the
reference implementation's own evaluator. Score files verified on the laptop: they join Anurag's
`src/rerank` frames on (`imp_row`, `cand_position`) with every `impression_id` and `article_id`
equal, and the metrics recompute from the file plus his labels to the same digits.

**EB-NeRD** — ledger `a2-nrms-ebnerd` v4 (`data/logs/kaggle/a2-nrms-ebnerd_v4.log`, repo `5a6bb76`).
Command: `.venv/bin/kaggle kernels push -p scripts/kaggle/nrms_ebnerd --accelerator NvidiaTeslaT4`
with `DEMO_CHECK=False`. `jppol-ai/ebnerd-benchmark` at `5164e2c`; fit on `ebnerd_small/train`
(232,887 impressions → 226,452 wu2019 samples, npratio 4; last train day 25 May = 7,825 samples
held out for early stopping), scored on all **244,647** `ebnerd_small/validation` impressions
(2,928,942 candidates). xlm-roberta-large word embeddings (0.318 s/step on the 200-step gate,
kept), 257.9M parameters, 5 epochs at 1,609 s each; scoring 5,836 s; **4 h 02 min**, 4.0 h of quota.

| EB-NeRD | AUC | MRR | nDCG@5 | nDCG@10 | on |
|---|---|---|---|---|---|
| **NRMS, ours** | **0.5600** [0.5588, 0.5612] | 0.3491 [0.3479, 0.3503] | 0.3884 [0.3870, 0.3897] | 0.4668 [0.4656, 0.4679] | small validation, all impressions |
| NRMS, published (Kruse et al. 2024, Table 3) | 0.6103 | 0.3975 | 0.4445 | 0.5124 | hidden test set (leaderboard) |
| Clicks (popularity), published, same table | 0.5970 | 0.3774 | 0.4236 | 0.4965 | hidden test set |
| Random, published, same table | 0.4998 | 0.3156 | 0.3489 | 0.4338 | hidden test set |
| reranker `config.FINAL` (Q2) | 0.6728 | — | — | — | 100k-impression sample of the same validation week |
| NRMS holdout `val_auc` during training | 0.6393 | | | | last train day, **sampled 5-candidate slates** — not comparable |

Holdout `val_auc` by epoch: 0.6250, 0.6348, 0.6364, 0.6294, **0.6393** (best = last; checkpoint kept it).

**MIND** — ledger `a2-nrms-mind` v5 (`data/logs/kaggle/a2-nrms-mind_v5.log`). Command:
`.venv/bin/kaggle kernels push -p scripts/kaggle/nrms_mind --accelerator NvidiaTeslaT4` with
`DEMO_CHECK=False`. `recommenders-team/recommenders` at `0bb4b36` under tf-keras; the quick-start
recipe (GloVe-300d `embedding.npy`, history 50, npratio 4, title 30, batch 32, Adam 1e-4,
dropout 0.2), 11.29M parameters; fit on `MINDsmall_train` (156,965 impressions), scored on all
**73,152** `MINDsmall_dev` impressions (2,740,998 candidates — the same candidate set as Q2).
5 epochs at 1,330 s + 177 s eval each; **2 h 10 min**, 2.2 h of quota.

| MIND | AUC | MRR | nDCG@5 | nDCG@10 | on |
|---|---|---|---|---|---|
| **NRMS, ours** | **0.6667** [0.6647, 0.6688] | 0.3220 [0.3195, 0.3243] | 0.3557 [0.3529, 0.3584] | 0.4184 [0.4159, 0.4208] | MINDsmall_dev, all impressions |
| NRMS, published (Wu et al. 2020, Table 3, "Overall") | 0.6776 | 0.3305 | 0.3594 | 0.4163 | full-MIND test set; trained on half the users of full MIND; mean of 10 runs |
| NRMS, recommenders quick-start notebook output | 0.6127 | 0.2697 | 0.2912 | 0.3625 | MIND-**demo** dev, 5 epochs (their recorded run, TF 2.6) |
| reranker `config.FINAL` (Q2, pointwise GBDT, 7 features) | 0.6747 [0.6725, 0.6768] | 0.3295 [0.3272, 0.3321] | 0.3630 [0.3603, 0.3657] | 0.4217 [0.4193, 0.4243] | same 73,152 impressions |
| stage 1 alone: MiniLM cosine (Q2) | 0.6353 | 0.3084 | 0.3371 | 0.3960 | same |

Dev `group_auc` by epoch: 0.6466, 0.6523, 0.6613, 0.6569, **0.6667** (the package keeps the last
epoch, not the best; here they coincide).

**The gap, explained (PLAN.md P3.1 step 2).**

*MIND: ours 0.6667 vs published 0.6776, −0.011.* (1) Training data: the paper trains on half the
users of full MIND (≈ 500k users); we train on MINDsmall_train (50k users) — the dominant
factor, and in the expected direction. (2) Evaluation set: their hidden test set vs our dev
split; A1 measured a dev-to-leaderboard offset of −0.03 to −0.09 for feature models (SPEC §7),
so this factor, if anything, flatters our number. (3) Their figure is the mean of 10 runs; ours
is one deterministic run. (4) Same implementation (the paper cites Microsoft Recommenders,
which is what we ran), same GloVe-300d initialisation, same architecture. Net: a −0.011 gap on a
data set 10× smaller is a faithful reproduction. The quick-start's own recorded 0.6127 on
MIND-demo shows the size-of-training-data slope directly (demo → small: +0.054).

*EB-NeRD: ours 0.5600 vs published 0.6103, −0.050.* (1) Evaluation set: the paper's number is
on the hidden test set (1–8 June) after training on `ebnerd_large`; ours is on the small
validation week (25 May – 1 June) after training on the small train week — 12× less training
data, the largest factor. (2) The paper gives no NRMS training details (epochs, embeddings,
history), so the recipe cannot be matched exactly; we followed the repository README recipe.
(3) The published script's dropped-embedding bug (C-021) means the published recipe may
effectively train a random 32k×300 embedding table; we passed the xlm-roberta-large embeddings,
which should help, not hurt. (4) Keras `mixed_float16` unusable, so float32 (C-022) — no
accuracy effect expected. (5) In the paper's own Table 3, NRMS (61.03) sits only 1.3 AUC above
the click-popularity baseline (59.70); on EB-NeRD, text-only NRMS is close to popularity, so a
−0.05 shift from far less training data is plausible.

**What the two numbers say together.** On MIND the text-only neural baseline lands 0.008 AUC
below the feature-engineered two-stage reranker; on EB-NeRD it lands 0.113 below. That asymmetry
matches A1's finding (RESULTS.md at `be15ee6`, Q9c) that recency dominates on EB-NeRD: NRMS has no
notion of article age, popularity or session, which are exactly the reranker's strongest EB-NeRD
features. No "beats" is claimed here — the reranker-vs-NRMS difference is measured by P3.4a's
paired bootstrap, and D4 (the one principled change) is chosen with this asymmetry as evidence.

**Determinism, measured** (ledger): EB-NeRD demo twins v2/v3 identical to every digit; MIND demo
twins v1/v2 differed by 0.002 AUC until Python's `random` was seeded (the package seeds TF and
numpy only; `newsample` draws the negatives from the stdlib), after which v3/v4 were identical.

**Serving-time honesty (Q9).** NRMS uses history and titles only; no serving-unavailable feature
exists in either baseline, so the "with vs without" row is not applicable and is stated as such.

Score files (gitignored): `data/scores/ebnerd/validation/nrms.{parquet,json}`,
`data/scores/mind/MINDsmall_dev/nrms.{parquet,json}`, SPEC §13.3 schema; the P3.4a harness
reads them.

### Q3.4 · Paired bootstrap harness — validation, 2026-09-14, Aayush Pandey (P3.4a, SPEC.md §14, C-026)

These rows validate the **judge**, not a model: both "B" systems are constructed from the real
EB-NeRD NRMS file (`_check_*.parquet`, gitignored, manifest `note` says so). Command:
`make paired A=data/scores/ebnerd/validation/nrms.parquet B=data/scores/ebnerd/validation/<B>.parquet JSON=data/processed/paired/<name>.json`
(all 244,647 validation impressions, 1,000 resamples, seed 0; records in `data/processed/paired/`).

| B | construction | Δ AUC (B − A) | Δ MRR | Δ nDCG@5 | Δ nDCG@10 | verdict | time / peak RSS |
|---|---|---|---|---|---|---|---|
| `nrms_rescaled` | score × 2 + 1 (rank-preserving) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | no significant difference | 64 s |
| `nrms_oracle_boost` | +0.5 on every clicked candidate | +0.4082 [+0.4070, +0.4093] | +0.5334 [+0.5322, +0.5346] | +0.5217 [+0.5205, +0.5229] | +0.4459 [+0.4449, +0.4469] | B beats A | 49 s / 1.3 GB |

Sensitivity (test, first 20,000 impressions, 300 resamples): N(0, 0.01) noise on the scores is
a real degradation — AUC −0.0011 [−0.0020, −0.0004], significant; MRR −0.0005 [−0.0013, +0.0002],
not significant. Calibration on synthetic pairs: `tests/test_paired.py` (coverage ≥ 90 % over
200 trials at n = 1,000 for Δ = 0 and Δ = 0.02).

**Incident.** The first attempt at the boosted row was OOM-killed (dmesg 01:16, python at
3.4 GB RSS): A1's `bootstrap_ci` allocated the (1,000 × 244,647) index array at once. Fixed by
blocked draws, bit-identical to the old ones (`tests/test_bootstrap.py`); every CI in this file
is unchanged.

**Pending, not run:** reranker `config.FINAL` vs NRMS on both datasets — needs Anurag's scores
file for the reranker (P0.8). The command above is the one that will produce it.


_Not started._

## Q4 · Serving and scale

_Not started._

## Q5 · Extended evaluation and leaderboard submissions

_Not started._

## Q9 · With and without serving-unavailable features

_Not started._
