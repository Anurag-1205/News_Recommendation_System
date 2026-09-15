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

### Q2.5 · Retrieve-then-rerank (D1 framing (b)): stage-1 recall@K, and why it is descoped — 2026-09-15 (C-037)

Q2.1 says the reranker sits on a top-K (100–200) *retrieved from the corpus*. The model above
re-ranks the impression's own slate (framing (a)), because that is what both leaderboards score.
The retrieval path exists and was served and timed in Q4 (`src/serving/request.retrieve`: BM25
top-K over the last-5-click query ∪ flat-ANN top-K over the mean-pooled user vector). This
subsection measures the one number that decides whether a reranker on that list could work at
all — **how often the clicked article is in the retrieved top-K** — on the same 1,000-impression
seeded validation sample Q4 timed, so recall and latency describe one and the same list.

Command (records in `data/processed/stage1_recall/<dataset>.json`):

```
PYTHONPATH=. .venv/bin/python -u scripts/stage1_recall.py --dataset ebnerd --n 1000 --seed 0
PYTHONPATH=. .venv/bin/python -u scripts/stage1_recall.py --dataset mind   --n 1000 --seed 0
```

hit@K = 1 if any clicked article is in the union; 95% CI by blocked bootstrap over impressions.

| dataset | K | union size (mean) | hit@K | Q4 p99 latency for this list |
|---|---|---|---|---|
| EB-NeRD | 100 | 198.0 | **0.0100** [0.0040, 0.0160] | 72.0 ms |
| EB-NeRD | 200 | 395.2 | **0.0140** [0.0070, 0.0220] | — |
| MIND | 100 | 189.3 | **0.0210 [0.0130, 0.0310]** | 86.5 ms |
| MIND | 200 | 378.3 | **0.0350 [0.0240, 0.0460]** | — |

**Reading.** On EB-NeRD the clicked article is inside the 198-article union **once in 100
impressions**; doubling K to a 395-article union lifts that to 1.4 in 100. A second seed on 300
impressions gave 0/300, and at K = 5,000 (a 9,654-article union, 7.7 % of the corpus) the first
sampled impression's click was still absent.

**Why, measured.** Every clicked id is in the index (302/302 checked), so this is ranking, not
coverage. The clicked article is a **median 4 hours old** at click time; the articles BM25 ∪ ANN
returns are a median **3,024 hours** (about four months) old, and retrieval overlaps the actual
slate on 0.2 % of its articles. Similarity to the user's last five reads pulls the most
content-similar articles out of a 125k-article archive, and news readers click what is new. The
publisher's slate is already a freshness-filtered candidate set; rebuilding it from the corpus
needs a time-windowed candidate generator (articles published in the last *N* hours, then rank),
which is a different stage 1 from A1's, and A1's is what the brief tells us to reuse.

**Decision (C-037, PLAN.md §4 ladder item 1).** Framing (b) is descoped with this number as the
reason. No framing-(b) AUC/MRR/nDCG row is built: a reranker cannot rank what stage 1 does not
return, so any such row is bounded above by the ~1 % hit rate and would measure the retriever, not
the reranker — and the labels exist only for the slate, so the number would be neither comparable
with (a) nor meaningful. What ships is framing (a); what the note reports for (b) is its latency
and cost (Q4), its 10× behaviour (Q4.4), and this recall ceiling.

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

### Q3.2–3.4 · One principled change: a freshness term in NRMS — 2026-09-14, Aayush Pandey (SPEC.md §15, C-028–C-030)

**The change** (C-028): `score = user · news + g(x, unknown)`, `g` = Dense(8, relu) → Dense(1) on the
standardised log-age of the candidate at impression time (§15.2, the reranker's own `freshness_hours`,
strictly before *t*). Everything else — encoders, recipe, seeds, epochs, data, sampling — is the
baseline's. Pre-flight, in the ledger: the model-level oracles (g ≡ 0 ⇒ baseline scores, per-candidate
locality, loader/iterator shapes and mask) passed on Kaggle before training; demo twins identical to
every digit on both datasets. Training-split stats: EB-NeRD μ = 2.632, σ = 2.655; MIND μ = 2.915, σ = 0.861.

**Pre-registered prediction (C-028, before any run):** EB-NeRD Δ AUC ≥ +0.03 with a CI excluding 0;
MIND Δ AUC within ±0.01, CI possibly including 0. **Pre-run diagnostic (C-029, before any run):**
freshness alone ranks within-slate clicks at AUC 0.501 on EB-NeRD and 0.519 on MIND
(`scripts/check_fresh_signal.py`), so the EB-NeRD prediction was expected to fail on magnitude.

**Runs.** EB-NeRD: ledger `a2-nrms-ebnerd-fresh` v5 (main account, repo `3b94316`, xlm-roberta-large
kept at 0.323 s/step, 5 epochs, 8,101 s train, two full-slate scoring passes, 3 h 54 min, 3.9 h quota).
MIND: ledger `a2-nrms-mind-fresh` v6 (alt account, 5 epochs, 7,545 s train, 2 h 12 min, 2.2 h quota).
Judgement: `make paired A=… B=… JSON=data/processed/paired/<dataset>_<rows>.json` (SPEC §14; records in
`data/processed/paired/`), all impressions of each evaluation split, 1,000 resamples, seed 0.

**The ablation (§15.3).**

| EB-NeRD, 244,647 validation impressions | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| 1 · NRMS (Q3.1) | 0.5600 [0.5588, 0.5612] | 0.3491 | 0.3884 | 0.4668 |
| 2 · NRMS + freshness | **0.5674** [0.5661, 0.5686] | 0.3585 | 0.4005 | 0.4759 |
| 3 · row 2 scored with the term masked | 0.5500 [0.5488, 0.5512] | 0.3427 | 0.3815 | 0.4602 |
| **Δ row 2 − row 1, paired** | **+0.0074 [+0.0066, +0.0081]** | **+0.0094 [+0.0085, +0.0101]** | **+0.0121 [+0.0112, +0.0129]** | **+0.0092 [+0.0085, +0.0098]** |
| Δ row 3 − row 2, paired | −0.0174 [−0.0180, −0.0169] | −0.0157 [−0.0164, −0.0151] | −0.0190 [−0.0196, −0.0183] | −0.0157 [−0.0163, −0.0152] |

| MIND, 73,152 dev impressions | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| 1 · NRMS (Q3.1) | 0.6667 [0.6647, 0.6688] | 0.3220 | 0.3557 | 0.4184 |
| 2 · NRMS + freshness | 0.6661 [0.6641, 0.6681] | 0.3227 | 0.3565 | 0.4182 |
| 3 · row 2 scored with the term masked | 0.6661 [0.6641, 0.6682] | 0.3226 | 0.3566 | 0.4183 |
| Δ row 2 − row 1, paired | −0.0006 [−0.0014, +0.0002] | +0.0007 [−0.0001, +0.0015] | +0.0008 [−0.0001, +0.0018] | −0.0002 [−0.0010, +0.0005] |
| Δ row 3 − row 2, paired | +0.0000 [−0.0002, +0.0002] | −0.0001 [−0.0002, +0.0001] | +0.0001 [−0.0001, +0.0002] | +0.0001 [−0.0001, +0.0003] |

**Verdicts (SPEC §14 wording; "beats" only when the paired CI excludes 0).**

- **EB-NeRD: the harness returns "NRMS + freshness beats NRMS" on all four metrics.**
  **Reviewed and verified by Anurag Kaushal, 2026-09-15 (C-019, C-033).** The review reran the
  recorded commands and checked the claim's mechanics; all four checks passed:
  - **Reproduction.** `make paired` on the two score files reproduced the record **bit-identically**
    (every value in `system_a`, `system_b`, `delta_b_minus_a`; both manifests; same seed 0 and 1,000
    resamples): Δ AUC +0.0074 [+0.0066, +0.0081]. Records: `ebnerd_row2_vs_row1_review_anurag.json`.
  - **No truncation, full overlap.** Both files carry 2,928,942 rows over 244,647 impressions, which
    equals the validation split exactly (its slate lengths sum to 2,928,942); the two files'
    `(imp_row, cand_position)` key sets are identical, and per-impression counts equal the slate
    lengths.
  - **Labels come from the split, not the files.** No score file has a label column. `split_labels`
    was cross-checked against an independent reconstruction (`src.rerank.ebnerd.candidate_frame`,
    which builds labels by clicked-id membership): **identical on all 2,928,942 rows**, 245,622
    positives each. The gap to the split's 246,289 clicked ids is exactly the 667 duplicate clicked
    ids inside impressions, deduped identically by both paths; every clicked id is present in its
    slate, so no impression has an unrankable click.
  - **Masked control.** `nrms_fresh_masked` − `nrms_fresh` reproduced bit-identically at
    −0.0174 AUC [−0.0180, −0.0169]; record `ebnerd_row3_vs_row2_review_anurag.json`.
  The claim stands as a result.
- **MIND: no significant difference on any metric** — a null, as pre-registered.

**Prediction vs outcome.** MIND: as predicted (within ±0.01; null). EB-NeRD: the sign and the
significance were as predicted, the **magnitude was not** — +0.007, a quarter of the pre-registered
+0.03. C-029 explains why: A1's +0.125 importance was pooled (across impressions), and the within-slate
signal an additive term can use is small. The term still buys 0.007 because `g` is non-monotone —
C-029's fresher-first/older-first check bounds only monotone signal — but the gap to the reranker
(0.673) barely moves: **0.113 → 0.106**.

**What row 3 means, and does not mean.** Masking the term at inference is a *dependence* check, not a
removal: on EB-NeRD the encoders co-adapted to the term (−0.017 without it, below the baseline itself),
on MIND they learned to ignore it (Δ 0.000 ± 0.0002). The clean "component removed" row is row 1, which
was trained without the term.

**Sampled vs full slates.** On the sampled 5-candidate holdout the variant's `val_auc` was 0.7149 vs the
baseline's 0.6393 (+0.076); on the full validation slates the gain is +0.007. Sampled-slate validation
overstated the change by an order of magnitude — the reason §13.2 evaluates on full slates.

**Serving-time honesty (Q9).** The term uses publish time (EB-NeRD) / first-seen time strictly before *t*
(MIND); both exist at serving time. No serving-unavailable feature is involved.

**Cost.** GPU: EB-NeRD 3.9 h (main), MIND 2.2 h (alt), pre-flight checks ≈ 1.3 h across both; totals
this week: main 12.2 h of 30, alt 2.5 h of 30. Laptop: each `make paired` on EB-NeRD 46 s at 1.4 GB peak.


_Not started._

## Q4 · Serving and scale — 2026-09-14, Aayush Pandey (P4, SPEC.md §16, CONTEXT.md C-031)

**Setup.** The locked two-stage reranker (`config.FINAL`) on A1's stage 1, assembled into a
per-request path (`src/serving/`) that reproduces the batch path **bit for bit** on 200 real
impressions per dataset (`tests/test_serving.py`; two skews caught on the way: the history snapshot
and a float64/float32 mismatch that flipped 11/8,360 MIND scores at tree thresholds). The served
model is `config.FINAL` refitted by `src/serving/models.py`, which reproduces Q2's locked numbers
exactly: EB-NeRD AUC 0.67283 [0.6710, 0.6745] (58 s fit), MIND 0.67472 [0.6725, 0.6768] (49 s).

Command: `make bench DATASET=ebnerd` / `make bench DATASET=mind` = `taskset -c 0 python scripts/bench.py`.
Machine: Intel i7-14650HX laptop, **one core**, LightGBM `num_threads=1`, OpenMP/BLAS 1 thread;
1,000 seeded validation impressions (seed 0) after 100 warm-ups. Records:
`data/processed/bench_ebnerd.json`, `bench_mind.json` (every number below, hardware, commit).

### Q4.1 · Memory

| component | what it holds | EB-NeRD disk / RAM | MIND disk / RAM |
|---|---|---|---|
| articles | title tokens, category, first-known time | 150.8 MB / 49.5 MB | 74.7 MB / (inside BM25) |
| BM25 inverted index | postings + forward index (Python dicts/lists) | — / **412.1 MB** (125,541 docs, 120,400 terms) | — / 203.4 MB (65,238 docs) |
| ANN index (FAISS flat) | `ntotal × d × 4` | 153.6 MB / 150.6 MB (125,541 × 300) | 179.1 MB / 192.9 MB (125,590 × 384) |
| popularity counts (`RollingCounts`) | one timestamp list per article per event type | 10.3 MB / 160.1 MB (1,892 articles, 2.8 M events) | 92.0 MB / 347.4 MB (7,713 articles) |
| user store | last-5 ids + click log with categories per user | 21.8 MB / 182.7 MB (15,342 users) | 42.8 MB / 156.7 MB (50,000 users) |
| session store | pre-*t* session position per impression | 11.3 MB / 48.6 MB (244,647) | n/a |
| **process RSS after build** | | **1.16 GB** (peak 2.41 GB incl. the batch-path comparison) | **1.38 GB** (peak 1.40 GB) |
| build time | | 7 s | 6 s |

RAM is the RSS delta while building each component, so it includes Python-object overhead — the
honest number for this implementation. Two of it are pathological and are the 10× story below:
the BM25 index (≈ 3.3 KB per document in dicts) and the event-level counts (≈ 57 B per event).

### Q4.2 · Latency (p50 / p95 / p99 of one request on one core, ms)

| request | EB-NeRD total | of which: retrieve (BM25 / ANN) | features | score | cands | req/s |
|---|---|---|---|---|---|---|
| **(a) rerank the impression** — what ships | **1.23 / 1.50 / 1.71** | 0.1 | 1.3 | 0.4 | 12.1 | **790** |
| (b) retrieve K=100 ∪ K=100, then rerank | 48.0 / 65.9 / **72.0** | 68.2 (**60.0** / 8.4) | 2.9 | 0.9 | 198 | 21 |
| (b) K=200 | 50.5 / 67.1 / 74.9 | 69.2 (60.7 / 8.8) | 4.2 | 1.6 | 395 | 20 |
| (a), naive — profile recomputed per candidate (first measurement) | 8.7 / 29.6 / 47.3 | 0.1 | 46.8 | 0.5 | 12.1 | 85 |
| P2 batch path, same 1,000 impressions | 0.85 mean | | | | | |

| request | MIND total | retrieve (BM25 / ANN) | features | score | cands | req/s |
|---|---|---|---|---|---|---|
| **(a)** | **1.66 / 2.66 / 3.37** | 0.2 | 1.1 | 2.2 | 37.2 | **550** |
| (b) K=100 | 61.9 / 78.7 / **86.5** | 82.5 (**73.9** / 8.9) | 1.7 | 2.5 | 189 | 17 |
| (b) K=200 | 63.7 / 80.4 / 88.2 | 81.9 (73.3 / 8.9) | 2.9 | 3.5 | 378 | 16 |
| P2 batch path | 1.96 mean | | | | | |

Reading it, by service demand (D = V·S):

- **(a) is cheap:** p99 1.7 / 3.4 ms; the per-request overhead over the batch path is ≈ 1.5×.
  The first measurement was 47 ms p99 — the two category-profile features recomputed the user's
  decayed click masses once *per candidate*; computing them once per request (same definitions,
  parity kept) removed 96 % of the features time. Found by measuring, not by guessing.
- **(b) is retrieval-bound, and BM25-bound:** the pure-Python postings scan for an 82-token
  (EB-NeRD) / 128-token (MIND) query — five clicked titles + subtitles/abstracts — costs 60–74 ms
  p99 and is **independent of K** (K=200 adds ≈ 3 ms of features/scoring). The flat FAISS scan of
  125k × 300–384 vectors costs 8.4–8.9 ms. Features for ≈ 200 candidates: 3 ms. GBDT: 1–3 ms.
- **MIND's (a) scoring is 2.2 ms for 37 candidates:** sklearn `predict_proba` call overhead, not
  tree work (LightGBM on EB-NeRD does 12 candidates in 0.4 ms).

### Q4.3 · Cost at p99 < 100 ms

Model (SPEC §16.2): cores = ⌈QPS × mean service / ρ⌉ with ρ = 0.5 (M/M/1 mean wait = one service
time, so p99 stays near the single-request p99); cost/1k = cores × price / (QPS × 3.6).
Price: **$0.0425 per vCPU-hour** = AWS EC2 on-demand c7i.large (2 vCPU, $0.085/h), us-east-1,
https://aws.amazon.com/ec2/pricing/on-demand/ read 2026-09-14. All numbers per one vCPU of
this laptop's class; a cloud vCPU is a hyperthread and typically slower, so treat cores as a
lower bound.

| dataset · framing | mean / p99 | 100 QPS | 1,000 QPS | SLA (p99 < 100 ms) |
|---|---|---|---|---|
| EB-NeRD (a) | 1.26 / 1.71 ms | 1 core, **$0.00012 / 1k** | 3 cores, **$0.00004 / 1k** | holds |
| EB-NeRD (b) K=100 | 48.0 / 72.0 ms | 10 cores, $0.00118 / 1k | 96 cores, $0.00113 / 1k | holds, 28 ms of headroom |
| MIND (a) | 1.81 / 3.37 ms | 1 core, $0.00012 / 1k | 4 cores, $0.00005 / 1k | holds |
| MIND (b) K=100 | 59.9 / 86.5 ms | 12 cores, $0.00142 / 1k | 120 cores, $0.00142 / 1k | holds, 13 ms of headroom |

Ten cents per million requests for the shipped path; a dollar per million for retrieve-then-rerank.

### Q4.4 · 10× — what breaks first (RUM: reads, updates, memory, from the measured shares)

| 10× in… | reads per request | updates per event | memory | what breaks |
|---|---|---|---|---|
| **articles** (1.25 M) | BM25 scans ≈ 10× the postings for the same 82–128 query tokens → **≈ 600–740 ms p99**; flat ANN does 10 × `ntotal × d` → ≈ 85–90 ms alone | none | BM25 dicts 4.1 GB, ANN 1.5–1.9 GB | **(b) breaks first, on BM25**: the SLA is gone by 6×. Fix in order of payoff: a compiled index with top-k pruning (WAND/MaxScore — reads ∝ K·log N, not N), query truncation (titles only halves the tokens), then IVF/HNSW for the ANN. **(a) is unaffected**: it never scans the corpus |
| **users** (150 k–500 k) | unchanged (per-user dict lookups) | history append per click | user store 1.6–1.8 GB (≈ 12 KB/user as polars frames) | nothing in latency; memory only. A compact log (int64 ids + timestamps, ≈ 100 B/click) cuts it 10× |
| **events** (counts) | unchanged (bisect on a per-article list) | one list append per view and per click | counts 1.6–3.5 GB — the store keeps **every** event as a Python datetime | the feature store's memory: the only component that grows without bound. A live system keeps windowed aggregates (`pop_total`/`ctr_total` need two integers per article), not events |
| **QPS** (10 k) | unchanged per request | unchanged | unchanged | cores scale linearly: (a) ≈ 26–37 cores, (b) ≈ 960–1,200; the (b) bill grows 10× with no way down except the BM25 fix |

So: the shipped path (a) scales to 10× on all three axes with only memory to buy; the literal
retrieve-then-rerank path (b) is already within 13–28 ms of the SLA and fails at ≈ 1.4× more
articles, because A1's BM25 is a Python postings scan. The stage that decides the 10× question is
not the model — GBDT scoring is 1–3 ms and O(K) — but the lexical index.

**Serving-time honesty.** Every served feature is in `config.FINAL`, which `model_features`
already filtered for serving safety (C-013); the per-request path uses strict `< t` lookups
(tested). Q4.5: this is a measured local benchmark plus a scaling argument, as the brief allows.


_Not started._

## Q5 · Extended evaluation and leaderboard submissions — 2026-09-15, Anurag Kaushal (P5, SPEC.md §17)

**Status: the harness and the extended metrics are done; the Codabench submissions are not.** The
locked reranker's scores exist for both evaluation splits, `make eval` reports every Q5 metric with
both required slices, and the reranker has been paired against Aayush's NRMS. Still open: test-set
inference and the two leaderboard uploads with their screenshots.

### The scores files (SPEC.md §13.3 contract)

Command: `PYTHONPATH=. .venv/bin/python -u scripts/score_final.py --dataset ebnerd|mind`. Each
fits `src/rerank/config.FINAL` (C-018) on the same seeded training sample as the measured runs and
scores **every impression** of the evaluation split, because the contract requires 1:1 alignment
with the other systems' files.

| File | Rows | Impressions | Model |
|---|---|---|---|
| `data/scores/ebnerd/validation/reranker_final.parquet` | 2,928,942 | 244,647 | lambdarank, 11 features |
| `data/scores/mind/MINDsmall_dev/reranker_final.parquet` | 2,740,998 | 73,152 | pointwise, 7 features |

Both match Aayush's NRMS files row for row, so every pairing below is over the full split.

### `make eval` — EB-NeRD, all 244,647 validation impressions

Command: `make eval SCORES=data/scores/ebnerd/validation/reranker_final.parquet JSON=data/processed/eval/ebnerd_reranker_final.json`
(top-10 for the beyond-accuracy trio; bootstrap 95% CI over impressions, 1,000 resamples, seed 0).

| slice | impressions | AUC | MRR | nDCG@5 | nDCG@10 | diversity | novelty | coverage | Gini |
|---|---|---|---|---|---|---|---|---|---|
| all | 244,647 | 0.6734 [0.6723, 0.6746] | 0.4376 [0.4362, 0.4388] | 0.4985 [0.4971, 0.4999] | 0.5509 [0.5497, 0.5520] | 0.7864 | 17.118 | 0.8568 | 0.8097 |
| cold | 887 | 0.6749 [0.6570, 0.6934] | 0.4308 [0.4094, 0.4518] | 0.4830 [0.4594, 0.5048] | 0.5431 [0.5245, 0.5612] | 0.7891 | 17.053 | 0.2370 | 0.4529 |
| warm | 243,760 | 0.6734 [0.6722, 0.6745] | 0.4376 [0.4363, 0.4389] | 0.4986 [0.4973, 0.4999] | 0.5509 [0.5498, 0.5521] | 0.7864 | 17.118 | 0.8558 | 0.8095 |
| head | 4,704 | 0.7537 [0.7461, 0.7617] | 0.4923 [0.4824, 0.5029] | 0.5403 [0.5297, 0.5513] | 0.5834 [0.5740, 0.5931] | 0.7739 | 13.391 | 0.4116 | 0.7675 |
| tail | 239,943 | 0.6718 [0.6707, 0.6730] | 0.4365 [0.4351, 0.4377] | 0.4977 [0.4963, 0.4990] | 0.5503 [0.5491, 0.5514] | 0.7867 | 17.191 | 0.8509 | 0.8107 |

### `make eval` — MIND, all 73,152 dev impressions

Command: `make eval SCORES=data/scores/mind/MINDsmall_dev/reranker_final.parquet JSON=data/processed/eval/mind_reranker_final.json`.

| slice | impressions | AUC | MRR | nDCG@5 | nDCG@10 | diversity | novelty | coverage | Gini |
|---|---|---|---|---|---|---|---|---|---|
| all | 73,152 | 0.6747 [0.6725, 0.6768] | 0.3295 [0.3272, 0.3321] | 0.3630 [0.3603, 0.3657] | 0.4217 [0.4193, 0.4243] | 0.8085 | 15.825 | 0.5733 | 0.9463 |
| cold | 12,982 | 0.6336 [0.6284, 0.6385] | 0.3412 [0.3356, 0.3471] | 0.3674 [0.3612, 0.3741] | 0.4236 [0.4180, 0.4295] | 0.8560 | 15.715 | 0.2984 | 0.9173 |
| warm | 60,170 | 0.6836 [0.6813, 0.6857] | 0.3270 [0.3244, 0.3296] | 0.3620 [0.3592, 0.3650] | 0.4213 [0.4186, 0.4241] | 0.7983 | 15.848 | 0.5496 | 0.9430 |
| head | 21,098 | 0.7304 [0.7271, 0.7335] | 0.3379 [0.3338, 0.3423] | 0.3907 [0.3859, 0.3957] | 0.4444 [0.4400, 0.4491] | 0.8010 | 14.730 | 0.3898 | 0.9268 |
| tail | 52,054 | 0.6521 [0.6496, 0.6548] | 0.3261 [0.3232, 0.3292] | 0.3518 [0.3486, 0.3552] | 0.4125 [0.4096, 0.4154] | 0.8116 | 16.268 | 0.5170 | 0.9439 |

**Two checks that the harness is measuring the right thing.**

- **MIND's "all" row reproduces Q2 exactly** (0.6747 AUC): Q2 already evaluated MIND on the whole
  dev split, so the two must agree, and they do to four decimals on every metric.
- **EB-NeRD's "all" row sits where Q2's two samples predict.** Q2 measured 0.6728 on a 100,000-impression
  sample and 0.6738 on its disjoint 144,647-impression holdout; their impression-weighted mean is
  0.67339, against 0.6734 measured here on the union.

### What the slices say

- **Head impressions are much easier on both datasets** (+0.080 AUC on EB-NeRD, +0.078 on MIND
  against their tails). The model ranks articles it has seen clicked in training far better than
  the ones it has not — the same coverage problem A1 measured, now quantified per slice.
- **Cold-start splits the datasets.** On MIND, cold users lose 0.050 AUC against warm ones
  (0.6336 vs 0.6836): with ≤ 5 history clicks the profile and category features have almost
  nothing to work with. On EB-NeRD cold users are statistically indistinguishable (0.6749 vs
  0.6734), but only **887 of 244,647** impressions are cold there, so that CI is wide
  [0.6570, 0.6934] and the comparison is weak.
- **Coverage is low where it is measured on fewer impressions** (cold 0.24 / 0.30) — coverage is a
  union over recommendations, so it scales with how many impressions the slice contains and is not
  comparable across slices of different sizes. Gini is the comparable concentration number.
- **Novelty is lower on head impressions** (13.4 vs 17.2 on EB-NeRD), which is the definition
  working: head impressions are the ones whose clicked article is popular.

### Reranker vs NRMS — the definitive comparison (paired, SPEC.md §14)

Command per row: `make paired A=data/scores/<ds>/<split>/<nrms|nrms_fresh>.parquet B=data/scores/<ds>/<split>/reranker_final.parquet JSON=data/processed/paired/<ds>_reranker_vs_<system>.json`.

| Comparison, Δ = reranker − NRMS | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| **EB-NeRD** vs NRMS (Q3.1 baseline) | **+0.1134 [+0.1117, +0.1151]** | +0.0885 [+0.0868, +0.0900] | +0.1101 [+0.1084, +0.1119] | +0.0841 [+0.0827, +0.0854] |
| **EB-NeRD** vs NRMS + freshness (Q3.2, their best) | **+0.1060 [+0.1045, +0.1076]** | +0.0791 [+0.0776, +0.0807] | +0.0980 [+0.0964, +0.0997] | +0.0750 [+0.0737, +0.0763] |
| **MIND** vs NRMS | **+0.0080 [+0.0059, +0.0101]** | +0.0075 [+0.0057, +0.0096] | +0.0073 [+0.0052, +0.0096] | +0.0033 [+0.0014, +0.0052] |
| **MIND** vs NRMS + freshness | **+0.0086 [+0.0065, +0.0108]** | +0.0068 [+0.0049, +0.0088] | +0.0065 [+0.0043, +0.0087] | +0.0035 [+0.0016, +0.0054] |

**The feature-based reranker beats the neural baseline on both datasets, on every metric, with
every CI excluding 0** — but the margins are of different orders. On EB-NeRD it is +0.11 AUC, a
gap the freshness term closes only slightly (0.1134 → 0.1060). On MIND it is +0.008, small but
significant. The honest reading is that NRMS was trained on `ebnerd_small` / `MINDsmall_train`
under a fixed recipe and 5 epochs, while the reranker consumes A1's stage-1 scores plus
point-in-time behavioural features; this is a comparison of *systems as built here*, not a claim
about the architectures in general.

### Q5.6 · Test-set inference and Codabench submissions — 2026-09-15 (C-038)

Both test files are scored with `config.FINAL` by `scripts/submit_a2.py`, run as a Kaggle CPU
kernel (`scripts/kaggle/submit_<dataset>/`). The driver fits once through
`src.serving.models.load_or_fit`, builds each chunk through the **same `build()` the measured runs
use**, converts scores to ranks with `ranks_from_scores` only, writes chunks atomically so a
killed session resumes from the first missing one, then validates the assembled file against the
test file's own ids and slate lengths before zipping.

| dataset | test impressions | candidate rows | chunks | kernel wall time | predictions | zip |
|---|---|---|---|---|---|---|
| MIND (`MINDlarge_test`) | 2,370,727 | 93,115,001 | 24 × 100k | 2,993 s (49.9 min) | 291.3 MB | 107.5 MB |
| EB-NeRD (`ebnerd_testset`) | 13,536,710 | 205,925,868 | 55 × 250k | 7,098 s (118.3 min) | 703.1 MB | 229.7 MB |

**Verification before upload** (`scripts/kaggle/fetch_submission.sh <dataset> <vN>`), all passed
on both: the downloaded zip's sha256 equals the one the kernel printed (MIND `9468cea3…82fa1`,
EB-NeRD `51f531a5…bf37`); each archive holds `predictions.txt` at its root; and `validate_file`
re-run locally over every line reports each rank list a valid permutation of 1..N and a line count
equal to the test file's impression count — MIND 2,370,727 lines / 2,370,727 distinct ids / 0
duplicates, EB-NeRD 13,536,710 lines / 13,336,711 distinct ids / **199,999 duplicates**, which is
exactly the 200,000 beyond-accuracy rows sharing `impression_id` 0 (SPEC §6) and is why the
EB-NeRD check runs with `allow_duplicate_ids=True`. Manifest, chunk ledger and kernel log are kept
under `data/submissions/_kaggle/<dataset>/<vN>/` and `data/logs/kaggle/` (both gitignored); each
run has a row in `scripts/kaggle/RUN_LEDGER.md`.

**Cost note (why CPU).** Measured on 20,000 EB-NeRD impressions: feature building is **96.0 %** of
the time (BM25 postings, polars joins, per-user profile lookups), GBDT scoring **3.9 %**. A perfect
GPU would therefore remove about 4 % of the run, and neither LightGBM's inference path nor
sklearn's `HistGradientBoostingClassifier` has a GPU one to begin with. The 2× T4 / fp16 setup
(C-020) is right for NRMS — dense matmuls — and wrong for a tree ensemble. The lever that mattered
was `InvertedIndex.avg_doc_length` (commit `b4c4fb7`), 37 % of the runtime before it was cached.

### Still open in Q5

Both zips are built and verified. What remains is the two Codabench uploads and their screenshots.

| dataset | competition | zip | leaderboard score | screenshot |
|---|---|---|---|---|
| MIND | [13967](https://www.codabench.org/competitions/13967/) | ready 2026-09-15 (107.5 MB) | _pending_ | _pending_ |
| EB-NeRD | [2469](https://www.codabench.org/competitions/2469/) | ready 2026-09-15 (229.7 MB) | _pending_ | _pending_ |

## Q9 · With and without serving-unavailable features — 2026-09-15, Anurag Kaushal (C-034, C-036)

Command (one per dataset; records in `data/processed/q9/<dataset>.json`):

```
PYTHONPATH=. .venv/bin/python -u scripts/ablation_q9.py --dataset ebnerd
PYTHONPATH=. .venv/bin/python -u scripts/ablation_q9.py --dataset mind
```

**What is measured.** `config.FINAL` is already the serving-honest model: it contains no member of
`UNSAFE_FEATURES`, and `model_features(for_submission=True)` also drops `ABSENT_FROM_TEST_FILE`
(SPEC §11.4). So the "without" row is the model that ships, and the ablation adds the forbidden
features back to see what a model that ignored the registry would have scored. Everything else is
held fixed — the same seeded fit sample (EB-NeRD 100,000 train-week impressions, MIND 80,000), the
same objective, the same evaluation split (every impression, not a sample) and the same seed —
so the delta is attributable to the feature set alone. Paired bootstrap over per-impression
metrics, 1,000 resamples, seed 0.

Rows:

- **serving_safe** — `config.FINAL` exactly (EB-NeRD 11 features, MIND 7). What ships.
- **plus_unsafe** — + every `UNSAFE_FEATURES` column the dataset builds: `session_len` (counts the
  session's impressions *after* *t*), `cur_read_time` and `cur_scroll_percentage` (describe a page
  view that outlasts the moment of serving). A live system cannot know any of them at request time.
- **plus_absent** — + `n_prior_clicks_in_session`, reported as its own row per the brief. It is
  serving-*safe* (a live service knows its own session's clicks) but it is built from
  `article_ids_clicked`, which the Codabench test file does not ship, so a model trained on it would
  meet a missing column at test time (the A1 submission-3 failure, SPEC §7).

### Q9.1 · EB-NeRD (`ebnerd_small/validation`, all 244,647 impressions, lambdarank)

| row | features | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|
| **serving_safe** (ships) | 11 | **0.6734** [0.6723, 0.6746] | **0.4376** [0.4362, 0.4388] | **0.4985** [0.4971, 0.4999] | **0.5509** [0.5497, 0.5520] |
| plus_unsafe | 14 | 0.6498 [0.6488, 0.6511] | 0.4119 [0.4106, 0.4132] | 0.4695 [0.4682, 0.4709] | 0.5295 [0.5284, 0.5306] |
| plus_absent | 12 | 0.6598 [0.6587, 0.6610] | 0.4219 [0.4206, 0.4232] | 0.4815 [0.4802, 0.4828] | 0.5383 [0.5372, 0.5394] |

Paired deltas against serving_safe (95% CI; every interval excludes 0):

| Δ row − serving_safe | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| plus_unsafe | **−0.0236** [−0.0245, −0.0227] | −0.0257 [−0.0266, −0.0247] | −0.0290 [−0.0300, −0.0280] | −0.0214 [−0.0222, −0.0206] |
| plus_absent | **−0.0136** [−0.0143, −0.0129] | −0.0157 [−0.0166, −0.0148] | −0.0170 [−0.0179, −0.0162] | −0.0126 [−0.0133, −0.0119] |

**Reading.** The usual Q9 story is that a serving-unavailable feature inflates the offline number
and the honest model looks worse. Here it is the reverse: **adding the forbidden features makes the
listwise model significantly worse on all four metrics**, by 0.024 AUC for the unsafe trio and
0.014 for `n_prior_clicks_in_session`. This matches the Phase 2 finding that the dwell family hurt
lambdarank (C-018): these columns are strong at the *impression* level (a long read time says the
user is engaged) but nearly constant *within* an impression, so a pairwise-loss model that only
sees within-list differences gets noise from them. Being serving-honest costs nothing on EB-NeRD;
it is the better model as well as the only deployable one. The serving_safe row reproduces Q5's
all-row AUC 0.6734 exactly, which cross-checks `scripts/ablation_q9.py` against `make eval`.

### Q9.2 · MIND (`MINDsmall_dev`, all 73,152 impressions, pointwise HistGBDT)

| row | features | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|
| **serving_safe** (ships) | 7 | **0.6747** [0.6725, 0.6768] | **0.3295** [0.3272, 0.3321] | **0.3630** [0.3603, 0.3657] | **0.4217** [0.4193, 0.4243] |

**MIND has no serving-unavailable feature to add.** It ships no session, dwell, read-time or
scroll columns (SPEC §11.5), so every column the registry marks unsafe is absent by construction,
and `n_prior_clicks_in_session` cannot exist without sessions. The script reports the single row
and says so rather than manufacturing a comparison. The row reproduces Q2's 0.6747 exactly.

### Q9.3 · What the leaderboard sees

Both submissions score the test file with `config.FINAL`, i.e. the serving_safe row. No number in
this report was produced with a feature the test file lacks or a live system could not have; the
two other rows exist only to be disclosed here.
