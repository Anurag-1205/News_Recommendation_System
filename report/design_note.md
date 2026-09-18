# Learning from Click-Logs on EB-NeRD and MIND

**CS4.406 Information Retrieval and Extraction — Assignment 2, design note**
Anurag Kaushal (2025202013) · Aayush Pandey (2025201058) · 18 September 2026 · branch `a2-click-logs`

**Scope.** Q6 of the brief: what we built and why, the baseline reproduced and then changed with an ablation and paired confidence intervals, the serving and scale measurements, and where the system breaks at ten times the load. Every number here has a command and a record in `RESULTS.md`; every decision has a numbered entry in `CONTEXT.md` (C-001–C-041); interfaces and verification are in `SPEC.md`. Nothing in this note was produced without a test or oracle that would have failed before the change.

**Statistical convention.** Every headline metric carries a bootstrap 95 % CI over impressions (1,000 resamples). A *difference* between two systems is always a **paired** bootstrap on the same impressions; the word "beats" appears only where that interval excludes zero, and the one Q3 claim was reviewed by the team member who did not build it (C-033).

## 1. What we built

A two-stage recommender that re-ranks the articles a user is shown by click likelihood, evaluated on two datasets that share almost nothing structurally — MIND (English, TSV, history inline, 2.37 M test impressions) and EB-NeRD (Danish, Parquet, history in a separate file, 13.5 M test impressions) — plus the official neural baseline (NRMS) reproduced on both, one principled change to it, and a per-request serving path that is held to the batch path by a parity test.

```
raw archives ─► unified schema ─► temporal split ─► feature store (as-of cutoff, strictly < t)
                                                             │
    stage 1 (A1, reused): BM25 over the last-5 clicked titles ∪ flat-FAISS cosine of the user vector
                                                             │
    Phase 1 features: recency-weighted category profile · category match · popularity/CTR
                      · freshness · list position · session position · (dwell — measured, dropped)
                                                             │
    stage 2: config.FINAL — EB-NeRD LightGBM lambdarank (11 features) · MIND HistGBDT pointwise (7)
                                                             │
    scores file (imp_row, cand_position, score) ─► make eval · make paired · Codabench writer
    ─────────────────────────────────────────────────────────────────────────────────────────
    baseline track: NRMS (ebnerd-benchmark on EB-NeRD; Microsoft Recommenders on MIND), 5 epochs,
                    deterministic; + one change (a freshness term); paired against the reranker
```

**Roles.** Anurag built the features, the reranker, the extended evaluation, the anti-gaming ablation and the submission pipeline (Q1, Q2, Q5, Q9); Aayush reproduced the baseline, built the paired-bootstrap judge, made and ablated the change, and built the serving benchmark (Q3, Q4). Each reviewed the other's claims (C-019, C-033).

| locked configuration (`src/rerank/config.FINAL`, C-018) | objective | features | AUC on the evaluation split |
|---|---|---|---|
| EB-NeRD (`ebnerd_small` train week → validation week, 244,647 impressions) | LightGBM lambdarank, one query per impression | bm25, semantic, pop_total, ctr_total, freshness_hours, n_candidates, history_len, recency_weighted_profile, category_match, cand_position, session_pos | **0.6734** [0.6723, 0.6746] |
| MIND (`MINDsmall_train` → `MINDsmall_dev`, 73,152 impressions) | HistGBDT pointwise | bm25, semantic (MiniLM), pop_total, ctr_total, n_candidates, cat_affinity, history_len | **0.6747** [0.6725, 0.6768] |

## 2. Design choices and the alternatives

**Temporal split only, and model selection that mimics the test gap.** Interaction data is never split at random (SPEC §3). A1 lost 0.090 AUC on a MIND submission to a subtler trap: trailing-window counts that are populated on an *adjacent* dev split and empty on the test split. The gap-aware protocol (SPEC §7) fits counts strictly before a cutoff, trains one day later and evaluates a day after that, so selection sees the same degenerate regime the test file imposes. Windowed counts were dropped from the shipped model for exactly this reason.

**Every feature at time *t* sees only events strictly before *t*, and a test asserts it.** Each Phase 1 family (Q1) landed with a hand-computed oracle on a toy log and a "append the future, nothing safe changes" leakage test. Two corrections came out of that discipline (C-013): `session_len` counts a session's *later* impressions and is unsafe; `n_prior_clicks_in_session` is safe in production but absent from the Codabench test file, so it is omitted rather than zero-filled — the train/serve skew that sank an A1 submission would otherwise have recurred. The recency half-life for the category profile was gridded (6 h … ∞) and the decay found not to help; h = ∞ ships (C-014).

**Listwise where it wins, pointwise where it does not.** Lambdarank (one LightGBM query per impression) beats pointwise HistGBDT by +0.034 AUC on EB-NeRD's base features but *loses* by −0.009 on MIND, in both halves of dev and at a truncation level covering the longest slate (C-016, C-017). The reranker is therefore lambdarank on EB-NeRD and pointwise on MIND — one configuration per dataset, each chosen on a paired CI, not on preference. On EB-NeRD the dwell family (`hist_read_time_mean`, `hist_scroll_mean`) hurt the listwise model by −0.015 AUC and was dropped after the drop held on 144,647 held-out impressions (C-018).

**The official baseline, as the brief names it, per dataset (D3, C-024).** NRMS from `ebnerd-benchmark` (pinned `5164e2c`) on EB-NeRD and the Microsoft Recommenders NRMS — the implementation the MIND paper itself cites — on MIND. Neither runs unmodified on today's Kaggle image: the benchmark pins polars 0.20 and needed a two-function compatibility shim held to its own docstrings (C-021); Recommenders is TF1-style Keras and runs under `tf-keras`. Keras mixed precision fails inside the benchmark's attention layer, so NRMS trains in float32 (C-022). Both baselines are **deterministic to every printed digit** across Kaggle sessions — after seeding Python's `random`, which the MIND package leaves unseeded while drawing its negatives from it (a 0.002 AUC run-to-run wobble that would have swamped a small ablation).

**One change, pre-registered (D4, C-028).** `score = user·news + g(freshness)`, a two-layer MLP on the candidate's standardised log-age, using the reranker's own point-in-time freshness feature so both systems agree on what "fresh" means. The prediction was written down before any run: EB-NeRD Δ AUC ≥ +0.03, MIND within ±0.01. A diagnostic run *before* the training runs (C-029) showed the prediction would likely fail on magnitude — freshness alone ranks within-slate clicks at AUC 0.501 on EB-NeRD — and said why: the A1 evidence for freshness was pooled importance across impressions, not within a slate. The runs went ahead unchanged; §3.3 reports the outcome against the prediction.

**The hand-off contract.** Neither of us imports the other's code. Every system writes one Parquet scores file per split, keyed by `imp_row` (the file-order row) and 1-based `cand_position`, Float64 scores, labels never in the file. `impression_id` cannot be the key: 200,000 EB-NeRD test rows share `impression_id` 0 (C-023). The judge (`make paired`) refuses files whose manifests differ in dataset, split or framing.

**Retrieve-then-rerank, measured and descoped (D1, C-037).** The brief's "retrieve top-K from the corpus, then rerank" was built end to end (§4 times it) and its stage-1 recall was measured: the clicked article is in the BM25 ∪ ANN union **1 in 100** impressions on EB-NeRD (hit@100 = 0.010 [0.004, 0.016]) and 2 in 100 on MIND — because clicked articles are a median 4 hours old and content similarity to the user's last five reads retrieves articles a median 3,024 hours old. The publisher's slate is already a freshness-filtered candidate set; the leaderboards score it; the shipped system re-ranks it.

## 3. Results

### 3.1 The reranker: before and after (Q2)

| EB-NeRD, 100k fit / 100k evaluation impressions | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| stage 1: BM25 only | 0.5036 [0.5020, 0.5054] | 0.3201 | 0.3521 | 0.4354 |
| stage 1: word2vec cosine only | 0.5059 [0.5038, 0.5079] | 0.3214 | 0.3526 | 0.4379 |
| pointwise GBDT, A1 base (7) | 0.6359 [0.6340, 0.6377] | 0.4039 | 0.4571 | 0.5194 |
| **lambdarank, A1 base (7)** | **0.6702** [0.6685, 0.6721] | 0.4345 | 0.4953 | 0.5495 |
| paired Δ, lambdarank A2 − BM25 (the two-stage gain) | **+0.1549** [+0.1523, +0.1570] | +0.1040 | +0.1309 | +0.1035 |
| paired Δ, lambdarank − pointwise, base | +0.0343 [+0.0328, +0.0359] | +0.0306 | +0.0382 | +0.0301 |

| MIND, 80k fit / all 73,152 dev impressions | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| stage 1: MiniLM cosine only | 0.6353 [0.6330, 0.6374] | 0.3084 | 0.3371 | 0.3960 |
| **pointwise GBDT, A1 v4 base (7) = FINAL** | **0.6747** [0.6725, 0.6768] | 0.3295 | 0.3630 | 0.4217 |
| lambdarank, base (7) | 0.6663 [0.6642, 0.6683] | 0.3239 | 0.3580 | 0.4169 |
| paired Δ, lambdarank − pointwise | −0.0084 [−0.0095, −0.0073] | −0.0056 | −0.0050 | −0.0048 |

Re-ranking lifts EB-NeRD by +0.155 AUC over its best single generator; on MIND the semantic stage is already strong and the reranker adds +0.039. The Phase 1 families themselves were a negative result on the listwise model: the leave-one-family-out ablation (RESULTS Q2) showed the category profile helping (−0.009 AUC when removed) and dwell hurting (+0.014 when removed), which is what `config.FINAL` encodes.

### 3.2 The baseline, reproduced (Q3.1)

| | ours (validation, every impression, full slates) | published | gap, explained |
|---|---|---|---|
| EB-NeRD NRMS, 5 epochs, xlm-roberta-large | **0.5600** [0.5588, 0.5612] AUC · MRR 0.3491 · nDCG@10 0.4668 | 0.6103 / 0.3975 / 0.5124 (Kruse et al. 2024, Table 3, hidden test set, trained on `large`) | −0.050: 12× less training data, validation vs hidden test; the published recipe also silently drops the word embeddings it builds (C-021), which we pass |
| MIND NRMS, 5 epochs, GloVe | **0.6667** [0.6647, 0.6688] · 0.3220 · 0.4184 | 0.6776 / 0.3305 / 0.4163 (Wu et al. 2020, Table 3, full-MIND test, half the users, mean of 10 runs) | −0.011: 10× less training data; same implementation and initialisation |

The reproduction's own finding: NRMS trails the reranker by **0.113 AUC on EB-NeRD but only 0.008 on MIND**. NRMS knows nothing about article age or popularity, and A1 had found EB-NeRD "recency-dominated". That asymmetry chose the change.

### 3.3 The change, its ablation, and the prediction (Q3.2–3.4)

| paired Δ (variant − baseline), every impression | AUC | MRR | nDCG@5 | nDCG@10 | verdict (SPEC §14) |
|---|---|---|---|---|---|
| **EB-NeRD** NRMS + freshness − NRMS (0.5674 vs 0.5600) | **+0.0074** [+0.0066, +0.0081] | +0.0094 [+0.0085, +0.0101] | +0.0121 [+0.0112, +0.0129] | +0.0092 [+0.0085, +0.0098] | beats — reviewed and signed off (C-033) |
| EB-NeRD, row 2 scored with the term masked − row 2 | −0.0174 [−0.0180, −0.0169] | −0.0157 | −0.0190 | −0.0157 | the encoders co-adapted to the term |
| **MIND** NRMS + freshness − NRMS (0.6661 vs 0.6667) | −0.0006 [−0.0014, +0.0002] | +0.0007 [−0.0001, +0.0015] | +0.0008 [−0.0001, +0.0018] | −0.0002 [−0.0010, +0.0005] | **no significant difference** |
| MIND, masked − row 2 | +0.0000 [−0.0002, +0.0002] | −0.0001 | +0.0001 | +0.0001 | the model learned to ignore it |

**Against the pre-registration.** MIND: as predicted, a null, and we report it as one. EB-NeRD: right on sign and significance, **wrong on magnitude** — +0.007, a quarter of the predicted +0.03 — for the reason C-029 gave before the runs: within-slate freshness signal is small; `g` is non-monotone and captures what a monotone check cannot, and no more. Two lessons travel to the next project: *inference-time masking is a dependence check, not a removal* (the clean "component removed" row is the baseline trained without the term), and *sampled-slate validation overstates changes* — on the 5-candidate training holdout the term looked like +0.076 AUC; on full slates it is +0.007. The judge itself was calibrated on synthetic pairs with known Δ (coverage ≥ 90 % over 200 trials) and on the real file (a rank-preserving rescale gives Δ exactly 0; N(0, 0.01) noise is correctly detected as a small real loss).

### 3.4 Reranker versus NRMS (Q5, C-035)

The feature-based two-stage reranker beats the neural baseline on both datasets, every metric, every paired CI excluding zero: EB-NeRD **+0.1134** AUC [+0.1117, +0.1151] against the baseline and +0.1060 against the freshness variant; MIND **+0.0080** [+0.0059, +0.0101] and +0.0086. Stated honestly: this compares the systems as built here — NRMS at the published recipe for five epochs on the small splits versus a reranker that consumes A1's stage-1 scores plus behavioural features — not the architectures in general.

### 3.5 Extended evaluation and slices (Q5)

| `make eval`, `config.FINAL` | impressions | AUC | nDCG@10 | diversity | novelty | coverage |
|---|---|---|---|---|---|---|
| EB-NeRD all | 244,647 | 0.6734 [0.6723, 0.6746] | 0.5509 | 0.786 | 17.12 | 0.857 |
| EB-NeRD cold (≤ 5 history clicks) / warm | 887 / 243,760 | 0.6749 [0.6570, 0.6934] / 0.6734 | 0.5431 / 0.5509 | | | |
| EB-NeRD head (top popularity quintile) / tail | 4,704 / 239,943 | **0.7537** [0.7461, 0.7617] / 0.6718 | 0.5834 / 0.5503 | | | |
| MIND all | 73,152 | 0.6747 [0.6725, 0.6768] | 0.4217 | 0.809 | 15.83 | 0.573 |
| MIND cold / warm | 12,982 / 60,170 | **0.6336** [0.6284, 0.6385] / 0.6836 | 0.4236 / 0.4213 | | | |
| MIND head / tail | 21,098 / 52,054 | **0.7304** [0.7271, 0.7335] / 0.6521 | 0.4444 / 0.4125 | | | |

Head impressions are ≈ 0.08 AUC easier than tail on both datasets — popularity is a strong prior and the model uses it. MIND's cold users cost 0.050 AUC against warm; EB-NeRD has only 887 cold impressions, so its cold interval is wide and that comparison is weak. Coverage is high on EB-NeRD (0.857 of the catalogue reached in top-k) and Gini concentrated on MIND (0.946): the MIND model recommends from a narrower set.

### 3.6 With and without serving-unavailable features (Q9)

| EB-NeRD lambdarank, all 244,647 validation impressions | features | AUC | paired Δ vs serving_safe |
|---|---|---|---|
| **serving_safe — what ships** | 11 | **0.6734** [0.6723, 0.6746] | — |
| + unsafe trio (`session_len`, `cur_read_time`, `cur_scroll_percentage`) | 14 | 0.6498 | **−0.0236** [−0.0245, −0.0227] |
| + `n_prior_clicks_in_session` (absent from the test file) | 12 | 0.6598 | −0.0136 [−0.0143, −0.0129] |

The usual Q9 story is an inflated offline number. Here the forbidden features make the *listwise* model significantly **worse**: they are strong at the impression level and nearly constant within an impression, so a pairwise loss gets noise from them. MIND ships none of these columns, so only the serving-safe row exists there. Both leaderboard submissions score the test files with the serving-safe row.

### 3.7 The leaderboards (Q5, Q7.3)

Both test files were scored with `config.FINAL` by a resumable driver on Kaggle CPU kernels (MIND 2,370,727 impressions in 50 min; EB-NeRD 13,536,710 in 118 min after a 37 % `avg_doc_length` fix, C-038), validated locally against the test files' own ids and slate lengths, and uploaded from a member account (the competitions have no team feature, C-023). A1's best entries were MIND 0.5714 and EB-NeRD 0.5110; A1 measured a dev-to-leaderboard offset of −0.03 to −0.09 AUC for feature models (SPEC §7), so the offline 0.67 figures are expected to land lower on the test week.

| competition | file | offline (dev / validation) | leaderboard AUC | offset | A1's best entry |
|---|---|---|---|---|---|
| MIND (13967), submission #930531, 2026-09-17 21:46, status Finished | `mind_reranker_final_v2.zip` | 0.6747 | **0.5606** | **−0.114** | 0.5714 (v4, offline 0.6447, offset −0.073) |
| RecSys 2024 / EB-NeRD (2469), uploaded 2026-09-17 | `ebnerd_reranker_final.zip` | 0.6734 | **pending** — the entry has shown "Submitted" since upload without being scored (A1's EB-NeRD scoring took 2.4 h; this one had not finished at the time of writing, 18 Sep) | — | 0.5110 |

**The MIND number is worse than A1's, and the reason is the one A1 documented.** Offline, `config.FINAL` is +0.030 AUC above A1's v4 model (same seven features); on the test week it is −0.011 below it, and the dev-to-leaderboard offset grew from −0.073 to −0.114. SPEC §7 records that A1 lost 0.090 AUC to selecting on the *adjacent* dev split, whose behavioural counts are still fresh while the test week's are frozen at the training boundary, and prescribes the gap-aware protocol as the authority for model selection. P2 selected `config.FINAL` — objective, features, the h = ∞ grid — on the adjacent dev split (§3.1), so the offline gain was measured in the regime the test file does not have. The number is reported as measured; it is the assignment's own invariant, restated with a fresh data point, and the first thing we would change (§5).

<figure><img src="leaderboard_mind.png" alt="MIND leaderboard"><figcaption>Figure 1. MIND leaderboard entry (Codabench 13967): submission #930531, 17 Sep 2026 21:46, Finished, AUC 0.5606.</figcaption></figure>
<figure><img src="leaderboard_ebnerd.png" alt="EB-NeRD leaderboard"><figcaption>Figure 2. RecSys 2024 / EB-NeRD (Codabench 2469): the entry as it stood at the time of writing — "Submitted", not yet scored.</figcaption></figure>

## 4. Serving and scale (Q4)

The serving state is the locked reranker assembled into a per-request path (`src/serving/`): article meta, the inverted index + BM25, the flat FAISS index, `RollingCounts` (the popularity feature store) sealed over the training split, per-user last-5 ids and click logs, a session store. On 200 real impressions per dataset the per-request features and scores are **bit-identical** to the batch path's — a test that caught two skews before any number was recorded (the user store had merged train ∪ validation history; the model was fed float64 where training used float32, flipping 11 of 8,360 MIND scores at tree thresholds). The served models are `config.FINAL` refits that reproduce Q2 exactly. Benchmark: this laptop (i7-14650HX), **one core** (`taskset`, single-thread LightGBM/FAISS/BLAS), 1,000 seeded validation impressions after 100 warm-ups; every number is in `data/processed/bench_<dataset>.json`.

| memory after build | EB-NeRD | MIND |
|---|---|---|
| BM25 inverted index (Python dicts) | 412 MB, 125,541 docs | 203 MB, 65,238 docs |
| ANN index (FAISS flat, `ntotal × d × 4`) | 151 MB (× 300) | 193 MB (× 384) |
| popularity counts (one timestamp list per article and event type) | 160 MB, 2.8 M events | 347 MB |
| user store (last-5 ids + click log with categories) | 183 MB, 15,342 users | 157 MB, 50,000 users |
| **process RSS** | **1.16 GB** | **1.38 GB** |

| latency, one core, ms (p50 / p95 / **p99**) | EB-NeRD | MIND |
|---|---|---|
| **(a) re-rank the impression's slate — what ships** (12 / 37 candidates) | 1.2 / 1.5 / **1.7**, 790 req/s | 1.7 / 2.7 / **3.4**, 550 req/s |
| (b) retrieve K=100 ∪ K=100 from the corpus, then re-rank (~200 candidates) | 48 / 66 / **72** — BM25 60, ANN 8.4, features 2.9, GBDT 0.9 | 62 / 79 / **87** — BM25 74, ANN 8.9 |
| (b) K=200 | **75** | **88** |
| (a) before caching the profile per request (first measurement) | 47 | — |

**Cost** (cores = ⌈QPS × mean service / 0.5⌉; $0.0425 per vCPU-hour, AWS c7i on-demand, 14 Sep 2026; p99 < 100 ms as the SLA): the shipped path needs **3–4 cores at 1,000 QPS, ≈ $0.00004–0.00005 per 1,000 queries**; retrieve-then-rerank needs 96–120 cores, ≈ $0.0011–0.0014 per 1,000 queries, with 13–28 ms of p99 headroom. Ten cents versus a dollar per million requests.

**What measuring found.** The first per-request number was 47 ms p99 because the two category-profile features recomputed the user's decayed click masses once per candidate; computing them once per request — same definitions, parity kept — removed 96 % of that. During test-set inference, 37 % of the time was A1's `InvertedIndex.avg_doc_length`, a property that summed 125,541 lengths on every BM25 call, invisible at 1,000 requests and dominant at 13.5 M (C-038). Framing (b) is retrieval-bound and BM25-bound: a pure-Python postings scan over an 82–128-token query costs 60–74 ms and does not depend on K.

## 5. Where it breaks at 10×

| 10× in… | reads per request | updates per event | memory | what breaks first |
|---|---|---|---|---|
| **articles** | BM25 scans 10× the postings for the same query → ≈ 600–740 ms p99; flat ANN does 10× `ntotal × d` → ≈ 85 ms alone | none | BM25 dicts ≈ 4 GB, ANN 1.5–1.9 GB | **retrieve-then-rerank, on BM25** — the SLA is gone by ≈ 1.4×. Fix: a compiled index with top-k pruning (WAND/MaxScore, reads ∝ K·log N), titles-only queries (halves the tokens), then IVF/HNSW. The shipped path never scans the corpus and is unaffected |
| **users** | unchanged (dict lookups) | one history append per click | user store 1.6–1.8 GB (≈ 12 KB/user as polars frames) | memory only; a compact log (int64 ids + timestamps, ≈ 100 B/click) is 10× smaller |
| **events** | unchanged (`bisect` on a per-article list) | one list append per view and per click | counts 1.6–3.5 GB — the store keeps **every** event as a Python datetime | the only unbounded component. `pop_total`/`ctr_total` need two integers per article; a live system keeps windowed aggregates, not events |
| **QPS** | unchanged | unchanged | unchanged | cores scale linearly: (a) ≈ 30–40 cores at 10 k QPS; (b) ≈ 1,000–1,200 |

The stage that decides the 10× question is not the model — GBDT scoring is 1–3 ms and O(K) — but the lexical index and the event-level feature store. The system we ship scales on every axis with only memory to buy; the literal retrieve-then-rerank path fails first, on a component we inherited from A1, and §2's recall measurement says it is also the wrong candidate generator for news.

**And the first thing we would change is not about scale.** §3.7's MIND result says the offline gain over A1 was measured in the wrong regime: model selection for `config.FINAL` used the adjacent dev split, against our own SPEC §7. The fix is procedural and cheap — select objective and features under the gap-aware protocol (counts frozen a day before the training day, evaluation a day after), where A1 measured the direction of leaderboard moves correctly — and it should have been the P2 exit gate.

## 6. What the process caught

Every task started with an oracle, and the oracles earned their keep: the batch/serving parity test (two real skews, §4); the benchmark's docstring examples as tests for the polars shim (C-021); the paired judge's calibration on constructed Δ (a first draft that assumed noise is a zero-Δ change was wrong — the harness detected the loss, C-026); determinism twins on every model (an unseeded RNG in the official MIND baseline, C-022/C-030); and the pre-registered prediction, which turned a "small gain" into a documented lesson about pooled versus within-slate importance (C-029). Process failures are on the record too: eleven Kaggle runs failed before the four that counted (a shell without `pipefail`, a column named `article_ids_fixed` against the benchmark's `article_id_fixed`, TF1 sessions that cannot share weights, a wrong archive member name on the first MIND upload), and one laptop session was killed by the OOM killer when A1's bootstrap allocated a 1,000 × 244,647 index array at once — fixed with blocked draws proven bit-identical (C-026). The MIND null is reported as a null. Both members' AI-usage logs (`AI_USAGE.md`, 26 entries) record what was asked, what was generated, what failed and how it was caught.

---

## Appendix A — Reproduce

`make env` · `make fetch-small` · `make fetch-testset` · `make fetch-mind` (HF login) · `make data` · `make test` (376 tests, incl. the leakage assertions) · `make eval SCORES=data/scores/<ds>/<split>/reranker_final.parquet` · `make paired A=… B=…` · `make bench DATASET=ebnerd|mind` · `make note`. GPU work runs as Kaggle kernels under `scripts/kaggle/` (NRMS baselines and variants, the assets kernel, the two submission kernels); every run — 40, failures included — is a row in `scripts/kaggle/RUN_LEDGER.md` with its commit, mode, outcome and log. External code is pinned: `jppol-ai/ebnerd-benchmark` `5164e2c`, `recommenders-team/recommenders` `0bb4b36`. Seeds: 0 (sampling, bootstrap), 123/42 (NRMS). Score files, models and submissions live under gitignored `data/`; the design-note sources are `report/design_note.md` + `scripts/make_note.py`.

## Appendix B — References

Kruse, J., Lindskow, K., Kalloori, S., Polignano, M., Pomo, C., Srivastava, A., Uppal, A., Andersen, M. R., Frellsen, J. (2024). EB-NeRD: A large-scale dataset for news recommendation. *Proc. ACM RecSys Challenge 2024.* — Table 3: NRMS 61.03 / 39.75 / 44.45 / 51.24.
Wu, C., Wu, F., Ge, S., Qi, T., Huang, Y., Xie, X. (2019). Neural news recommendation with multi-head self-attention. *EMNLP-IJCNLP*, 6389–6394.
Wu, F., Qiao, Y., Chen, J.-H., et al. (2020). MIND: A large-scale dataset for news recommendation. *ACL*, 3597–3606. — Table 3: NRMS 67.76 / 33.05 / 35.94 / 41.63; §5.1 setup.
Repositories: `github.com/jppol-ai/ebnerd-benchmark` (5164e2c); `github.com/recommenders-team/recommenders` (0bb4b36); our code `github.com/Anurag-1205/News_Recommendation_System`, branch `a2-click-logs`.
