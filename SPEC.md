# Specification

**CS4.406 Information Retrieval and Extraction — Assignment 1, Component 1**

Defines each component, the interfaces between them, and how each is verified. Decisions are
recorded at the point they were taken, with the evidence that settled them.

Measured results live in `RESULTS.md` and are not repeated here; this document states the
contract, not the numbers.

---

## 1 · Scope and the two evaluation modes

One scorer, `score(user, candidates) -> list[float]`, wrapped by two harnesses. Settling this
before any retrieval code was written was deliberate: retrofitting mode (b) onto a corpus-only
retriever costs a day.

| | Mode (a) — candidate generation | Mode (b) — in-impression re-ranking |
|---|---|---|
| Scores against | the whole article corpus | that impression's candidates only |
| Metric | recall@K, K ∈ {50, 100, 200} | AUC · MRR · nDCG@5 · nDCG@10 |
| Required by | Q2.4, Q3.4 | Q4.1, and both leaderboards |

Beyond-accuracy metrics (Q4.2) attach to mode (b)'s top-k output.

**Verified:** `test_search_agrees_with_score_candidates` asserts both harnesses return identical
scores for the same query, so they cannot drift apart.

---

## 2 · Unified schema

Two dissimilar sources normalise to one shape so everything downstream is dataset-agnostic:

```
articles(article_id, title, abstract, body, category, entities, published_ts)
impressions(impression_id, user_id, ts, candidates[], labels[])
history(user_id, article_ids[], timestamps[])
```

Implemented in `src/pipeline/mind.py` and `src/pipeline/ebnerd.py`. The differences the two
readers reconcile:

| | MIND | EB-NeRD |
|---|---|---|
| Format | TSV, no header | Parquet |
| Language | English | **Danish** — no English stemmer or stoplist |
| History | inline per impression | separate file, keyed by user |
| Labels | `-1` / `-0` suffix per candidate | list of clicked ids |
| Publish time | absent | present |

The label difference is the trap: a 0/1 mask against a set of ids. `labels_from_clicked`
converts one to the other and is covered by `tests/test_ebnerd.py`, because getting it wrong
yields a well-formed submission that scores like noise.

---

## 3 · Temporal split

Interaction data is split by time or not at all. Enforced by
`test_random_split_would_be_rejected`, which shuffles a frame and asserts the detector rejects it.

**Rule.** Last N days = test, preceding M days = validation, derived from each dataset's observed
range rather than an assumed calendar. **N = 1, M = 1** for both, justified by the training files
spanning six and seven days respectively.

**Shipped splits are honoured as published.** Both datasets already provide strictly temporal,
disjoint train/validation/test splits; re-splitting them would discard the organisers' protocol.
The N/M rule applies to the internal split carved from the training file for tuning.

### Observed windows

| Dataset | train | validation | test |
|---|---|---|---|
| MIND | 9–14 Nov 2019 | 15 Nov 2019 | 16–22 Nov 2019 |
| EB-NeRD | 18–25 May 2023 | 25 May – 1 Jun 2023 | 1–8 Jun 2023 |

**EB-NeRD's three windows are contiguous and seven days each**, validation ending one second
before test begins. This has a consequence recorded in §7: models fitted on `train` alone sit
seven to fourteen days from the test window, while `validation` is immediately adjacent to it.

**A correction.** §9 of an earlier revision recorded `MINDlarge_test` as 19–22 Nov, taken from a
reference notebook. Measured from the file it is 16–22 Nov. Facts of this kind are re-derived from
the data before use.

---

## 4 · Feature store

Parquet on disk, keyed lookups, built by `make data` at `data/processed/feature_store/`:

```
manifest.json                    seeds, split boundaries, cutoffs, row counts
mind/article_popularity.parquet  article_id, click_count, cutoff
mind/user_activity.parquet       user_id, n_impressions, n_clicks, last_seen, history_len, cutoff
ebnerd/article_popularity.parquet
```

Every frame carries the `cutoff` it was built as-of. That column makes the leakage invariant
checkable after the fact rather than trusted: the build asserts `max_source_timestamp < cutoff`
before writing, and `tests/test_no_leakage.py` re-derives the same assertion from the stored value.

---

## 5 · Verification strategy

Every component has an oracle written before its implementation.

| Component | Oracle | Location |
|---|---|---|
| Temporal split | leaked fixture caught; shuffled split rejected; as-of cutoff enforced | `tests/test_no_leakage.py` |
| Point-in-time counts | event at exactly *t* excluded; windows; unsorted input refused | `tests/test_rolling.py` |
| BM25 scorer | 5-document corpus, scores computed by hand | `tests/test_bm25.py` |
| BM25 ranking | **exact** score agreement with `rank_bm25` (Okapi variant) | `tests/test_bm25.py` |
| Forward index | identical scores to the inverted path — an optimisation must change nothing | `tests/test_bm25.py` |
| nDCG | worked example: relevances 2,0,1,0,2 → nDCG@5 = 0.8642 | `tests/test_metrics.py` |
| AUC | `sklearn.metrics.roc_auc_score`, including heavy ties | `tests/test_metrics.py` |
| MRR | hand-computed; MIND's all-relevant definition against the textbook first-relevant | `tests/test_metrics.py` |
| Bootstrap CI | interval narrows with more data; covers a known mean; deterministic per seed | `tests/test_bootstrap.py` |
| Harness sanity | random scorer → AUC ≈ 0.5 over 3,000 impressions | `tests/test_metrics.py` |
| Submission file | line count; ranks a permutation of 1..N; duplicate policy | `tests/test_submission_format.py` |
| Model selection | evaluated at the test split's temporal distance, not the dev split's | §8 |

212 tests. Library dependencies used **as oracles only** — `rank_bm25` and `scikit-learn`'s
`roc_auc_score` — exist to disagree with our implementations, never to be them.

---

## 6 · Codabench submission format

A format is never trusted to memory; a rejected upload costs a full regeneration pass. Both
competitions use the same line format:

```
impression_id [rank_order]
```

`rank_order` is a permutation of 1..N over the impression's candidates, **in the candidate list's
own order** — position *i* holds candidate *i*'s rank, not the identity of the *i*-th ranked
candidate. Rank 1 is the most likely click. Transposing this produces a file that validates
perfectly and scores like noise, so `ranks_from_scores` is the only sanctioned constructor and its
orientation is pinned by test.

| Competition | File | Archive |
|---|---|---|
| MIND (13967) | `prediction.txt` | `.zip`, text at archive root |
| EB-NeRD (2469) | `predictions.txt` | `.zip`, text at archive root |

**Confirmed by acceptance:** six submissions in this format were processed by the two scorers —
four on MIND, two on EB-NeRD.

**Duplicate identifiers.** MIND ids are unique. EB-NeRD sets `impression_id = 0` on exactly the
200,000 rows flagged `is_beyond_accuracy`, a 1:1 correspondence confirmed against the file, so a
faithful EB-NeRD submission repeats that id legitimately. `validate_file` therefore takes
`allow_duplicate_ids` and always reports the count, so permitting duplicates never conceals them.

---

## 7 · Validation protocol for model selection

The dev split cannot detect one class of failure, and submission 3 was scored 0.090 below its
offline figure because of it.

**The failure.** Behavioural features count clicks in trailing windows. MIND's training events end
14 Nov; dev is 15 Nov; test is 16–22 Nov. A 24-hour lookback from a dev impression still reaches
training data; the same lookback from a test impression reaches an empty window. `pop_24h`,
`pop_1h` and `ctr_24h` are zero for 100% of test candidates and for 57.7%, 99.7% and 57.7% of dev
candidates.

This is **train/serve skew, not leakage** — no future information is used; the features are
degenerate at serving time. A validation split adjacent to training cannot reveal it.

**The protocol** (`scripts/gap_aware_mind.py`). Counts fitted on events strictly before 13 Nov;
ranker trained on 14 Nov; evaluated on 15 Nov — two and three days past the last counted event,
against the test split's two to eight. Both sides therefore sit in the degenerate regime the test
split imposes.

**Where a model must be selected, this protocol is the authority, not the dev split.**

**Its limits, stated.** Its evaluation day sits two to three days past the counts where the test
split reaches eight, and `MINDsmall_train` spans six days, so a fully matched window cannot be
built from it. Measured against the leaderboard it predicts the *direction* of a change correctly
and overstates the *size*.

**Unexploited on EB-NeRD.** Because EB-NeRD's validation window is contiguous with test (§3),
fitting counts on `train ∪ validation` would place the model immediately adjacent to the test
window rather than seven days from it. The shipped EB-NeRD models fit on `train` alone; this is a
known limitation, not a design choice.

---

## 8 · Environment

Python 3.12.3, virtual environment at `.venv/`. Resolved versions:
polars 1.43.2 · pyarrow 25.0.1 · numpy 2.5.2 · scikit-learn 1.9.0 · faiss-cpu 1.15.0 ·
rank-bm25 0.2.2 · pytest 9.1.1 · matplotlib 3.11.1 · huggingface_hub 1.28.0 ·
torch 2.13.0+cpu · sentence-transformers 6.0.0.

**Polars over Dask.** Dask was available; both reference notebooks use Polars lazy scans, and
matching them keeps this code cross-referenceable against the course material.

---

## 9 · Repository layout

`src/{pipeline,lexical,semantic,features,baselines,eval}/`, `tests/`, `scripts/`, `report/`,
`data/` (gitignored), plus `README.md`, `SPEC.md`, `RESULTS.md`, `Makefile`, `requirements*.txt`.

Deviations from the standard layout, recorded rather than assumed:

| Addition | Reason |
|---|---|
| `src/features/` | point-in-time aggregation, distinct from the static feature store |
| `src/baselines/` | non-retrieval reference scorers; keeps them out of `lexical/` and `semantic/` |
| `scripts/` | data acquisition and experiment drivers |
| `report/` | Q6 design note and leaderboard screenshots |

Not in the repository, by decision: the AI usage log and the prompt record (both submitted with
the report), prediction files (delivered through Codabench; A1 Q8 forbids large files in git, which
overrides Q7.1's mention of them), and third-party reference material.

---

## 10 · Decision log

| # | Decision | Resolution |
|---|---|---|
| 1 | Build order | Q1 → **Q4** → Q2 → Q3 — the harness is the oracle for the retrievers |
| 2 | Scorer interface | one scorer, two harnesses (§1) |
| 3 | N, M per dataset | N=1, M=1, from the observed ranges (§3) |
| 4 | BM25 implementation | own; `rank_bm25` as oracle only |
| 5 | IDF variant | Lucene default; Okapi retained for the external check |
| 6 | Cold-start / head-tail thresholds | ≤5 history clicks; top popularity quintile |
| 7 | MIND embeddings | TF-IDF + SVD (128d) for the Q3 comparison; `all-MiniLM-L6-v2` (384d) for the final submission, selected under §7 |
| 8 | EB-NeRD embeddings | publisher-provided word2vec, in preference to multilingual BERT on Danish text |
| 9 | User vector pooling | mean — recency weighting was ablated and lost |
| 10 | ANN index | flat (exact), so recall measures the model and not the index's approximation error |
| 11 | Popularity aggregation | point-in-time; the frozen variant leaks *and* scores worse |
| 12 | Model selection protocol | gap-aware (§7), not the dev split |
| 13 | C2 boundary | see below |

**C2 boundary.** Nothing here forecloses Component 2. The scorer interface is model-agnostic, so a
learned click model drops in beside BM25 and the semantic retriever using the same harness,
submission writer and leakage guarantees, and the feature store already carries as-of user activity
keyed by cutoff. Deliberately not built: per-user learned parameters, interaction-matrix
factorisation, and any sequence model over click history.

---

## 11 · A2 Phase 1 — Behavioural features (Q1)

**Assignment 2 sections start here.** §1–§10 are the A1 contract and remain in force for the
retrieval stage that A2 builds on.

Every Phase 1 feature is a function of `(user, candidate, t)` that reads **only events strictly
before t**. Each one carries a `serving_ok` flag, so the Q9 with/without ablation is driven by
data rather than by hand edits. Each one is defined here, with its oracle, before any code exists.
Implementations live in `src/features/behavioural.py`; oracles live in
`tests/test_behavioural_features.py`.

### 11.1 · `recency_weighted_profile`: category profile with exponential time decay

**What it measures.** How strongly the user's *recent* reading leans towards the candidate's
category. A click from an hour ago counts for more than one from last week, because news
interest decays in days (A1: the train→test gap alone is 2–8 days).

**Inputs**

| Name | Type | Meaning |
|---|---|---|
| `log` | `pl.DataFrame` with `user_id`, `article_id`, `category`, `ts` | click events; `ts` is a non-null naive `Datetime` on the dataset's own clock |
| `user_id` | same type as `log.user_id` | the user being scored |
| `candidate_category` | same type as `log.category` | the candidate article's category (the caller looks it up from `articles`) |
| `t` | `datetime` | the impression time: the moment the recommendation is requested |
| `half_life` | `timedelta`, > 0 | *h*: the age at which a click counts half as much as one made just now |
| `untimed_ts` | `datetime \| None`, keyword-only, default `None` | MIND fallback (P1-D1): the stamp given to clicks whose `ts` is null. `None` means a null `ts` is an error |

**The behaviour-window boundary.** Only events with `user_id == user` **and `ts < t` (strict)**
are eligible. An event at exactly `t` is excluded: it is part of, or simultaneous with, the
request being scored, so a real system could not know it. Events after `t` are excluded too.
This is the Q1.4 / Q9 invariant for this feature.

**The decay math.** For each eligible event *i*, with age Δᵢ = t − tsᵢ > 0:

```
wᵢ = 2^(−Δᵢ / h) = exp(−ln 2 · Δᵢ / h)          0 < wᵢ < 1
```

One half-life of age halves the weight, two quarter it, and so on. The user's profile is the
normalised decayed category distribution, and the feature is the profile's mass on the
candidate's category:

```
P(c) = Σ_{i : categoryᵢ = c} wᵢ  /  Σ_i wᵢ         Σ_c P(c) = 1
f(user, candidate, t) = P(category(candidate))
```

**Outputs**

| Function | Returns |
|---|---|
| `decay_weights(log, user_id, t, half_life)` | `pl.DataFrame` with `article_id`, `category`, `ts`, `weight`: eligible events only, sorted by `ts` ascending |
| `recency_weighted_profile(log, user_id, candidate_category, t, half_life)` | `float` in [0, 1]; `0.0` if the user has eligible history but none in that category; **`NaN` if the user has no eligible history** |

**Properties the contract fixes, and why**

- **Decay is by timestamp, not by row position.** The result is identical under any reordering
  of `log`, so an out-of-order event is handled by its `ts`. A1's `semantic.user_vector.recency_pool`
  weights by *position* in the history list (`exp(−i/τ)`) and would fail this contract. That is
  the difference between the A1 and A2 notions of recency.
- **NaN, not 0, for no history.** A score of 0 would conflate "never read this category" with "never
  read anything". Both GBDT candidates for Q2 (sklearn HistGBDT, LightGBM) handle missing values
  natively. History length is a separate feature.
- **Normalisation keeps only *relative* recency.** Scaling every age by the same amount leaves P
  unchanged. How long ago the user was last active is a different signal and belongs to a
  separate feature.
- **Invalid input is rejected, not guessed at.** `half_life ≤ 0` raises `ValueError`, and so does
  a null `ts` in the user's events **unless `untimed_ts` is passed**. A click without a time
  cannot be placed relative to `t`: keeping it risks leakage and dropping it silently loses data.
  The MIND fallback is therefore an explicit opt-in, never a silent default.
- `serving_ok = True`: every input exists at request time.

**Which log it reads (train/serve consistency).** The feature reads only click events that also
exist when the *test* split is scored: each split's provided history plus, where labels exist at
serving time, earlier clicks. A1's submission 3 lost 0.090 AUC by training on signal that was
absent at test time (§7). The data sources differ between the datasets:

| | click times available | consequence |
|---|---|---|
| EB-NeRD | ✓ per click: `history.parquet` `impression_time_fixed` + `article_id_fixed`; category from `articles.category` | direct |
| MIND | ✗ `behaviors.history` is an id list with **no click times**; only clicks inside impressions carry `time` | **open decision P1-D1** |

**P1-D1 (decided, Anurag, CONTEXT.md C-008): option (a).** MIND history clicks are stamped with
their split's start via `untimed_ts`. The alternatives were (b) use only timed in-window clicks
and (c) position-based decay for MIND alone. The facts behind the choice were measured on
2026-09-11 from `behaviors.tsv`:

| Split | `untimed_ts` (split start) | first impression |
|---|---|---|
| `MINDsmall_train` | 2019-11-09 00:00:00 | 00:00:19 |
| `MINDsmall_dev` | 2019-11-15 00:00:00 | 00:00:01 |
| `MINDlarge_test` | 2019-11-16 00:00:00 | 00:00:05 |

- **The stamp is strictly before every impression in its split.**
- **The stamp is an upper bound on the true click time.** MIND history is a frozen snapshot: it
  varies within a split for 0 of 33,617 / 14,826 / 484,059 repeat users (train / dev / test),
  and is identical in train and dev for all 5,943 users present in both. It therefore predates
  9 Nov.
- **Even a wrong stamp cannot leak.** A stamp at or after `t` is excluded by the strict boundary.
- **Per-split stamps keep train and test comparable.** History ages span 0–6 days in train and 0–7
  days in test. One global stamp (9 Nov) would be truer, but would age test history to 7–14 days
  against 0–6 in training, which is the train/serve skew behind A1 submission 3 (§7).

**Consequence, stated because it limits the feature.** Every MIND history click shares one stamp,
so they all share one weight, and normalisation cancels it. **On MIND, a history-only profile is
the undecayed category distribution for every h**, which is pinned by
`test_history_only_profile_does_not_depend_on_half_life`. Decay can only matter on MIND if timed
in-window clicks are added, and those are unlabelled in the test split, so adding them in training
would recreate the skew above. The half-life ablation is therefore informative on EB-NeRD only.

**P1-D2 (grid decided, value open).** The value of *h* is chosen from the grid h ∈ {6 h, 24 h,
72 h, ∞} under the gap-aware protocol (§7). h = ∞ gives wᵢ = 1, i.e. the plain undecayed
distribution, so the no-decay baseline is one row of the same code. The functions take no default
`half_life`; every caller states it.

**Cost.** The reference implementation filters the whole log per call, which is O(N). The batch
path over millions of impressions is a separate task, and **must reproduce the reference on the
toy log exactly**, as A1's forward index had to reproduce the inverted index (§5).

**Verification — oracle: a 20-event toy log** (`tests/test_behavioural_features.py`)

Three users, `t` = 2026-01-10 12:00, h = 24 h. The target user U1 has eight events. Six are
eligible, with hand-computed weights: 96 h → 2⁻⁴ = 0.0625, 72 h → 0.125, 48 h → 0.25,
24 h → 0.5, 12 h → 2^−½ ≈ 0.7071, 1 s → 2^(−1/86400) ≈ 0.999992. That gives
P(sports) ≈ 0.6145, P(politics) ≈ 0.1182, P(tech) ≈ 0.2674. The edge cases each shift a
score detectably:

| Case | Toy event | Asserted |
|---|---|---|
| strictly before t | U1 sports, t − 1 s | included, weight 2^(−1/86400) |
| exactly at t | U1 *entertainment*, ts = t | excluded; P(entertainment) = 0.0 exactly |
| after t | U1 *weather*, t + 1 h | excluded; P(weather) = 0.0 exactly |
| out of order | U1's oldest event (96 h) listed after newer ones; the future event listed early | weights by `ts`; output identical under shuffled row order |
| other users | U2 reads entertainment and weather before t | nothing of U2's reaches U1's profile |
| no eligible history | U3's events are all at or after t | NaN |
| invalid input | h ≤ 0; a null `ts` without `untimed_ts` | `ValueError` |
| MIND fallback | U1's 24 h click untimed, stamped at t − 120 h | weight 2⁻⁵ = 0.03125; other weights unchanged |
| fallback stamp ≥ t | stamp = t, or t + 1 h | click excluded, not leaked |
| history only | four untimed clicks, h ∈ {6 h, 24 h, 72 h, 1000 d} | P(sports) = 0.5, P(tech) = 0.25 for every h |

The oracle is checked against itself too: one test recomputes every hand-written weight from the
toy log's timestamps, so a typo in an expected value cannot pass unnoticed.

**Mutation check: the oracle catches each planted bug on its own.** Each bug was inserted alone
into an otherwise-correct implementation (C-007):

| Planted bug | Dedicated test that fails | Tests failing |
|---|---|---|
| `ts <= t` instead of `ts < t` | `test_event_at_exactly_t_is_excluded` | 8 / 16 |
| decay by list position instead of t − ts | `test_out_of_order_event_is_weighted_by_timestamp_not_row_order` | 6 / 16 |
| no `user_id` filter | `test_other_users_events_are_excluded` | 7 / 16 |
| none (control) | — | 0 / 16 |

### 11.2 · `recency_profile_batch`: the same feature for millions of requests

**Why a second implementation.** The reference (`recency_weighted_profile`) filters the whole log
for every call. That is the right thing to read and test, and the wrong thing to run 13.5M times.
The batch path computes the identical quantity for a whole frame of requests in one vectorised
Polars pass. **The reference is its oracle.**

**Contract**

| | |
|---|---|
| Signature | `recency_profile_batch(log, requests, half_life, *, untimed_ts=None) -> pl.DataFrame` |
| `requests` | columns `user_id`, `t`, `candidate_category`; any other columns (e.g. `impression_id`, `article_id`) pass through untouched |
| Returns | `requests`, same row order, with `recency_weighted_profile` appended |
| Semantics | identical to §11.1 per row: strict `ts < t`, timestamp decay, 0.0 / NaN rules, `untimed_ts`, `ValueError` cases |

**Algorithm** (each step is one Polars operation, lazily planned):

1. **Anchors** = distinct `(user_id, t)`. The profile depends on the user and the moment, not on
   the candidate, so it is built once per impression instead of once per candidate. Measured
   mean candidates per impression: EB-NeRD small validation 12.0; MIND 37.2 / 37.5 / 39.3 in
   small train, small dev and large test.
2. **Pair** each anchor with that user's events (an equi-join on `user_id`), then keep `ts < t`.
3. **Weight** each pair with 2^(−(t − ts)/h); sum per (anchor, category) and per anchor.
4. **Look up** each request's category: its mass divided by the anchor total. 0.0 if absent; NaN
   if the anchor has no eligible events.

**Cost, and where it breaks.** Step 2 materialises Σ over anchors of that user's history length:
reads and memory are linear in (impressions × history length). This is the bound to watch.
The Kaggle driver passes `requests` one chunk at a time (a Parquet row group), and only log rows
for users in the chunk are read. **Not yet measured on real data.** Throughput and peak memory
on EB-NeRD come when this is wired into the Kaggle driver, and are recorded in `RESULTS.md` Q1.
The known cheaper alternative, if the measurement demands it, is per-user prefix sums of decayed
mass with an as-of lookup at `t`. That is linear in (events + requests) with no pair table, but it
is subtler (duplicate timestamps, strictness of the as-of match), so it is deferred until a
measured number justifies it.

**Verification — parity oracle** (`TestBatchParity`)

- **Grid.** Every (user, t, category) combination over both toy logs, plus a user and a category
  never seen: **360 requests**, of which 128 have a NaN reference value, 150 an exact 0.0 and 82 a
  non-zero value.
- **Boundary per anchor.** The scoring times include moments equal to event timestamps
  (t − 72 h, t − 12 h, t, t + 1 h), so the strict boundary is exercised per anchor.
- **Also covered:** the MIND fallback, request-order preservation with shuffled requests and a
  passthrough column, and the same `ValueError` cases as the reference.
- **Absolute check.** One test pins the batch to the hand-computed U1 values directly, because
  parity alone would pass if both paths shared a bug.
- **"Same output" means:** NaN meets NaN and 0.0 meets 0.0 **exactly**; every other value agrees
  within **1e-12 relative**. Bitwise identity is not required: the batch path sums the same
  weights in a different order, and float addition is not associative.

**Mutation check.** Planted in the batch path alone:

| Planted bug | Parity tests failing |
|---|---|
| `ts <= t` | 4 / 6 |
| no user filter (cross join) | 3 / 6 |

The hand-computed-values test alone does **not** catch the second bug: with a single requested
user, the upstream semi-join already restricts the log to that user. The many-user parity grid is
what catches it.

### 11.3 · `category_match`: cosine between the recency profile and the candidate's category

**Status: implemented.** The cosine definition was confirmed by Anurag (CONTEXT.md C-010, C-012).
There are two versions: row-by-row `category_match`, and `category_match_batch`, which is held to
the row-by-row one by the §11.2 parity grid (`TestCategoryMatchBatchParity`). Both batch functions
share the private helpers `_decayed_category_mass` (steps 1–3) and `_lookup` (step 4).

**What it measures.** How close the candidate's category is to the centre of the user's recent
interests, as a cosine similarity. The recency profile *P* (§11.1) is a vector over categories;
the candidate is the one-hot vector **e**_c for its category:

```
category_match(user, candidate, t) = cos(P, e_c) = P(c) / ‖P‖₂ ,   ‖P‖₂ = sqrt(Σ_k P(k)²)
```

Because cosine ignores scale, the same value comes from the un-normalised decayed masses:
W_c / ‖W‖₂. The oracle checks both forms give identical values.

**Why cosine, not the dot product.** P · **e**_c = P(c), which is exactly `recency_weighted_profile`,
so a dot-product "match" would hand the reranker a duplicate column. The cosine divides by ‖P‖₂,
which measures how concentrated the user's interests are:

- 1.0 when the candidate is the user's only recent category;
- 1/√k when the user reads k categories equally.

So category_match = recency_weighted_profile / ‖P‖₂. It adds exactly one piece of information,
profile concentration. Alternatives, if that is judged too thin, are open for Anurag:

- a top-1 indicator: is c the user's heaviest category;
- lift over a point-in-time population prior, P(c) / P_all(c);
- a subcategory-level match (both datasets carry subcategories).

**Inputs, boundary, outputs.** These are identical to §11.1, including the signature, strict
`ts < t`, timestamp decay, `untimed_ts` and `ValueError` cases:

- `category_match(log, user_id, candidate_category, t, half_life, *, untimed_ts=None) -> float`;
- value in [0, 1]; **0.0** for a category the user has no eligible clicks in; **NaN** with no
  eligible history;
- `serving_ok = True`.

**Verification — oracle** (`TestCategoryMatch`, extended toy log). The 20-event log gains U4 and U5,
7 events, 27 in total. The original 20 rows are unchanged, so every §11.1 value still holds.

| Case | Toy events | Asserted |
|---|---|---|
| multi-category user | U1 (profile as §11.1) | sports 0.9030165, politics 0.1736579, tech 0.3929429 (= W_c / ‖W‖₂, ‖W‖₂ = 1.7995153) |
| only one category | U4: science ×2 before t | 1.0; **a leaked at-t music click would give 0.7287** |
| equal mass | U5: sports and tech at one timestamp | 1/√2 each; **a leaked future sports click would give 0.9589** |
| exactly at t / after t | U4 music at t and at t + 2 h | P = 0 → exactly 0.0 |
| other users | U1 never reads science (U4 does) | exactly 0.0 |
| no eligible history | U3 | NaN |
| out of order | extended log shuffled, 5 seeds | identical values |
| invalid input | null `ts` without `untimed_ts` | `ValueError` |

**Mutation check.** Two planted bugs, each run alone:

| Planted bug | Tests failing |
|---|---|
| dot product P(c) instead of the cosine, in the row-by-row version | 4 / 10 (the U1 values and U5) |
| L1 sum instead of the L2 norm, in the batch version | 4 / 6 batch parity tests |

U4 passes under the first bug, correctly: with a single category, the cosine and the dot
product are both 1.0.

---

## 11.4 – 11.7 · Phase 1.2: slate, session and dwell features, and the unsafe registry

These features come from Aayush's schema mapping (CONTEXT.md C-013). Every definition below
rests on a fact measured on 2026-09-11. The ones that changed the mapping are marked **(correction)**.

**What the EB-NeRD files contain.** The Codabench test `behaviors.parquet` has `impression_time`,
`read_time`, `scroll_percentage`, `session_id` and `article_ids_inview`. It **lacks**
`article_ids_clicked`, `article_id`, `next_read_time` and `next_scroll_percentage`. Train and
validation have all of them.

| Feature | Datasets | `serving_ok` | In Codabench test file | Section |
|---|---|---|---|---|
| `cand_position` | MIND, EB-NeRD | ✓ | ✓ | 11.4 |
| `n_candidates` | MIND, EB-NeRD | ✓ | ✓ | 11.4 |
| `session_pos` | EB-NeRD | ✓ | ✓ | 11.5 |
| `n_prior_clicks_in_session` | EB-NeRD | ✓ | **✗ (correction)** | 11.5 |
| `session_len` | EB-NeRD | **✗ (correction)** | ✓ | 11.5 |
| `hist_read_time_mean` | EB-NeRD | ✓ | ✓ | 11.6 |
| `hist_scroll_mean` | EB-NeRD | ✓ | ✓ | 11.6 |
| `cur_read_time` | EB-NeRD | **✗** | ✓ | 11.7 |
| `cur_scroll_percentage` | EB-NeRD | **✗** | ✓ | 11.7 |

The two columns are different questions:

- **`serving_ok`** asks whether a live system could know the value when the request arrives.
  If not, the feature is an ablation row only (Q9).
- **In test file** asks whether the offline Codabench test set supplies the input. If not, the
  feature cannot be used by the submission model, even though it is fine in production. Training
  on it recreates the A1 submission-3 skew (§7).

Code lives in `src/features/behavioural.py`, and oracles in `tests/test_session_features.py`.

### 11.4 · Slate features: `cand_position`, `n_candidates`

`slate_features(impressions) -> pl.DataFrame`

- **Input:** `impression_id` and `candidates` (a list). **Nothing else is read, in particular not
  labels,** so the features are computable on an unlabelled test impression. That is this
  feature's behaviour-window boundary: its only input is the slate the request itself carries.
- **Output:** one row per candidate, in input order and then list order: `impression_id`,
  `article_id`, `cand_position` (1 = first in the list) and `n_candidates` (the list length).
  An empty list yields no rows.

**Measured** with `scripts/check_phase1_data.py`, output in `RESULTS.md` Q1. Click rates come from
the first 50,000 impressions of each split:

- **The order is not a label artifact.** Click rate by position quintile is nearly flat:
  - EB-NeRD train 0.087–0.094, validation 0.080–0.087;
  - MIND small train 0.036–0.045, dev 0.037–0.044.

  A "clicked items first" construction would put quintile 0 far above the rest.
- **EB-NeRD lists are not sorted by article id,** so position is not an article-age proxy. Only
  0.2% of lists are sorted in either direction, and the figure is identical in train, validation
  and the test file.
- **A weak position effect is present, with the same shape in train and dev**, so it is safe to
  use. MIND quintile 0 vs 4 is 0.0449 vs 0.0362 in train and 0.0440 vs 0.0373 in dev.
- **Open:** whether list order equals the order shown on screen is not established by anything we
  have verified. The feature is therefore described as *list position*, not display position.

### 11.5 · Session features: `session_pos`, `n_prior_clicks_in_session`, `session_len`

`session_features(behaviors) -> pl.DataFrame`

- **Input:** `impression_id`, `user_id`, `session_id` and `t`, plus `clicked` (list) if available.
- **Output:** one row per input row, in input order.

**The session key is `(user_id, session_id)` (correction).** In the test file, `session_id = 0`
is shared by all 200,000 `is_beyond_accuracy` placeholder rows, one per user. Keyed on
`session_id` alone, that becomes a 200,000-impression "session", and the self-join would need
about 4×10¹⁰ pairs. Keyed on `(user_id, session_id)`:

- the largest test session is 118 impressions (p99 = 9);
- the whole 13.5M-row file needs **47.3M** pairs;
- each placeholder row is a single-impression session.

In `ebnerd_small` train, no session spans more than one user.

| Feature | Definition | Boundary |
|---|---|---|
| `session_pos` | 1 + number of the session's impressions with `t' < t` | strict: a tie at exactly `t` is not "prior" (sessions containing tied timestamps: 5 in small train, 195 in test) |
| `n_prior_clicks_in_session` | Σ `len(clicked)` over those same prior impressions | strict |
| `session_len` | number of the session's impressions, **including those after t** | **none, which is why it is unsafe** |

- **`session_len` is serving-unsafe (correction; the mapping listed it as safe).** A live system
  does not know how long a session will last. Its value changes when a future impression is
  appended, and a test demonstrates this. The safe "length so far" is `session_pos` − 1, which
  `session_pos` already carries.
- **`n_prior_clicks_in_session` cannot feed the submission model (correction).** It is built from
  `article_ids_clicked`, which the test file does not ship. When `clicked` is absent, the column is
  **omitted, not zero-filled**: a silent zero is exactly the degenerate-at-test value that caused
  A1's skew.
- **It is also nearly redundant.** Every `ebnerd_small` train impression has at least one click
  (0 without, mean 1.006 clicks), so it is ≈ 1.006 × (`session_pos` − 1).
- **Cost:** a self-join within sessions, Σ over sessions of length² pairs. That is 731k pairs for
  small train, which is fine.

### 11.6 · Dwell features: `hist_read_time_mean`, `hist_scroll_mean`

`dwell_features(history, requests) -> pl.DataFrame`

- **Inputs:**
  - `history`: one row per past click, with `user_id`, `ts`, `read_time`, `scroll_percentage`.
    For EB-NeRD this is the exploded `history.parquet` (`impression_time_fixed`,
    `read_time_fixed`, `scroll_percentage_fixed`).
  - `requests`: `user_id` and `t`; other columns pass through.
- **Definition:** the mean over the user's history clicks with **`ts < t`**, strict as in §11.1.
  Null values are skipped (10.5% of `scroll_percentage_fixed` is null in small train;
  `read_time_fixed` has none).
- **Missing values:** **NaN** when no eligible values exist, same convention as §11.1. A null `ts`
  raises `ValueError`.

### 11.7 · Serving-unsafe features and the feature registry

`current_page_features(behaviors) -> pl.DataFrame` returns `impression_id`, `cur_read_time` and
`cur_scroll_percentage`: the impression's own `read_time` and `scroll_percentage`.

- **Why they are unsafe:** they describe the page view the impression belongs to, which continues
  after the moment the recommendation is served. They are in the test file, so a model could
  exploit them on the leaderboard, but no live system has them at request time.
- **Consequence:** they exist only to supply the Q9 "with" row. (A1 measured the analogous
  `next_read_time`: adding it moved AUC from 0.50 to 0.96.)
- **Data note:** `scroll_percentage` is null for 70.3% of train rows and 71.6% of test rows.

**The registry** is in `src/features/behavioural.py`, so the Q9 ablation is driven by data rather
than by memory:

- `SERVING_OK`: feature → bool, covering every feature in §11.
- `UNSAFE_FEATURES`: derived from it; currently `cur_read_time`, `cur_scroll_percentage` and
  `session_len`.
- `ABSENT_FROM_TEST_FILE`: currently `n_prior_clicks_in_session`.
- `drop_unsafe(df)`: removes exactly the unsafe columns.

**Verification** (`tests/test_session_features.py`, hand-computed toy frames):

| Case | Asserted |
|---|---|
| slate: 3-, 1-, 0- and 2-candidate lists, string and integer ids | positions 1..n in list order, `n_candidates` = n, no rows for the empty list |
| slate never reads labels | same output with labels dropped or permuted |
| session: 10 impressions, 5 sessions | hand-computed `session_pos`, `n_prior_clicks_in_session`, `session_len` for every row |
| tie at exactly t | two impressions at t in one session get the same `session_pos` (4) and click count (3) |
| future impressions | appending one after t changes none of the safe features and changes `session_len` 6 → 7, which is why it is unsafe |
| session key | U2's session 100 is separate from U1's session 100; the two `session_id = 0` rows stay separate |
| unlabelled input | without `clicked`, no `n_prior_clicks_in_session` column at all, rather than zeros |
| dwell: one user at two times | at t: read 30.0, scroll 60.0; at t − 2 d: 10.0 and 40.0 (the click at exactly that time is excluded) |
| dwell leak trap | a click at exactly t (999) and after t (777) would move the read mean to 272.25 or beyond |
| dwell missing values | U3 with no history → NaN, NaN; U4 with no scroll values → 20.0, NaN |
| registry | the unsafe list is exact; every emitted feature column is registered; `drop_unsafe` removes exactly those columns |

**Mutation check.** Each bug was planted alone:

| Planted bug | Tests failing |
|---|---|
| session boundary `t' <= t` | 4 / 19, including `test_tie_at_exactly_t_is_not_prior` |
| session keyed on `session_id` alone (the sentinel trap) | 5 / 19, including `test_sessions_are_keyed_by_user_and_session_id` |
| dwell boundary `ts <= t` | 4 / 19, including `test_click_at_exactly_t_is_excluded` |

The session `<=` bug does **not** fail the append-the-future test, correctly: `<=` still excludes
events strictly after t. It is the tie test that catches it, which is why both tests exist.

### 11.8 · Freshness (Q1.3): `freshness_hours`

`freshness_batch(first_known, requests, *, untimed_ts=None) -> pl.DataFrame`

**Definition.** One definition covers both datasets: how long before *t* the article was first
known to exist.

```
first_seen(a, t)   = min { ts : (a, ts) ∈ first_known,  ts < t }        (strict)
freshness_hours    = (t − first_seen(a, t)) / 1 h ,   NaN if no such ts
```

The first-known times come from different sources:

| Dataset | `first_known` rows | Meaning |
|---|---|---|
| EB-NeRD | one per article, `ts = published_time` (`articles.parquet`, 0 nulls in 125,541) | publication age |
| MIND | one per candidate appearance in any impression, `ts` = impression time; plus history articles stamped with `untimed_ts` | first-seen proxy, because MIND has no publish time |

**Why a single group-by is enough.** The minimum over sightings strictly before *t* equals the
overall minimum sighting whenever that minimum is before *t*, and is empty otherwise. So the
feature is: the overall first sighting per article, then a strict comparison with *t*. There is no
pair table, and the cost is linear in sightings + requests.

**The boundary, and two consequences:**

- **A sighting at exactly *t* does not count.** On MIND, the current impression contains its own
  candidates. If that counted, every candidate would be "first seen" at *t* with freshness 0,
  which says nothing about age.
- **EB-NeRD: a `published_time` ≥ *t* gives NaN, not a negative age.** An article cannot be shown
  before it exists, so a recorded publish time after the impression is a later rewrite of the
  metadata, and using it would leak that rewrite. The fraction of such requests is measured and
  reported by the reranker script.
- **MIND history articles.** Clicks in the frozen history (C-008) prove an article existed before
  the dataset's first impression. They are stamped with `untimed_ts`; the reranker uses the first
  day of `MINDsmall_train`, 2019-11-09 00:00.

**Missing values and invalid input:** NaN when nothing qualifies; a null `ts` without
`untimed_ts` raises `ValueError` (same rule as §11.1). `serving_ok = True`: publish times and
earlier impressions are both known at request time.

**Test-file availability.** EB-NeRD `published_time` ships with the test bundle's articles. The
MIND test impressions' candidate lists are unlabelled but present, so first sightings within the
test file are available too.

**Verification** (`tests/test_freshness.py`)

| Case | Asserted |
|---|---|
| publish time T − 5 h | 5.0 h at T; 1.0 h at T − 4 h; NaN at T − 6 h |
| publish time exactly T, or T + 1 h | NaN (not strictly before t) |
| article unknown to `first_known` | NaN |
| sightings at T − 10 h, T − 2 h, T + 1 h | 10.0 h at T; 5.0 h at T − 5 h; NaN at exactly T − 10 h |
| seen only at exactly T, or only after T | NaN |
| history article with null `ts`, `untimed_ts` = T − 48 h | 48.0 h |
| future sightings appended | no value changes |
| null `ts` without `untimed_ts` | `ValueError` |

**Mutation check.** A `first_seen <= t` boundary fails 3 of 8 tests.

---

## 12 · A2 Phase 2 — Two-stage reranker (Q2), first measured version

`scripts/rerank_ebnerd_a2.py` and `scripts/rerank_mind_a2.py`, built on `src/rerank/`
(`common.py`, `ebnerd.py`, `mind.py`). Measured numbers are in `RESULTS.md` Q2.

**Framing (PLAN D1: option (a) here; (b) and (c) still open).** Each impression's own candidates
are re-ranked, which is what both leaderboards score. Stage 1 is A1's two generators, BM25 over
the last 5 clicked titles and the embedding cosine of the mean-pooled history vector, used as
*scores* on those candidates. Re-ranking a retrieved top-K from the whole corpus (option (b)) is
not in this version.

**Protocol: the shipped temporal split.** Fit on the train period and evaluate on the next one:
EB-NeRD small train week → validation week; MINDsmall_train → MINDsmall_dev.

- **Sampling.** Impressions are *seeded random samples* within a split (100k fit and 100k
  evaluation on EB-NeRD; 80k fit and all 73,152 dev impressions on MIND). That is sampling, not
  splitting, so the temporal boundary is untouched.
- **Correction of A1.** A1's EB-NeRD reranker fitted and evaluated inside the validation file,
  split 70/30 by row order. `behaviors.parquet` is **not sorted by time** (measured: `is_sorted`
  is false for both train and validation), so that was not a temporal split. Its AUC 0.7084 is not
  comparable, and this version replaces the protocol (CONTEXT.md C-015).
- **Popularity counts** (`pop_total`, `ctr_total`) come from train events only. Train rows see
  point-in-time counts; evaluation rows see counts that stop at the split boundary, as test rows
  would. The windowed counts are dropped, as A1 v4 did (§7).

**Feature sets.** Every list passes through `src/rerank/common.model_features`, which removes
`UNSAFE_FEATURES` and `ABSENT_FROM_TEST_FILE`. Both scripts assert the result.

| Set | EB-NeRD | MIND |
|---|---|---|
| base (A1) | bm25, semantic (word2vec), pop_total, ctr_total, freshness_hours, n_candidates, history_len | bm25, semantic (MiniLM), pop_total, ctr_total, n_candidates, cat_affinity, history_len |
| A2 = base + | recency_weighted_profile, category_match, cand_position, session_pos, hist_read_time_mean, hist_scroll_mean | category_match, cand_position, freshness_hours |
| computed but excluded | n_prior_clicks_in_session (absent from test file), session_len, cur_read_time, cur_scroll_percentage (unsafe) | — |

- **Freshness in the base set.** On EB-NeRD, `freshness_hours` replaces A1's `age_hours` (same
  quantity, with §11.8's NaN rules), so the base-vs-A2 difference measures only the *new*
  features.
- **No duplicate profile on MIND.** `recency_weighted_profile` equals A1's `cat_affinity` there
  (C-008), so it is not added twice. The script asserts that the two agree to within 1e-9 on
  every row with history.

**Models.** Both use the same capacity (300 trees, learning rate 0.08, 31 leaves, seed 0), so
comparing them compares objectives, not model size. Both handle NaN features natively.

| | Pointwise (A1) | Listwise (PLAN D2, CONTEXT.md C-016) |
|---|---|---|
| Model | sklearn `HistGradientBoostingClassifier` | LightGBM 4.7.0 `LGBMRanker(objective="lambdarank")` |
| Loss | log-loss per candidate row, all impressions pooled | pairwise swaps *within* one impression, weighted by their nDCG change |
| Rewards | any feature that separates impressions (e.g. slate size) | only features that order candidates inside an impression |
| Grouping | none | one query per impression: `group_sizes(frame)` = contiguous run lengths of `imp_row` |
| Determinism | seeded | `deterministic=True`, `force_row_wise=True`, `n_jobs=4`: refits are bit-identical (tested) |

- **Grouping key.** It is `imp_row`, not `impression_id`, because the EB-NeRD test file repeats
  `impression_id` 0 across 200,000 rows. Frames are sorted by (`imp_row`, `cand_position`), and
  `group_sizes` raises if any impression's rows are not contiguous. LightGBM reads groups as
  consecutive counts, so a split impression would otherwise become two queries without any error.
- **Scores.** The ranker outputs unnormalised scores. That is fine here, because every metric is
  computed within an impression.

**Conditional ablation (leave one Phase 1 family out, lambdarank).** The trigger is fixed in code
before the numbers are seen (`common.drop_reasons`): it fires if the paired Δ(A2 − base) has an
AUC point estimate below 0, *or* any metric's 95% CI entirely below 0.

- **When it fires,** the model is refitted once per family with that family's columns removed.
  Each refit is reported as (without − with) and (without − base).
- **Reading it:** a positive (without − with) whose CI excludes 0 means the family costs
  performance.
- **Families:**
  - EB-NeRD: category profile {`recency_weighted_profile`, `category_match`}, list position
    {`cand_position`}, session {`session_pos`}, dwell {`hist_read_time_mean`, `hist_scroll_mean`};
  - MIND: {`category_match`}, {`cand_position`}, {`freshness_hours`}.

**Half-life.** h = ∞, from the P1-D2 grid (CONTEXT.md C-014). The EB-NeRD script also fits the A2
model with h = 72 h, to test that choice inside the full model.

**Evaluation.**

- **Metrics:** A1's per-impression AUC, MRR, nDCG@5 and nDCG@10, each with a bootstrap 95% CI
  (1,000 resamples of impressions).
- **Differences:** `src/rerank/common.paired_delta`, a paired bootstrap over the same
  impressions, provisional until P3.4a.
- **Importance:** permutation importance (pooled ROC-AUC) for the A2 model.

**Test-file check.** Both scripts run the full feature pipeline on the first 5,000 impressions of
the unlabelled Codabench test file:

- no `label` column appears, and EB-NeRD produces no `n_prior_clicks_in_session`;
- the A2 model scores every row with finite values.

This proves the pipeline runs without click columns. It is not a submission.

### 12.1 · The locked configuration (CONTEXT.md C-018)

`src/rerank/config.FINAL` is the single source of truth for what ships. The Q5 submission runs
read it, and both reranker scripts assert that it equals the model they measured.

| | EB-NeRD | MIND |
|---|---|---|
| Objective | lambdarank (`fit_final` → `fit_lambdarank`) | pointwise (`fit_final` → `fit_gbdt`) |
| Features | A1 base (7) + `recency_weighted_profile`, `category_match`, `cand_position`, `session_pos` = **11** | A1 v4 base, **7** |
| Left out, with reason | `hist_read_time_mean`, `hist_scroll_mean`: they hurt the listwise model, confirmed on a 144,647-impression holdout | every Phase 1 addition: none beat the base under either objective |
| Evidence | `RESULTS.md` Q2, follow-up 2 | `RESULTS.md` Q2, follow-up 1 |

`tests/test_rerank_config.py` pins the invariants:

- no final feature is unsafe or absent from the test file;
- EB-NeRD keeps the category profile and has no dwell;
- MIND is exactly A1 v4;
- `fit_final` dispatches on the objective.

**Still open against the brief:** Q2.1 asks for A1's generators to *retrieve* the top K
(100–200) from the corpus before re-ranking, which is PLAN D1 framing (b). This configuration
re-ranks each impression's own candidates (framing (a)), which is what the leaderboards score.
Framing (b) remains to be built and measured.

**Verification of the frame builders** (`tests/test_rerank_*.py`):

- EB-NeRD labels map by clicked-id membership: a duplicated click id counts once, and duplicate
  `impression_id`s stay distinct rows.
- MIND labels map by position.
- Unlabelled splits carry no label column.
- `model_features` drops exactly the banned columns.
- `paired_delta` returns exactly 0 for a system against itself, and recovers a constant shift
  with a zero-width CI.
- **Lambdarank:**
  - `group_sizes` returns contiguous runs and rejects interleaved rows;
  - `fit_lambdarank` rejects group sizes that do not cover every row;
  - two fits are bit-identical;
  - on a toy where one feature separates impressions and another decides the click inside them,
    the ranker puts the clicked candidate first in ≥ 95% of impressions.
- **The ablation trigger** (`drop_reasons`) is tested on its three cases.
