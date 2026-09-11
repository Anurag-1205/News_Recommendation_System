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
