# SPEC.md — A1 Component-1

What each component is, and **how it will be verified**. Graded artifact — keep it current, and
prefer an honest `TBD` to a plausible guess.

Phase labels (P0–P6) refer to the build order: P0 setup · P1 data pipeline · P2 eval harness ·
P3 lexical/BM25 · P4 semantic/ANN · P5 scale + Codabench · P6 design note. The eval harness is
built *before* the retrievers because it is their oracle.

Status key: **[P0]** settled · **[TBD@Pn]** decided in that phase, from measured data.

---

## §1 Scope & the two evaluation modes **[P0]**

One scorer, `score(user, article) -> float`, wrapped by two harnesses. Settling this before any
retrieval code is written is the point — retrofitting mode (b) onto a corpus-only retriever costs
a day we do not have.

| | Mode (a) — candidate generation | Mode (b) — in-impression re-ranking |
|---|---|---|
| Scores against | the whole article corpus | only `article_ids_inview` for that impression |
| Metrics | recall@K, K ∈ {50, 100, 200} | AUC · MRR · nDCG@5 · nDCG@10 |
| Asked by | Q2.4, Q3.4 | Q4.1 |
| Used for | ablations, lexical-vs-semantic comparison | **both Codabench leaderboards** |

Beyond-accuracy metrics (Q4.2: diversity, novelty, coverage) attach to mode (b)'s top-k output.

## §2 Unified schema **[TBD@P1]**

Two very different sources normalize to one shape, so everything downstream is dataset-agnostic.
Target (field names to be finalised against the real files):

```
articles(article_id, title, abstract, body, category, entities, published_ts)
impressions(impression_id, user_id, ts, candidates[], labels[])
history(user_id, article_ids[], timestamps[])
```

Source-specific facts that the two readers must reconcile (from the TA notebooks, §9 below):

| | MIND | EB-NeRD |
|---|---|---|
| Format | TSV, no header | Parquet |
| Language | English | **Danish** — no English stemmer/stoplist |
| Body text | ❌ absent (URLs expired) | ✅ present |
| History | inline in `behaviors.tsv` | separate `history.parquet` |
| Candidates | `impressions` str, `N123-1 N456-0` | `article_ids_inview` list[int32] |
| Labels | `-1` / `-0` suffix | `article_ids_clicked` list[int32] |
| Entity embeddings | ✅ TransE 100-dim, shipped | ❌ (separate download) |

## §3 Temporal split — N and M **[P1, measured 2026-08-22]**

Never random — interaction data is split by time or not at all. Enforced by
`tests/test_no_leakage.py::TestTemporalSplit::test_random_split_would_be_rejected`, which
shuffles a frame and asserts the detector rejects it.

### MIND — measured from the files, not from documentation

| Split | Range | Days | Rows | Users |
|---|---|---:|---:|---:|
| `MINDsmall_train` | 2019-11-09 00:00:19 → 2019-11-14 23:59:13 | 6 | 156,965 | 50,000 |
| `MINDsmall_dev` | 2019-11-15 00:00:01 → 2019-11-15 23:58:03 | **1** | 73,152 | 50,000 |
| `MINDlarge_test` | 2019-11-16 00:00:05 → 2019-11-22 23:59:58 | 7 | 2,370,727 | 702,005 |

**Correction:** §9 previously recorded `MINDlarge_test` as 19–22 Nov, taken from the TA
reference notebook. Measured from the file it is **16–22 Nov**. The notebook was wrong, which
is why every fact of this kind is re-derived from the data before it is used.

The three splits are already strictly temporal and disjoint, with dev occupying the single day
between train and test. **The shipped boundary is honoured as-is for the leaderboard path** —
re-splitting would discard the organisers' own protocol for no gain.

### Our internal split — N = 1, M = 1

Carved out of `MINDsmall_train` only, for tuning without touching the official dev set:

- **N = 1** — last 1 day (14 Nov) is internal test
- **M = 1** — preceding 1 day (13 Nov) is internal validation
- train = 9–12 Nov

Justified by the observed range: the training file spans six days, so N=1/M=1 leaves four days
of training data while mirroring the dataset's own one-day dev window. A larger N would both
shrink training and diverge from the protocol the leaderboard actually uses.
`compute_boundaries` derives these from the data's real maximum and raises if the range is too
short, so a dataset that cannot support N+M days fails loudly rather than silently emptying a
split.

### EB-NeRD **[TBD]**
Ships `train/` and `validation/` as separate directories. Ranges to be measured once
`ebnerd_testset.zip` finishes downloading; the same honour-the-shipped-boundary rule applies.

## §4 Feature store layout **[TBD@P1]**

Parquet on disk, keyed lookups, small and reusable — not a database.

## §5 Verification strategy **[P0, seeded]**

Every component gets its oracle *before* its implementation. No oracle → build the oracle first;
"looks right" is not a result.

| Component | Oracle | Where |
|---|---|---|
| Temporal split | no history event ts ≥ its impression ts, asserted row-wise | `tests/test_no_leakage.py` |
| BM25 scorer | 5-doc toy corpus, scores computed **by hand** | `tests/test_bm25.py` [TBD@P3] |
| BM25 ranking | agreement with `rank_bm25` on the same corpus | same |
| nDCG | L1 slide-17 worked example: labels 2,0,1,0,2 → nDCG@5 ≈ 0.86 | `tests/test_metrics.py` [TBD@P2] |
| AUC | `sklearn.metrics.roc_auc_score` on the same vectors | same |
| MRR | one-line reference implementation | same |
| Harness sanity | random scorer → AUC ≈ 0.5, CI contains 0.5 | `make eval` [TBD@P2] |
| Submission file | line count == impression count; every id present; ranks a permutation | `tests/test_submission_format.py` [TBD@P5] |

## §6 Codabench submission formats **[P0 — from TA notebooks, CONFIRM against pages]**

Formats are copied verbatim from the competition page and never trusted to memory — a rejected
upload late in the schedule costs a full re-run. These came from the TA-provided reference
notebooks, which is second-hand — **better than memory, not yet confirmed.**
Re-read both pages when the network allows and mark confirmed.

Both competitions use the **same line format**:

```
impression_id [rank_order]
```

`rank_order` is a permutation of 1..N, N = number of candidates in that impression.
**Rank 1 = most likely to be clicked.** Ranks are positional: the i-th rank belongs to the i-th
candidate in the impression's candidate list, so candidate order must be preserved.

### §6.1 MIND — Codabench 13967
- Source: TA reference notebook (MIND), cell 22, citing MIND's official `evaluate.py`.
  Competition page: **not yet read.**
- Example line: `1 [5,4,9,16,11,2,1,15,7,12,13,3,6,14,8,10]`
- Scoring: `1/rank` — lower rank = higher score
- File: `prediction.txt` → zipped to `mind_prediction.zip`, .txt at the archive root
- Covers: all 2,370,727 impressions of `MINDlarge_test` — **confirmed against the real file**
- Enforced by `tests/test_submission_format.py` and `validate_file()`, run before every upload

### §6.2 EB-NeRD — Codabench 2469
- Source: TA reference notebook (EB-NeRD), cell 22. Competition page: **not yet read.**
- Example line: `6451339 [8,1,6,7,4,2,9,5,3]`
- File: `predictions.txt` → zipped to `predictions.zip`
- Covers: all 13,536,710 impressions of `ebnerd_testset`

### §6.3 Submission cadence **[P0 — course update, 20 Aug 2026]**
Course staff require **at least 2 submissions per person**, the second demonstrating improvement
over the first. This splits P5 from one big submission day into two checkpoints:

| # | Model | Phase | Rationale |
|---|---|---|---|
| 1 | popularity or BM25 | end of P3 | banks the mandatory submission early, de-risks format |
| 2 | semantic / fusion | P5 | the improvement, measured offline first |

Submission 1 exists to prove the format and the plumbing while there is still time to fix a
rejection. Do not defer both to 25 Aug.

## §7 Environment & pinned dependencies **[P0 — versions TBD, see note]**

Python 3.12.3, venv at `.venv/`. Versions are filled from `pip freeze` **after** install —
pinning to versions not yet resolved would be a guess. Install is currently blocked on the
network (§10).

`requirements.txt` — the P0.5 list:

| Package | Why |
|---|---|
| `polars` | lazy scans — memory is the binding constraint (§11), and the TA notebooks use it |
| `pyarrow` | parquet feature store; row-group-at-a-time reads |
| `numpy` | — |
| `scikit-learn` | **ORACLE ONLY** — `roc_auc_score` cross-check |
| `faiss-cpu` | P4 ANN index |
| `rank-bm25` | **ORACLE ONLY** — cross-check against our own BM25 |
| `pytest` | test runner |
| `tqdm` | progress on multi-hour passes |
| `matplotlib` | the notebooks import it |
| `huggingface_hub[cli]` | gated MIND download |

The two **ORACLE ONLY** entries are load-bearing: BM25 is hand-written (§10.3 — it is Module 2
course material and exam-probed). `rank_bm25` exists solely to disagree with ours and expose bugs.
Anyone reading this manifest later must not mistake it for the implementation.

`requirements-embed.txt` — deferred to P4, ~2.5 GB, not installed yet:
`torch`, `transformers`, `sentence-transformers`.

**Dask:** offered by the TAs as an option. Not adopted — both TA notebooks use Polars lazy scans,
and matching them keeps our code cross-referenceable against the reference material. Revisit only
if a specific step proves Polars cannot stream it.

## §8 Repo shape — deviations from the standard layout **[P0]**

The agreed layout is `src/{pipeline,lexical,semantic,eval}/`, `tests/`, `prompts/`, `data/`
(gitignored), plus `SPEC.md` · `AI_USAGE.md` · `RESULTS.md` · `README.md` · `Makefile`. It permits
deviation only if recorded here, so the additions are written down rather than assumed:

| Addition | Why |
|---|---|
| `.gitignore` | mandated by A1 Q8 |
| `requirements.txt`, `requirements-embed.txt` | P0 pins dependency versions |
| `scripts/fetch_data.sh` | P0 downloads; resilience flags documented in-file |
| `report/` | P5 leaderboard screenshots land here |
| `src/baselines/` | non-retrieval reference scorers (popularity); keeps them out of `lexical/`/`semantic/` |

Not committed, by decision: the assignment's own working notes, and the TA-provided reference
notebooks (`ebnerd_analysis` / `mind_analysis`) — third-party material rather than deliverables.
They remain the cited source for §6 and §9.

## §9 Dataset facts **[P0 — from TA notebooks, re-verify from files at P1]**

### EB-NeRD large
| File | Rows |
|---|---|
| `articles.parquet` | 125,541 articles |
| `train/behaviors.parquet` | 12,063,890 impressions |
| `train/history.parquet` | 788,090 users (avg 144 articles, max 1,530) |
| `validation/behaviors.parquet` | 12,566,385 impressions |
| `validation/history.parquet` | 791,582 users |
| `ebnerd_testset/ebnerd_testset/test/behaviors.parquet` | **13,536,710 impressions, no labels** |

Test **lacks** `article_id`, `article_ids_clicked`, `next_read_time`, `next_scroll_percentage`;
**adds** `is_beyond_accuracy` (1.5% of impressions).

### MIND
| File | Rows |
|---|---|
| `MINDsmall_train/behaviors.tsv` | 156,965 impressions |
| `MINDsmall_train/news.tsv` | 51,282 articles |
| `MINDsmall_dev/behaviors.tsv` | 73,152 impressions |
| `MINDlarge_test/behaviors.tsv` | **2,370,727 impressions, no labels** |
| `MINDlarge_test/news.tsv` | 120,961 articles |
| `entity_embedding.vec` | 100-dim TransE, Wikidata |

### Q9 serving-time honesty — the ablation pair **[P0]**
EB-NeRD hands us the ablation directly. `next_read_time` and `next_scroll_percentage` exist in
train/validation and are **absent from test**, because they describe the *next* impression —
future information by construction. They are the "features unavailable at serving time" that Q9
asks us to report with and without. `is_beyond_accuracy` is likewise test-only.

MIND has no equivalent pair; its ablation is history-length truncation instead. **[TBD@P4]**

## §10 Open decisions

Carried from the pre-Phase-1 decision list, plus what P0 surfaced.

| # | Decision | Status |
|---|---|---|
| 1 | N, M per dataset | **[TBD@P1]** — needs real timestamps |
| 2 | Cold-start threshold; head/tail percentile | **[TBD@P2]** |
| 3 | Own BM25 vs `rank_bm25` | **settled: own**, library as oracle |
| 4 | MIND embedding model | **[TBD@P4]** — MIND's TransE entity vectors may remove the need for torch |
| 5 | Submission formats verbatim | §6 — from notebooks, **pages not yet read** |
| 6 | C2 boundary (click-log modelling) | **[TBD@P6]** — do not block 10 Sep |
| 7 | **Where the large-scale run executes** | **open — see §11** |

## §11 The binding constraint: this laptop **[P0, measured 20 Aug 2026]**

Two hardware measurements change the plan, and both belong in the Q6 "breaks at 10×" section as
evidence rather than speculation.

**Memory: 7 GB total, ~2 GB available.** The TA notebook annotates its MIND batching as
"Safe for 10GB RAM" and calls `pl.read_csv` on all 2,370,727 test rows non-lazily. That will not
fit here. Every large-scale step must be lazy or row-group batched, and the MIND reader in
particular needs `scan_csv`, not `read_csv`.

**Network: 4–13 KB/s, 50% packet loss, 305–1225 ms RTT on `wifi@iiith`.** Measured against three
independent endpoints (S3 eu-west-1, Cloudflare, HuggingFace), so it is the link, not a server.
At 13 KB/s the 2.97 GB `ebnerd_large.zip` needs ~63 hours of *uninterrupted* transfer; the full
~5 GB EB-NeRD set plus MIND exceeds the time remaining before the deadline.

Consequence: the large-scale path likely cannot run on this laptop at all, and the recommendation
is to move download **and** prediction for the large bundles to Colab/Kaggle, where the fetch is
gigabit and RAM is 12–16 GB — keeping the laptop to demo/small for development. Decision §10.7.
