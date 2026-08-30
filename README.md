# Lexical and Semantic Retrieval on MIND and EB-NeRD

**CS4.406 Information Retrieval and Extraction — Assignment 1, Component 1**

| | |
|---|---|
| **Name** | Anurag Kaushal |
| **Roll number** | 2025202013 |
| **Programme** | M.Tech CSIS |
| **Repository** | https://github.com/Anurag-1205/News_Recommendation_System |
| **Submission** | Individual · due 27 August 2026 |

**Leaderboards.** MIND ([Codabench 13967](https://www.codabench.org/competitions/13967/)) —
rank 78, AUC 0.5714. EB-NeRD / RecSys 2024
([Codabench 2469](https://www.codabench.org/competitions/2469/)) — rank 190, AUC 0.5110.

A reproducible pipeline that ranks the candidate articles of an impression by click
likelihood, using click history, session context and article content, on **EB-NeRD** (Danish,
RecSys 2024 Challenge) and **MIND** (English, Microsoft). Covers lexical (BM25) and semantic
(embedding) candidate generation, an offline evaluation harness with bootstrap confidence
intervals, and submissions to both Codabench leaderboards.

The design note required by Q6 is `report/design_note.md`. Component specifications and the
verification strategy are in `SPEC.md`; every measured figure with its originating command is
in `RESULTS.md`. The AI usage log required by Q7.4 is submitted with the report rather than
through this repository.

## Reproduce

```bash
make env           # .venv + pinned deps
make fetch-small   # EB-NeRD demo + small (~0.10 GB)
make fetch-mind    # MIND — needs `hf auth login` first (dataset is gated)
make data          # raw -> unified schema -> temporal split -> feature store
make test          # incl. the no-leakage assertion (Q9)
make eval          # AUC · MRR · nDCG@5 · nDCG@10 + diversity/novelty/coverage, with 95% CIs
```

`make` on its own lists every target.

The large bundles needed for Codabench are fetched separately, since they are ~5 GB:

```bash
make fetch-large   # EB-NeRD large + testset + embeddings
```

## Layout

| Path | What |
|---|---|
| `SPEC.md` | interfaces, decisions, and **how each piece is verified** |
| `RESULTS.md` | every measured number, with the command that produced it |
| `src/pipeline/` | readers, unified schema, temporal split, feature store |
| `src/lexical/` | inverted index + BM25 |
| `src/semantic/` | embeddings + ANN index |
| `src/eval/` | metrics, slices, bootstrap CIs |
| `tests/` | oracles, incl. `test_no_leakage.py` |
| `src/features/` | point-in-time popularity / CTR, strictly-before-t |
| `src/baselines/` | popularity, score fusion (RRF, weighted sum) |
| `report/design_note.md` | Q6 design note |
| `scripts/fetch_data.sh` | raw downloads (resumable) |

## Status

| Deliverable | State |
|---|---|
| **Q1** reproducible pipeline, temporal split | done — `make data`, disjointness asserted at build |
| **Q2** BM25 lexical retrieval, recall@K | done — own index + scorer, `n_recent` ablation, both datasets (MIND `n_recent`=5 wins, EB-NeRD `n_recent`=20 wins — opposite, measured not assumed) |
| **Q3** semantic retrieval, ANN, lexical-vs-semantic by slice | done — both datasets. MIND: LSA→MiniLM + FAISS. EB-NeRD: provided word2vec (100% coverage) + FAISS. Pooling ablation reverses between them (mean wins on MIND, recency on EB-NeRD) |
| **Q4** eval harness, beyond-accuracy, slices, bootstrap CIs | done |
| **Q5** both Codabench leaderboards | MIND 0.5036 → 0.5258 → 0.5554 → **0.5714** (four, monotonic); EB-NeRD **0.5110**, second submission pending |
| **Q6** design note | `report/design_note.md` |
| **Q9** leakage test + serving-time ablations | done — 26 tests; two ablations, see below |

212 tests. `make bench` remains a stub; its numbers are collected inline and live in `RESULTS.md`.

**Two Q9 ablations**, because they probe different failures:
1. *Serving-unavailable columns* — adding EB-NeRD's `next_read_time` moves AUC 0.5029 → 0.9629
   (**+0.46**). A chance-level model looks near-perfect via a column that cannot exist at
   request time.
2. *Frozen vs point-in-time popularity* — on MIND, rolling 0.6447 vs frozen 0.6058, disjoint.
   The frozen variant leaks yet scores **worse**: staleness costs more than the leak gains.
   On EB-NeRD the same comparison shows **no difference** — because there, recency dominates and
   popularity barely contributes. The mechanism is in `RESULTS.md` Q9c.

**Train/serve skew, and the validation rebuilt to detect it.** Submission 3's offline-to-
leaderboard offset was 0.090 against 0.03 for its predecessors. Measured cause: `pop_24h`,
`pop_1h` and `ctr_24h` are zero for **100%** of test candidates, because training events end
14 Nov and the test split runs 16–22 Nov — while on dev, adjacent to training, two of the three
carried signal 42% of the time. Not leakage; the validation split simply sat in a temporal
position the test set never occupies. The fix was a gap-aware protocol (`scripts/gap_aware_mind.py`)
that evaluates the same distance past the last counted event as the test split does.

## Reproduce in order

```bash
make env && make fetch-small && make fetch-mind   # MIND needs `hf auth login` first
make data      # raw -> unified schema -> temporal split -> feature store
make test      # 212 tests, incl. the no-leakage assertions
PYTHONPATH=. .venv/bin/python scripts/eval_bm25_mind.py       # Q2
PYTHONPATH=. .venv/bin/python scripts/eval_semantic_mind.py   # Q3 + Q3.5 + Q4.2
PYTHONPATH=. .venv/bin/python scripts/ablation_serving_time.py # Q9  (serving-unavailable)
PYTHONPATH=. .venv/bin/python scripts/rerank_mind.py           # Q9b (frozen vs rolling)
PYTHONPATH=. .venv/bin/python scripts/encode_mind_minilm.py    # one-off sentence-transformer cache
PYTHONPATH=. .venv/bin/python scripts/gap_aware_mind.py        # gap-aware model selection
```

## What the baseline taught us

The popularity baseline scored **AUC 0.5036** on the MIND leaderboard. Chance is 0.5000. The
model contributes almost nothing — and understanding *why* is what determines the next model.

**The diagnosis.** Popularity can only rank an article it watched being clicked during training.
On the test split it has almost never seen one:

| | Candidate coverage | Impressions with zero known candidates |
|---|---:|---:|
| `MINDsmall_dev` | 34.2% | — |
| `MINDlarge_test` | **6.5%** | **28.6%** |

MIND trains on 9–14 Nov 2019 and tests on 16–22 Nov 2019, against a news set of 120,961 articles
versus small-train's 51,282. News turns over in days, so 93.5% of the articles we are asked to
rank are ones we have no click evidence for at all. For 28.6% of impressions *every* candidate is
unknown, all scores tie at zero, and the output degenerates to the candidate list's own order.
Roughly a third of the submission is therefore not a prediction.

**What changes as a result.**

1. **Score text, not identifiers.** Every one of the 120,961 test articles has a title and an
   abstract. A model that scores those has 100% coverage by construction, against popularity's
   6.5%. This single change is why BM25 is next, and the 6.5% → 100% gap is a measured reason to
   expect improvement rather than a hopeful one.
2. **Use the click history — we currently ignore it entirely.** MIND puts each user's prior clicks
   inline in `behaviors.tsv` and the popularity baseline reads none of it. It is the same score for
   every user, which caps AUC near chance no matter how good the popularity estimate gets. The
   history is the only per-user signal in the dataset and it is untouched.
3. **Never let a third of the output be a tie.** Any signal that separates candidates — category
   match against history, title overlap, article recency — beats arbitrary ordering on the 28.6%.
4. **Trust the offline harness for *direction*, not for size.** All four offline metrics were
   optimistic by 0.013–0.028 here and none inverted. That consistency was later over-read as a
   calibration constant: submission 3's offset was 0.090, and EB-NeRD's ran the other way
   entirely (0.5110 against an offline 0.5030). Offline comparisons predict which way a change
   moves; they overstate how far, by an amount that grows with the test window's distance from
   training.
5. **Fit on the largest training split available.** Coverage is partly a sample-size problem;
   `MINDlarge_train` sees far more of the article space than `MINDsmall_train`'s 51,282.

**What this does not mean.** The baseline was not a mistake. It is the floor that makes any later
improvement measurable, it proved the submission format end to end while there was still time to
fix a rejection, and it produced the coverage measurement that determines what to build next.
Grading is on pipeline correctness, ablation rigour and scale analysis — never on rank.

## Notes

- Data is **never** split randomly — temporal only. See `SPEC.md` §3.
- `data/` is gitignored, as are `*.zip *.pt *.ckpt __pycache__/`.
