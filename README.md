# Learning from Click-Logs on MIND and EB-NeRD

**CS4.406 Information Retrieval and Extraction — Assignment 2**

| | |
|---|---|
| **Team** | Anurag Kaushal (2025202013) · _partner name, roll number_ |
| **Repository** | https://github.com/Anurag-1205/News_Recommendation_System, branch `a2-click-logs` |
| **Submission** | Team of 2 · due 20 September 2026 |
| **Builds on** | Assignment 1, frozen on `main` at `be15ee6` |

A two-stage news recommender. Assignment 1's lexical (BM25) and semantic (FAISS) candidate
generators retrieve the top-K articles, and a reranker trained on behavioural features from
click logs orders them: recency-weighted history, session context, popularity and freshness. The
system is evaluated against a reproduced NRMS baseline with paired bootstrap confidence
intervals, benchmarked for serving cost and latency, and submitted to both Codabench
leaderboards ([MIND 13967](https://www.codabench.org/competitions/13967/),
[RecSys 2024 / EB-NeRD 2469](https://www.codabench.org/competitions/2469/)).

`SPEC.md` holds the interfaces and verification strategy, `RESULTS.md` every measured figure with
its command, and `PLAN.md` / `CONTEXT.md` the team's plan and decision log.

## Reproduce

```bash
make env           # .venv + pinned deps
make fetch-small   # EB-NeRD demo + small
make fetch-mind    # MIND — needs `hf auth login` first (dataset is gated)
make fetch-large   # EB-NeRD large + testset + embeddings (~5 GB, needed for Codabench)
make data          # raw -> unified schema -> temporal split -> feature store
make test          # incl. the no-leakage assertions (Q9)
make eval          # A2 Q5 — not yet implemented
make bench         # A2 Q4 — not yet implemented
```

`make` on its own lists every target.

## Layout

| Path | What |
|---|---|
| `src/pipeline/` | readers, unified schema, temporal split, feature store |
| `src/lexical/` | inverted index + BM25 (stage-1 candidates) |
| `src/semantic/` | embeddings, user vectors, FAISS index (stage-1 candidates) |
| `src/features/` | point-in-time behavioural features, strictly before *t* |
| `src/baselines/` | popularity, score fusion |
| `src/eval/` | metrics, beyond-accuracy, slices, bootstrap CIs, submission writer |
| `tests/` | oracles, incl. `test_no_leakage.py` |
| `scripts/` | data acquisition and experiment drivers |

## Status

| Deliverable | State |
|---|---|
| **Q1** click-history & session features | not started |
| **Q2** two-stage reranker | not started (A1 pointwise GBDT is the starting point) |
| **Q3** baseline reproduced, then beaten | not started |
| **Q4** serving & scale | not started |
| **Q5** extended eval + leaderboards | not started |
| **Q6** design note | not started |
| **Q9** leakage test + serving-time ablation | A1 tests carried over; A2 features pending |

## Notes

- Data is **never** split randomly — temporal only. See `SPEC.md` §3.
- `data/`, `external/`, `*.zip *.pt *.ckpt __pycache__/` are gitignored.
