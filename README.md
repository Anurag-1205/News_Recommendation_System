# Learning from Click-Logs on MIND and EB-NeRD

**CS4.406 Information Retrieval and Extraction — Assignment 2**

| | |
|---|---|
| **Team** | Anurag Kaushal (2025202013) · Aayush Pandey (2025201058) |
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
make env             # .venv + pinned deps (Python 3.12.3)
make fetch-small     # EB-NeRD demo + small (~0.1 GB, public S3)
make fetch-testset   # EB-NeRD test set (~1.6 GB) — the minimum `make data` needs for EB-NeRD
make fetch-mind      # MIND small train/dev + large test — needs `hf auth login` first (gated)
make fetch-large     # EB-NeRD large + embeddings (~5 GB) — only for the full-scale submission runs
make data            # raw -> unified schema -> temporal split -> feature store (idempotent)
make test            # 376 tests, incl. the no-leakage assertions (Q9) and the batch/serving parity test
make eval SCORES=data/scores/ebnerd/validation/reranker_final.parquet     # Q5: all metrics, slices, CIs
make paired A=<scores.parquet> B=<scores.parquet> [JSON=out.json]          # Q3.4: paired bootstrap judge
make bench DATASET=ebnerd|mind                                             # Q4: memory, p50/p95/p99, cost
make note            # Q6: report/design_note.md -> report/design_note.pdf
```

`make` on its own lists every target. GPU work (the NRMS baselines and their freshness variant,
the embedding assets, the two test-set submission kernels) runs on Kaggle from `scripts/kaggle/`;
every run has a row in `scripts/kaggle/RUN_LEDGER.md`. Score files, models, submissions and
screenshots live under the gitignored `data/` and `report/` (A2 Q8).

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
| **Q1** click-history & session features | **done** — recency-decayed profile, session, dwell, position, popularity, freshness, category match, each with a leakage test (`RESULTS.md` Q1) |
| **Q2** two-stage reranker | **done** — locked in `src/rerank/config.FINAL` (C-018): EB-NeRD lambdarank AUC 0.6734, MIND pointwise 0.6747, before/after tables in `RESULTS.md` Q2. D1 framing (b) descoped on measured stage-1 recall (Q2.5, C-037) |
| **Q3** baseline reproduced, then beaten | **done** — NRMS on both datasets; EB-NeRD + freshness beats NRMS, ΔAUC +0.0074 [+0.0066, +0.0081]; MIND a pre-registered null. Claim independently reviewed (C-033) |
| **Q4** serving & scale | **done** — memory per component, p99 1.7 ms (EB-NeRD) / 3.4 ms (MIND) served, cost/1k queries, and what breaks at 10× (`RESULTS.md` Q4) |
| **Q5** extended eval + leaderboards | **done** except the leaderboard lines: `make eval` metrics + both slices (`RESULTS.md` Q5), reranker vs NRMS paired (C-035), both test files scored and validated (Q5.6); Codabench uploads in progress (17 Sep) |
| **Q6** design note | **drafted** — `report/design_note.md`, `make note` → 6-page PDF; leaderboard scores/screenshots pending |
| **Q9** leakage test + serving-time ablation | **done** — with/without serving-unavailable features on both datasets with paired CIs (`RESULTS.md` Q9). On EB-NeRD the unavailable features make the model *worse*, −0.0236 AUC [−0.0245, −0.0227] |

## Notes

- Data is **never** split randomly — temporal only. See `SPEC.md` §3.
- `data/`, `external/`, `*.zip *.pt *.ckpt __pycache__/` are gitignored.
