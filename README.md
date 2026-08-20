# News Recommendation System — IRE A1, Component 1

Lexical & semantic retrieval on **EB-NeRD** (Danish, RecSys 2024 Challenge) and **MIND**
(English, Microsoft). Ranks the candidate articles in an impression by click likelihood from
click history, session context, and article content.

CS4.406 Information Retrieval & Extraction · Individual · Due 27 Aug 2026

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
| `AI_USAGE.md` | tools, prompts, what worked and what failed |
| `src/pipeline/` | readers, unified schema, temporal split, feature store |
| `src/lexical/` | inverted index + BM25 |
| `src/semantic/` | embeddings + ANN index |
| `src/eval/` | metrics, slices, bootstrap CIs |
| `tests/` | oracles, incl. `test_no_leakage.py` |
| `scripts/fetch_data.sh` | raw downloads (resumable) |

## Status

Phase 0 (setup) — in progress. `make data` / `make eval` / `make bench` are stubs that exit
non-zero until their phase lands; `make test` is intentionally red until P1 (see
`tests/test_no_leakage.py`).

## Notes

- Data is **never** split randomly — temporal only. See `SPEC.md` §3.
- `data/` is gitignored, as are `*.zip *.pt *.ckpt __pycache__/`.
