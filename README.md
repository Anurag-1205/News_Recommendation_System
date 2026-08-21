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

| Deliverable | State |
|---|---|
| Ranking metrics + submission format | done, 65 oracle tests |
| MIND submission 1 (popularity) | **submitted** — Codabench 13967, AUC 0.5036 |
| EB-NeRD submission | pending `ebnerd_testset.zip` |
| BM25 / semantic retrieval | not started |
| Data pipeline, temporal split | not started — `make data` exits non-zero |

`make eval` / `make bench` are stubs until their phase lands, and `tests/test_no_leakage.py` is
intentionally red until the feature store exists.

## What submission 1 taught us

The popularity baseline scored **AUC 0.5036** on the MIND leaderboard. Chance is 0.5000. The
model contributes almost nothing — and understanding *why* is what determines the next model.

**The diagnosis.** Popularity can only rank an article it watched being clicked during training.
On the test split it has almost never seen one:

| | Candidate coverage | Impressions with zero known candidates |
|---|---:|---:|
| `MINDsmall_dev` | 34.2% | — |
| `MINDlarge_test` | **6.5%** | **28.6%** |

MIND trains on 11 Nov 2019 and tests on 19–22 Nov 2019, against a news set of 120,961 articles
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
4. **Trust the offline harness, with a known offset.** All four offline metrics were optimistic by
   0.013–0.028 and none inverted, so offline comparisons predict the direction of a leaderboard
   move. Model selection can happen locally without spending submissions to learn things.
5. **Fit on the largest training split available.** Coverage is partly a sample-size problem;
   `MINDlarge_train` sees far more of the article space than `MINDsmall_train`'s 51,282.

**What this does not mean.** The baseline was not a mistake. It is the floor that makes any later
improvement measurable, it proved the submission format end to end while there was still time to
fix a rejection, and it produced the coverage measurement that determines what to build next.
Grading is on pipeline correctness, ablation rigour and scale analysis — never on rank.

## Notes

- Data is **never** split randomly — temporal only. See `SPEC.md` §3.
- `data/` is gitignored, as are `*.zip *.pt *.ckpt __pycache__/`.
