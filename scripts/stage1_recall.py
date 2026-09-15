#!/usr/bin/env python3
"""Q2.1 / D1 framing (b): stage-1 recall@K of the retrieve-then-rerank path (SPEC.md §16.1).

    PYTHONPATH=. .venv/bin/python -u scripts/stage1_recall.py --dataset ebnerd [--n 1000] [--seed 0]
    PYTHONPATH=. .venv/bin/python -u scripts/stage1_recall.py --dataset mind

The brief asks for the reranker to sit on a top-K (100-200) *retrieved from the corpus*. The
reranker that ships re-ranks the impression's own slate instead (framing (a)), because that is
what both leaderboards score. This script measures the one number that decides whether framing
(b) could even work here: **how often the article the user actually clicked is inside the
retrieved top-K at all.** A reranker cannot rank what stage 1 did not return, so recall@K is a
hard ceiling on any framing-(b) accuracy metric. Full framing-(b) AUC/nDCG are not reported:
the labels only exist for the impression's slate, so a retrieved list that mostly contains
unlabelled articles would produce a number that is neither comparable with (a) nor meaningful.

Retrieval is `src.serving.request.retrieve` -- the *served* path P4 timed -- on the same seeded
validation sample `scripts/bench.py` used (n + warmup impressions, the last n measured), so the
recall and the latency describe one and the same list. Per impression: hit@K = 1 if any clicked
article is in the union (BM25 top-K ∪ ANN top-K), and clicked-recall@K = the fraction of that
impression's clicked articles retrieved. Both are reported with a blocked bootstrap 95% CI over
impressions. The slate-based upper bound (recall if the candidates were the impression's own
slate) is 1.0 by construction, which is the whole point of framing (a).
"""
from __future__ import annotations

import argparse, json, subprocess, time
from pathlib import Path

import numpy as np
import polars as pl

from src.eval.bootstrap import bootstrap_ci
from src.serving.request import Request, retrieve

OUT = Path("data/processed/stage1_recall")


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sample_requests(dataset: str, n: int, warmup: int, seed: int):
    """bench.py's sample exactly: sorted seeded rows, the first `warmup` discarded."""
    if dataset == "ebnerd":
        from src.rerank.ebnerd import SMALL, load_behaviors
        val = load_behaviors(SMALL / "validation/behaviors.parquet")
        clicked_of = lambda r: [int(a) for a in r["clicked"]]
    else:
        from src.rerank.mind import load_behaviors
        val = load_behaviors("MINDsmall_dev")
        clicked_of = lambda r: [a for a, y in zip(r["candidates"], r["labels"]) if y == 1]
    rows = np.sort(np.random.default_rng(seed).choice(val.height, size=n + warmup, replace=False))
    sample = val.filter(pl.col("imp_row").is_in(rows)).sort("imp_row")
    reqs = []
    for r in sample.iter_rows(named=True):
        req = Request(r["user_id"], r["t"], list(r["candidates"]), r["imp_row"],
                      history=list(r["history_ids"]) if "history_ids" in sample.columns else None)
        reqs.append((req, clicked_of(r), len(r["candidates"])))
    return reqs[warmup:], val.height


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("ebnerd", "mind"), required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ks", type=int, nargs="+", default=[100, 200])
    args = ap.parse_args()

    from src.serving.state import ServingState
    t0 = time.perf_counter()
    st = ServingState.build(args.dataset)
    log(f"serving state built in {time.perf_counter() - t0:.0f}s")

    reqs, n_split = sample_requests(args.dataset, args.n, args.warmup, args.seed)
    log(f"{len(reqs)} measured impressions (seed {args.seed}, bench.py's sample) out of {n_split:,}")

    result = {}
    for k in args.ks:
        hit, rec, n_ret, n_clicked, n_bm25_only = [], [], [], [], []
        t1 = time.perf_counter()
        for req, clicked, _ in reqs:
            ids, _, uv = retrieve(st, req, k)
            got = set(ids)
            n_ret.append(len(ids))
            n_clicked.append(len(clicked))
            if not clicked:
                continue                       # nothing to recall; excluded from the rates
            found = sum(1 for a in clicked if a in got)
            hit.append(1.0 if found else 0.0)
            rec.append(found / len(clicked))
            n_bm25_only.append(1.0 if uv is None else 0.0)
        wall = time.perf_counter() - t1
        h, r = bootstrap_ci(hit, seed=args.seed), bootstrap_ci(rec, seed=args.seed)
        result[f"k{k}"] = {
            "k": k, "n_impressions": len(hit),
            "hit_at_k": {"mean": h.mean, "lo": h.lo, "hi": h.hi},
            "clicked_recall_at_k": {"mean": r.mean, "lo": r.lo, "hi": r.hi},
            "mean_retrieved": float(np.mean(n_ret)), "max_retrieved": int(np.max(n_ret)),
            "mean_clicked_per_impression": float(np.mean(n_clicked)),
            "share_without_user_vector": float(np.mean(n_bm25_only)) if n_bm25_only else 0.0,
            "wall_s": wall,
        }
        log(f"K={k}: hit@K {h.mean:.4f} [{h.lo:.4f}, {h.hi:.4f}]  clicked-recall@K {r.mean:.4f} "
            f"[{r.lo:.4f}, {r.hi:.4f}]  over {len(hit)} impressions; union size mean {np.mean(n_ret):.1f} "
            f"max {np.max(n_ret)}  ({wall:.0f}s)")

    OUT.mkdir(parents=True, exist_ok=True)
    record = {
        "dataset": args.dataset, "framing": "retrieved-topk (b)", "retriever": "src.serving.request.retrieve: BM25 top-K ∪ flat-ANN top-K",
        "sample": {"n": args.n, "warmup": args.warmup, "seed": args.seed, "split_size": n_split, "same_as": "scripts/bench.py"},
        "results": result, "seed": args.seed,
        "repo_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "command": f"PYTHONPATH=. .venv/bin/python -u scripts/stage1_recall.py --dataset {args.dataset} --n {args.n} --seed {args.seed}",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = OUT / f"{args.dataset}.json"
    path.write_text(json.dumps(record, indent=2))
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
