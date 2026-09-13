#!/usr/bin/env python3
"""How much can a freshness-only ranker do within a slate? (CONTEXT.md C-029; SPEC.md §15)

    PYTHONPATH=. .venv/bin/python scripts/check_fresh_signal.py [ebnerd|mind ...]

For each dataset: fit the §15.2 stats on the training split, rank every evaluation slate by
x (fresher-first) and by −x (older-first), unknown last, and report the reranker's own
metrics. This bounds what an additive g(freshness) term in NRMS can capture: the term sees the
candidate's age only, so its ceiling is the within-slate marginal signal measured here.
"""
import sys

import numpy as np
import polars as pl

from src.baselines.nrms_fresh_features import fit_stats, fresh_inputs
from src.eval.metrics import per_impression_metrics
from src.eval.paired import split_labels
from src.rerank.common import impression_rows

SPLITS = {"ebnerd": ("ebnerd_small/train", "ebnerd_small/validation"), "mind": ("MINDsmall_train", "MINDsmall_dev")}


def main(datasets):
    for ds in datasets:
        tr, ev = SPLITS[ds]
        stats = fit_stats(fresh_inputs(ds, tr, None))
        f = fresh_inputs(ds, ev, stats)
        j = f.join(split_labels(ds, ev), on=["imp_row", "cand_position"], how="left")
        assert j["label"].null_count() == 0
        for name, sign in [("fresher-first (-x)", -1.0), ("older-first (+x)", 1.0)]:
            jj = j.with_columns(score=(sign * pl.col("x")).cast(pl.Float64) - 10.0 * pl.col("unknown"))
            m = per_impression_metrics(impression_rows(jj, "score", label_col="label"))
            print(f"{ds:7s} {name:20s} AUC {np.nanmean(m['auc']):.4f}  MRR {np.mean(m['mrr']):.4f}  "
                  f"nDCG@5 {np.mean(m['ndcg@5']):.4f}  nDCG@10 {np.mean(m['ndcg@10']):.4f}  n={len(m['mrr']):,}")
        r = j.group_by("imp_row").agg((pl.col("x").max() - pl.col("x").min()).alias("r"))["r"]
        print(f"{ds:7s} within-slate range of x: median {r.median():.3f}; slates with range < 0.05: {(r < 0.05).mean():.3%}; stats μ={stats.mu:.3f} σ={stats.sigma:.3f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["ebnerd", "mind"])
