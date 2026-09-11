#!/usr/bin/env python3
"""Choose the recency half-life h (P1-D2) on EB-NeRD validation.

    PYTHONPATH=. .venv/bin/python -u scripts/tune_half_life_ebnerd.py [--impressions 50000]

Each h in {6 h, 24 h, 72 h, inf} is scored as a *single-feature ranker*: candidates in an
impression are ordered by `recency_weighted_profile` (and, separately, `category_match`) alone.
That isolates the hyperparameter from everything a GBDT would add. inf is `timedelta.max`: every
weight is 1 to within 1e-8, i.e. the undecayed category distribution — the no-decay baseline.

Evaluated on a seeded random sample of validation impressions (random *sampling* of evaluation
rows, not a split; the split is the shipped temporal one). Every h is compared with inf by a paired
bootstrap over the same impressions, so the question "does decay help?" gets a CI of its own.

MIND is not tuned: its history has no click times, so h cannot change its profile (C-008).
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from src.rerank.common import SEED, category_profile_features, evaluate, paired_delta, per_impression
from src.rerank.ebnerd import SMALL, candidate_frame, load_articles, load_behaviors, load_history

OUT = Path("data/processed")
H_GRID = {"6h": timedelta(hours=6), "24h": timedelta(hours=24), "72h": timedelta(hours=72), "inf": timedelta.max}
FEATURES = ("recency_weighted_profile", "category_match")


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impressions", type=int, default=50_000)
    args = ap.parse_args()

    articles = load_articles()
    beh = load_behaviors(SMALL / "validation/behaviors.parquet")
    rng = np.random.default_rng(SEED)
    sample = np.sort(rng.choice(beh.height, size=min(args.impressions, beh.height), replace=False))
    long = candidate_frame(beh.filter(pl.col("imp_row").is_in(sample)), articles)
    history = load_history(SMALL / "validation/history.parquet", articles)
    log(f"validation sample: {len(sample):,} of {beh.height:,} impressions -> {long.height:,} candidate rows")
    log(f"history: {history.height:,} clicks, {history['user_id'].n_unique():,} users; "
        f"category missing for {history['category'].null_count():,} clicks and "
        f"{long['candidate_category'].null_count():,} candidates")
    log(f"latest history ts {history['ts'].max()}  <  earliest sampled t {long['t'].min()}")
    del beh

    per_imp, results = {}, {}
    for name, h in H_GRID.items():
        t0 = time.perf_counter()
        scored = long.join(category_profile_features(long, history, h), on=["imp_row", "cand_position"], how="left")
        for f in FEATURES:
            per_imp[(name, f)] = per_impression(scored, f)
            ci = evaluate(per_imp[(name, f)])
            results.setdefault(f, {})[name] = {m: {"mean": c.mean, "lo": c.lo, "hi": c.hi, "n": c.n} for m, c in ci.items()}
            log(f"  h={name:4s} {f:25s} AUC {ci['auc']}  nDCG@10 {ci['ndcg@10']}  ({time.perf_counter() - t0:.0f}s)")

    log("paired difference vs inf (no decay), same impressions:")
    for f in FEATURES:
        for name in ("6h", "24h", "72h"):
            for m in ("auc", "ndcg@10"):
                d = paired_delta(per_imp[("inf", f)][m], per_imp[(name, f)][m])
                results[f][name][f"delta_vs_inf_{m}"] = {"mean": d.mean, "lo": d.lo, "hi": d.hi, "n": d.n}
                verdict = "excludes 0" if (d.lo > 0 or d.hi < 0) else "includes 0"
                log(f"  {f:25s} h={name:4s} delta {m:7s} {d.mean:+.4f} [{d.lo:+.4f}, {d.hi:+.4f}]  {verdict}")

    best = max(("6h", "24h", "72h", "inf"), key=lambda n: results["recency_weighted_profile"][n]["auc"]["mean"])
    results["chosen_half_life"] = best
    results["sample_impressions"] = int(len(sample))
    (OUT / "halflife_ebnerd.json").write_text(json.dumps(results, indent=2))
    log(f"highest AUC for recency_weighted_profile: h={best}  -> {OUT / 'halflife_ebnerd.json'}")


if __name__ == "__main__":
    main()
