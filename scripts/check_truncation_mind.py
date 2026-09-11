#!/usr/bin/env python3
"""C-016 option (c): does a lambdarank truncation level covering MIND's longest slate rescue the
listwise objective on MIND?

    PYTHONPATH=. .venv/bin/python -u scripts/check_truncation_mind.py

LightGBM's lambdarank builds its gradients from the top `lambdarank_truncation_level` positions of
each query (documented default 30). MIND slates average 37 candidates and reach 299, against
EB-NeRD's 11 and 100 — so on MIND the default ignores much of each list. Here the truncation level
is raised to cover the longest slate seen in train or dev, and everything else is held fixed: the
same 80,000 train impressions, the same features, the same 73,152 dev impressions, the frames built
by `scripts/rerank_mind_a2.py` itself.

Rows: pointwise GBDT (reference), lambdarank at the default, lambdarank at the raised level — for
both the A1 v4 base and A2 feature sets. The truncation level is itself a choice read on dev, so
dev is also split at its median impression time and the key differences are reported per half:
a real effect should hold in both.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from scripts.rerank_mind_a2 import BASE, DEV, PHASE1_NEW, TRAIN, Stage1, build, log
from src.rerank.common import SEED, fit_gbdt, fit_lambdarank, group_sizes, matrix, model_features, report
from src.rerank.mind import first_sightings, load_behaviors, load_categories

OUT = Path("data/processed")
FIT_N = 80_000                                  # as in rerank_mind_a2.py


def main() -> None:
    a2 = model_features(BASE + PHASE1_NEW)
    train, dev = load_behaviors(TRAIN), load_behaviors(DEV)
    categories = load_categories([TRAIN, DEV])
    sightings = first_sightings([train, dev])
    stage1 = Stage1([TRAIN, DEV], train)
    fit_sample = np.sort(np.random.default_rng(SEED).choice(train.height, size=min(FIT_N, train.height),
                                                            replace=False))  # rerank_mind_a2.py's draw
    tr = build(train, TRAIN, fit_sample, categories, sightings, stage1)
    va = build(dev, DEV, np.arange(dev.height), categories, sightings, stage1)
    del train

    max_slate = int(max(tr["n_candidates"].max(), va["n_candidates"].max()))
    trunc = max(300, max_slate)
    log(f"longest slate: {max_slate} candidates (train fit sample and dev) -> truncation level {trunc}")

    y, groups = tr["label"].to_numpy(), group_sizes(tr)
    fits = {"gbdt_base": (BASE, None), "gbdt_a2": (a2, None),
            "lr_base": (BASE, {}), "lr_a2": (a2, {}),
            "lr_base_t": (BASE, {"lambdarank_truncation_level": trunc}),
            "lr_a2_t": (a2, {"lambdarank_truncation_level": trunc})}
    for name, (cols, overrides) in fits.items():
        t0 = time.perf_counter()
        if overrides is None:
            s = fit_gbdt(matrix(tr, cols), y).predict_proba(matrix(va, cols))[:, 1]
        else:
            s = fit_lambdarank(matrix(tr, cols), y, groups, **overrides).predict(matrix(va, cols))
        va = va.with_columns(pl.Series(name, s))
        log(f"  fitted {name} ({len(cols)} features, {time.perf_counter() - t0:.0f}s)")

    rows = {"gbdt_base": "pointwise: A1 v4 base", "gbdt_a2": "pointwise: A2",
            "lr_base": "lambdarank default (30): A1 v4 base", "lr_a2": "lambdarank default (30): A2",
            "lr_base_t": f"lambdarank trunc {trunc}: A1 v4 base", "lr_a2_t": f"lambdarank trunc {trunc}: A2"}
    pairs = [("lr_base", "lr_base_t"), ("lr_a2", "lr_a2_t"), ("gbdt_base", "lr_base_t"),
             ("gbdt_a2", "lr_a2_t"), ("gbdt_base", "lr_a2_t"), ("lr_base_t", "lr_a2_t")]
    results = {"truncation_level": trunc, "max_slate": max_slate, "eval_impressions": int(va["imp_row"].n_unique())}
    log(f"== all dev ({results['eval_impressions']:,} impressions)")
    results["all_dev"] = {}
    report(va, rows, pairs, log, results["all_dev"])

    median_t = va.group_by("imp_row").agg(pl.col("t").first())["t"].median()
    for half, cond in (("earlier_half", pl.col("t") < median_t), ("later_half", pl.col("t") >= median_t)):
        part = va.filter(cond)
        log(f"== dev {half} (t {'<' if half == 'earlier_half' else '>='} {median_t}, "
            f"{part['imp_row'].n_unique():,} impressions)")
        results[half] = {}
        report(part, {k: rows[k] for k in ("gbdt_base", "lr_base", "lr_base_t", "lr_a2_t")},
               [("lr_base", "lr_base_t"), ("gbdt_base", "lr_base_t"), ("gbdt_base", "lr_a2_t"),
                ("lr_base_t", "lr_a2_t")], log, results[half])

    (OUT / "truncation_mind.json").write_text(json.dumps(results, indent=2, default=str))
    log(f"done -> {OUT / 'truncation_mind.json'}")


if __name__ == "__main__":
    main()
