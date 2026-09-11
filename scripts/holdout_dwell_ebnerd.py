#!/usr/bin/env python3
"""Holdout test of the EB-NeRD dwell finding (CONTEXT.md C-016, C-017).

    PYTHONPATH=. .venv/bin/python -u scripts/holdout_dwell_ebnerd.py

The leave-one-family-out ablation found that dropping the dwell family (`hist_read_time_mean`,
`hist_scroll_mean`) improved lambdarank A2 — but it read that on the same 100,000 validation
impressions the choice would be made on (the *selection* sample, seed SEED + 1). This script asks
whether the gain generalises:

  * the three lambdarank models (A1 base; A2; A2 without dwell) are refitted on the identical
    100,000 train-week impressions, with the identical features and parameters;
  * they are scored on the selection sample again — which must reproduce the earlier numbers —
    and on the **holdout**: every other validation-week impression (144,647), never looked at.

Same week, disjoint impressions: a pure test of whether the dwell decision was fitted to the
particular sample it was read on. Features and frames come from `scripts/rerank_ebnerd_a2.py`
itself, so nothing is re-implemented.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from scripts.rerank_ebnerd_a2 import (BASE, COMPUTED_BUT_BANNED, PHASE1_FAMILIES, PHASE1_SAFE, Stage1,
                                      build, log, seeded_sample)
from src.rerank.common import SEED, fit_lambdarank, group_sizes, matrix, model_features, report
from src.rerank.ebnerd import SMALL, load_articles, load_behaviors

OUT = Path("data/processed")
FIT_N, SELECTION_N = 100_000, 100_000          # as in rerank_ebnerd_a2.py


def main() -> None:
    a2 = model_features(BASE + PHASE1_SAFE + COMPUTED_BUT_BANNED)
    assert a2 == BASE + PHASE1_SAFE
    no_dwell = [f for f in a2 if f not in PHASE1_FAMILIES["dwell"]]
    feature_sets = {"lr_base": BASE, "lr_a2": a2, "lr_a2_no_dwell": no_dwell}
    log(f"A2 without dwell ({len(no_dwell)}): {no_dwell}")

    articles = load_articles()
    train_beh = load_behaviors(SMALL / "train/behaviors.parquet")
    val_beh = load_behaviors(SMALL / "validation/behaviors.parquet")
    stage1 = Stage1(train_beh)

    log("train frame: the same 100,000 train-week impressions as rerank_ebnerd_a2.py")
    tr, _ = build(train_beh, SMALL / "train/history.parquet", seeded_sample(train_beh.height, FIT_N, SEED),
                  articles, stage1)
    y, groups = tr["label"].to_numpy(), group_sizes(tr)
    models = {}
    for name, cols in feature_sets.items():
        t0 = time.perf_counter()
        models[name] = fit_lambdarank(matrix(tr, cols), y, groups)
        log(f"  fitted {name} ({len(cols)} features, {time.perf_counter() - t0:.0f}s)")
    del tr, train_beh

    selection = seeded_sample(val_beh.height, SELECTION_N, SEED + 1)
    holdout = np.setdiff1d(np.arange(val_beh.height), selection)
    assert np.intersect1d(selection, holdout).size == 0 and len(selection) + len(holdout) == val_beh.height
    log(f"validation week: selection {len(selection):,} impressions (the ablation's), "
        f"holdout {len(holdout):,} (never scored before), disjoint")

    rows = {"lr_base": "lambdarank: A1 base", "lr_a2": "lambdarank: A2",
            "lr_a2_no_dwell": "lambdarank: A2 without dwell"}
    pairs = [("lr_base", "lr_a2"), ("lr_a2", "lr_a2_no_dwell"), ("lr_base", "lr_a2_no_dwell")]
    results = {"features_no_dwell": no_dwell, "selection_impressions": int(len(selection)),
               "holdout_impressions": int(len(holdout))}
    for part, imp_rows in (("selection", selection), ("holdout", holdout)):
        log(f"== {part} ({len(imp_rows):,} impressions)")
        va, _ = build(val_beh, SMALL / "validation/history.parquet", imp_rows, articles, stage1)
        for name, cols in feature_sets.items():
            va = va.with_columns(pl.Series(name, models[name].predict(matrix(va, cols))))
        results[part] = {}
        report(va, rows, pairs, log, results[part])
        del va

    (OUT / "holdout_dwell_ebnerd.json").write_text(json.dumps(results, indent=2))
    log(f"done -> {OUT / 'holdout_dwell_ebnerd.json'}")


if __name__ == "__main__":
    main()
