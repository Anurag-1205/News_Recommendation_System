#!/usr/bin/env python3
"""Q9: `config.FINAL` with and without the features a live system could not have (SPEC.md §11.4).

    PYTHONPATH=. .venv/bin/python -u scripts/ablation_q9.py --dataset ebnerd
    PYTHONPATH=. .venv/bin/python -u scripts/ablation_q9.py --dataset mind

The shipped model is already the honest one: `config.FINAL` contains no member of
`UNSAFE_FEATURES`, and `model_features(..., for_submission=True)` also drops
`ABSENT_FROM_TEST_FILE`. So this ablation does not ask "how much do we lose by being honest"
after the fact -- it measures what a dishonest model *would* have scored, which is the number Q9
wants disclosed. Three rows per dataset:

  serving_safe      config.FINAL exactly. This is what ships and what the leaderboard sees.
  plus_unsafe       + every UNSAFE_FEATURES column the dataset actually builds. Cannot be served:
                    `session_len` counts the session's impressions *after* t, and `cur_read_time` /
                    `cur_scroll_percentage` describe a page view that outlasts the moment of serving.
  plus_absent       + `n_prior_clicks_in_session`, reported as its own row per the Q9 brief. It is
                    serving-*safe* in a real system (a live service knows its own session's clicks),
                    but it is built from `article_ids_clicked`, which the Codabench test file does
                    not ship -- so a model trained on it would meet a missing column at test time.

Everything else is held fixed: the same seeded fit sample, the same objective, the same evaluation
split, the same seed. Only the feature set moves, so the delta is attributable to it. Each pair is
compared with the paired bootstrap over per-impression metrics, and a row only "beats" if the CI
excludes 0 (CLAUDE.md §4).
"""
from __future__ import annotations

import argparse, json, subprocess, time
from pathlib import Path

import numpy as np
import polars as pl

from src.eval.bootstrap import CI
from src.rerank.common import (SEED, evaluate, fit_final, group_sizes, matrix, paired_delta,
                               per_impression, predict_scores)
from src.rerank.config import FINAL
from src.features.behavioural import ABSENT_FROM_TEST_FILE, UNSAFE_FEATURES

OUT = Path("data/processed/q9")
METRICS = ("auc", "mrr", "ndcg@5", "ndcg@10")


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build(dataset: str):
    """The same frames the measured runs and scripts/score_final.py use -- imported, not rebuilt."""
    from scripts.score_final import build_ebnerd, build_mind
    return build_ebnerd() if dataset == "ebnerd" else build_mind()


def variants(dataset: str, columns: set[str]) -> dict[str, list[str]]:
    """The three feature sets, each restricted to columns the dataset actually builds."""
    base = list(FINAL[dataset]["features"])
    unsafe = [c for c in UNSAFE_FEATURES if c in columns and c not in base]
    absent = [c for c in sorted(ABSENT_FROM_TEST_FILE) if c in columns and c not in base]
    out = {"serving_safe": base}
    if unsafe:
        out["plus_unsafe"] = base + unsafe
    if absent:
        out["plus_absent"] = base + absent
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("ebnerd", "mind"), required=True)
    ap.add_argument("--iterations", type=int, default=1000)
    args = ap.parse_args()

    objective = FINAL[args.dataset]["objective"]
    t0 = time.perf_counter()
    tr, va, split = build(args.dataset)
    log(f"frames built in {time.perf_counter() - t0:.0f}s: fit {tr['imp_row'].n_unique():,} imp, "
        f"eval {va['imp_row'].n_unique():,} imp ({va.height:,} rows)")

    sets = variants(args.dataset, set(va.columns))
    for name, feats in sets.items():
        log(f"  {name}: {len(feats)} features" + (f" (+{sorted(set(feats) - set(sets['serving_safe']))})"
                                                  if name != "serving_safe" else ""))
    if len(sets) == 1:
        log(f"NOTE: {args.dataset} builds no serving-unsafe column, so there is nothing to add. "
            "The 'without' row is the only row that exists here.")

    y_tr, groups, y_va = tr["label"].to_numpy(), group_sizes(tr), va["label"].to_numpy()
    per_imp, rows = {}, {}
    for name, feats in sets.items():
        t1 = time.perf_counter()
        model = fit_final(objective, matrix(tr, feats), y_tr, groups)
        scored = va.select("imp_row", "label").with_columns(
            score=pl.Series(predict_scores(model, matrix(va, feats)), dtype=pl.Float64))
        per_imp[name] = per_impression(scored, "score")
        cis = evaluate(per_imp[name], iterations=args.iterations, seed=SEED)
        rows[name] = {m: {"mean": cis[m].mean, "lo": cis[m].lo, "hi": cis[m].hi} for m in METRICS}
        log(f"  {name}: " + "  ".join(f"{m} {cis[m].mean:.4f} [{cis[m].lo:.4f}, {cis[m].hi:.4f}]"
                                      for m in METRICS) + f"  ({time.perf_counter() - t1:.0f}s)")

    deltas = {}
    for name in sets:
        if name == "serving_safe":
            continue
        # paired_delta(a, b) is mean(b - a): baseline first, so the delta reads variant - serving_safe.
        d = {m: paired_delta(per_imp["serving_safe"][m], per_imp[name][m],
                             iterations=args.iterations, seed=SEED) for m in METRICS}
        deltas[name] = {m: {"mean": c.mean, "lo": c.lo, "hi": c.hi, "excludes_zero": c.lo > 0 or c.hi < 0}
                        for m, c in d.items()}
        log(f"  delta {name} - serving_safe: " +
            "  ".join(f"{m} {c.mean:+.4f} [{c.lo:+.4f}, {c.hi:+.4f}]" for m, c in d.items()))

    OUT.mkdir(parents=True, exist_ok=True)
    record = {
        "dataset": args.dataset, "split": split, "objective": objective,
        "features": sets, "unsafe_registry": UNSAFE_FEATURES,
        "absent_from_test_file": sorted(ABSENT_FROM_TEST_FILE),
        "n_eval_impressions": int(va["imp_row"].n_unique()), "n_eval_rows": va.height,
        "n_fit_impressions": int(tr["imp_row"].n_unique()), "seed": SEED,
        "iterations": args.iterations, "metrics": rows, "deltas_vs_serving_safe": deltas,
        "repo_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                      text=True).stdout.strip(),
        "command": f"PYTHONPATH=. .venv/bin/python -u scripts/ablation_q9.py --dataset {args.dataset}",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = OUT / f"{args.dataset}.json"
    path.write_text(json.dumps(record, indent=2))
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
