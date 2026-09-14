#!/usr/bin/env python3
"""Write the locked reranker's scores for a split, in the SPEC §13.3 scores-file contract (P5).

    PYTHONPATH=. .venv/bin/python -u scripts/score_final.py --dataset ebnerd
    PYTHONPATH=. .venv/bin/python -u scripts/score_final.py --dataset mind

`src/rerank/config.FINAL` says what to fit (C-018): EB-NeRD lambdarank on 11 features, MIND
pointwise on 7. The fit uses the same seeded training sample as the measured runs
(`scripts/rerank_*_a2.py`), and the model then scores **every impression of the evaluation split**,
not a sample, because the contract requires 1:1 alignment with the other systems' files.

The frames come from the reranker scripts themselves, so the features are the ones that were
measured; nothing is re-implemented here.
"""
from __future__ import annotations

import argparse, json, subprocess, time
from pathlib import Path

import numpy as np
import polars as pl

from src.eval.paired import write_scores
from src.rerank.common import SEED, fit_final, matrix, group_sizes, predict_scores
from src.rerank.config import FINAL

OUT = Path("data/scores")


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_ebnerd():
    from scripts.rerank_ebnerd_a2 import Stage1, build, seeded_sample
    from src.rerank.ebnerd import SMALL, load_articles, load_behaviors
    articles = load_articles()
    train_beh = load_behaviors(SMALL / "train/behaviors.parquet")
    val_beh = load_behaviors(SMALL / "validation/behaviors.parquet")
    stage1 = Stage1(train_beh)
    log("train frame (fit): the same 100,000 train-week impressions as rerank_ebnerd_a2.py")
    tr, _ = build(train_beh, SMALL / "train/history.parquet",
                  seeded_sample(train_beh.height, 100_000, SEED), articles, stage1)
    log(f"evaluation frame: ALL {val_beh.height:,} validation impressions")
    va, _ = build(val_beh, SMALL / "validation/history.parquet",
                  np.arange(val_beh.height), articles, stage1)
    return tr, va, "ebnerd_small/validation"


def build_mind():
    from scripts.rerank_mind_a2 import DEV, Stage1, TRAIN, build
    from src.rerank.mind import first_sightings, load_behaviors, load_categories
    train, dev = load_behaviors(TRAIN), load_behaviors(DEV)
    categories = load_categories([TRAIN, DEV])
    sightings = first_sightings([train, dev])
    stage1 = Stage1([TRAIN, DEV], train)
    fit_sample = np.sort(np.random.default_rng(SEED).choice(train.height, size=min(80_000, train.height),
                                                            replace=False))
    log("train frame (fit): the same 80,000 MINDsmall_train impressions as rerank_mind_a2.py")
    tr = build(train, TRAIN, fit_sample, categories, sightings, stage1)
    log(f"evaluation frame: ALL {dev.height:,} dev impressions")
    va = build(dev, DEV, np.arange(dev.height), categories, sightings, stage1)
    return tr, va, "MINDsmall_dev"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("ebnerd", "mind"), required=True)
    ap.add_argument("--system", default="reranker_final")
    args = ap.parse_args()
    cfg = FINAL[args.dataset]
    log(f"config.FINAL[{args.dataset}]: {cfg['objective']}, {len(cfg['features'])} features")

    t0 = time.perf_counter()
    tr, va, split = build_ebnerd() if args.dataset == "ebnerd" else build_mind()
    model = fit_final(cfg["objective"], matrix(tr, cfg["features"]), tr["label"].to_numpy(), group_sizes(tr))
    log(f"  fitted on {tr['imp_row'].n_unique():,} impressions ({time.perf_counter() - t0:.0f}s)")
    scores = predict_scores(model, matrix(va, cfg["features"]))

    frame = (va.select("imp_row", "impression_id", "article_id", "cand_position")
             .with_columns(score=pl.Series(scores, dtype=pl.Float64))
             .sort("imp_row", "cand_position"))
    manifest = {
        "dataset": args.dataset, "split": split, "system": args.system, "framing": "in-impression",
        "model": cfg["objective"], "features": cfg["features"], "config": "src/rerank/config.FINAL (C-018)",
        "n_rows": frame.height, "n_impressions": int(frame["imp_row"].n_unique()),
        "fit_impressions": int(tr["imp_row"].n_unique()), "seed": SEED,
        "repo_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "command": f"PYTHONPATH=. .venv/bin/python -u scripts/score_final.py --dataset {args.dataset}",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = write_scores(frame, manifest, OUT / args.dataset / split.split("/")[-1] / f"{args.system}.parquet")
    log(f"wrote {frame.height:,} rows over {manifest['n_impressions']:,} impressions -> {path}")


if __name__ == "__main__":
    main()
