#!/usr/bin/env python3
"""Build the MIND Codabench submission from the popularity baseline.

    .venv/bin/python scripts/make_submission_mind.py

Three stages, in this order on purpose:

  1. Fit click counts on MINDsmall_train.
  2. **Evaluate on MINDsmall_dev, which has labels.** Measuring before submitting is the
     point — an unmeasured submission teaches nothing and cannot be improved on
     deliberately in submission #2.
  3. Score MINDlarge_test in batches, validate the file offline, and zip it.

Memory: the test split is 2.37M impressions and this machine has ~2 GB free (see the
environment baseline in RESULTS.md),
so the test pass streams in slices and appends to the output file rather than building the
predictions in memory. The reference notebooks read the whole frame; that is the one place
we deliberately diverge.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import polars as pl

from src.baselines.popularity import click_counts, coverage, score_map
from src.eval.metrics import evaluate_impressions
from src.eval.submission import format_line, ranks_from_scores, validate_file, zip_submission
from src.pipeline.mind import (
    scan_behaviors,
    split_dir,
    with_labelled_candidates,
    with_unlabelled_candidates,
)

MIND_ROOT = Path("data/interim/mind")
OUT_DIR = Path("data/processed")
BATCH = 200_000


def fit(train_dir: Path) -> dict[str, int]:
    lf = with_labelled_candidates(scan_behaviors(train_dir))
    counts = click_counts(lf)
    print(f"  fitted on {counts.height:,} distinct clicked articles")
    print(f"  most-clicked: {counts.row(0)}")
    return score_map(counts)


def evaluate(dev_dir: Path, scores: dict[str, int]) -> dict:
    """Offline metrics on the labelled dev split."""
    df = with_labelled_candidates(scan_behaviors(dev_dir)).select("candidates", "labels").collect()

    def rows():
        for cands, labels in zip(df["candidates"], df["labels"]):
            yield labels.to_list(), [scores.get(c, 0) for c in cands]

    metrics = evaluate_impressions(rows())

    all_cands = {c for cands in df["candidates"] for c in cands}
    metrics["candidate_coverage"] = coverage(all_cands, scores)["coverage"]
    return metrics


def predict(test_dir: Path, scores: dict[str, int], out_txt: Path) -> dict:
    """Stream the test split, writing one line per impression."""
    lf = with_unlabelled_candidates(scan_behaviors(test_dir))
    total = lf.select(pl.len()).collect().item()
    print(f"  {total:,} test impressions, batch={BATCH:,}")

    lengths: dict[int, int] = {}
    written = 0
    t0 = time.perf_counter()
    with out_txt.open("w") as fh:
        for start in range(0, total, BATCH):
            batch = lf.slice(start, BATCH).select("impression_id", "candidates").collect()
            for imp_id, cands in zip(batch["impression_id"], batch["candidates"]):
                cands = cands.to_list()
                ranks = ranks_from_scores([scores.get(c, 0) for c in cands])
                fh.write(format_line(imp_id, ranks) + "\n")
                lengths[imp_id] = len(cands)
                written += 1
            print(f"    {written:,}/{total:,}  ({time.perf_counter()-t0:.0f}s)", flush=True)
            del batch
    return {"written": written, "lengths": lengths, "seconds": time.perf_counter() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-eval", action="store_true", help="skip the dev-set measurement")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_dir = split_dir(MIND_ROOT, "MINDsmall_train")
    dev_dir = split_dir(MIND_ROOT, "MINDsmall_dev")
    test_dir = split_dir(MIND_ROOT, "MINDlarge_test")

    print("[1/3] fitting popularity on MINDsmall_train")
    scores = fit(train_dir)

    metrics = {}
    if not args.skip_eval:
        print("[2/3] evaluating on MINDsmall_dev (labelled)")
        metrics = evaluate(dev_dir, scores)
        for k, v in metrics.items():
            print(f"  {k:20s} {v:.4f}" if isinstance(v, float) else f"  {k:20s} {v}")

    print("[3/3] scoring MINDlarge_test")
    out_txt = OUT_DIR / "prediction.txt"
    info = predict(test_dir, scores, out_txt)

    print("  validating offline before upload")
    stats = validate_file(out_txt, expected_lengths=info["lengths"])
    print(f"  {stats}")

    out_zip = zip_submission(out_txt, OUT_DIR / "mind_prediction.zip")
    print(f"  wrote {out_zip} ({out_zip.stat().st_size/1e6:.1f} MB)")

    (OUT_DIR / "mind_popularity_metrics.json").write_text(
        json.dumps({"dev_metrics": metrics, "test": {k: v for k, v in info.items() if k != "lengths"},
                    "validation": stats}, indent=2)
    )


if __name__ == "__main__":
    main()
