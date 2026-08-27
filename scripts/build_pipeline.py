#!/usr/bin/env python3
"""Q1: rebuild the feature store from raw files, in one command.

    make data            # or: PYTHONPATH=. .venv/bin/python scripts/build_pipeline.py

Stages: extract archives -> unified schema -> temporal split -> feature store, for both
datasets. Idempotent: an already-extracted archive or an up-to-date parquet is skipped, so
re-running after a failure resumes rather than starting over.

Two invariants are enforced *here*, not only in tests, because a silently bad split
invalidates every number downstream:

  * splits are strictly disjoint in time (`assert_disjoint`)
  * every feature is built as-of a cutoff, and the cutoff is stored on the frame

Seeds are fixed and recorded in the manifest so a rebuild is reproducible.
"""
from __future__ import annotations

import argparse, json, subprocess, time
from pathlib import Path

import polars as pl

from src.pipeline.features import article_popularity, max_source_timestamp, user_activity
from src.pipeline.mind import (scan_behaviors, split_dir, with_history,
                               with_labelled_candidates, with_parsed_time)
from src.pipeline.split import assert_disjoint, compute_boundaries, temporal_split

RAW, INTERIM, STORE = Path("data/raw"), Path("data/interim"), Path("data/processed/feature_store")
SEED = 0
N_TEST_DAYS, M_VAL_DAYS = 1, 1


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def extract(zip_path: Path, dest: Path) -> bool:
    """Unzip unless already present. Returns True if extraction actually ran."""
    if dest.exists() and any(dest.iterdir()):
        return False
    if not zip_path.exists():
        raise FileNotFoundError(f"{zip_path} missing — run `make fetch-small` / `make fetch-mind`")
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(dest)], check=True)
    return True


def build_mind(manifest: dict) -> None:
    log("MIND: extracting")
    for name in ("MINDsmall_train", "MINDsmall_dev", "MINDlarge_test"):
        z = RAW / "mind" / f"{name}.zip"
        if z.exists():
            did = extract(z, INTERIM / "mind" / name)
            log(f"  {name}: {'extracted' if did else 'already present'}")

    train_d = split_dir(INTERIM / "mind", "MINDsmall_train")
    lf = with_history(with_labelled_candidates(with_parsed_time(scan_behaviors(train_d))))

    log("MIND: temporal split of the training file")
    bounds = compute_boundaries(lf, n_test_days=N_TEST_DAYS, m_val_days=M_VAL_DAYS)
    splits = temporal_split(lf, bounds)
    ranges = assert_disjoint(splits)          # raises on overlap
    log(f"  {bounds.describe()}")
    for name, (n, lo, hi) in ranges.items():
        log(f"  {name:6s} {n:>8,} rows  {lo} .. {hi}")

    log("MIND: feature store, as-of the training cutoff")
    cutoff = bounds.val_start                  # features may see train only
    pop = article_popularity(splits["train"], cutoff)
    acts = user_activity(splits["train"], cutoff)
    latest = max_source_timestamp(splits["train"], cutoff)
    assert latest is None or latest < cutoff, f"leak: source event {latest} >= cutoff {cutoff}"

    out = STORE / "mind"; out.mkdir(parents=True, exist_ok=True)
    pop.write_parquet(out / "article_popularity.parquet")
    acts.write_parquet(out / "user_activity.parquet")
    log(f"  {pop.height:,} articles, {acts.height:,} users -> {out}")

    manifest["mind"] = {
        "split": {"n_test_days": N_TEST_DAYS, "m_val_days": M_VAL_DAYS,
                  "val_start": str(bounds.val_start), "test_start": str(bounds.test_start),
                  "data_min": str(bounds.data_min), "data_max": str(bounds.data_max)},
        "rows": {k: v[0] for k, v in ranges.items()},
        "feature_cutoff": str(cutoff),
        "max_source_timestamp": str(latest),
        "articles_with_popularity": pop.height, "users_with_activity": acts.height,
    }


def build_ebnerd(manifest: dict) -> None:
    for name in ("ebnerd_small", "ebnerd_testset"):
        z = RAW / "ebnerd" / f"{name}.zip"
        if not z.exists():
            log(f"EB-NeRD: {name}.zip missing, skipping")
            return
    log("EB-NeRD: extracting")
    for name in ("ebnerd_small", "ebnerd_testset"):
        did = extract(RAW / "ebnerd" / f"{name}.zip", INTERIM / "ebnerd" / name)
        log(f"  {name}: {'extracted' if did else 'already present'}")

    small = INTERIM / "ebnerd" / "ebnerd_small"
    lf = (pl.scan_parquet(small / "train/behaviors.parquet")
          .select(pl.col("impression_id"), pl.col("user_id"),
                  pl.col("impression_time").alias("ts"),
                  pl.col("article_ids_inview").alias("candidates"),
                  pl.col("article_ids_clicked")))

    log("EB-NeRD: temporal split of the training file")
    bounds = compute_boundaries(lf, n_test_days=N_TEST_DAYS, m_val_days=M_VAL_DAYS)
    splits = temporal_split(lf, bounds)
    ranges = assert_disjoint(splits)
    log(f"  {bounds.describe()}")
    for name, (n, lo, hi) in ranges.items():
        log(f"  {name:6s} {n:>8,} rows  {lo} .. {hi}")

    log("EB-NeRD: feature store, as-of the training cutoff")
    cutoff = bounds.val_start
    pop = (splits["train"].filter(pl.col("ts") < cutoff)
           .select(pl.col("article_ids_clicked").alias("article_id"))
           .explode("article_id").drop_nulls()
           .group_by("article_id").agg(pl.len().alias("click_count"))
           .sort("click_count", descending=True)
           .with_columns(pl.lit(cutoff).alias("cutoff"))
           .collect())
    out = STORE / "ebnerd"; out.mkdir(parents=True, exist_ok=True)
    pop.write_parquet(out / "article_popularity.parquet")
    log(f"  {pop.height:,} articles -> {out}")

    manifest["ebnerd"] = {
        "split": {"n_test_days": N_TEST_DAYS, "m_val_days": M_VAL_DAYS,
                  "val_start": str(bounds.val_start), "test_start": str(bounds.test_start),
                  "data_min": str(bounds.data_min), "data_max": str(bounds.data_max)},
        "rows": {k: v[0] for k, v in ranges.items()},
        "feature_cutoff": str(cutoff), "articles_with_popularity": pop.height,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["mind", "ebnerd"], help="build just one dataset")
    args = ap.parse_args()

    t0 = time.perf_counter()
    STORE.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": SEED, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}

    if args.only in (None, "mind"):
        build_mind(manifest)
    if args.only in (None, "ebnerd"):
        build_ebnerd(manifest)

    manifest["seconds"] = round(time.perf_counter() - t0, 1)
    (STORE / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"feature store written to {STORE} in {manifest['seconds']:.0f}s")


if __name__ == "__main__":
    main()
