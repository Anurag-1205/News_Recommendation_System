"""MIND candidate frames for the reranker (SPEC.md §12).

Same shape as src/rerank/ebnerd.py — one row per candidate, sorted by (`imp_row`, `cand_position`)
— with MIND's differences:

* labels arrive as a list aligned with the candidates, so they are attached **by position**;
* history carries no click times, so its clicks are stamped with the split start (C-008);
* there is no publish time, so freshness is first-seen (§11.8): the earliest impression that
  listed the article, with history articles stamped at the dataset start;
* MIND has no session, dwell or read-time columns, so those features do not exist here.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from src.features.behavioural import freshness_batch, slate_features
from src.pipeline.mind import (read_news, scan_behaviors, split_dir, with_history, with_labelled_candidates,
                               with_parsed_time, with_unlabelled_candidates)
from src.rerank.common import category_profile_features

ROOT = Path("data/interim/mind")
# First midnight of each split, measured to precede its first impression (RESULTS.md Q1, C-008).
SPLIT_START = {"MINDsmall_train": datetime(2019, 11, 9), "MINDsmall_dev": datetime(2019, 11, 15),
               "MINDlarge_test": datetime(2019, 11, 16)}
DATASET_START = SPLIT_START["MINDsmall_train"]


def load_behaviors(name: str, *, labelled: bool = True, limit: int | None = None) -> pl.DataFrame:
    lf = with_history(with_parsed_time(scan_behaviors(split_dir(ROOT, name))))
    lf = with_labelled_candidates(lf) if labelled else with_unlabelled_candidates(lf)
    cols = ["impression_id", "user_id", pl.col("ts").alias("t"), "history_ids", "candidates"]
    lf = lf.select(cols + (["labels"] if labelled else []))
    return (lf.head(limit) if limit else lf).collect().with_row_index("imp_row")


def load_categories(names: list[str]) -> pl.DataFrame:
    """article_id -> category, from the news files of the given splits."""
    return (pl.concat([read_news(split_dir(ROOT, n)).select(article_id="news_id", category="category") for n in names])
            .unique("article_id", keep="first"))


def candidate_frame(beh: pl.DataFrame, categories: pl.DataFrame) -> pl.DataFrame:
    long = (slate_features(beh.select(impression_id=pl.col("imp_row"), candidates=pl.col("candidates")))
            .rename({"impression_id": "imp_row"})
            .join(beh.select("imp_row", "impression_id", "user_id", "t"), on="imp_row", how="left"))
    if "labels" in beh.columns:
        labels = (beh.select("imp_row", "labels", cand_position=pl.int_ranges(1, pl.col("labels").list.len() + 1))
                  .explode("labels", "cand_position", empty_as_null=False)
                  .select("imp_row", pl.col("cand_position").cast(long["cand_position"].dtype),
                          label=pl.col("labels").cast(pl.Int8)))
        long = long.join(labels, on=["imp_row", "cand_position"], how="left")
    long = long.join(categories.rename({"category": "candidate_category"}), on="article_id", how="left")
    return long.sort("imp_row", "cand_position")


def history_log(beh: pl.DataFrame, categories: pl.DataFrame) -> pl.DataFrame:
    """One row per (user, history click) with the article's category and a null `ts`. MIND history
    is a frozen per-user snapshot (C-008), so each user's first row carries all of it."""
    return (beh.group_by("user_id").agg(pl.col("history_ids").first())
            .explode("history_ids", empty_as_null=False).drop_nulls("history_ids")
            .rename({"history_ids": "article_id"})
            .join(categories, on="article_id", how="left")
            .with_columns(ts=pl.lit(None, pl.Datetime("us"))))


def first_sightings(behs: list[pl.DataFrame]) -> pl.DataFrame:
    """`first_known` for freshness: each article's earliest impression time across the given
    splits, plus every history article with a null ts (stamped with DATASET_START by the caller)."""
    shown = pl.concat([b.select("t", article_id=pl.col("candidates")).explode("article_id", empty_as_null=False)
                       .group_by("article_id").agg(ts=pl.col("t").min()) for b in behs])
    in_history = pl.concat([b.select(article_id=pl.col("history_ids")).explode("article_id", empty_as_null=False)
                            .drop_nulls().unique() for b in behs]).unique()
    return pl.concat([shown.select("article_id", pl.col("ts").cast(pl.Datetime("us"))),
                      in_history.with_columns(ts=pl.lit(None, pl.Datetime("us")))])


def add_phase1_features(long: pl.DataFrame, beh_split: pl.DataFrame, split_name: str,
                        categories: pl.DataFrame, sightings: pl.DataFrame, half_life: timedelta) -> pl.DataFrame:
    """`recency_weighted_profile`, `category_match` and `freshness_hours` for MIND candidates.
    On MIND the profile does not depend on h (C-008); `half_life` is passed only for uniformity."""
    prof = category_profile_features(long, history_log(beh_split, categories), half_life,
                                     untimed_ts=SPLIT_START[split_name])
    fresh = freshness_batch(sightings, long.select("imp_row", "cand_position", "article_id", "t"),
                            untimed_ts=DATASET_START)
    return (long.join(prof, on=["imp_row", "cand_position"], how="left")
            .join(fresh.drop("article_id", "t"), on=["imp_row", "cand_position"], how="left")
            .sort("imp_row", "cand_position"))
