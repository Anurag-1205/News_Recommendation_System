"""EB-NeRD candidate frames for the reranker (SPEC.md §12).

An impression becomes one row per candidate — label, keys, and every Phase 1 feature — so a GBDT
can train on it and the evaluation harness can regroup it by `imp_row`. Rows are always sorted by
(`imp_row`, `cand_position`), i.e. list order, so per-impression arrays computed elsewhere (the
stage-1 scores) line up with them by position.

`imp_row` is the row number within the *full* split. It is the key, not `impression_id`: the
EB-NeRD test file repeats impression_id 0 for 200,000 rows.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import polars as pl

from src.features.behavioural import (current_page_features, dwell_features, freshness_batch,
                                      session_features, slate_features)
from src.rerank.common import category_profile_features

SMALL = Path("data/interim/ebnerd/ebnerd_small")
TEST = Path("data/interim/ebnerd/ebnerd_testset/ebnerd_testset")
ARTICLES = TEST / "articles.parquet"           # 125,541 articles: covers small, large and test


def load_articles() -> pl.DataFrame:
    return pl.read_parquet(ARTICLES, columns=["article_id", "category", "published_time"])


def load_behaviors(path: Path, limit: int | None = None) -> pl.DataFrame:
    """Every impression of a split (or its first `limit`), in file order, with `imp_row` and
    unified column names. `clicked` exists only when the file has labels — the test file has none."""
    lf = pl.scan_parquet(path)
    cols = ["impression_id", "user_id", "session_id", pl.col("impression_time").alias("t"),
            pl.col("article_ids_inview").alias("candidates"), "read_time", "scroll_percentage"]
    if "article_ids_clicked" in lf.collect_schema().names():
        cols.append(pl.col("article_ids_clicked").alias("clicked"))
    lf = lf.select(cols)
    return (lf.head(limit) if limit else lf).collect().with_row_index("imp_row")


def load_history(path: Path, articles: pl.DataFrame, users: pl.Series | None = None) -> pl.DataFrame:
    """`history.parquet`, exploded to one row per past click, with the clicked article's category.
    `users` restricts it to those users *before* exploding (the test history file is 1.16 GB)."""
    lf = pl.scan_parquet(path)
    if users is not None:
        lf = lf.filter(pl.col("user_id").is_in(users.unique().implode()))
    return (lf
            .select("user_id", article_id=pl.col("article_id_fixed"), ts=pl.col("impression_time_fixed"),
                    read_time=pl.col("read_time_fixed"), scroll_percentage=pl.col("scroll_percentage_fixed"))
            .explode("article_id", "ts", "read_time", "scroll_percentage", empty_as_null=False)
            .join(articles.lazy().select("article_id", "category"), on="article_id", how="left")
            .collect())


def candidate_frame(beh: pl.DataFrame, articles: pl.DataFrame) -> pl.DataFrame:
    """One row per (impression, candidate): keys, `cand_position`, `n_candidates`, the candidate's
    category, and `label` when the split is labelled (clicked ids -> 0/1 by membership)."""
    long = (slate_features(beh.select(impression_id=pl.col("imp_row"), candidates=pl.col("candidates")))
            .rename({"impression_id": "imp_row"})
            .join(beh.select("imp_row", "impression_id", "user_id", "session_id", "t"), on="imp_row", how="left"))
    if "clicked" in beh.columns:
        clicks = (beh.select("imp_row", article_id=pl.col("clicked")).explode("article_id", empty_as_null=False)
                  .drop_nulls().unique().with_columns(label=pl.lit(1, pl.Int8)))
        long = (long.join(clicks, on=["imp_row", "article_id"], how="left")
                .with_columns(pl.col("label").fill_null(0)))
    long = long.join(articles.select("article_id", candidate_category=pl.col("category")),
                     on="article_id", how="left")
    return long.sort("imp_row", "cand_position")


def add_phase1_features(long: pl.DataFrame, beh_full: pl.DataFrame, history: pl.DataFrame,
                        articles: pl.DataFrame, half_life: timedelta) -> pl.DataFrame:
    """Every Phase 1 feature, joined onto the candidate frame by key. Unsafe and test-absent
    features are computed too, so the frame is complete; `model_features` decides what a model sees.

    Session features use the **full** split, not the sample: `session_pos` counts every earlier
    impression of the session, sampled or not.
    """
    sess_cols = ["imp_row", "user_id", "session_id", "t"] + (["clicked"] if "clicked" in beh_full.columns else [])
    sess = session_features(beh_full.select(sess_cols)).drop("user_id", "session_id", "t", "clicked", strict=False)
    cur = (current_page_features(beh_full.select(impression_id=pl.col("imp_row"), read_time=pl.col("read_time"),
                                                 scroll_percentage=pl.col("scroll_percentage")))
           .rename({"impression_id": "imp_row"}))
    anchors = long.select("imp_row", "user_id", "t").unique(maintain_order=True)
    dwell = dwell_features(history.select("user_id", "ts", "read_time", "scroll_percentage"), anchors)
    fresh = freshness_batch(articles.select("article_id", ts=pl.col("published_time")),
                            long.select("imp_row", "cand_position", "article_id", "t"))
    prof = category_profile_features(long, history, half_life)
    return (long.join(sess, on="imp_row", how="left")
            .join(cur, on="imp_row", how="left")
            .join(dwell.drop("user_id", "t"), on="imp_row", how="left")
            .join(fresh.drop("article_id", "t"), on=["imp_row", "cand_position"], how="left")
            .join(prof, on=["imp_row", "cand_position"], how="left")
            .sort("imp_row", "cand_position"))
