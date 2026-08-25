"""Feature store: user and article features computed **as of** a cutoff timestamp.

The anti-gaming requirement (A1 Q9) is that a feature used at impression time *t* may only
see events strictly before *t*. Rather than compute features and then check them, every
builder here takes an explicit `cutoff` and filters to `ts < cutoff` as its first operation,
so a leaking feature is not something you have to remember not to write.

Each frame carries the `cutoff` it was built at as a column. That is what makes the
invariant externally checkable — `tests/test_no_leakage.py` re-derives the assertion from
the stored value instead of trusting the builder's word for it.

MIND-specific caveat worth stating plainly: MIND's `history` field is a bare list of news
ids with **no per-event timestamps**, unlike EB-NeRD's `impression_time_fixed`. So a
row-level "history event precedes impression" assertion is not computable from MIND's files.
What *is* computable, and what we enforce, is that every feature is derived only from
impressions whose own timestamp precedes the cutoff.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl


def article_popularity(impressions: pl.LazyFrame, cutoff: datetime) -> pl.DataFrame:
    """Clicks per article, counting only impressions strictly before `cutoff`.

    Requires `ts`, `candidates`, `labels`. The filter comes first so no future click can
    reach the aggregation even transiently.
    """
    return (
        impressions
        .filter(pl.col("ts") < cutoff)
        .select("candidates", "labels")
        .explode(["candidates", "labels"])
        .filter(pl.col("labels") == 1)
        .group_by("candidates")
        .agg(pl.len().alias("click_count"))
        .rename({"candidates": "article_id"})
        .sort("click_count", descending=True)
        .with_columns(pl.lit(cutoff).alias("cutoff"))
        .collect()
    )


def user_activity(impressions: pl.LazyFrame, cutoff: datetime) -> pl.DataFrame:
    """Per-user counts and last-seen time, from impressions strictly before `cutoff`.

    `last_seen` is kept because recency is the feature news recommendation actually needs,
    and because it gives the leakage test something sharp to assert: it must be < cutoff.
    """
    return (
        impressions
        .filter(pl.col("ts") < cutoff)
        .group_by("user_id")
        .agg(
            pl.len().alias("n_impressions"),
            pl.col("labels").list.sum().sum().alias("n_clicks"),
            pl.col("ts").max().alias("last_seen"),
            pl.col("history_ids").last().list.len().alias("history_len"),
        )
        .with_columns(pl.lit(cutoff).alias("cutoff"))
        .collect()
    )


def max_source_timestamp(impressions: pl.LazyFrame, cutoff: datetime) -> datetime | None:
    """The latest event any as-of-`cutoff` feature could have seen.

    Exists so the leakage assertion is computed from the same filter the builders use,
    rather than re-implemented in the test where it could drift.
    """
    row = (
        impressions.filter(pl.col("ts") < cutoff)
        .select(pl.col("ts").max().alias("hi"))
        .collect()
    )
    return row["hi"][0]
