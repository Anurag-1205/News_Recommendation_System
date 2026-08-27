"""Popularity baseline: rank candidates by how often they were clicked in training.

This is submission #1. It is deliberately the simplest thing that is not
random: it uses no user history, no article text, and no personalisation, so it sets the
floor that BM25 and the semantic retriever must clear in submission #2. A baseline that
already encodes half the eventual model cannot do that job.

Known weakness, measured rather than assumed (see RESULTS.md): MIND's train split is
11 Nov 2019 and the test split is 19-22 Nov 2019. News turns over fast, so a large share
of test candidates were never clicked in training and score 0. `coverage()` reports
exactly how large, and that number is the argument for content-based retrieval.
"""

from __future__ import annotations

import polars as pl


def click_counts(behaviors: pl.LazyFrame) -> pl.DataFrame:
    """Count clicks per news_id from labelled impressions.

    Expects `candidates` and `labels` columns (see `mind.with_labelled_candidates`).
    Explode-then-filter is the streaming-friendly shape: polars never materialises the
    full cross product, and the result is one small frame of ~50K rows.
    """
    return (
        behaviors
        .select("candidates", "labels")
        .explode(["candidates", "labels"])
        .filter(pl.col("labels") == 1)
        .group_by("candidates")
        .agg(pl.len().alias("click_count"))
        .rename({"candidates": "news_id"})
        .sort("click_count", descending=True)
        .collect()
    )


def score_map(counts: pl.DataFrame) -> dict[str, int]:
    """Materialise the lookup as a plain dict.

    ~50K entries, so a dict is both small and the fastest per-candidate lookup available.
    A join would be tidier but forces the 2.37M-row test frame through an explode, which
    is precisely what this machine's ~2 GB of free RAM cannot afford.
    """
    return dict(zip(counts["news_id"].to_list(), counts["click_count"].to_list()))


def coverage(candidate_ids: set[str], scores: dict[str, int]) -> dict:
    """What fraction of the candidates we are asked to rank did we ever observe?

    Reported rather than assumed — it is the honest measure of how much signal this
    baseline actually has on the test distribution.
    """
    known = len(candidate_ids & scores.keys())
    return {
        "candidates": len(candidate_ids),
        "known": known,
        "unknown": len(candidate_ids) - known,
        "coverage": known / len(candidate_ids) if candidate_ids else 0.0,
    }
