"""Behavioural features for the A2 reranker (SPEC.md §11).

Every feature here is a function of (user, candidate, t) that reads **only events strictly
before t** — the behaviour-window boundary A2 Q1.4 and Q9 require. An event at exactly t is part
of the request being scored, not history, so a real system could not know it.

`SERVING_OK` records, per feature, whether every input exists at request time. The Q9
"with and without serving-unavailable features" ablation reads this flag instead of relying on
someone remembering which features to drop.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import polars as pl

SERVING_OK = {"recency_weighted_profile": True}


def decay_weights(log: pl.DataFrame, user_id, t: datetime, half_life: timedelta, *,
                  untimed_ts: datetime | None = None) -> pl.DataFrame:
    """One user's click events strictly before `t`, each weighted by exponential time decay.

        w = 2 ** (-(t - ts) / h)

    A click one half-life old counts half as much as a click made just now, two half-lives a
    quarter, and so on. The weight depends on the click's **timestamp**, never on its row position,
    so the output is identical under any reordering of `log` (SPEC.md §11.1). A1's
    `semantic.user_vector.recency_pool` decays by list position instead — two clicks a minute apart
    and two a week apart get the same weights there.

    `untimed_ts` is the MIND fallback (SPEC.md §11.1, P1-D1). MIND's `history` carries no click
    times, so its clicks arrive here with a null `ts`. Passing the split's start time stamps them
    with it: MIND history is a frozen snapshot that predates every split, so the split start is an
    upper bound on the true click time. The fallback cannot leak — a stamp at or after `t` is
    still excluded by the strict boundary below. Without `untimed_ts`, a null `ts` is an error:
    a click with no time cannot be placed relative to t, and silently keeping or dropping it is
    either leakage or data loss.

    Returns `article_id, category, ts, weight`, sorted by `ts` then `article_id`.
    """
    if half_life <= timedelta(0):
        raise ValueError(f"half_life must be positive, got {half_life}")

    events = log.filter(pl.col("user_id") == user_id)
    if untimed_ts is not None:
        events = events.with_columns(pl.col("ts").fill_null(untimed_ts))
    elif events["ts"].null_count():
        raise ValueError(f"user {user_id!r} has {events['ts'].null_count()} click(s) with a null ts; "
                         "pass untimed_ts to stamp them explicitly")

    h_seconds = half_life.total_seconds()
    age_seconds = (pl.lit(t) - pl.col("ts")).dt.total_microseconds() / 1e6
    return (events
            .filter(pl.col("ts") < t)                       # strictly before t: the Q9 boundary
            .with_columns(weight=pl.lit(2.0).pow(-age_seconds / h_seconds))
            .select("article_id", "category", "ts", "weight")
            .sort("ts", "article_id"))


def recency_weighted_profile(log: pl.DataFrame, user_id, candidate_category, t: datetime,
                             half_life: timedelta, *, untimed_ts: datetime | None = None) -> float:
    """Share of the user's decayed click mass that falls on the candidate's category.

        P(c) = sum of weights of clicks in category c / sum of all weights

    Returns a value in [0, 1]: `0.0` when the user has eligible history but none in this category,
    and **NaN when the user has no eligible history at all**. A 0 there would conflate "never
    read this category" with "never read anything"; both GBDT candidates handle NaN natively.

    Normalising keeps only *relative* recency: shifting every click's age by the same amount
    leaves P unchanged. One consequence, measured on MIND: history clicks stamped with a single
    `untimed_ts` all get the same weight, so a history-only profile equals the undecayed category
    distribution for every half-life.
    """
    w = decay_weights(log, user_id, t, half_life, untimed_ts=untimed_ts)
    if w.height == 0:
        return math.nan
    in_category = w.filter(pl.col("category") == candidate_category)["weight"].sum()
    return in_category / w["weight"].sum()
