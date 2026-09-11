"""Behavioural features for the A2 reranker (SPEC.md §11).

Every feature here is a function of (user, candidate, t) that reads **only events strictly
before t** — the behaviour-window boundary A2 Q1.4 and Q9 require. An event at exactly t is part
of the request being scored, not history, so a real system could not know it. The exceptions are
listed in `UNSAFE_FEATURES` and exist only as Q9 ablation rows.

Two registries classify every feature, because they answer different questions (SPEC.md §11.4):

* `SERVING_OK` — could a live system know this value when the request arrives? The Q9
  "with and without serving-unavailable features" ablation reads this flag (`drop_unsafe`)
  instead of relying on someone remembering which columns to drop.
* `ABSENT_FROM_TEST_FILE` — serving-safe in production, but built from columns the Codabench
  test file does not ship. The submission model must not use these, or it trains on a signal that
  is missing at test time (A1 submission 3, SPEC.md §7).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import polars as pl

SERVING_OK = {
    "recency_weighted_profile": True,
    "category_match": True,
    "cand_position": True,
    "n_candidates": True,
    "session_pos": True,
    "n_prior_clicks_in_session": True,
    "hist_read_time_mean": True,
    "hist_scroll_mean": True,
    "freshness_hours": True,
    "session_len": False,               # counts the session's impressions *after* t
    "cur_read_time": False,             # the current page view outlasts the moment of serving
    "cur_scroll_percentage": False,     # likewise
}
UNSAFE_FEATURES = sorted(name for name, ok in SERVING_OK.items() if not ok)

# Needs `article_ids_clicked`, which the EB-NeRD test file does not ship (SPEC.md §11.5).
ABSENT_FROM_TEST_FILE = {"n_prior_clicks_in_session"}


def drop_unsafe(df: pl.DataFrame) -> pl.DataFrame:
    """The Q9 "without" frame: every serving-unsafe feature column removed, nothing else."""
    return df.drop([c for c in df.columns if c in UNSAFE_FEATURES])


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


def category_match(log: pl.DataFrame, user_id, candidate_category, t: datetime,
                   half_life: timedelta, *, untimed_ts: datetime | None = None) -> float:
    """Cosine between the user's recency-weighted category profile and the candidate's category.

        cos(P, e_c) = P(c) / ||P||_2        (SPEC.md §11.3, CONTEXT.md C-010 / C-012)

    `e_c` is the one-hot vector of the candidate's category, so the dot product P . e_c is just
    P(c) — which *is* `recency_weighted_profile`. Dividing by ||P||_2 is what makes this a separate
    feature: it measures how concentrated the user's recent interests are. 1.0 when c is the
    user's only recent category; 1/sqrt(k) across k equally-read categories.

    Cosine ignores scale, so the un-normalised decayed masses give the same value: W_c / ||W||_2.
    Same boundary, decay, fallback and 0.0 / NaN rules as `recency_weighted_profile`.
    """
    w = decay_weights(log, user_id, t, half_life, untimed_ts=untimed_ts)
    if w.height == 0:
        return math.nan
    # One mass per category; the frame arrives sorted by ts, so the sums are order-deterministic.
    masses = w.group_by("category", maintain_order=True).agg(pl.col("weight").sum())["weight"]
    in_category = w.filter(pl.col("category") == candidate_category)["weight"].sum()
    return in_category / math.sqrt((masses * masses).sum())


def recency_profile_batch(log: pl.DataFrame, requests: pl.DataFrame, half_life: timedelta, *,
                          untimed_ts: datetime | None = None) -> pl.DataFrame:
    """`recency_weighted_profile` for every row of `requests`, in one vectorised pass.

    `requests` needs `user_id`, `t` and `candidate_category`. Any other columns (impression_id,
    article_id, ...) pass through unchanged, and row order is preserved. Returns `requests` with
    a `recency_weighted_profile` column appended. It must agree with the row-by-row reference on
    every request (SPEC.md §11.2, `TestBatchParity`); the reference is the oracle for this path.

    Same maths, arranged for volume:

      1. anchors — distinct (user_id, t). The profile depends on the user and the moment, not on
         the candidate, so it is computed once per impression, not once per candidate.
      2. pair each anchor with that user's events and keep ts < t — the strict boundary.
      3. weight each pair, then sum per (anchor, category) and per anchor.
      4. look each request's category up: 0.0 if the anchor has mass elsewhere, NaN if none.

    Cost: step 2 materialises, for every anchor, all of that user's events — memory grows with
    sum-over-anchors of history length. That is the bound to watch: a Kaggle driver passes
    `requests` in chunks (one Parquet row group at a time) to cap it. Only log rows for users
    that appear in `requests` are read.
    """
    by_category = _decayed_category_mass(log, requests, half_life, untimed_ts)
    by_anchor = by_category.group_by("user_id", "t").agg(total_mass=pl.col("category_mass").sum())
    return _lookup(requests, by_category, by_anchor, "recency_weighted_profile",
                   pl.col("category_mass").fill_null(0.0) / pl.col("total_mass"))


def category_match_batch(log: pl.DataFrame, requests: pl.DataFrame, half_life: timedelta, *,
                         untimed_ts: datetime | None = None) -> pl.DataFrame:
    """`category_match` for every row of `requests`, in one vectorised pass (SPEC.md §11.3).

    Same contract as `recency_profile_batch` — requests in, requests plus a `category_match`
    column out, order and extra columns preserved — and the same per-(anchor, category) masses.
    Only the denominator differs: the L2 norm of the anchor's masses instead of their sum.
    Must agree with the row-by-row `category_match` on every request (`TestCategoryMatchBatchParity`).
    """
    by_category = _decayed_category_mass(log, requests, half_life, untimed_ts)
    by_anchor = by_category.group_by("user_id", "t").agg(
        norm=pl.col("category_mass").pow(2).sum().sqrt())
    return _lookup(requests, by_category, by_anchor, "category_match",
                   pl.col("category_mass").fill_null(0.0) / pl.col("norm"))


def _decayed_category_mass(log: pl.DataFrame, requests: pl.DataFrame, half_life: timedelta,
                           untimed_ts: datetime | None) -> pl.LazyFrame:
    """Steps 1-3 shared by the batch features: decayed click mass per (user_id, t, category),
    over events strictly before t. Validates exactly as `decay_weights` does."""
    if half_life <= timedelta(0):
        raise ValueError(f"half_life must be positive, got {half_life}")

    events = log.join(requests.select("user_id").unique(), on="user_id", how="semi")
    if untimed_ts is not None:
        events = events.with_columns(pl.col("ts").fill_null(untimed_ts))
    elif events["ts"].null_count():
        raise ValueError(f"{events['ts'].null_count()} requested-user click(s) have a null ts; "
                         "pass untimed_ts to stamp them explicitly")

    h_seconds = half_life.total_seconds()
    age_seconds = (pl.col("t") - pl.col("ts")).dt.total_microseconds() / 1e6
    return (requests.lazy().select("user_id", "t").unique()
            .join(events.lazy(), on="user_id")
            .filter(pl.col("ts") < pl.col("t"))            # strictly before t, per anchor
            .with_columns(weight=pl.lit(2.0).pow(-age_seconds / h_seconds))
            .group_by("user_id", "t", "category")
            .agg(category_mass=pl.col("weight").sum()))


def _lookup(requests: pl.DataFrame, by_category: pl.LazyFrame, by_anchor: pl.LazyFrame,
            name: str, value: pl.Expr) -> pl.DataFrame:
    """Step 4 shared by the batch features: attach each request's anchor aggregate and category
    mass, compute `value`, and return `requests` in its original order plus column `name`.
    An anchor with no eligible events has no row in `by_anchor`, and gets NaN."""
    anchor_cols = [c for c in by_anchor.collect_schema().names() if c not in ("user_id", "t")]
    return (requests.lazy().with_row_index("_row")
            .join(by_anchor, on=["user_id", "t"], how="left")
            .join(by_category, left_on=["user_id", "t", "candidate_category"],
                  right_on=["user_id", "t", "category"], how="left")
            .with_columns(pl.when(pl.col(anchor_cols[0]).is_null()).then(pl.lit(math.nan))
                          .otherwise(value).alias(name))
            .sort("_row")
            .drop("_row", "category_mass", *anchor_cols)
            .collect())


# ----------------------------------------------------------------------------------------------
# Phase 1.2 (SPEC.md §11.4-11.7): slate, session and dwell features.
# ----------------------------------------------------------------------------------------------

def slate_features(impressions: pl.DataFrame) -> pl.DataFrame:
    """`cand_position` and `n_candidates`, one row per candidate (SPEC.md §11.4). MIND and EB-NeRD.

    Reads `impression_id` and `candidates` (a list) and **nothing else** — in particular not the
    labels — so it is computable on an unlabelled test impression. `cand_position` is 1 for the
    first candidate in the list. Rows come out in input order, then list order; an empty list
    yields no rows.

    Measured: click rate is nearly flat across list positions in both datasets, so the order is not
    a label artifact; whether it equals on-screen order is not established (§11.4).
    """
    return (impressions.select("impression_id", "candidates")
            .filter(pl.col("candidates").list.len() > 0)
            .with_columns(n_candidates=pl.col("candidates").list.len(),
                          cand_position=pl.int_ranges(1, pl.col("candidates").list.len() + 1))
            .explode("candidates", "cand_position", empty_as_null=False)   # empties already removed
            .select("impression_id", pl.col("candidates").alias("article_id"),
                    "cand_position", "n_candidates"))


def session_features(behaviors: pl.DataFrame) -> pl.DataFrame:
    """`session_pos`, `n_prior_clicks_in_session` and the UNSAFE `session_len` (SPEC.md §11.5).

    Reads `user_id`, `session_id`, `t` and, if present, `clicked` (list of clicked ids). Returns
    the input, in its original order, with the features appended.

    A session is keyed by **(user_id, session_id)**: the EB-NeRD test file reuses session_id 0
    for 200,000 placeholder rows from 200,000 different users.

    * session_pos — 1 + number of the session's impressions strictly before t. Two impressions at
      the same t are not prior to each other.
    * n_prior_clicks_in_session — clicks on those same prior impressions. Only computed when
      `clicked` exists; otherwise the column is **omitted, never zero-filled** — the EB-NeRD test
      file ships no clicks, and a column of zeros there would be a silent train/test skew.
    * session_len — all of the session's impressions, *including later ones*. Unknowable at request
      time, so it is in UNSAFE_FEATURES and exists only for the Q9 ablation.
    """
    key = ["user_id", "session_id"]
    has_clicks = "clicked" in behaviors.columns
    rows = behaviors.lazy().with_row_index("_row")

    earlier = rows.select(*key, t_prior=pl.col("t"),
                          **({"clicks_prior": pl.col("clicked").list.len()} if has_clicks else {}))
    per_row = [pl.len().alias("_n_prior")]
    if has_clicks:
        per_row.append(pl.col("clicks_prior").sum().alias("n_prior_clicks_in_session"))
    prior = (rows.select("_row", *key, "t")
             .join(earlier, on=key)                               # every pair within a session
             .filter(pl.col("t_prior") < pl.col("t"))             # strictly before t
             .group_by("_row").agg(per_row))
    length = rows.group_by(key).agg(session_len=pl.len())

    out = (rows.join(prior, on="_row", how="left")
           .join(length, on=key, how="left")
           .with_columns(session_pos=1 + pl.col("_n_prior").fill_null(0)))
    if has_clicks:
        out = out.with_columns(pl.col("n_prior_clicks_in_session").fill_null(0))
    return out.sort("_row").drop("_row", "_n_prior").collect()


def dwell_features(history: pl.DataFrame, requests: pl.DataFrame) -> pl.DataFrame:
    """`hist_read_time_mean` and `hist_scroll_mean` over the user's past clicks (SPEC.md §11.6).

    `history`: one row per past click — `user_id`, `ts`, `read_time`, `scroll_percentage`.
    `requests`: `user_id`, `t`, plus any passthrough columns; order preserved.

    Only clicks with ts < t count — strict, as everywhere in this module. Null values are skipped
    by the mean; NaN when nothing eligible remains (no history, or every value null). A null `ts`
    raises, as in `decay_weights`: a click with no time cannot be placed relative to t.
    """
    events = history.join(requests.select("user_id").unique(), on="user_id", how="semi")
    if events["ts"].null_count():
        raise ValueError(f"{events['ts'].null_count()} requested-user history click(s) have a null ts")

    means = (requests.lazy().select("user_id", "t").unique()
             .join(events.lazy(), on="user_id")
             .filter(pl.col("ts") < pl.col("t"))                 # strictly before t
             .group_by("user_id", "t")
             .agg(hist_read_time_mean=pl.col("read_time").mean(),
                  hist_scroll_mean=pl.col("scroll_percentage").mean()))
    return (requests.lazy().with_row_index("_row")
            .join(means, on=["user_id", "t"], how="left")
            .with_columns(pl.col("hist_read_time_mean", "hist_scroll_mean").fill_null(math.nan))
            .sort("_row")
            .drop("_row")
            .collect())


def freshness_batch(first_known: pl.DataFrame, requests: pl.DataFrame, *,
                    untimed_ts: datetime | None = None) -> pl.DataFrame:
    """`freshness_hours`: how long before t the candidate was first known to exist (SPEC.md §11.8).

    `first_known`: `article_id`, `ts` — EB-NeRD: one row per article with its `published_time`;
    MIND (no publish time): one row per appearance in any impression, plus history articles with a
    null `ts` stamped by `untimed_ts`. `requests`: `article_id`, `t` and passthrough columns.

    The earliest sighting *strictly before t* is the article's overall earliest sighting whenever
    that one is before t — so one group-by gives it, no pair table. NaN when nothing qualifies:
    unknown article, first seen at or after t, or (EB-NeRD) a publish time recorded as ≥ t, which
    can only be a later rewrite of the metadata.
    """
    if untimed_ts is not None:
        first_known = first_known.with_columns(pl.col("ts").fill_null(untimed_ts))
    elif first_known["ts"].null_count():
        raise ValueError(f"{first_known['ts'].null_count()} sighting(s) have a null ts; "
                         "pass untimed_ts to stamp them explicitly")

    first_seen = first_known.lazy().group_by("article_id").agg(first_seen=pl.col("ts").min())
    age_hours = (pl.col("t") - pl.col("first_seen")).dt.total_microseconds() / 3.6e9
    return (requests.lazy().with_row_index("_row")
            .join(first_seen, on="article_id", how="left")
            .with_columns(freshness_hours=pl.when(pl.col("first_seen") < pl.col("t"))   # strict
                          .then(age_hours).otherwise(pl.lit(math.nan)))
            .sort("_row")
            .drop("_row", "first_seen")
            .collect())


def current_page_features(behaviors: pl.DataFrame) -> pl.DataFrame:
    """UNSAFE: the impression's own `read_time` and `scroll_percentage` (SPEC.md §11.7).

    They describe the page view the impression belongs to, which continues after the moment the
    recommendation is served — no live system has them at request time. The EB-NeRD test file
    does ship them, which is exactly why they are dangerous: a model can exploit them on the
    leaderboard. Registered in UNSAFE_FEATURES; they exist only to supply the Q9 "with" row.
    """
    return behaviors.select("impression_id", cur_read_time=pl.col("read_time"),
                            cur_scroll_percentage=pl.col("scroll_percentage"))
