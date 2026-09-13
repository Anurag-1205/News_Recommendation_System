"""Freshness inputs for the NRMS variant (SPEC.md §15.2).

The number the model sees is the reranker's own `freshness_hours` (§11.8, `freshness_batch`,
strictly before t, NaN when unknown), transformed to two inputs per candidate:

    x       = (log1p(freshness_hours) − μ) / σ      μ, σ fitted on the TRAINING split's known candidates
    unknown = 1 if freshness_hours is NaN else 0 ;  x = 0 when unknown

`fresh_from_frames` is the pure function (tested by hand on a toy split); `fresh_inputs` wires
it to a real split with the same first-known sources the reranker uses, so the two systems
agree on what "fresh" means; `as_lists` gives the loaders a list column aligned with
`article_ids_inview`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl

from src.features.behavioural import freshness_batch

COLUMNS = ["imp_row", "cand_position", "freshness_hours", "x", "unknown"]


@dataclass(frozen=True)
class FreshStats:
    mu: float
    sigma: float

    def to_dict(self) -> dict:
        return asdict(self)


def fresh_from_frames(beh: pl.DataFrame, first_known: pl.DataFrame, stats: FreshStats, *,
                      untimed_ts=None, key: str = "imp_row") -> pl.DataFrame:
    """`beh`: `key`, `t`, `article_ids_inview`. `first_known`: `article_id`, `ts` (§11.8).
    One row per candidate, slate order, with the raw hours and the two model inputs.

    `key` must identify a *row* of `beh`. For a split it is `imp_row`; for a wu2019-sampled
    training frame it must be a fresh row index — sampling turns an impression with k clicks into
    k rows that share `imp_row` (EB-NeRD demo run v2 died on exactly that)."""
    req = (beh.select(pl.col(key).alias("imp_row"), "t", article_id=pl.col("article_ids_inview"))
           .with_columns(cand_position=pl.int_ranges(1, pl.col("article_id").list.len() + 1, dtype=pl.Int64))
           .explode("article_id", "cand_position", empty_as_null=False))
    out = freshness_batch(first_known, req, untimed_ts=untimed_ts)
    x = ((pl.col("freshness_hours").log1p() - stats.mu) / stats.sigma)
    return (out.with_columns(unknown=pl.col("freshness_hours").is_nan().cast(pl.Int8))
            .with_columns(x=pl.when(pl.col("unknown") == 1).then(0.0).otherwise(x).cast(pl.Float32))
            .select(COLUMNS).sort("imp_row", "cand_position"))


def fit_stats(feats: pl.DataFrame) -> FreshStats:
    """μ, σ of log1p(hours) over the known candidates of a frame (use the TRAINING split)."""
    v = np.log1p(feats.filter(pl.col("unknown") == 0)["freshness_hours"].to_numpy())
    return FreshStats(mu=float(v.mean()), sigma=float(v.std()))


def as_lists(beh: pl.DataFrame, feats: pl.DataFrame, *, key: str = "imp_row") -> pl.DataFrame:
    """`(key, fresh_inview)`: per row of `beh`, a list of [x, unknown] pairs in slate order.
    `feats` is `fresh_from_frames(beh, …, key=key)`, whose `imp_row` column holds that key;
    `beh[key]` must be unique (a fresh row index on a sampled frame)."""
    if beh[key].n_unique() != beh.height:
        raise ValueError(f"{key} is not unique per row; use a row index as the key on sampled frames")
    pairs = (feats.sort("imp_row", "cand_position")
             .with_columns(pair=pl.concat_list(pl.col("x").cast(pl.Float32), pl.col("unknown").cast(pl.Float32)))
             .group_by("imp_row", maintain_order=True).agg(fresh_inview=pl.col("pair"))
             .rename({"imp_row": key}))
    out = beh.select(key).join(pairs, on=key, how="left", maintain_order="left")
    lens = out["fresh_inview"].list.len().fill_null(0)
    if not (lens == beh["article_ids_inview"].list.len()).all():
        raise ValueError("fresh_inview lengths do not match article_ids_inview")
    return out


def _ebnerd_frames(split: str, limit: int | None):
    from src.baselines.nrms_data import ebnerd_behaviors
    from src.rerank.ebnerd import load_articles
    beh = ebnerd_behaviors(Path("data/interim/ebnerd") / split, history_size=1, limit=limit)
    beh = beh.select("imp_row", t=pl.col("impression_time"), article_ids_inview=pl.col("article_ids_inview"))
    first_known = load_articles().select("article_id", ts=pl.col("published_time"))
    return beh, first_known, None


def _mind_frames(split: str, limit: int | None):
    from src.rerank.mind import DATASET_START, first_sightings, load_behaviors
    train, dev = load_behaviors("MINDsmall_train"), load_behaviors("MINDsmall_dev")
    sightings = first_sightings([train, dev])          # strict < t keeps dev sightings out of train rows
    beh = {"MINDsmall_train": train, "MINDsmall_dev": dev}[split]
    if limit:
        beh = beh.head(limit)
    beh = beh.select("imp_row", "t", article_ids_inview=pl.col("candidates"))
    return beh, sightings, DATASET_START


def fresh_inputs(dataset: str, split: str, stats: FreshStats | None, *, limit: int | None = None) -> pl.DataFrame:
    """Freshness inputs for every candidate of a real split. `stats=None` fits μ, σ on this split
    (do that on the training split only, then pass the result for the evaluation split)."""
    beh, first_known, untimed = {"ebnerd": _ebnerd_frames, "mind": _mind_frames}[dataset](split, limit)
    raw = fresh_from_frames(beh, first_known, FreshStats(0.0, 1.0), untimed_ts=untimed)
    if stats is None:
        stats = fit_stats(raw)
    return fresh_from_frames(beh, first_known, stats, untimed_ts=untimed)
