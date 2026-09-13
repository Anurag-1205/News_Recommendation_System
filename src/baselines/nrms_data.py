"""Frames for the NRMS baselines and the scores file they write (SPEC.md §13.2–§13.3).

Two jobs, both keyed by `imp_row` = the row index of the split file in file order, assigned
before any join, exactly as `src.rerank.ebnerd.load_behaviors` assigns it (parity is tested):

* `ebnerd_behaviors`: one row per impression in the shape the benchmark's `NRMSDataLoader`
  expects — `article_id_fixed` (history, left-padded with 0 to `history_size`),
  `article_ids_inview`, `labels` (0/1 per slot, from `article_ids_clicked`) — plus our keys.
* `scores_frame`: per-slate score lists -> the agreed scores file, one row per candidate,
  `cand_position` 1-based, sorted by (`imp_row`, `cand_position`).

No TensorFlow here: everything is polars, so it runs and is tested on the laptop.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import polars as pl

SCORE_COLUMNS = ["imp_row", "impression_id", "article_id", "cand_position", "score"]


def truncate_history(df: pl.DataFrame, column: str, history_size: int, padding_value: Any = None) -> pl.DataFrame:
    """Keep the last `history_size` items of a list column; with `padding_value`, left-pad so every
    list has exactly `history_size` items. Same semantics as the benchmark's
    `ebrec.utils._behaviors.truncate_history` (its docstring example is the test), re-implemented
    so this module does not need `external/`."""
    if padding_value is not None:
        df = df.with_columns(
            pl.col(column).list.reverse()
            .list.eval(pl.element().extend_constant(padding_value, n=history_size))
            .list.reverse()
        )
    return df.with_columns(pl.col(column).list.tail(history_size))


def ebnerd_behaviors(split_dir: Path, *, history_size: int, limit: int | None = None) -> pl.DataFrame:
    """Every impression of an EB-NeRD split (or its first `limit` rows), in file order.

    Columns: `imp_row`, `impression_id`, `user_id`, `impression_time`, `article_ids_inview`,
    `article_ids_clicked` (absent on the unlabelled test file), `labels` (with it), and
    `article_id_fixed` from `history.parquet` (an empty list for a user without history).
    """
    split_dir = Path(split_dir)
    lf = pl.scan_parquet(split_dir / "behaviors.parquet")
    labelled = "article_ids_clicked" in lf.collect_schema().names()
    cols = ["impression_id", "user_id", "impression_time", "article_ids_inview"] + (["article_ids_clicked"] if labelled else [])
    beh = (lf.select(cols).head(limit) if limit else lf.select(cols)).collect().with_row_index("imp_row")

    hist = (pl.scan_parquet(split_dir / "history.parquet")
            .select("user_id", pl.col("article_id_fixed"))
            .filter(pl.col("user_id").is_in(beh["user_id"].implode()))
            .collect())
    hist = truncate_history(hist, "article_id_fixed", history_size, padding_value=0)
    beh = beh.join(hist, on="user_id", how="left", maintain_order="left")
    pad = pl.lit([0] * history_size, dtype=beh["article_id_fixed"].dtype)
    beh = beh.with_columns(pl.col("article_id_fixed").fill_null(pad))

    if labelled:
        # 0/1 per slot by membership of the clicked set: the same rule as the reranker frame.
        labels = (beh.select("imp_row", "article_ids_inview", "article_ids_clicked")
                  .explode("article_ids_inview")
                  .with_columns(label=pl.col("article_ids_inview").is_in(pl.col("article_ids_clicked")).cast(pl.Int8))
                  .group_by("imp_row", maintain_order=True).agg(labels=pl.col("label")))
        beh = beh.join(labels, on="imp_row", how="left", maintain_order="left")
    return beh.sort("imp_row")


def scores_frame(beh: pl.DataFrame, scores: Sequence[Sequence[float]]) -> pl.DataFrame:
    """The scores file (SPEC.md §13.3) from per-impression score lists aligned with `beh` rows.

    `beh` needs `imp_row`, `impression_id`, `article_ids_inview`; `scores[i]` must have exactly
    one value per candidate of row i. Native dtypes for `impression_id` and `article_id` are
    kept; `cand_position` is 1-based Int64; `score` Float64.
    """
    if len(scores) != beh.height:
        raise ValueError(f"{len(scores)} score lists for {beh.height} impressions")
    slate_len = beh["article_ids_inview"].list.len().to_list()
    bad = [i for i, (s, n) in enumerate(zip(scores, slate_len)) if len(s) != n]
    if bad:
        raise ValueError(f"score list length != slate length at rows {bad[:5]}")
    scores_col = pl.Series("score", [[float(x) for x in s] for s in scores], dtype=pl.List(pl.Float64))
    return (beh.select("imp_row", "impression_id", article_id=pl.col("article_ids_inview"))
            .with_columns(scores_col)
            .with_columns(cand_position=pl.int_ranges(1, pl.col("article_id").list.len() + 1, dtype=pl.Int64))
            .explode("article_id", "cand_position", "score", empty_as_null=False)
            .select(SCORE_COLUMNS)
            .sort("imp_row", "cand_position"))
