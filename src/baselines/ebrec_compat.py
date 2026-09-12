"""Compatibility shim so ebnerd-benchmark (pinned to polars 0.20.8) runs on polars >= 1.0.

The benchmark's `map_list_article_id_to_value` (src/ebrec/utils/_articles_behaviors.py) does
`pl.col(col).replace(mapping, default=None)` with a dict whose values are *lists* (token ids per
article). polars 0.20 accepted that. polars 1.35, the version on Kaggle's GPU image, routes it to
`replace_strict` and raises "not yet implemented: Nested object types" (smoke run v1, RESULTS.md
P0); polars 1.43, our venv, accepts it again. The shim makes the baseline independent of which
side of that line the runtime falls on (CONTEXT.md C-021).

A second function, `add_prediction_scores` (src/ebrec/utils/_behaviors.py), ends with
`.drop("_groupby_id")` on a frame that never had that column; polars 0.20 ignored a missing
column in `drop`, polars 1.x raises ColumnNotFoundError (smoke run v2).

Both functions below keep the benchmark's signature and semantics and are verified against the
benchmark's own docstring examples in tests/test_ebrec_compat.py. `install()` patches them into
every module that imported the originals by name.
"""
from __future__ import annotations

from typing import Any

import polars as pl


def _unique_name(existing: list[str], base: str) -> str:
    name, i = base, 0
    while name in existing:
        i += 1
        name = f"{base}_{i}"
    return name


def map_list_article_id_to_value(
    behaviors: pl.DataFrame,
    behaviors_column: str,
    mapping: dict[Any, Any],
    drop_nulls: bool = False,
    fill_nulls: Any = None,
) -> pl.DataFrame:
    """Drop-in for ebrec.utils._articles_behaviors.map_list_article_id_to_value on polars >= 1."""
    row_id = _unique_name(behaviors.columns, "_groupby_id")
    val_col = _unique_name(behaviors.columns + [row_id], "_mapped")
    keys = list(mapping.keys())
    values = [list(v) if isinstance(v, pl.Series) else v for v in mapping.values()]
    map_df = pl.DataFrame({behaviors_column: pl.Series(keys), val_col: pl.Series(values)})

    with_id = behaviors.with_row_index(row_id)
    exploded = (
        with_id.select(row_id, behaviors_column)
        .explode(behaviors_column)
        .join(map_df, on=behaviors_column, how="left", maintain_order="left")
        .drop(behaviors_column)
        .rename({val_col: behaviors_column})
    )
    if drop_nulls:
        exploded = exploded.drop_nulls()
    elif fill_nulls is not None:
        exploded = exploded.with_columns(pl.col(behaviors_column).fill_null(fill_nulls))
    agg = exploded.group_by(row_id, maintain_order=True).agg(behaviors_column)
    return with_id.drop(behaviors_column).join(agg, on=row_id, how="left").drop(row_id)


def add_prediction_scores(
    df: pl.DataFrame,
    scores,
    inview_col: str = "article_ids_inview",
    prediction_scores_col: str = "scores",
) -> pl.DataFrame:
    """Drop-in for ebrec.utils._behaviors.add_prediction_scores: attach one flat score list to a
    frame of slates, re-nested to match `inview_col`. Same body as the original except the final
    `.drop` of a column `df` never had."""
    row_id = _unique_name(df.columns, "_groupby_id")
    nested = (
        df.lazy()
        .select(pl.col(inview_col))
        .with_row_index(row_id)
        .explode(inview_col)
        .with_columns(pl.Series(prediction_scores_col, scores).explode())
        .group_by(row_id)
        .agg(inview_col, prediction_scores_col)
        .sort(row_id)
        .collect()
    )
    return df.with_columns(nested.get_column(prediction_scores_col))


def install() -> None:
    """Replace the benchmark's functions everywhere they were imported by name."""
    import importlib

    patches = {
        map_list_article_id_to_value: [
            "ebrec.utils._articles_behaviors",
            "ebrec.models.newsrec.dataloader",
            "ebrec.models.fastformer.dataloader",
        ],
        add_prediction_scores: ["ebrec.utils._behaviors"],
    }
    for fn, modules in patches.items():
        for name in modules:
            try:
                mod = importlib.import_module(name)
            except ImportError:  # fastformer pulls torch; absent on some images
                continue
            setattr(mod, fn.__name__, fn)
