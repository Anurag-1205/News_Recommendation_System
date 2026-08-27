"""MIND reader: raw TSV -> the unified schema in SPEC.md §2.

MIND ships four TSVs per split with no header row. Only two matter here:

    news.tsv       news_id, category, subcategory, title, abstract, url,
                   title_entities, abstract_entities
    behaviors.tsv  impression_id, user_id, time, history, impressions

Two shapes of `impressions` exist and the difference is the whole point of the test set:

    train/dev   "N55689-1 N35729-0"   candidate-label pairs
    test        "N55689 N35729"       candidates only, labels are what we predict

`scan_behaviors` therefore returns a LazyFrame and leaves parsing to the two explicit
functions below, so a caller can never accidentally read labels that do not exist.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

BEHAVIOR_COLS = ["impression_id", "user_id", "time", "history", "impressions"]
NEWS_COLS = [
    "news_id", "category", "subcategory", "title", "abstract", "url",
    "title_entities", "abstract_entities",
]

# MIND timestamps look like "11/11/2019 9:05:58 AM" — US month-first, 12-hour with AM/PM.
TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"

# The TSVs contain bare " characters inside titles and abstracts. Without quote_char=None
# polars treats them as quoting and silently mis-parses rows; MIND's own loaders disable
# quoting for the same reason.
_CSV_OPTS = dict(separator="\t", quote_char=None, has_header=False)


def split_dir(root: Path | str, name: str) -> Path:
    """Locate a split directory, tolerating the doubled nesting the zips unpack into.

    `unzip MINDsmall_train.zip -d .../MINDsmall_train` yields
    `.../MINDsmall_train/MINDsmall_train/behaviors.tsv`, but a flat layout is just as
    plausible depending on how the archive was extracted. Check both rather than
    hard-coding one and failing confusingly later.
    """
    root = Path(root)
    for candidate in (root / name / name, root / name, root):
        if (candidate / "behaviors.tsv").exists():
            return candidate
    raise FileNotFoundError(f"no behaviors.tsv under {root}/{name}")


def scan_behaviors(path: Path | str) -> pl.LazyFrame:
    """Lazily scan behaviors.tsv. Nothing is read until the caller collects.

    Lazy by default because MINDlarge_test is 2.37M rows and this machine has ~2 GB free
    with ~2 GB free. The reference notebooks call `read_csv` here; that is the one place we
    deliberately diverge from them.
    """
    return pl.scan_csv(
        Path(path) / "behaviors.tsv",
        new_columns=BEHAVIOR_COLS,
        schema_overrides={"impression_id": pl.Int64, "history": pl.Utf8, "impressions": pl.Utf8},
        **_CSV_OPTS,
    )


def read_news(path: Path | str) -> pl.DataFrame:
    """Read news.tsv eagerly — 121K rows at most, comfortably small."""
    return pl.read_csv(
        Path(path) / "news.tsv",
        new_columns=NEWS_COLS,
        schema_overrides={"title_entities": pl.Utf8, "abstract_entities": pl.Utf8},
        **_CSV_OPTS,
    )


def with_parsed_time(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add `ts`, the impression timestamp, parsed from MIND's US-format `time` string."""
    return lf.with_columns(
        pl.col("time").str.strptime(pl.Datetime, TIME_FORMAT, strict=True).alias("ts")
    )


def with_history(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add `history_ids`: the user's prior clicks, oldest first.

    A null history means a cold-start user and becomes an empty list, not a null, so that
    downstream list operations do not have to special-case it.
    """
    return lf.with_columns(
        pl.col("history").fill_null("").str.split(" ")
        .list.eval(pl.element().filter(pl.element() != ""))
        .alias("history_ids")
    )


def with_labelled_candidates(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Split `impressions` into `candidates` and `labels`. Train/dev only.

    "N55689-1 N35729-0" -> candidates ["N55689","N35729"], labels [1, 0].
    Only the final "-1"/"-0" is stripped: news ids never contain "-", but stripping with a
    plain replace would be wrong if they ever did.
    """
    items = pl.col("impressions").str.split(" ")
    return lf.with_columns(
        items.list.eval(pl.element().str.head(-2)).alias("candidates"),
        items.list.eval(pl.element().str.tail(1).cast(pl.Int8)).alias("labels"),
    )


def with_unlabelled_candidates(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Split `impressions` into `candidates`. Test only — there are no labels to read."""
    return lf.with_columns(pl.col("impressions").str.split(" ").alias("candidates"))
