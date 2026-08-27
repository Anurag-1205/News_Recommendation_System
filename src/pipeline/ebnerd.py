"""EB-NeRD reader: raw Parquet -> the unified schema in SPEC.md §2.

EB-NeRD differs from MIND in four ways that matter to everything downstream:

* **Danish**, so tokenisation uses the Danish stoplist. Applying English resources here is
  the easiest way to produce numbers that look fine and are wrong.
* **Parquet, not TSV**, and large enough that the test behaviours file is read row-group at
  a time rather than scanned whole.
* **History lives in its own file**, keyed by user_id, instead of inline per impression.
* **History carries timestamps** (`impression_time_fixed`), which MIND does not. That makes
  the row-level leakage assertion — every history event strictly before its impression —
  actually computable here.

Field mapping to the unified schema:

    article_id            <- article_id            (int32, kept as int)
    title / abstract      <- title / subtitle
    published_ts          <- published_time
    impression_id, user_id, ts <- impression_id, user_id, impression_time
    candidates            <- article_ids_inview
    labels                <- article_ids_clicked   (a list of ids, not a 0/1 mask)

That last row is the trap: MIND gives a parallel 0/1 label per candidate, EB-NeRD gives the
*set* of clicked ids. `labels_from_clicked` converts one to the other.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

LANG = "da"


def article_text(articles_path: Path | str, limit: int | None = None) -> dict[int, tuple]:
    """article_id -> (title, subtitle), for indexing.

    `body` is deliberately excluded: it is present in EB-NeRD and would blow the index up by
    two orders of magnitude for a corpus this size. Title+subtitle mirrors MIND's
    title+abstract, which also keeps the two datasets' lexical results comparable.
    """
    df = pl.read_parquet(articles_path, columns=["article_id", "title", "subtitle"])
    if limit:
        df = df.head(limit)
    return dict(zip(df["article_id"].to_list(), zip(df["title"].to_list(), df["subtitle"].to_list())))


def recent_history(history_path: Path | str, n_recent: int = 5,
                   users: set | None = None) -> dict[int, list[int]]:
    """user_id -> their last `n_recent` clicked article ids, oldest first.

    Truncated at read time rather than after loading. The test history file is 1.16 GB and
    averages 144 articles per user across 807,677 users; materialising all of it would be
    ~116M ids in Python objects, which this machine cannot hold (SPEC.md §11). Keeping only
    the tail we actually query drops that to ~4M.

    `users` restricts the result to a known set of user ids, and the filter is pushed into the
    lazy scan so the rows are never materialised. When only part of the test set is being
    scored -- resuming an interrupted pass, say -- this is the difference between holding
    807,677 users and holding the few hundred thousand actually needed. Loading the full
    dictionary was what repeatedly exhausted memory during a resume.
    """
    lf = pl.scan_parquet(history_path).select(
        "user_id", pl.col("article_id_fixed").list.tail(n_recent).alias("recent")
    )
    if users is not None:
        lf = lf.filter(pl.col("user_id").is_in(list(users)))
    df = lf.collect()
    return dict(zip(df["user_id"].to_list(), df["recent"].to_list()))


def scan_behaviors(behaviors_path: Path | str) -> pl.LazyFrame:
    return pl.scan_parquet(behaviors_path)


def labels_from_clicked(candidates: list[int], clicked: list[int]) -> list[int]:
    """Convert EB-NeRD's clicked-id list into a 0/1 mask aligned to `candidates`.

    The membership test uses a set: impressions reach 100+ candidates, and a list scan per
    candidate would make this quadratic for no reason.
    """
    clicked_set = set(clicked)
    return [1 if c in clicked_set else 0 for c in candidates]


def iter_row_groups(behaviors_path: Path | str, columns: list[str]):
    """Yield one row group at a time as a polars DataFrame.

    The test file holds 13,536,710 rows in 51 row groups (~265K rows each). Row groups are
    the file's own physical chunking, so reading one is a single sequential read rather than
    a scattered seek — and it bounds peak memory at one group regardless of file size.
    """
    pf = pq.ParquetFile(behaviors_path)
    for i in range(pf.metadata.num_row_groups):
        yield pl.from_arrow(pf.read_row_group(i, columns=columns))


def n_rows(path: Path | str) -> int:
    """Row count from Parquet metadata — no data is read."""
    return pq.ParquetFile(path).metadata.num_rows
