"""Temporal splitting. Never random — interaction data is split by time or not at all.

Two separate boundaries exist and conflating them is the classic leak:

**The dataset's own split** is already temporal and disjoint. Measured from the files
(RESULTS.md), MIND is train 9-14 Nov / dev 15 Nov / test 16-22 Nov 2019. Nothing here
re-splits it; the leaderboard path uses it as shipped.

**Our internal split**, carved out of the training file, is what tunes hyper-parameters
without touching the official dev set. `MINDsmall_train` spans six days, so N=1 and M=1
mirror the dataset's own one-day dev window rather than inventing a different shape.

The cut is on `ts`, and the returned boundaries are the timestamps the split actually used
— not the ones requested — so a caller can assert against reality instead of intent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl


@dataclass(frozen=True)
class SplitBoundaries:
    """The two cut points, plus the observed range. Recorded so results are reproducible."""

    data_min: datetime
    val_start: datetime
    test_start: datetime
    data_max: datetime

    def describe(self) -> str:
        return (
            f"train [{self.data_min} .. {self.val_start}) | "
            f"val [{self.val_start} .. {self.test_start}) | "
            f"test [{self.test_start} .. {self.data_max}]"
        )


def compute_boundaries(lf: pl.LazyFrame, n_test_days: int = 1, m_val_days: int = 1,
                       ts_col: str = "ts") -> SplitBoundaries:
    """Derive cut points from the data's real range, not from an assumed calendar.

    The last N days become test and the M days before them become validation, both measured
    back from the observed maximum. Deriving from the data means a dataset that does not
    span N+M days fails loudly here rather than silently producing an empty split.
    """
    stats = lf.select(
        pl.col(ts_col).min().alias("lo"), pl.col(ts_col).max().alias("hi")
    ).collect()
    lo, hi = stats["lo"][0], stats["hi"][0]

    # Days are counted from the end of the last calendar day present, so a test window of
    # "1 day" means that whole final day rather than a 24h slice ending mid-afternoon.
    last_day_end = datetime.combine(hi.date(), datetime.min.time()) + timedelta(days=1)
    test_start = last_day_end - timedelta(days=n_test_days)
    val_start = test_start - timedelta(days=m_val_days)

    if val_start <= lo:
        raise ValueError(
            f"range {lo}..{hi} is too short for {n_test_days}d test + {m_val_days}d val; "
            f"validation would start at {val_start}, at or before the first event"
        )
    return SplitBoundaries(data_min=lo, val_start=val_start, test_start=test_start, data_max=hi)


def temporal_split(lf: pl.LazyFrame, bounds: SplitBoundaries, ts_col: str = "ts",
                   ) -> dict[str, pl.LazyFrame]:
    """Cut into train / val / test on the given boundaries.

    Half-open intervals throughout: an event exactly on a boundary belongs to the later
    split. That makes the three sets provably disjoint and their union the whole frame,
    which is what `tests/test_no_leakage.py` asserts.
    """
    return {
        "train": lf.filter(pl.col(ts_col) < bounds.val_start),
        "val": lf.filter((pl.col(ts_col) >= bounds.val_start) & (pl.col(ts_col) < bounds.test_start)),
        "test": lf.filter(pl.col(ts_col) >= bounds.test_start),
    }


def assert_disjoint(splits: dict[str, pl.LazyFrame], ts_col: str = "ts") -> dict[str, tuple]:
    """Verify train < val < test with no overlap, and return each split's observed range.

    Called by the pipeline itself rather than only by tests: a split that silently overlaps
    invalidates every downstream number, so it is worth re-checking on every build.
    """
    ranges = {}
    for name, lf in splits.items():
        row = lf.select(
            pl.len().alias("n"), pl.col(ts_col).min().alias("lo"), pl.col(ts_col).max().alias("hi")
        ).collect().row(0)
        ranges[name] = row

    for earlier, later in (("train", "val"), ("val", "test")):
        n_e, _, hi_e = ranges[earlier]
        n_l, lo_l, _ = ranges[later]
        if n_e == 0 or n_l == 0:
            continue
        if hi_e >= lo_l:
            raise AssertionError(
                f"temporal overlap: {earlier} ends {hi_e} but {later} starts {lo_l}"
            )
    return ranges
