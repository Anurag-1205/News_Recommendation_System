"""The Q9 anti-gaming assertion: no future-click leakage.

A feature used at impression time *t* may only see events strictly before *t*.

The load-bearing tests here are the ones that feed a **deliberately leaked** input and
assert it is caught. A test that only checks correct data passes is not evidence the
invariant holds — it is evidence the invariant was never exercised.

MIND has no per-event history timestamps (see `src/pipeline/features.py`), so the assertion
is made at the level where timestamps do exist: the impressions feeding each feature.
"""

from datetime import datetime, timedelta

import polars as pl
import pytest

from src.pipeline.features import article_popularity, max_source_timestamp, user_activity
from src.pipeline.split import (
    SplitBoundaries,
    assert_disjoint,
    compute_boundaries,
    temporal_split,
)

CUTOFF = datetime(2019, 11, 14)


def _impressions(rows) -> pl.LazyFrame:
    """rows: (impression_id, user_id, ts, candidates, labels)."""
    return pl.LazyFrame(
        {
            "impression_id": [r[0] for r in rows],
            "user_id": [r[1] for r in rows],
            "ts": [r[2] for r in rows],
            "candidates": [r[3] for r in rows],
            "labels": [r[4] for r in rows],
            "history_ids": [[] for _ in rows],
        },
        schema_overrides={"ts": pl.Datetime},
    )


@pytest.fixture
def clean():
    """All events strictly before the cutoff."""
    return _impressions([
        (1, "U1", datetime(2019, 11, 12), ["NA", "NB"], [1, 0]),
        (2, "U2", datetime(2019, 11, 13), ["NA", "NC"], [0, 1]),
    ])


@pytest.fixture
def leaked(clean):
    """The same data plus one impression AFTER the cutoff — the row that must not leak.

    Its click on NA is a future click: any feature that counts it has leaked.
    """
    future = _impressions([(3, "U1", datetime(2019, 11, 20), ["NA", "ND"], [1, 0])])
    return pl.concat([clean, future])


class TestFeaturesIgnoreTheFuture:
    def test_popularity_excludes_post_cutoff_clicks(self, clean, leaked):
        """NA has one click before the cutoff and one after. Only the first may count."""
        before = article_popularity(clean, CUTOFF)
        after = article_popularity(leaked, CUTOFF)
        na = lambda df: df.filter(pl.col("article_id") == "NA")["click_count"][0]
        assert na(before) == 1
        assert na(after) == 1, "future click leaked into popularity"

    def test_popularity_would_change_without_the_filter(self, leaked):
        """Proves the fixture is genuinely adversarial rather than a no-op.

        Without the cutoff filter NA scores 2. If this assertion ever fails, the leaked
        fixture has stopped containing a future click and the test above proves nothing.
        """
        unfiltered = (
            leaked.select("candidates", "labels").explode(["candidates", "labels"])
            .filter((pl.col("labels") == 1) & (pl.col("candidates") == "NA"))
            .select(pl.len()).collect().item()
        )
        assert unfiltered == 2

    def test_article_never_appears_if_only_seen_after_cutoff(self, leaked):
        """ND is clicked only in the future, so it must be absent entirely."""
        ids = article_popularity(leaked, CUTOFF)["article_id"].to_list()
        assert "ND" not in ids

    def test_user_activity_last_seen_precedes_cutoff(self, leaked):
        acts = user_activity(leaked, CUTOFF)
        assert acts["last_seen"].max() < CUTOFF
        u1 = acts.filter(pl.col("user_id") == "U1")
        assert u1["n_impressions"][0] == 1, "U1's post-cutoff impression was counted"

    def test_max_source_timestamp_is_strictly_before_cutoff(self, leaked):
        assert max_source_timestamp(leaked, CUTOFF) < CUTOFF

    def test_cutoff_is_recorded_on_every_feature_frame(self, clean):
        """The stored cutoff is what makes the invariant checkable after the fact."""
        assert article_popularity(clean, CUTOFF)["cutoff"][0] == CUTOFF
        assert user_activity(clean, CUTOFF)["cutoff"][0] == CUTOFF

    def test_boundary_is_strict_not_inclusive(self):
        """An event exactly at the cutoff is future, not past."""
        exact = _impressions([(1, "U1", CUTOFF, ["NA"], [1])])
        assert article_popularity(exact, CUTOFF).height == 0


class TestTemporalSplit:
    def _six_days(self):
        base = datetime(2019, 11, 9)
        rows = [
            (i, f"U{i}", base + timedelta(days=d, hours=i % 12), ["NA"], [1])
            for d in range(6)
            for i in range(4)
        ]
        return _impressions([(i, u, t, c, l) for i, (_, u, t, c, l) in enumerate(rows)])

    def test_splits_are_disjoint_in_time(self):
        lf = self._six_days()
        bounds = compute_boundaries(lf, n_test_days=1, m_val_days=1)
        ranges = assert_disjoint(temporal_split(lf, bounds))
        assert ranges["train"][2] < ranges["val"][1]
        assert ranges["val"][2] < ranges["test"][1]

    def test_splits_partition_the_data(self):
        """Disjoint is not enough — nothing may be silently dropped."""
        lf = self._six_days()
        bounds = compute_boundaries(lf, n_test_days=1, m_val_days=1)
        splits = temporal_split(lf, bounds)
        total = sum(s.select(pl.len()).collect().item() for s in splits.values())
        assert total == lf.select(pl.len()).collect().item()

    def test_overlapping_split_is_rejected(self):
        """The detector must fail on a bad split, or it proves nothing on a good one."""
        lf = self._six_days()
        overlapping = {
            "train": lf,                                            # spans everything
            "val": lf.filter(pl.col("ts") >= datetime(2019, 11, 13)),
            "test": lf.filter(pl.col("ts") >= datetime(2019, 11, 14)),
        }
        with pytest.raises(AssertionError, match="temporal overlap"):
            assert_disjoint(overlapping)

    def test_random_split_would_be_rejected(self):
        """A shuffled split interleaves in time, so the detector must catch it.

        This is the assertion that makes "never random" enforceable rather than a comment.
        """
        lf = self._six_days().collect()
        shuffled = lf.sample(fraction=1.0, shuffle=True, seed=0)
        n = shuffled.height // 3
        bad = {
            "train": shuffled.head(n).lazy(),
            "val": shuffled.slice(n, n).lazy(),
            "test": shuffled.tail(n).lazy(),
        }
        with pytest.raises(AssertionError, match="temporal overlap"):
            assert_disjoint(bad)

    def test_range_too_short_fails_loudly(self):
        one_day = _impressions([(1, "U1", datetime(2019, 11, 9, h), ["NA"], [1]) for h in range(3)])
        with pytest.raises(ValueError, match="too short"):
            compute_boundaries(one_day, n_test_days=3, m_val_days=3)

    def test_boundaries_describe_themselves(self):
        b = SplitBoundaries(
            datetime(2019, 11, 9), datetime(2019, 11, 13),
            datetime(2019, 11, 14), datetime(2019, 11, 14, 23, 59),
        )
        assert "train [2019-11-09" in b.describe()
