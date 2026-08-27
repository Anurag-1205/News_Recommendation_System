"""Oracles for point-in-time features (SPEC.md §5).

The load-bearing property is *strictly before*: an event at exactly the query time must not
be counted, because at serving time it has not happened yet.
"""

from datetime import datetime, timedelta

import pytest

from src.features.rolling import RollingCounts, frozen_counts

T0 = datetime(2019, 11, 10, 12, 0, 0)


@pytest.fixture
def counts():
    c = RollingCounts()
    for hours in (0, 1, 2, 30, 50):          # clicks at T0, +1h, +2h, +30h, +50h
        c.add_click("A", T0 + timedelta(hours=hours))
    for hours in range(0, 60, 2):
        c.add_view("A", T0 + timedelta(hours=hours))
    c.add_click("B", T0 + timedelta(hours=5))
    c.seal()
    return c


class TestStrictlyBefore:
    def test_counts_only_earlier_events(self, counts):
        assert counts.clicks_before("A", T0 + timedelta(hours=2, minutes=30)) == 3

    def test_event_at_exactly_query_time_is_excluded(self, counts):
        """The Q9 boundary: an event at t has not happened yet, as of t."""
        assert counts.clicks_before("A", T0) == 0
        assert counts.clicks_before("A", T0 + timedelta(microseconds=1)) == 1

    def test_far_future_sees_everything(self, counts):
        assert counts.clicks_before("A", T0 + timedelta(days=365)) == 5

    def test_unknown_article_is_zero_not_error(self, counts):
        assert counts.clicks_before("NOPE", T0) == 0


class TestWindows:
    def test_trailing_window_excludes_older_events(self, counts):
        """At T0+31h with a 24h window: clicks at +30h only -> 1 (the +0/+1/+2h are too old)."""
        assert counts.clicks_before("A", T0 + timedelta(hours=31), window=timedelta(hours=24)) == 1

    def test_window_matches_unwindowed_when_wide_enough(self, counts):
        t = T0 + timedelta(hours=60)
        assert (counts.clicks_before("A", t, window=timedelta(days=365))
                == counts.clicks_before("A", t))

    def test_narrow_window_can_be_empty(self, counts):
        assert counts.clicks_before("A", T0 + timedelta(hours=49), window=timedelta(minutes=1)) == 0


class TestCTR:
    def test_zero_views_gives_zero_not_division_error(self, counts):
        assert counts.ctr_before("NOPE", T0) == 0.0

    def test_smoothing_damps_low_volume_articles(self):
        """One click in one view must not outrank a well-evidenced article."""
        c = RollingCounts()
        c.add_click("lucky", T0); c.add_view("lucky", T0)
        for i in range(500):
            c.add_click("solid", T0 + timedelta(seconds=i))
        for i in range(2000):
            c.add_view("solid", T0 + timedelta(seconds=i))
        c.seal()
        t = T0 + timedelta(days=1)
        assert c.ctr_before("solid", t) > c.ctr_before("lucky", t)

    def test_ctr_is_between_zero_and_one(self, counts):
        t = T0 + timedelta(hours=60)
        assert 0.0 <= counts.ctr_before("A", t) <= 1.0


class TestSealing:
    def test_query_before_seal_is_refused(self):
        """An unsorted bisect returns a wrong count silently — so refuse instead."""
        c = RollingCounts()
        c.add_click("A", T0)
        with pytest.raises(RuntimeError, match="seal"):
            c.clicks_before("A", T0)

    def test_out_of_order_insertion_still_correct_after_seal(self):
        c = RollingCounts()
        for h in (50, 1, 30, 0, 2):          # deliberately unsorted
            c.add_click("A", T0 + timedelta(hours=h))
        c.seal()
        assert c.clicks_before("A", T0 + timedelta(hours=3)) == 3


def test_frozen_differs_from_rolling(counts):
    """The ablation only means something if the two actually disagree."""
    frozen = frozen_counts(counts, ["A"])
    rolling = counts.clicks_before("A", T0 + timedelta(hours=2, minutes=30))
    assert frozen["A"] == 5 and rolling == 3
