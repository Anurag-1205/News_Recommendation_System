"""Oracle for the freshness feature (SPEC.md §11.8). Hand-computed; imported lazily, as in the
other Phase 1 oracles, so it could be seen failing before the implementation existed.

freshness_hours = (t - first time the article was known, strictly before t) in hours.
"""

import importlib
import math
from datetime import datetime, timedelta

import polars as pl
import pytest

T = datetime(2026, 1, 10, 12, 0, 0)


def _at(**offset) -> datetime:
    return T + timedelta(**offset)


def _impl():
    return importlib.import_module("src.features.behavioural")


def first_known(rows) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=[("article_id", pl.Utf8), ("ts", pl.Datetime("us"))], orient="row")


# EB-NeRD style: one row per article, ts = published_time.
PUBLISHED = [("A", _at(hours=-5)), ("B", _at(hours=+1)), ("C", T)]

# MIND style: one row per sighting (impression time); H is a history-only article (no time).
SIGHTINGS = [("X", _at(hours=-2)), ("X", _at(hours=-10)), ("X", _at(hours=+1)),   # out of order
             ("Y", T), ("Z", _at(hours=+3)), ("H", None)]


def requests(rows) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=[("article_id", pl.Utf8), ("t", pl.Datetime("us"))], orient="row")


def _hours(df: pl.DataFrame) -> list[float]:
    return df["freshness_hours"].to_list()


def _same(got: list[float], expected: list[float]) -> bool:
    return len(got) == len(expected) and all(
        (math.isnan(e) and math.isnan(g)) or g == pytest.approx(e, rel=1e-12) for g, e in zip(got, expected))


NAN = float("nan")


class TestPublishTime:
    def test_matches_hand_computation(self):
        req = requests([("A", T), ("A", _at(hours=-4)), ("A", _at(hours=-6)),
                        ("B", T), ("C", T), ("D", T)])
        got = _hours(_impl().freshness_batch(first_known(PUBLISHED), req))
        #              A@T  A@T-4h A@T-6h (published after t)  B (after t)  C (at t)  D (unknown)
        assert _same(got, [5.0, 1.0,   NAN,                    NAN,         NAN,      NAN])

    def test_publish_time_at_exactly_t_is_not_before_t(self):
        got = _hours(_impl().freshness_batch(first_known(PUBLISHED), requests([("C", T)])))
        assert math.isnan(got[0]), "a publish time equal to t is not strictly before it"


class TestFirstSeen:
    def test_matches_hand_computation(self):
        req = requests([("X", T), ("X", _at(hours=-5)), ("X", _at(hours=-10)),
                        ("Y", T), ("Z", T)])
        got = _hours(_impl().freshness_batch(first_known(SIGHTINGS[:-1]), req))
        #              X@T (first T-10h)  X@T-5h  X@T-10h (at t)  Y (only at t)  Z (only after t)
        assert _same(got, [10.0,          5.0,    NAN,            NAN,           NAN])

    def test_history_article_is_stamped_with_untimed_ts(self):
        got = _hours(_impl().freshness_batch(first_known(SIGHTINGS), requests([("H", T)]),
                                             untimed_ts=_at(hours=-48)))
        assert _same(got, [48.0])

    def test_null_ts_without_untimed_ts_is_rejected(self):
        with pytest.raises(ValueError):
            _impl().freshness_batch(first_known(SIGHTINGS), requests([("X", T)]))

    def test_appending_future_sightings_changes_nothing(self):
        req = requests([("X", T), ("X", _at(hours=-5)), ("Y", T), ("Z", T)])
        impl = _impl()
        before = impl.freshness_batch(first_known(SIGHTINGS[:-1]), req)
        future = SIGHTINGS[:-1] + [("X", _at(days=+1)), ("Y", _at(hours=+1)), ("W", _at(hours=+2))]
        assert _same(_hours(impl.freshness_batch(first_known(future), req)), _hours(before))

    def test_keeps_request_order_and_other_columns(self):
        req = requests([("Z", T), ("X", T), ("X", _at(hours=-5))]).with_row_index("impression_id")
        out = _impl().freshness_batch(first_known(SIGHTINGS[:-1]), req)
        assert out.columns == req.columns + ["freshness_hours"]
        assert out.drop("freshness_hours").equals(req)
        assert _same(_hours(out), [NAN, 10.0, 5.0])


def test_freshness_is_registered_as_serving_safe():
    assert _impl().SERVING_OK["freshness_hours"] is True
