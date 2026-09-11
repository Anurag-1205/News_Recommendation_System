"""Oracles for A2 Phase 1.2: slate, session and dwell features, and the unsafe registry
(SPEC.md §11.4-11.7).

Every expected value is hand-computed from the definitions in SPEC.md and written as a literal.
As in tests/test_behavioural_features.py, the module under test is imported inside each test, so
these oracles could be seen failing on their own before the implementation existed (CONTEXT.md C-013).

The central leakage check for time-dependent features is *append the future, nothing changes*:
adding events after t must leave every serving-safe feature untouched. `session_len` is the
counter-example, and is tagged unsafe for exactly that reason.
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


# ----------------------------------------------------------------------------------------------
# 11.4 Slate features — cand_position, n_candidates (MIND and EB-NeRD)
# ----------------------------------------------------------------------------------------------
def mind_slates() -> pl.DataFrame:
    return pl.DataFrame({
        "impression_id": [10, 11, 12, 13],
        "candidates": [["N3", "N1", "N2"], ["N9"], [], ["N1", "N3"]],
        "labels": [[0, 1, 0], [1], [], [0, 1]],           # present only to prove it is never read
    }, schema={"impression_id": pl.Int64, "candidates": pl.List(pl.Utf8), "labels": pl.List(pl.Int8)})


EXPECTED_MIND_SLATE = [
    # impression_id, article_id, cand_position, n_candidates
    (10, "N3", 1, 3), (10, "N1", 2, 3), (10, "N2", 3, 3),
    (11, "N9", 1, 1),
    # impression 12 has an empty list: no rows
    (13, "N1", 1, 2), (13, "N3", 2, 2),
]


def _slate_rows(df: pl.DataFrame) -> list[tuple]:
    return list(df.select("impression_id", "article_id", "cand_position", "n_candidates").iter_rows())


class TestSlateFeatures:
    def test_positions_and_counts_match_hand_computation(self):
        assert _slate_rows(_impl().slate_features(mind_slates())) == EXPECTED_MIND_SLATE

    def test_integer_article_ids_as_in_ebnerd(self):
        eb = pl.DataFrame({"impression_id": [7], "candidates": [[901, 305, 777]]},
                          schema={"impression_id": pl.UInt32, "candidates": pl.List(pl.Int32)})
        assert _slate_rows(_impl().slate_features(eb)) == [(7, 901, 1, 3), (7, 305, 2, 3), (7, 777, 3, 3)]

    def test_never_reads_labels(self):
        """The slate is all these features see. Dropping or scrambling the labels changes nothing,
        which is what makes them computable on an unlabelled test impression."""
        impl = _impl()
        reference = impl.slate_features(mind_slates())
        assert impl.slate_features(mind_slates().drop("labels")).equals(reference)
        scrambled = mind_slates().with_columns(pl.col("labels").list.reverse())
        assert impl.slate_features(scrambled).equals(reference)


# ----------------------------------------------------------------------------------------------
# 11.5 Session features — session_pos, n_prior_clicks_in_session (safe), session_len (UNSAFE)
# ----------------------------------------------------------------------------------------------
_SESSION_ROWS = [
    # impression_id user  session  t                 clicked    role
    (1, "U1", 100, _at(minutes=-30), [11]),        # first in U1's session 100
    (2, "U1", 100, _at(minutes=-20), [12, 13]),    # two clicks
    (3, "U1", 100, _at(minutes=-10), []),          # no click
    (4, "U1", 100, T,                [14]),        # TARGET
    (5, "U1", 100, T,                [15]),        # same session, same t as 4: a tie, not "prior"
    (6, "U1", 100, _at(minutes=+10), [16]),        # after 4
    (7, "U1", 200, _at(minutes=-5),  [17]),        # same user, another session
    (8, "U2", 100, _at(minutes=-40), [18]),        # another user reusing session_id 100
    (0, "U3", 0,   T,                []),          # beyond-accuracy-style sentinel: id 0, session 0
    (0, "U4", 0,   T,                []),          # ... shared by another user, as in the test file
]


def sessions(rows=_SESSION_ROWS) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=[("impression_id", pl.Int64), ("user_id", pl.Utf8),
                                      ("session_id", pl.Int64), ("t", pl.Datetime("us")),
                                      ("clicked", pl.List(pl.Int64))], orient="row")


# Hand-computed per row, in input order. Sessions are keyed by (user_id, session_id).
#   U1/100 prior sets: imp1 {} | imp2 {1} | imp3 {1,2} | imp4 {1,2,3} | imp5 {1,2,3} | imp6 {1..5}
#   clicks:            0       | 1        | 1+2=3     | 1+2+0=3      | 3            | 1+2+0+1+1=5
EXPECTED_SESSION = [
    # impression_id, session_pos, n_prior_clicks_in_session, session_len
    (1, 1, 0, 6), (2, 2, 1, 6), (3, 3, 3, 6), (4, 4, 3, 6), (5, 4, 3, 6), (6, 6, 5, 6),
    (7, 1, 0, 1),
    (8, 1, 0, 1),
    (0, 1, 0, 1), (0, 1, 0, 1),
]

SAFE_SESSION = ["session_pos", "n_prior_clicks_in_session"]


def _session_rows(df: pl.DataFrame) -> list[tuple]:
    return list(df.select("impression_id", "session_pos", "n_prior_clicks_in_session",
                          "session_len").iter_rows())


class TestSessionFeatures:
    def test_every_row_matches_hand_computation(self):
        assert _session_rows(_impl().session_features(sessions())) == EXPECTED_SESSION

    def test_tie_at_exactly_t_is_not_prior(self):
        """Impressions 4 and 5 share t; neither is before the other, so both are position 4 with
        3 prior clicks. A `<=` boundary would give 5 and 4."""
        out = _impl().session_features(sessions()).filter(pl.col("impression_id").is_in([4, 5]))
        assert out["session_pos"].to_list() == [4, 4]
        assert out["n_prior_clicks_in_session"].to_list() == [3, 3]

    def test_appending_the_future_changes_no_safe_feature(self):
        """The Q9 boundary for session features: an impression after t cannot move any safe
        value — but it does move session_len, which is why session_len is unsafe."""
        impl = _impl()
        before = impl.session_features(sessions())
        future = _SESSION_ROWS + [(99, "U1", 100, _at(hours=+1), [19])]
        after = impl.session_features(sessions(future)).head(len(_SESSION_ROWS))
        assert after.select(SAFE_SESSION).equals(before.select(SAFE_SESSION))
        u1_s100 = pl.col("impression_id").is_in([1, 2, 3, 4, 5, 6])
        assert before.filter(u1_s100)["session_len"].unique().to_list() == [6]
        assert after.filter(u1_s100)["session_len"].unique().to_list() == [7]

    def test_sessions_are_keyed_by_user_and_session_id(self):
        """U2's session 100 is not U1's session 100, and the two session-0 sentinel rows (different
        users, as in the EB-NeRD test file) are not one 2-impression session."""
        out = _impl().session_features(sessions())
        assert out.filter(pl.col("user_id") == "U2")["session_pos"].to_list() == [1]
        assert out.filter(pl.col("session_id") == 0)["session_len"].to_list() == [1, 1]

    def test_unlabelled_input_omits_the_click_count_instead_of_zero_filling(self):
        """The EB-NeRD test file has no article_ids_clicked. The click count must be absent,
        not a column of zeros that a model would read as 'no prior clicks'."""
        out = _impl().session_features(sessions().drop("clicked"))
        assert "n_prior_clicks_in_session" not in out.columns
        assert list(out.select("impression_id", "session_pos", "session_len").iter_rows()) == [
            (i, p, n) for i, p, _, n in EXPECTED_SESSION]

    def test_keeps_input_order_and_other_columns(self):
        shuffled = sessions().with_row_index("row").sample(fraction=1.0, shuffle=True, seed=3)
        out = _impl().session_features(shuffled)
        assert out["row"].to_list() == shuffled["row"].to_list()
        assert out.select(shuffled.columns).equals(shuffled)


# ----------------------------------------------------------------------------------------------
# 11.6 Dwell features — hist_read_time_mean, hist_scroll_mean (EB-NeRD)
# ----------------------------------------------------------------------------------------------
_HISTORY_ROWS = [
    # user   ts               read_time scroll
    ("U1", _at(days=-3),      10.0,     40.0),
    ("U1", _at(days=-2),      30.0,     None),     # no scroll value: skipped by the mean
    ("U1", _at(days=-1),      50.0,     80.0),
    ("U1", T,                 999.0,    999.0),    # EXACTLY AT T -> excluded
    ("U1", _at(days=+1),      777.0,    777.0),    # future -> excluded
    ("U2", _at(days=-1),      100.0,    100.0),
    ("U4", _at(days=-1),      20.0,     None),     # an eligible read time but no scroll at all
]


def history(rows=_HISTORY_ROWS) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=[("user_id", pl.Utf8), ("ts", pl.Datetime("us")),
                                      ("read_time", pl.Float64), ("scroll_percentage", pl.Float64)],
                        orient="row")


def dwell_requests() -> pl.DataFrame:
    return pl.DataFrame({"user_id": ["U1", "U1", "U2", "U3", "U4"],
                         "t": [T, _at(days=-2), T, T, T]},
                        schema={"user_id": pl.Utf8, "t": pl.Datetime("us")})


NAN = float("nan")
EXPECTED_DWELL = [
    # hist_read_time_mean,  hist_scroll_mean
    (30.0, 60.0),     # U1 at T:      read (10+30+50)/3, scroll (40+80)/2 (the null skipped)
    (10.0, 40.0),     # U1 at T-2d:   only the T-3d click; the T-2d click is AT t -> excluded
    (100.0, 100.0),   # U2 at T
    (NAN, NAN),       # U3: no history
    (20.0, NAN),      # U4: a read time, but every scroll value is null
]


def _dwell_values(df: pl.DataFrame) -> list[tuple]:
    return list(df.select("hist_read_time_mean", "hist_scroll_mean").iter_rows())


def _same(got: list[tuple], expected: list[tuple]) -> bool:
    return len(got) == len(expected) and all(
        (math.isnan(e) and math.isnan(g)) or g == pytest.approx(e, rel=1e-12)
        for gr, er in zip(got, expected) for g, e in zip(gr, er))


class TestDwellFeatures:
    def test_means_match_hand_computation(self):
        assert _same(_dwell_values(_impl().dwell_features(history(), dwell_requests())), EXPECTED_DWELL)

    def test_click_at_exactly_t_is_excluded(self):
        """If U1's click at exactly T (read 999) leaked, U1's read mean at T would be 272.25."""
        got = _impl().dwell_features(history(), dwell_requests())
        assert got["hist_read_time_mean"][0] == pytest.approx(30.0, rel=1e-12)

    def test_appending_the_future_changes_nothing(self):
        impl = _impl()
        future = _HISTORY_ROWS + [("U2", _at(days=+2), 5.0, 5.0), ("U3", _at(hours=+1), 5.0, 5.0)]
        assert _same(_dwell_values(impl.dwell_features(history(future), dwell_requests())), EXPECTED_DWELL)

    def test_keeps_request_order_and_other_columns(self):
        req = dwell_requests().with_row_index("impression_id").reverse()
        out = _impl().dwell_features(history(), req)
        assert out.columns == req.columns + ["hist_read_time_mean", "hist_scroll_mean"]
        assert out.drop("hist_read_time_mean", "hist_scroll_mean").equals(req)
        assert _same(_dwell_values(out), EXPECTED_DWELL[::-1])

    def test_null_timestamp_is_rejected(self):
        broken = history().with_columns(
            pl.when(pl.col("read_time") == 30.0).then(None).otherwise(pl.col("ts")).alias("ts"))
        with pytest.raises(ValueError):
            _impl().dwell_features(broken, dwell_requests())


# ----------------------------------------------------------------------------------------------
# 11.7 Serving-unsafe features and the registry
# ----------------------------------------------------------------------------------------------
class TestUnsafeFeaturesAndRegistry:
    def test_current_page_features_copy_the_impressions_own_values(self):
        b = pl.DataFrame({"impression_id": [1, 2], "read_time": [12.0, 3.5],
                          "scroll_percentage": [None, 80.0]})
        out = _impl().current_page_features(b)
        assert list(out.iter_rows()) == [(1, 12.0, None), (2, 3.5, 80.0)]
        assert out.columns == ["impression_id", "cur_read_time", "cur_scroll_percentage"]

    def test_unsafe_list_is_exact(self):
        assert _impl().UNSAFE_FEATURES == ["cur_read_time", "cur_scroll_percentage", "session_len"]

    def test_click_count_is_flagged_absent_from_the_test_file(self):
        assert _impl().ABSENT_FROM_TEST_FILE == {"n_prior_clicks_in_session"}

    def test_every_emitted_feature_is_registered(self):
        """A feature nobody registered could slip into a model unclassified."""
        impl = _impl()
        emitted = (set(impl.slate_features(mind_slates()).columns) - {"impression_id", "article_id"}
                   | set(impl.session_features(sessions()).columns) - set(sessions().columns)
                   | set(impl.dwell_features(history(), dwell_requests()).columns) - {"user_id", "t"}
                   | set(impl.current_page_features(pl.DataFrame(
                       {"impression_id": [1], "read_time": [1.0], "scroll_percentage": [1.0]})).columns)
                   - {"impression_id"})
        assert emitted <= set(impl.SERVING_OK), emitted - set(impl.SERVING_OK)

    def test_drop_unsafe_removes_exactly_the_unsafe_columns(self):
        impl = _impl()
        frame = pl.DataFrame({"impression_id": [1], "session_pos": [1], "session_len": [3],
                              "cur_read_time": [2.0], "hist_read_time_mean": [4.0]})
        assert impl.drop_unsafe(frame).columns == ["impression_id", "session_pos", "hist_read_time_mean"]
