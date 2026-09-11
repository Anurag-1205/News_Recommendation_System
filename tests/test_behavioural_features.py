"""Oracles for A2 Phase 1 behavioural features (SPEC.md §11).

The toy log is 20 hand-built click events. Every expected value below was computed by hand
from the decay formula in SPEC.md §11.1,

    w = 2 ** (-(t - ts) / h)

and is written out as a literal, with its arithmetic in a comment. Nothing is derived from the
implementation.

**Why the module under test is imported inside each test, not at the top of the file.** The
oracle was written before `src/features/behavioural.py` existed. Importing it lazily let
`TestOracleIsSound` pass on its own, proving the toy log and hand-computed values right, while
each feature test failed separately with `ModuleNotFoundError` (CONTEXT.md C-006). The
implementation was then checked against three planted bugs, each of which its dedicated test
caught (C-007). The lazy import costs nothing now and keeps that history reproducible.
"""

import importlib
import math
from datetime import datetime, timedelta

import polars as pl
import pytest

T = datetime(2026, 1, 10, 12, 0, 0)          # the impression time being scored
H = timedelta(hours=24)                      # half-life h


def _at(**offset) -> datetime:
    """T shifted by a timedelta; negative = before T."""
    return T + timedelta(**offset)


# ----------------------------------------------------------------------------------------------
# The toy log: 20 click events, three users. Row order is deliberately NOT time order.
# ----------------------------------------------------------------------------------------------
_ROWS = [
    # user  article  category         ts                   role in the oracle
    ("U1", "a01", "sports",        _at(hours=-72)),     # eligible: 2^-3
    ("U1", "a02", "politics",      _at(hours=-48)),     # eligible: 2^-2
    ("U1", "a03", "weather",       _at(hours=+1)),      # FUTURE, listed early -> excluded
    ("U1", "a04", "sports",        _at(hours=-24)),     # eligible: 2^-1
    ("U1", "a05", "tech",          _at(hours=-12)),     # eligible: 2^-0.5
    ("U1", "a06", "entertainment", _at(hours=0)),       # EXACTLY AT T -> excluded
    ("U1", "a07", "sports",        _at(seconds=-1)),    # strictly before T: 2^(-1/86400)
    ("U1", "a08", "politics",      _at(hours=-96)),     # OUT OF ORDER: oldest, listed last: 2^-4
    ("U2", "b01", "entertainment", _at(hours=-30)),     # other user: must not reach U1
    ("U2", "b02", "weather",       _at(hours=-6)),      # other user: must not reach U1
    ("U2", "b03", "entertainment", _at(hours=-2)),
    ("U2", "b04", "sports",        _at(hours=0)),       # other user, exactly at T
    ("U2", "b05", "politics",      _at(hours=-50)),
    ("U2", "b06", "weather",       _at(hours=+3)),
    ("U2", "b07", "entertainment", _at(hours=-80)),
    ("U3", "c01", "tech",          _at(hours=0)),       # U3: every event at or after T
    ("U3", "c02", "sports",        _at(seconds=+1)),    #     -> no eligible history -> NaN
    ("U3", "c03", "tech",          _at(hours=+5)),
    ("U3", "c04", "politics",      _at(hours=+24)),
    ("U3", "c05", "sports",        _at(hours=+48)),
]


def toy_log() -> pl.DataFrame:
    return pl.DataFrame(_ROWS, schema=["user_id", "article_id", "category", "ts"], orient="row")


# Hand-computed weights for U1 at T, h = 24 h. The six eligible events, sorted by ts ascending
# (the output order SPEC.md §11.1 fixes).
EXPECTED_U1_WEIGHTS = {
    "a08": 0.0625,                  # 96 h = 4h  -> 2^-4
    "a01": 0.125,                   # 72 h = 3h  -> 2^-3
    "a02": 0.25,                    # 48 h = 2h  -> 2^-2
    "a04": 0.5,                     # 24 h = 1h  -> 2^-1
    "a05": 0.7071067811865476,      # 12 h = h/2 -> 2^-0.5
    "a07": 0.9999919774953684,      # 1 s = h/86400 -> 2^(-1/86400)
}

# Category totals:  sports   = 0.125 + 0.5 + 0.9999919774953684 = 1.6249919774953683
#                   politics = 0.0625 + 0.25                     = 0.3125
#                   tech     = 0.7071067811865476
#                   total                                        = 2.6445987586819157
EXPECTED_U1_PROFILE = {
    "sports":   0.6144569085048177,     # 1.6249919774953683 / 2.6445987586819157
    "politics": 0.11816537347077631,    # 0.3125             / 2.6445987586819157
    "tech":     0.2673777180244061,     # 0.7071067811865476 / 2.6445987586819157
}


def _impl():
    """The module under test (see the module docstring for why this is lazy)."""
    return importlib.import_module("src.features.behavioural")


# ----------------------------------------------------------------------------------------------
# The oracle checks itself. These pass without any implementation.
# ----------------------------------------------------------------------------------------------
class TestOracleIsSound:
    def test_toy_log_contains_every_required_edge_case(self):
        log = toy_log()
        assert log.height == 20
        u1 = log.filter(pl.col("user_id") == "U1")
        assert (u1["ts"] < T).any(), "an event strictly before t"
        assert (u1["ts"] == T).any(), "an event at exactly t"
        assert (u1["ts"] > T).any(), "an event after t"
        assert not u1["ts"].is_sorted(), "an out-of-order event (row order != time order)"

    def test_hand_written_weights_follow_the_formula(self):
        """Guards the oracle against a typo: recompute each literal from the log's timestamps."""
        ts = dict(toy_log().filter(pl.col("user_id") == "U1").select("article_id", "ts").iter_rows())
        for article, expected in EXPECTED_U1_WEIGHTS.items():
            age = (T - ts[article]) / H
            assert expected == pytest.approx(2 ** -age, rel=1e-15), article
        assert sum(EXPECTED_U1_PROFILE.values()) == pytest.approx(1.0, rel=1e-15)


# ----------------------------------------------------------------------------------------------
# decay_weights — the core of the feature.
# ----------------------------------------------------------------------------------------------
class TestDecayWeights:
    def test_weights_match_hand_computation(self):
        w = _impl().decay_weights(toy_log(), "U1", T, H)
        assert w["article_id"].to_list() == list(EXPECTED_U1_WEIGHTS), "eligible rows, ts ascending"
        assert w["weight"].to_list() == pytest.approx(list(EXPECTED_U1_WEIGHTS.values()), rel=1e-12)

    def test_event_strictly_before_t_is_included_and_decayed(self):
        w = dict(_impl().decay_weights(toy_log(), "U1", T, H).select("article_id", "weight").iter_rows())
        assert "a07" in w
        assert w["a07"] == pytest.approx(0.9999919774953684, rel=1e-12)
        assert w["a07"] < 1.0, "decayed, not counted at full weight"

    def test_event_at_exactly_t_is_excluded(self):
        """The Q9 boundary: an event at t is part of the request being scored, not history."""
        w = _impl().decay_weights(toy_log(), "U1", T, H)
        assert "a06" not in w["article_id"].to_list()

    def test_event_after_t_is_excluded(self):
        w = _impl().decay_weights(toy_log(), "U1", T, H)
        assert "a03" not in w["article_id"].to_list()

    def test_other_users_events_are_excluded(self):
        w = _impl().decay_weights(toy_log(), "U1", T, H)
        assert set(w["article_id"].to_list()) <= {"a01", "a02", "a04", "a05", "a07", "a08"}

    def test_out_of_order_event_is_weighted_by_timestamp_not_row_order(self):
        """a08 is U1's oldest click but sits last in the log. Its weight must come from its ts
        (96 h -> 2^-4), and no reordering of the log may change any output. A position-based
        decay (A1's `recency_pool`) fails this."""
        impl = _impl()
        reference = impl.decay_weights(toy_log(), "U1", T, H)
        w = dict(reference.select("article_id", "weight").iter_rows())
        assert w["a08"] == pytest.approx(0.0625, rel=1e-12)
        for seed in range(5):
            shuffled = toy_log().sample(fraction=1.0, shuffle=True, seed=seed)
            assert impl.decay_weights(shuffled, "U1", T, H).equals(reference), f"seed {seed}"


# ----------------------------------------------------------------------------------------------
# recency_weighted_profile — the feature value for (user, candidate, t).
# ----------------------------------------------------------------------------------------------
class TestRecencyWeightedProfile:
    @pytest.mark.parametrize("category", sorted(EXPECTED_U1_PROFILE))
    def test_profile_matches_hand_computation(self, category):
        got = _impl().recency_weighted_profile(toy_log(), "U1", category, T, H)
        assert got == pytest.approx(EXPECTED_U1_PROFILE[category], rel=1e-12)

    def test_leaked_categories_score_exactly_zero(self):
        """entertainment appears for U1 only at exactly t; weather only after t (U2 reads
        both, before t). A leak through any of those three paths makes the score > 0."""
        impl = _impl()
        assert impl.recency_weighted_profile(toy_log(), "U1", "entertainment", T, H) == 0.0
        assert impl.recency_weighted_profile(toy_log(), "U1", "weather", T, H) == 0.0

    def test_no_eligible_history_is_nan_not_zero(self):
        """U3 has five events, all at or after t, so nothing is eligible."""
        got = _impl().recency_weighted_profile(toy_log(), "U3", "tech", T, H)
        assert math.isnan(got)


class TestInvalidInput:
    @pytest.mark.parametrize("half_life", [timedelta(0), timedelta(hours=-1)])
    def test_non_positive_half_life_is_rejected(self, half_life):
        with pytest.raises(ValueError):
            _impl().decay_weights(toy_log(), "U1", T, half_life)

    def test_null_timestamp_is_rejected(self):
        """A click with no time cannot be placed relative to t; keeping or dropping it
        silently is either leakage or data loss."""
        with pytest.raises(ValueError):
            _impl().decay_weights(_untime("a04"), "U1", T, H)


# ----------------------------------------------------------------------------------------------
# MIND fallback (P1-D1): history clicks have no timestamp; stamp them with the split start.
# ----------------------------------------------------------------------------------------------
SPLIT_START = _at(hours=-120)                # 5 half-lives before T -> weight 2^-5 = 0.03125


def _untime(*articles: str) -> pl.DataFrame:
    """The toy log with the given articles' ts set to null, as MIND history arrives."""
    return toy_log().with_columns(
        pl.when(pl.col("article_id").is_in(articles)).then(None).otherwise(pl.col("ts")).alias("ts"))


class TestMindUntimedFallback:
    def test_untimed_click_is_stamped_at_split_start_and_decayed(self):
        """a04 (24 h old, weight 0.5) loses its timestamp; stamped at T - 120 h it weighs
        2^-5 = 0.03125. Every other U1 weight is unchanged."""
        w = _impl().decay_weights(_untime("a04"), "U1", T, H, untimed_ts=SPLIT_START)
        got = {a: (ts, wt) for a, _, ts, wt in w.iter_rows()}
        assert got["a04"] == (SPLIT_START, pytest.approx(0.03125, rel=1e-12))
        for article in ("a08", "a01", "a02", "a05", "a07"):
            assert got[article][1] == pytest.approx(EXPECTED_U1_WEIGHTS[article], rel=1e-12)
        assert w["article_id"].to_list()[0] == "a04", "now the oldest event, so sorted first"

    @pytest.mark.parametrize("stamp", [T, _at(hours=+1)])
    def test_stamp_at_or_after_t_cannot_leak(self, stamp):
        """If the stamp is not strictly before t, the strict boundary still excludes the click:
        the fallback can lose a click, never leak one."""
        w = _impl().decay_weights(_untime("a04"), "U1", T, H, untimed_ts=stamp)
        assert "a04" not in w["article_id"].to_list()

    def test_timed_clicks_are_untouched_by_the_fallback(self):
        impl = _impl()
        assert impl.decay_weights(toy_log(), "U1", T, H, untimed_ts=SPLIT_START).equals(
            impl.decay_weights(toy_log(), "U1", T, H))

    @pytest.mark.parametrize("half_life", [timedelta(hours=6), timedelta(hours=24),
                                           timedelta(hours=72), timedelta(days=1000)])
    def test_history_only_profile_does_not_depend_on_half_life(self, half_life):
        """MIND's consequence, pinned down. All untimed history clicks share one stamp, hence one
        weight, and normalisation cancels it: the profile is the plain category distribution.
        U9 read sports, sports, tech, politics -> P(sports) = 2/4 = 0.5, P(tech) = 1/4 = 0.25."""
        log = pl.DataFrame(
            {"user_id": ["U9"] * 4, "article_id": ["d1", "d2", "d3", "d4"],
             "category": ["sports", "sports", "tech", "politics"], "ts": [None] * 4},
            schema={"user_id": pl.Utf8, "article_id": pl.Utf8, "category": pl.Utf8,
                    "ts": pl.Datetime("us")})
        impl = _impl()
        assert impl.recency_weighted_profile(log, "U9", "sports", T, half_life,
                                             untimed_ts=SPLIT_START) == pytest.approx(0.5, rel=1e-12)
        assert impl.recency_weighted_profile(log, "U9", "tech", T, half_life,
                                             untimed_ts=SPLIT_START) == pytest.approx(0.25, rel=1e-12)
