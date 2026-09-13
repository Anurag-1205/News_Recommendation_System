"""Oracles for the freshness inputs of the NRMS variant (SPEC.md §15.2, §15.4).

The feature is §11.8's `freshness_hours`, so the tests are: a hand-computed transform on a toy
split, parity with the reranker's own column on real rows, the no-leakage rule re-asserted for
this consumer (Q9), and alignment of the list form with the slate.
"""
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.baselines.nrms_fresh_features import FreshStats, as_lists, fresh_from_frames, fresh_inputs, fit_stats

T0 = datetime(2023, 5, 24, 12, 0, 0)
H = timedelta(hours=1)

# toy split: 2 impressions, 3 + 2 candidates; article 9 is unknown (no first-known row)
BEH = pl.DataFrame({
    "imp_row": pl.Series([0, 1], dtype=pl.UInt32),
    "t": [T0, T0 + 2 * H],
    "article_ids_inview": pl.Series([[5, 6, 9], [6, 7]], dtype=pl.List(pl.Int32)),
})
FIRST_KNOWN = pl.DataFrame({"article_id": pl.Series([5, 6, 7], dtype=pl.Int32),
                            "ts": [T0 - 3 * H, T0 - 1 * H, T0 + 1 * H]})   # 7 is published after imp 0, before imp 1


def test_hand_computed_transform_and_unknown():
    stats = FreshStats(mu=1.0, sigma=0.5)
    out = fresh_from_frames(BEH, FIRST_KNOWN, stats)
    # imp 0 at T0: 5 -> 3 h, 6 -> 1 h, 9 -> unknown; imp 1 at T0+2h: 6 -> 3 h, 7 -> 1 h
    hours = out["freshness_hours"].to_list()
    assert hours[0] == pytest.approx(3.0) and hours[1] == pytest.approx(1.0) and math.isnan(hours[2])
    assert hours[3] == pytest.approx(3.0) and hours[4] == pytest.approx(1.0)
    x = out["x"].to_list(); unk = out["unknown"].to_list()
    assert x[0] == pytest.approx((math.log1p(3.0) - 1.0) / 0.5)
    assert x[1] == pytest.approx((math.log1p(1.0) - 1.0) / 0.5)
    assert (x[2], unk[2]) == (0.0, 1)
    assert unk == [0, 0, 1, 0, 0]
    assert out.columns == ["imp_row", "cand_position", "freshness_hours", "x", "unknown"]
    assert out["cand_position"].to_list() == [1, 2, 3, 1, 2]


def test_stats_fit_on_known_rows_only():
    stats = fit_stats(fresh_from_frames(BEH, FIRST_KNOWN, FreshStats(0.0, 1.0)))
    vals = np.log1p(np.array([3.0, 1.0, 3.0, 1.0]))
    assert stats.mu == pytest.approx(vals.mean()) and stats.sigma == pytest.approx(vals.std())


def test_no_leakage_from_first_known_at_or_after_t():
    """Q9: a publish time / sighting at or after t must not change any input."""
    before = fresh_from_frames(BEH, FIRST_KNOWN, FreshStats(1.0, 0.5))
    future = pl.concat([FIRST_KNOWN, pl.DataFrame({"article_id": pl.Series([9, 5], dtype=pl.Int32),
                                                    "ts": [T0, T0 + 5 * H]})])      # 9 exactly at t; 5 later again
    after = fresh_from_frames(BEH, future, FreshStats(1.0, 0.5))
    assert before.equals(after)


def test_as_lists_aligns_with_inview_position_by_position():
    out = fresh_from_frames(BEH, FIRST_KNOWN, FreshStats(1.0, 0.5))
    lists = as_lists(BEH, out)
    assert lists.columns == ["imp_row", "fresh_inview"]
    assert lists["fresh_inview"].list.len().to_list() == BEH["article_ids_inview"].list.len().to_list()
    row0 = lists["fresh_inview"][0].to_list()
    assert row0[2] == [0.0, 1.0]                      # unknown article 9 at slot 3
    assert row0[0][0] == pytest.approx((math.log1p(3.0) - 1.0) / 0.5)


SMALL = Path("data/interim/ebnerd/ebnerd_small")
needs_data = pytest.mark.skipif(not (SMALL / "validation" / "behaviors.parquet").exists(), reason="run make data")


@needs_data
def test_parity_with_reranker_freshness_on_real_rows():
    from src.rerank.ebnerd import add_phase1_features, candidate_frame, load_articles, load_behaviors, load_history
    from datetime import timedelta
    out = fresh_inputs("ebnerd", "ebnerd_small/validation", FreshStats(0.0, 1.0), limit=300)
    beh = load_behaviors(SMALL / "validation" / "behaviors.parquet", limit=300)
    arts = load_articles()
    long = candidate_frame(beh, arts)
    ref = add_phase1_features(long, beh, load_history(SMALL / "validation" / "history.parquet", arts, beh["user_id"]),
                              arts, timedelta.max)                       # h = inf, C-014
    j = out.join(ref.select("imp_row", "cand_position", ref_h=pl.col("freshness_hours")), on=["imp_row", "cand_position"], how="inner")
    assert j.height == ref.height
    a, b = j["freshness_hours"].to_numpy(), j["ref_h"].to_numpy()
    assert np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)])
