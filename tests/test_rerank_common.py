"""Oracles for src/rerank/common.py: the feature-exclusion guarantee and the paired bootstrap."""

import math

import numpy as np
import polars as pl
import pytest

from src.features.behavioural import SERVING_OK
from src.rerank.common import impression_rows, model_features, paired_delta


class TestModelFeatures:
    ALL = ["bm25", "session_pos", "session_len", "cur_read_time", "cur_scroll_percentage",
           "n_prior_clicks_in_session", "freshness_hours"]

    def test_submission_model_drops_unsafe_and_test_file_absent(self):
        assert model_features(self.ALL) == ["bm25", "session_pos", "freshness_hours"]

    def test_non_submission_model_still_drops_unsafe(self):
        assert model_features(self.ALL, for_submission=False) == [
            "bm25", "session_pos", "n_prior_clicks_in_session", "freshness_hours"]

    def test_no_unsafe_feature_survives_from_the_full_registry(self):
        kept = model_features(list(SERVING_OK), for_submission=False)
        assert all(SERVING_OK[f] for f in kept)


class TestPairedDelta:
    def test_system_against_itself_is_exactly_zero(self):
        a = np.array([0.5, 0.6, 0.7, 0.9])
        ci = paired_delta(a, a)
        assert (ci.mean, ci.lo, ci.hi) == (0.0, 0.0, 0.0)

    def test_constant_shift_is_recovered_with_zero_width(self):
        """Every impression improves by exactly 0.1: the paired CI must be the point 0.1. An
        unpaired comparison of these two lists would show wide, overlapping intervals."""
        a = np.array([0.2, 0.5, 0.9, 0.4, 0.6])
        ci = paired_delta(a, a + 0.1)
        assert ci.mean == pytest.approx(0.1) and ci.lo == pytest.approx(0.1) and ci.hi == pytest.approx(0.1)

    def test_nan_pairs_are_dropped(self):
        ci = paired_delta([0.5, math.nan, 0.7], [0.6, 0.9, math.nan])
        assert ci.n == 1 and ci.mean == pytest.approx(0.1)

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError):
            paired_delta([0.1, 0.2], [0.1])


class TestLambdarank:
    """D2 (CONTEXT.md C-016): LightGBM lambdarank, grouped by impression."""

    def test_group_sizes_are_contiguous_runs_in_frame_order(self):
        from src.rerank.common import group_sizes
        frame = pl.DataFrame({"imp_row": [3, 3, 7, 7, 7, 1]})
        assert group_sizes(frame).tolist() == [2, 3, 1]

    def test_interleaved_impressions_are_rejected(self):
        """LightGBM reads groups as consecutive row counts; an impression split across two runs
        would silently become two queries."""
        from src.rerank.common import group_sizes
        with pytest.raises(ValueError):
            group_sizes(pl.DataFrame({"imp_row": [3, 7, 3]}))

    @staticmethod
    def _toy(seed: int = 0):
        """400 impressions x 5 candidates. x0 is constant within an impression (separates
        impressions, cannot order candidates); x1 decides the click: the max-x1 candidate."""
        rng = np.random.default_rng(seed)
        g, n = 400, 5
        x0 = np.repeat(rng.normal(size=g), n)
        x1 = rng.normal(size=g * n)
        y = np.zeros(g * n, dtype=np.int8)
        for i in range(g):
            y[i * n + int(np.argmax(x1[i * n:(i + 1) * n]))] = 1
        return np.column_stack([x0, x1]), y, np.full(g, n)

    def test_learns_the_within_impression_signal(self):
        from src.rerank.common import fit_lambdarank
        X, y, groups = self._toy()
        s = fit_lambdarank(X, y, groups).predict(X)
        top_is_click = [y[i * 5 + int(np.argmax(s[i * 5:(i + 1) * 5]))] == 1 for i in range(len(groups))]
        assert np.mean(top_is_click) >= 0.95

    def test_is_deterministic(self):
        from src.rerank.common import fit_lambdarank
        X, y, groups = self._toy()
        assert np.array_equal(fit_lambdarank(X, y, groups).predict(X), fit_lambdarank(X, y, groups).predict(X))

    def test_parameter_override_is_applied_and_stays_deterministic(self):
        """C-016 option (c): the truncation level must reach LightGBM, and a refit with it must
        still be bit-identical."""
        from src.rerank.common import LAMBDARANK_PARAMS, fit_lambdarank
        X, y, groups = self._toy()
        m1 = fit_lambdarank(X, y, groups, lambdarank_truncation_level=300)
        m2 = fit_lambdarank(X, y, groups, lambdarank_truncation_level=300)
        assert m1.get_params()["lambdarank_truncation_level"] == 300
        assert "lambdarank_truncation_level" not in LAMBDARANK_PARAMS, "the default must stay LightGBM's own"
        assert np.array_equal(m1.predict(X), m2.predict(X))

    def test_truncation_level_actually_changes_training(self):
        """get_params() would echo back even a parameter LightGBM silently ignored (verbose=-1
        hides the warning). A different fitted model proves LightGBM consumes it."""
        from src.rerank.common import fit_lambdarank
        X, y, groups = self._toy()
        p1 = fit_lambdarank(X, y, groups, lambdarank_truncation_level=1).predict(X)
        p5 = fit_lambdarank(X, y, groups, lambdarank_truncation_level=5).predict(X)
        assert not np.array_equal(p1, p5)

    def test_group_sizes_must_cover_every_row(self):
        from src.rerank.common import fit_lambdarank
        X, y, groups = self._toy()
        with pytest.raises(ValueError):
            fit_lambdarank(X, y, groups[:-1])


class TestDropTrigger:
    """The rule that decides whether the leave-one-family-out ablation runs, fixed in code."""

    @staticmethod
    def _ci(mean, lo, hi):
        from src.eval.bootstrap import CI
        return CI(mean, lo, hi, 100, 1000)

    def test_negative_auc_point_estimate_triggers(self):
        from src.rerank.common import drop_reasons
        d = {"auc": self._ci(-0.001, -0.003, 0.001), "mrr": self._ci(0.001, -0.001, 0.003)}
        assert drop_reasons(d) == ["auc point estimate < 0"]

    def test_any_metric_significantly_below_zero_triggers(self):
        from src.rerank.common import drop_reasons
        d = {"auc": self._ci(0.001, -0.001, 0.003), "ndcg@10": self._ci(-0.002, -0.004, -0.001)}
        assert drop_reasons(d) == ["ndcg@10 CI entirely below 0"]

    def test_no_drop_is_empty(self):
        from src.rerank.common import drop_reasons
        d = {"auc": self._ci(0.002, 0.001, 0.003), "mrr": self._ci(0.0, -0.001, 0.001)}
        assert drop_reasons(d) == []


def test_impression_rows_groups_in_order_and_zero_fills_missing_scores():
    frame = pl.DataFrame({"imp_row": [7, 7, 3, 3, 3], "label": [1, 0, 0, 1, 0],
                          "s": [0.9, float("nan"), None, 0.2, 0.1]})
    assert impression_rows(frame, "s") == [([1, 0], [0.9, 0.0]), ([0, 1, 0], [0.0, 0.2, 0.1])]
