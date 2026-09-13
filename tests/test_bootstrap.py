"""Oracles for the bootstrap CI (SPEC.md §5)."""

import numpy as np
import pytest

from src.eval.bootstrap import CI, bootstrap_ci, bootstrap_metrics, compare


class TestBootstrapCI:
    def test_mean_is_the_sample_mean_not_a_resample_mean(self):
        """The point estimate must be the data's own mean, not an average of replicates."""
        vals = [0.1, 0.5, 0.9]
        assert bootstrap_ci(vals, iterations=200).mean == pytest.approx(0.5)

    def test_interval_brackets_the_mean(self):
        ci = bootstrap_ci(list(np.random.default_rng(0).normal(0.6, 0.1, 500)), iterations=500)
        assert ci.lo < ci.mean < ci.hi

    def test_constant_data_gives_zero_width(self):
        ci = bootstrap_ci([0.7] * 100, iterations=200)
        assert ci.lo == pytest.approx(0.7) and ci.hi == pytest.approx(0.7)

    def test_interval_narrows_with_more_data(self):
        """The defining property of a CI: more evidence, tighter bound."""
        rng = np.random.default_rng(1)
        narrow = bootstrap_ci(list(rng.normal(0.5, 0.2, 5000)), iterations=400)
        wide = bootstrap_ci(list(rng.normal(0.5, 0.2, 50)), iterations=400)
        assert (narrow.hi - narrow.lo) < (wide.hi - wide.lo)

    def test_covers_true_mean_for_a_known_distribution(self):
        """Sanity check against a known answer: a 95% CI on 2000 draws from N(0.5, 0.1)
        should contain 0.5."""
        ci = bootstrap_ci(list(np.random.default_rng(3).normal(0.5, 0.1, 2000)), iterations=1000)
        assert ci.lo < 0.5 < ci.hi

    def test_wider_confidence_gives_wider_interval(self):
        vals = list(np.random.default_rng(4).normal(0.5, 0.2, 300))
        c95 = bootstrap_ci(vals, iterations=500, confidence=0.95)
        c99 = bootstrap_ci(vals, iterations=500, confidence=0.99)
        assert (c99.hi - c99.lo) > (c95.hi - c95.lo)

    def test_is_deterministic_for_a_fixed_seed(self):
        vals = list(np.random.default_rng(5).normal(0.5, 0.1, 200))
        assert bootstrap_ci(vals, seed=42) == bootstrap_ci(vals, seed=42)

    def test_different_seeds_differ_slightly(self):
        vals = list(np.random.default_rng(6).normal(0.5, 0.1, 200))
        assert bootstrap_ci(vals, seed=1).lo != bootstrap_ci(vals, seed=2).lo

    def test_nan_values_are_dropped(self):
        """AUC is NaN for all-click impressions; those must not poison a replicate."""
        ci = bootstrap_ci([0.5, float("nan"), 0.5, float("nan")], iterations=100)
        assert ci.n == 2 and ci.mean == pytest.approx(0.5)

    def test_empty_and_single_inputs(self):
        assert np.isnan(bootstrap_ci([], iterations=10).mean)
        single = bootstrap_ci([0.3], iterations=10)
        assert single.mean == single.lo == single.hi == 0.3


class TestOverlapAndComparison:
    def test_overlapping_intervals_detected(self):
        a, b = CI(0.50, 0.48, 0.52, 100, 1000), CI(0.51, 0.49, 0.53, 100, 1000)
        assert a.overlaps(b) and b.overlaps(a)

    def test_disjoint_intervals_detected(self):
        a, b = CI(0.50, 0.48, 0.52, 100, 1000), CI(0.70, 0.68, 0.72, 100, 1000)
        assert not a.overlaps(b)

    def test_touching_intervals_count_as_overlapping(self):
        """Conservative on the boundary — a shared endpoint is not separation."""
        assert CI(0.5, 0.4, 0.6, 10, 10).overlaps(CI(0.7, 0.6, 0.8, 10, 10))

    def test_compare_refuses_beats_on_overlap(self):
        msg = compare(CI(0.50, 0.48, 0.52, 100, 1000), CI(0.51, 0.49, 0.53, 100, 1000))
        assert "OVERLAP" in msg and "beats" not in msg

    def test_compare_allows_beats_when_disjoint(self):
        msg = compare(CI(0.70, 0.68, 0.72, 100, 1000), CI(0.50, 0.48, 0.52, 100, 1000),
                      "bm25", "popularity")
        assert "bm25 beats popularity" in msg

    def test_str_format(self):
        assert str(CI(0.5451, 0.5402, 0.5500, 73152, 1000)) == "0.5451 [0.5402, 0.5500]"


def test_bootstrap_metrics_handles_several_at_once():
    out = bootstrap_metrics({"auc": [0.5, 0.6, 0.7], "mrr": [0.2, 0.3, 0.4]}, iterations=100)
    assert set(out) == {"auc", "mrr"}
    assert out["auc"].mean == pytest.approx(0.6)


def test_bootstrap_ci_blocked_draw_is_identical_to_the_a1_one_shot_draw():
    """P3.4a: `bootstrap_ci` now draws its resample indices in blocks of 100 iterations, because the
    one-shot (iterations x n) int64 array is 1.96 GB at EB-NeRD's 244,647 impressions and the OOM
    killer took the laptop session down (CONTEXT.md C-026). Every recorded CI in RESULTS.md came
    from the one-shot draw, so the blocked draw must reproduce it bit for bit."""
    import numpy as np
    from src.eval.bootstrap import bootstrap_ci
    values = np.random.default_rng(11).random(3_001)
    got = bootstrap_ci(values, iterations=1_000, seed=0)
    rng = np.random.default_rng(0)                                   # A1's one-shot form, verbatim
    idx = rng.integers(0, len(values), size=(1_000, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    assert (got.mean, got.lo, got.hi) == (float(values.mean()), float(lo), float(hi))
