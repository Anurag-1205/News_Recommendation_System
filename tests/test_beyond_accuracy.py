"""Oracles for diversity, novelty and coverage (SPEC.md §5).

Each value below is hand-computed from the stated definition, so a change to the formula
fails loudly rather than shifting a number nobody checks.
"""

import math

import pytest

from src.eval.beyond_accuracy import (
    CoverageTracker,
    NoveltyModel,
    intra_list_diversity,
    top_k_items,
)


class TestIntraListDiversity:
    def test_all_same_category_is_zero(self):
        assert intra_list_diversity(["sport", "sport", "sport"]) == 0.0

    def test_all_distinct_is_one(self):
        assert intra_list_diversity(["sport", "krimi", "økonomi"]) == 1.0

    def test_hand_computed_mixed_case(self):
        """[a,a,b]: pairs (0,1) same, (0,2) differ, (1,2) differ -> 2/3."""
        assert intra_list_diversity(["a", "a", "b"]) == pytest.approx(2 / 3)

    def test_four_items_two_categories(self):
        """[a,a,b,b]: 6 pairs, the 4 cross-category ones differ -> 4/6."""
        assert intra_list_diversity(["a", "a", "b", "b"]) == pytest.approx(4 / 6)

    def test_single_item_and_empty_are_zero(self):
        assert intra_list_diversity(["a"]) == 0.0
        assert intra_list_diversity([]) == 0.0


class TestNovelty:
    def test_hand_computed_self_information(self):
        """Counts a=75, b=25, total=100. novelty(a) = -log2(0.75), novelty(b) = -log2(0.25)=2."""
        nm = NoveltyModel({"a": 75, "b": 25})
        assert nm.of("a") == pytest.approx(-math.log2(0.75))
        assert nm.of("b") == pytest.approx(2.0)

    def test_rarer_item_is_more_novel(self):
        nm = NoveltyModel({"common": 900, "rare": 1})
        assert nm.of("rare") > nm.of("common")

    def test_unseen_item_is_finite_and_maximal(self):
        """An unseen item must not be infinite, or one of them dominates every mean."""
        nm = NoveltyModel({"a": 10, "b": 90})
        assert math.isfinite(nm.of("never_seen"))
        assert nm.of("never_seen") >= max(nm.of("a"), nm.of("b"))

    def test_mean_over_a_list(self):
        nm = NoveltyModel({"a": 50, "b": 50})
        assert nm.mean(["a", "b"]) == pytest.approx(1.0)   # both -log2(0.5) = 1

    def test_empty_inputs(self):
        assert NoveltyModel({}).mean([]) == 0.0
        assert NoveltyModel({"a": 1}).mean([]) == 0.0


class TestCoverage:
    def test_fraction_of_catalogue_touched(self):
        t = CoverageTracker(catalogue_size=10)
        t.observe(["a", "b"])
        t.observe(["b", "c"])
        assert t.coverage == pytest.approx(3 / 10)

    def test_repeated_items_counted_once_for_coverage(self):
        t = CoverageTracker(catalogue_size=4)
        for _ in range(50):
            t.observe(["a"])
        assert t.coverage == pytest.approx(1 / 4)

    def test_empty_catalogue_does_not_divide_by_zero(self):
        assert CoverageTracker(0).coverage == 0.0

    def test_gini_is_zero_for_uniform_exposure(self):
        t = CoverageTracker(3)
        t.observe(["a", "b", "c"])
        assert t.gini() == pytest.approx(0.0, abs=1e-12)

    def test_gini_rises_when_exposure_concentrates(self):
        flat = CoverageTracker(2); flat.observe(["a", "b"])
        skew = CoverageTracker(2)
        skew.observe(["a"] * 99 + ["b"])
        assert skew.gini() > flat.gini()


class TestTopKItems:
    def test_returns_highest_scoring_in_order(self):
        assert top_k_items(["a", "b", "c"], [0.1, 0.9, 0.5], 2) == ["b", "c"]

    def test_k_larger_than_list(self):
        assert top_k_items(["a", "b"], [1.0, 2.0], 10) == ["b", "a"]

    def test_ties_break_by_position_deterministically(self):
        assert top_k_items(["a", "b", "c"], [1.0, 1.0, 1.0], 2) == ["a", "b"]
