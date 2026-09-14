"""Oracles for the Q5 slice definitions (SPEC.md §17)."""

import pytest

from src.eval.slices import COLD_MAX_HISTORY, HEAD_FRACTION, head_articles, is_cold


class TestHeadArticles:
    def test_takes_the_top_quintile_of_clicked_articles(self):
        counts = {f"a{i}": i for i in range(1, 11)}          # a10 most clicked ... a1 least
        assert head_articles(counts) == {"a10", "a9"}         # round(10 * 0.2) = 2

    def test_articles_with_no_training_click_are_tail(self):
        """They are absent from click_counts, so they can never be head — that is the point of
        the slice: does the model rank items it has never seen clicked?"""
        assert "unseen" not in head_articles({"a": 5, "b": 1})

    def test_ties_are_broken_deterministically(self):
        counts = {"b": 7, "a": 7, "c": 7, "d": 7, "e": 1}
        assert head_articles(counts) == head_articles(dict(reversed(list(counts.items())))) == {"a"}

    def test_at_least_one_head_article_when_anything_was_clicked(self):
        assert head_articles({"only": 3}) == {"only"}

    def test_empty_counts_give_an_empty_head_set(self):
        assert head_articles({}) == set() and head_articles({"a": 1}, fraction=0) == set()

    def test_fraction_default_is_the_quintile(self):
        assert HEAD_FRACTION == 0.2


class TestCold:
    @pytest.mark.parametrize("n,cold", [(0, True), (5, True), (6, False), (50, False)])
    def test_boundary_is_inclusive(self, n, cold):
        assert is_cold(n) is cold

    def test_threshold_matches_a1(self):
        assert COLD_MAX_HISTORY == 5
