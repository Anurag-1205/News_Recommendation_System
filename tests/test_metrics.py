"""Oracles for the ranking metrics (SPEC.md §5).

Two independent kinds of check:
  1. Hand-computed values, worked out on paper, for nDCG / MRR / AUC.
  2. Agreement with sklearn.metrics.roc_auc_score, which is the oracle for AUC only —
     never the implementation (SPEC.md §7).
"""

import math

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.eval.metrics import auc, dcg, evaluate_impressions, mrr, mrr_first_relevant, ndcg


class TestNDCGHandComputed:
    def test_graded_worked_example(self):
        """Labels 2,0,1,0,2 in the given order, scores already descending.

        DCG = (2^2-1)/log2(2) + 0 + (2^1-1)/log2(4) + 0 + (2^2-1)/log2(6)
            = 3/1 + 1/2 + 3/2.58496 = 3 + 0.5 + 1.16054 = 4.66054
        Ideal order is 2,2,1,0,0:
        IDCG = 3/1 + 3/1.58496 + 1/2 = 3 + 1.89279 + 0.5 = 5.39279
        nDCG@5 = 4.66054 / 5.39279 = 0.86423
        """
        labels = [2, 0, 1, 0, 2]
        scores = [5.0, 4.0, 3.0, 2.0, 1.0]
        # Gains are 2^rel - 1, so the rel=1 item contributes 1, not 3. Writing 3 here was the
        # first version's bug: the assertion failed, the implementation was right.
        expected_dcg = 3 / math.log2(2) + 1 / math.log2(4) + 3 / math.log2(6)
        expected_idcg = 3 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)
        assert dcg(labels, scores, 5) == pytest.approx(expected_dcg, abs=1e-9)
        assert dcg(labels, labels, 5) == pytest.approx(expected_idcg, abs=1e-9)
        assert ndcg(labels, scores, 5) == pytest.approx(expected_dcg / expected_idcg, abs=1e-9)
        assert ndcg(labels, scores, 5) == pytest.approx(0.8642, abs=1e-4)

    def test_perfect_ranking_is_one(self):
        assert ndcg([1, 0, 0], [9.0, 1.0, 0.0], 5) == pytest.approx(1.0)

    def test_worst_ranking_binary(self):
        """Single relevant item pushed to position 3: DCG = 1/log2(4) = 0.5, IDCG = 1."""
        assert ndcg([1, 0, 0], [0.0, 1.0, 2.0], 5) == pytest.approx(0.5)

    def test_k_truncates(self):
        """Relevant item at position 3 falls outside k=2, so nDCG@2 is 0."""
        assert ndcg([0, 0, 1], [3.0, 2.0, 1.0], 2) == pytest.approx(0.0)

    def test_no_relevant_item_is_zero_not_nan(self):
        assert ndcg([0, 0, 0], [1.0, 2.0, 3.0], 5) == 0.0


class TestMRRHandComputed:
    def test_first_relevant_at_rank_two(self):
        assert mrr([0, 1, 0], [3.0, 2.0, 1.0]) == pytest.approx(0.5)

    def test_relevant_at_rank_one(self):
        assert mrr([1, 0, 0], [3.0, 2.0, 1.0]) == pytest.approx(1.0)

    def test_multi_click_averages_all_relevant(self):
        """Relevant at ranks 1 and 3: MIND's MRR = (1/1 + 1/3)/2 = 0.6667."""
        labels, scores = [1, 0, 1], [3.0, 2.0, 1.0]
        assert mrr(labels, scores) == pytest.approx((1 + 1 / 3) / 2)

    def test_textbook_mrr_differs_on_multi_click(self):
        """The distinction the docstring warns about, pinned by a test."""
        labels, scores = [1, 0, 1], [3.0, 2.0, 1.0]
        assert mrr_first_relevant(labels, scores) == pytest.approx(1.0)
        assert mrr(labels, scores) != pytest.approx(mrr_first_relevant(labels, scores))

    def test_no_relevant_item(self):
        assert mrr([0, 0], [1.0, 2.0]) == 0.0


class TestAUCAgainstSklearn:
    @pytest.mark.parametrize("seed", range(25))
    def test_matches_sklearn_on_random_impressions(self, seed):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(4, 60))
        labels = rng.integers(0, 2, n)
        if labels.sum() in (0, n):          # AUC undefined; covered separately
            labels[0], labels[-1] = 0, 1
        scores = rng.normal(size=n)
        assert auc(labels, scores) == pytest.approx(roc_auc_score(labels, scores), abs=1e-9)

    def test_matches_sklearn_with_heavy_ties(self):
        """Ties are where a hand-rolled AUC most often diverges from sklearn."""
        labels = [1, 0, 1, 0, 1, 0]
        scores = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        assert auc(labels, scores) == pytest.approx(roc_auc_score(labels, scores), abs=1e-12)

    def test_hand_computed_perfect_and_inverted(self):
        assert auc([1, 0], [1.0, 0.0]) == pytest.approx(1.0)
        assert auc([1, 0], [0.0, 1.0]) == pytest.approx(0.0)
        assert auc([1, 0], [0.5, 0.5]) == pytest.approx(0.5)

    def test_undefined_returns_nan(self):
        assert math.isnan(auc([1, 1, 1], [1.0, 2.0, 3.0]))
        assert math.isnan(auc([0, 0], [1.0, 2.0]))


class TestEvaluateImpressions:
    def test_skips_auc_undefined_impressions(self):
        rows = [([1, 0], [1.0, 0.0]), ([1, 1], [1.0, 2.0])]
        out = evaluate_impressions(rows)
        assert out["n_impressions"] == 2
        assert out["n_auc_defined"] == 1
        assert out["auc"] == pytest.approx(1.0)

    def test_random_scores_give_auc_near_half(self):
        """The harness sanity check from SPEC.md §5: a random scorer must land near 0.5."""
        rng = np.random.default_rng(0)
        rows = []
        for _ in range(3000):
            n = int(rng.integers(5, 30))
            labels = np.zeros(n, dtype=int)
            labels[rng.integers(0, n)] = 1
            rows.append((labels, rng.normal(size=n)))
        assert evaluate_impressions(rows)["auc"] == pytest.approx(0.5, abs=0.02)
