"""Ranking metrics for mode (b), in-impression re-ranking (SPEC.md §1).

Each metric is computed *per impression* and then averaged across impressions — not
pooled over all candidate rows. Pooling would let one impression with 300 candidates
outweigh fifty impressions with 5, and it is not what either leaderboard reports.

The MRR and nDCG definitions follow MIND's official `evaluate.py` so that offline numbers
are comparable to the leaderboard's. Note that MIND's MRR sums the reciprocal rank of
*every* relevant item and divides by their count, rather than taking only the first
relevant item. With MIND's near-universal single click the two coincide, but they differ
on multi-click impressions, and matching the official definition matters more than
matching the textbook one. `mrr_first_relevant` implements the textbook version for
comparison.
"""

from __future__ import annotations

import numpy as np


def _as_arrays(labels, scores) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=float)
    s = np.asarray(scores, dtype=float)
    if y.shape != s.shape:
        raise ValueError(f"labels {y.shape} and scores {s.shape} differ in length")
    return y, s


def dcg(labels, scores, k: int) -> float:
    """Discounted cumulative gain at k, with exponential gain 2^rel - 1."""
    y, s = _as_arrays(labels, scores)
    order = np.argsort(-s, kind="stable")[:k]
    gains = 2 ** y[order] - 1
    discounts = np.log2(np.arange(len(order)) + 2)
    return float(np.sum(gains / discounts))


def ndcg(labels, scores, k: int) -> float:
    """nDCG@k. Returns 0.0 for an impression with no relevant item (ideal DCG is 0)."""
    ideal = dcg(labels, labels, k)
    return dcg(labels, scores, k) / ideal if ideal > 0 else 0.0


def mrr(labels, scores) -> float:
    """MIND's MRR: mean reciprocal rank over all relevant items."""
    y, s = _as_arrays(labels, scores)
    total = y.sum()
    if total == 0:
        return 0.0
    order = np.argsort(-s, kind="stable")
    return float(np.sum(y[order] / (np.arange(len(y)) + 1)) / total)


def mrr_first_relevant(labels, scores) -> float:
    """Textbook MRR: reciprocal rank of the first relevant item only."""
    y, s = _as_arrays(labels, scores)
    order = np.argsort(-s, kind="stable")
    hits = np.flatnonzero(y[order] > 0)
    return float(1.0 / (hits[0] + 1)) if len(hits) else 0.0


def auc(labels, scores) -> float:
    """ROC AUC via the rank-sum identity, tie-aware.

    Hand-rolled rather than imported so the implementation is explainable, and
    cross-checked against sklearn.metrics.roc_auc_score in tests/test_metrics.py —
    sklearn is the oracle here, never the implementation (SPEC.md §7).

    Returns NaN when an impression is all-clicks or all-non-clicks: AUC is undefined
    there, and such impressions must be skipped rather than scored 0.5, which would
    quietly drag the mean toward chance.
    """
    y, s = _as_arrays(labels, scores)
    n_pos = int((y > 0).sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Average ranks so tied scores share rank mass, matching sklearn's tie handling.
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return float((ranks[y > 0].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def evaluate_impressions(rows) -> dict[str, float]:
    """Mean of each metric over impressions. `rows` yields (labels, scores) pairs.

    AUC ignores undefined impressions; the count that contributed is returned so the
    denominator is never silently different from the impression count.
    """
    aucs, mrrs, n5, n10 = [], [], [], []
    for labels, scores in rows:
        a = auc(labels, scores)
        if not np.isnan(a):
            aucs.append(a)
        mrrs.append(mrr(labels, scores))
        n5.append(ndcg(labels, scores, 5))
        n10.append(ndcg(labels, scores, 10))
    return {
        "auc": float(np.mean(aucs)) if aucs else float("nan"),
        "mrr": float(np.mean(mrrs)),
        "ndcg@5": float(np.mean(n5)),
        "ndcg@10": float(np.mean(n10)),
        "n_impressions": len(mrrs),
        "n_auc_defined": len(aucs),
    }
