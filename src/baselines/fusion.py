"""Score fusion: combining a lexical ranking with a popularity prior.

BM25 and click counts live on incomparable scales — a BM25 score of 12 and a click count of
12 mean nothing alike — so they cannot simply be added. Two standard ways to reconcile that,
both implemented so the choice can be made by measurement:

**Reciprocal rank fusion (RRF)** throws the scores away and keeps only the ranks:

    rrf(d) = sum over rankers of  1 / (k + rank(d))

Scale-free by construction, which is exactly why it is robust when one input is a sparse
integer count and the other a continuous score. k=60 damps the influence of the top rank so
one confident ranker cannot dominate; it is the value from the original Cormack et al. paper
and is left as a parameter rather than a constant.

**Normalised weighted sum** min-max scales each score *within the impression* and takes a
weighted average. It keeps score magnitudes — the margin between first and second place —
which RRF discards. That is an advantage when the margins are meaningful and a liability
when one ranker is poorly calibrated.
"""

from __future__ import annotations


def _ranks(scores: list[float]) -> list[int]:
    """1-based ranks, highest score first, ties broken by position for determinism."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    ranks = [0] * len(scores)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks


def rrf(score_lists: list[list[float]], k: int = 60) -> list[float]:
    """Reciprocal rank fusion over several score lists covering the same candidates."""
    if not score_lists:
        return []
    n = len(score_lists[0])
    if any(len(s) != n for s in score_lists):
        raise ValueError("all score lists must cover the same candidates")
    fused = [0.0] * n
    for scores in score_lists:
        for i, rank in enumerate(_ranks(scores)):
            fused[i] += 1.0 / (k + rank)
    return fused


def _minmax(scores: list[float]) -> list[float]:
    """Scale to [0,1] within this impression. All-equal input maps to 0.0, not 0.5.

    Mapping a flat input to zero is deliberate: a ranker with no opinion should contribute
    nothing to the sum rather than a constant that shifts every candidate equally.
    """
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.0] * len(scores)
    span = hi - lo
    return [(s - lo) / span for s in scores]


def weighted_sum(score_lists: list[list[float]], weights: list[float]) -> list[float]:
    """Min-max normalise each ranker within the impression, then weight and sum."""
    if len(score_lists) != len(weights):
        raise ValueError("need one weight per score list")
    if not score_lists:
        return []
    n = len(score_lists[0])
    fused = [0.0] * n
    for scores, w in zip(score_lists, weights):
        for i, s in enumerate(_minmax(scores)):
            fused[i] += w * s
    return fused
