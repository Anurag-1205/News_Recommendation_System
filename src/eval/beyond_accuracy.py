"""Beyond-accuracy metrics: diversity, novelty, coverage (A1 Q4.2).

These three terms are used loosely in the literature, so the definitions actually used are
stated here and referenced from RESULTS.md. A number is meaningless without them.

**Intra-list diversity** — the fraction of adjacent-pair comparisons in a top-k list whose
two items differ in category, averaged over impressions. 0 means every recommendation shares
one category; 1 means no two items in the list share a category. Category is used rather
than embedding distance so the metric stays defined for the lexical models too, which have
no vector space of their own.

**Novelty** — mean self-information of the recommended items, `-log2(p(item))`, where
`p(item)` is the item's share of training clicks. Popular items carry little information and
score low; obscure ones score high. An item never seen in training is assigned the novelty
of a single-click item rather than infinity, which would make one unknown item dominate the
mean.

**Coverage** — the fraction of the catalogue that appears in *any* impression's top-k. This
is a property of the whole run, not of one impression, so it is accumulated across the pass
and reported once.
"""

from __future__ import annotations

import math
from collections import Counter


def intra_list_diversity(categories: list) -> float:
    """Fraction of distinct-category pairs among all pairs in one recommendation list.

    Returns 0.0 for a list of fewer than two items: diversity of a single item is undefined,
    and 0 is the conservative reading (no variety demonstrated).
    """
    n = len(categories)
    if n < 2:
        return 0.0
    different = sum(
        1 for i in range(n) for j in range(i + 1, n) if categories[i] != categories[j]
    )
    return different / (n * (n - 1) / 2)


class NoveltyModel:
    """Self-information novelty from a training click distribution."""

    def __init__(self, click_counts: dict) -> None:
        total = sum(click_counts.values())
        self.total = total
        # Unknown items are treated as if seen once. Assigning them infinite novelty would
        # let a single never-seen article dominate the mean and make the metric useless.
        self.unseen = -math.log2(1 / total) if total else 0.0
        self.novelty = {
            item: -math.log2(count / total) for item, count in click_counts.items()
        } if total else {}

    def of(self, item) -> float:
        return self.novelty.get(item, self.unseen)

    def mean(self, items: list) -> float:
        return sum(self.of(i) for i in items) / len(items) if items else 0.0


class CoverageTracker:
    """Accumulates which catalogue items were ever recommended."""

    def __init__(self, catalogue_size: int) -> None:
        self.catalogue_size = catalogue_size
        self.seen: set = set()
        self.counts: Counter = Counter()

    def observe(self, items: list) -> None:
        self.seen.update(items)
        self.counts.update(items)

    @property
    def coverage(self) -> float:
        return len(self.seen) / self.catalogue_size if self.catalogue_size else 0.0

    def gini(self) -> float:
        """Concentration of recommendations across items, 0 = uniform, 1 = all on one item.

        Reported alongside coverage because they answer different questions: coverage says
        how much of the catalogue was touched at all, Gini says whether the exposure was
        spread evenly or piled onto a handful of articles.
        """
        if not self.counts:
            return 0.0
        values = sorted(self.counts.values())
        n = len(values)
        cumulative = sum((i + 1) * v for i, v in enumerate(values))
        return (2 * cumulative) / (n * sum(values)) - (n + 1) / n


def top_k_items(candidates: list, scores: list[float], k: int) -> list:
    """The k highest-scoring candidates, ties broken by position for determinism."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
    return [candidates[i] for i in order]
