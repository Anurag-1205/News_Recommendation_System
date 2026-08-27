"""Point-in-time behavioural features: popularity and CTR, strictly before an impression.

The difference between this and a precomputed count is the whole Q9 argument, and it is
worth stating precisely.

**Frozen popularity** counts every click in the training window once, then applies that
number to every impression. It does two wrong things at once: an impression early in the
window is scored using clicks that happened *after* it (leakage), and an impression late in
the evaluation window is scored using counts that have gone stale (drift).

**Rolling popularity** counts, for each impression at time *t*, only the clicks strictly
before *t* — and optionally only those within a trailing window. It is leakage-safe by
construction, and it tracks the news cycle instead of averaging over it.

Implementation: click timestamps are sorted per article once, then each query is a binary
search. Building the whole structure is O(n log n); every subsequent lookup is O(log n),
which is what makes per-impression recomputation affordable over millions of rows.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from datetime import timedelta

import numpy as np


class RollingCounts:
    """Per-article sorted event timestamps, queryable as-of any time.

    `clicks` and `impressions` are tracked separately so CTR can be formed from both without
    a second pass over the data.
    """

    def __init__(self) -> None:
        self._clicks: dict = defaultdict(list)
        self._views: dict = defaultdict(list)
        self._sealed = False

    def add_click(self, article_id, ts) -> None:
        self._clicks[article_id].append(ts)

    def add_view(self, article_id, ts) -> None:
        self._views[article_id].append(ts)

    def seal(self) -> None:
        """Sort every article's timestamps. Must be called before any query.

        Queries are refused before sealing rather than silently sorting on demand: an
        unsorted list would make `bisect` return a wrong count without erroring, which is
        exactly the class of bug that produces plausible-but-wrong numbers.
        """
        for store in (self._clicks, self._views):
            for k in store:
                store[k].sort()
        self._sealed = True

    def _count(self, store: dict, article_id, t, window: timedelta | None) -> int:
        if not self._sealed:
            raise RuntimeError("seal() must be called before querying")
        arr = store.get(article_id)
        if not arr:
            return 0
        hi = bisect_left(arr, t)          # strictly before t
        if window is None:
            return hi
        lo = bisect_left(arr, t - window)
        return hi - lo

    def clicks_before(self, article_id, t, window: timedelta | None = None) -> int:
        return self._count(self._clicks, article_id, t, window)

    def views_before(self, article_id, t, window: timedelta | None = None) -> int:
        return self._count(self._views, article_id, t, window)

    def ctr_before(self, article_id, t, window: timedelta | None = None,
                   prior: float = 10.0) -> float:
        """Smoothed click-through rate as of `t`.

        The `prior` is additive smoothing in the denominator: without it an article seen once
        and clicked once has CTR 1.0 and outranks an article with 500 clicks in 5,000 views.
        prior=10 means an article needs roughly ten views before its CTR is taken seriously.
        """
        views = self.views_before(article_id, t, window)
        if views == 0:
            return 0.0
        return self.clicks_before(article_id, t, window) / (views + prior)

    @property
    def n_articles(self) -> int:
        return len(self._clicks)


def frozen_counts(rolling: RollingCounts, articles) -> dict:
    """Total click count per article, ignoring time — the *leaky* comparison baseline.

    Exists so the frozen-vs-rolling ablation can be run against an identical event stream,
    isolating the point-in-time property as the only difference between the two.
    """
    return {a: rolling.clicks_before(a, _FAR_FUTURE) for a in articles}


_FAR_FUTURE = np.datetime64("2100-01-01").astype("datetime64[us]").item()
