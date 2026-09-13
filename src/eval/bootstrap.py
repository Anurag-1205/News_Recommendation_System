"""Bootstrap confidence intervals for ranking metrics (A1 Q4.4).

**Resample impressions, not candidate rows.** The impression is the independent unit: the
candidates inside one impression share a user, a timestamp and a candidate pool, so treating
them as independent draws would understate the variance and produce intervals that are too
narrow to be honest.

Every headline number in RESULTS.md carries one of these. The rule that gives them teeth:
two overlapping intervals are **not** a win, and the word "beats" is reserved for the case
where they separate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CI:
    """A point estimate with a percentile interval."""

    mean: float
    lo: float
    hi: float
    n: int
    iterations: int

    def __str__(self) -> str:
        return f"{self.mean:.4f} [{self.lo:.4f}, {self.hi:.4f}]"

    def overlaps(self, other: "CI") -> bool:
        """True when the two intervals intersect — i.e. when 'beats' is not justified."""
        return not (self.hi < other.lo or other.hi < self.lo)


def bootstrap_ci(per_impression: list[float] | np.ndarray, iterations: int = 1000,
                 confidence: float = 0.95, seed: int = 0) -> CI:
    """Percentile bootstrap over per-impression metric values.

    `per_impression` is one metric value per impression — not a pooled score. The seed is
    fixed and recorded so a reported interval can be reproduced exactly.

    NaNs are dropped first: AUC is undefined for an impression whose candidates are all
    clicks or all non-clicks, and resampling a NaN would poison the whole replicate.
    """
    values = np.asarray(per_impression, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return CI(float("nan"), float("nan"), float("nan"), 0, iterations)
    if n == 1:
        v = float(values[0])
        return CI(v, v, v, 1, iterations)

    rng = np.random.default_rng(seed)
    # One (iterations x n) index draw, then a single row-wise mean: the vectorised form is
    # ~100x faster than a Python loop and matters at 1000 iterations over 73K impressions.
    idx = rng.integers(0, n, size=(iterations, n))
    means = values[idx].mean(axis=1)

    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return CI(float(values.mean()), float(lo), float(hi), n, iterations)


def bootstrap_metrics(per_impression: dict[str, list[float]], iterations: int = 1000,
                      confidence: float = 0.95, seed: int = 0) -> dict[str, CI]:
    """Bootstrap several metrics that share the same impressions.

    Each metric gets the same seed, so all of them are resampled on comparable draws.
    """
    return {
        name: bootstrap_ci(values, iterations=iterations, confidence=confidence, seed=seed)
        for name, values in per_impression.items()
    }


def compare(a: CI, b: CI, label_a: str = "A", label_b: str = "B") -> str:
    """Phrase a comparison honestly, refusing 'beats' when the intervals overlap."""
    if a.overlaps(b):
        return (f"{label_a} {a} vs {label_b} {b}: intervals OVERLAP — "
                f"no significant difference at this sample size")
    better, worse = (label_a, label_b) if a.mean > b.mean else (label_b, label_a)
    return f"{better} beats {worse}: {a} vs {b}, intervals disjoint"


def paired_delta(a, b, *, iterations: int = 1000, confidence: float = 0.95, seed: int = 0,
                 block: int = 100) -> CI:
    """Paired bootstrap of mean(b - a) over impressions (A2 Q3.4; SPEC.md §14).

    Both systems are scored on the same impressions, so resampling the per-impression
    *differences* cancels the impression-to-impression variance they share. That makes it far
    tighter than comparing two independent CIs. Impressions where either value is NaN (AUC with a
    single class) are dropped. Resampled in blocks to bound memory.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError(f"paired inputs differ in length: {a.shape} vs {b.shape}")
    d = (b - a)[~(np.isnan(a) | np.isnan(b))]
    n = len(d)
    if n == 0:
        return CI(float("nan"), float("nan"), float("nan"), 0, iterations)
    rng = np.random.default_rng(seed)
    means = np.concatenate([d[rng.integers(0, n, size=(min(block, iterations - s), n))].mean(axis=1)
                            for s in range(0, iterations, block)])
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return CI(float(d.mean()), float(lo), float(hi), n, iterations)
