"""User representations from click history (A1 Q3.3).

Two poolings, and the difference is the point of the ablation:

**Mean pooling** treats every past click as equally informative. Simple, and the natural
baseline.

**Recency-weighted pooling** applies exponential decay, weight `exp(-i/tau)` for the i-th
most recent click. News decays in days -- MIND's train/test gap reaches eight days, and
RESULTS.md shows popularity fitted a week earlier is worthless -- so an article read a month
ago is weak evidence of what a user wants today. `tau` is the number of clicks over which
influence falls by 1/e and is a hyper-parameter, not a constant.

Both return an L2-normalised vector, so downstream inner products are cosine similarities.
"""

from __future__ import annotations

import numpy as np


def _normalise(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return (vector / norm).astype(np.float32) if norm > 0 else vector.astype(np.float32)


def mean_pool(vectors: np.ndarray) -> np.ndarray:
    """Unweighted mean of the history vectors."""
    if len(vectors) == 0:
        return np.zeros(0, dtype=np.float32)
    return _normalise(vectors.mean(axis=0))


def recency_pool(vectors: np.ndarray, tau: float = 5.0) -> np.ndarray:
    """Exponentially decayed mean, **most recent last** in `vectors`.

    The ordering convention matters and is the easy thing to get backwards: history arrives
    oldest-first from both readers, so index -1 is the newest click and gets weight 1.
    """
    n = len(vectors)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    age = np.arange(n - 1, -1, -1, dtype=np.float32)   # oldest gets the largest age
    weights = np.exp(-age / tau)
    return _normalise((vectors * weights[:, None]).sum(axis=0) / weights.sum())


def build_user_vector(history_ids: list, id_to_row: dict, matrix: np.ndarray,
                      pooling: str = "mean", tau: float = 5.0) -> np.ndarray | None:
    """Pool the embeddings of a user's history. None when nothing is resolvable.

    Returning None rather than a zero vector keeps "cold-start user" distinguishable from
    "user whose interests happen to sit at the origin" -- the caller must decide what to do
    with a user it cannot represent, rather than silently scoring them against noise.
    """
    rows = [id_to_row[a] for a in history_ids if a in id_to_row]
    if not rows:
        return None
    vectors = matrix[rows]
    return mean_pool(vectors) if pooling == "mean" else recency_pool(vectors, tau=tau)
