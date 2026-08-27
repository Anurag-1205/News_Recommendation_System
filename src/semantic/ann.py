"""ANN index over article embeddings (A1 Q3.2).

`IndexFlatIP` -- exhaustive inner product -- is the default and it is a deliberate choice,
not laziness. At 125K articles a flat index answers a query in about a millisecond and is
*exact*, so recall@K measures the retrieval model rather than the index's approximation
error. Introducing IVF or HNSW here would confound the two: a drop in recall could be the
embedding or the index, and disentangling them costs more than it buys at this scale.

`IndexIVFFlat` is provided so the recall-versus-latency trade can be *measured* rather than
asserted, which is what the scale section needs. Vectors are L2-normalised upstream, so inner
product is cosine similarity.
"""

from __future__ import annotations

import numpy as np


class ANNIndex:
    """Thin wrapper over FAISS that keeps article ids alongside the vectors."""

    def __init__(self, ids: list, matrix: np.ndarray, kind: str = "flat",
                 nlist: int = 256, nprobe: int = 16, seed: int = 0) -> None:
        import faiss

        if len(ids) != len(matrix):
            raise ValueError(f"{len(ids)} ids but {len(matrix)} vectors")
        self.ids = ids
        self.id_to_row = {a: i for i, a in enumerate(ids)}
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        self.kind = kind
        dim = self.matrix.shape[1]

        if kind == "flat":
            self.index = faiss.IndexFlatIP(dim)
        elif kind == "ivf":
            # nlist cannot exceed the training set; FAISS warns and degrades otherwise.
            nlist = min(nlist, max(1, len(ids) // 40))
            quantiser = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantiser, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            # FAISS 1.15 removed `faiss.cvar.rand_seed`, so IVF's k-means initialisation uses
            # the library's internal RNG and cannot be seeded from here. That is acceptable
            # because IVF is only ever used to *measure* the recall-vs-latency trade; every
            # reported retrieval number comes from the exact flat index, which is
            # deterministic by construction.
            self.index.train(self.matrix)
            self.index.nprobe = min(nprobe, nlist)
        else:
            raise ValueError(f"unknown index kind {kind!r}")
        self.index.add(self.matrix)

    def search(self, query: np.ndarray, top_k: int = 100) -> list[tuple]:
        """Top-k (article_id, similarity) for one query vector."""
        if query is None or query.size == 0:
            return []
        scores, rows = self.index.search(query.reshape(1, -1).astype(np.float32), top_k)
        return [(self.ids[r], float(s)) for r, s in zip(rows[0], scores[0]) if r != -1]

    def score_candidates(self, query: np.ndarray, candidate_ids: list) -> list[float]:
        """Mode (b): cosine similarity against a fixed candidate list (SPEC.md §1).

        Bypasses the index entirely -- with a handful of known candidates, a direct dot
        product against their rows is both exact and faster than an ANN lookup followed by a
        membership filter. Candidates absent from the embedding matrix score 0.0.
        """
        if query is None or query.size == 0:
            return [0.0] * len(candidate_ids)
        rows = [self.id_to_row.get(c, -1) for c in candidate_ids]
        known = [r for r in rows if r >= 0]
        if not known:
            return [0.0] * len(candidate_ids)
        sims = self.matrix[known] @ query
        out, it = [], iter(sims)
        for r in rows:
            out.append(float(next(it)) if r >= 0 else 0.0)
        return out
