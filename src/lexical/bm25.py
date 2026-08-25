"""BM25 scoring over the inverted index.

    score(D, Q) = sum over terms t in Q of

                          tf(t,D) * (k1 + 1)
        IDF(t) * -----------------------------------------
                 tf(t,D) + k1 * (1 - b + b * |D| / avgdl)

`k1` controls how fast term-frequency saturates: a term appearing ten times is worth more
than once, but nowhere near ten times as much. `b` controls length normalisation — b=1
normalises fully by document length, b=0 not at all. The defaults k1=1.2, b=0.75 are the
conventional starting point and are stated here as a choice to be ablated, not a constant.

**Two IDF variants, deliberately.** They differ on common terms and the difference is not
cosmetic:

    okapi   idf = ln((N - df + 0.5) / (df + 0.5))
            The original. Goes *negative* for terms in more than half the documents, which
            means a common term can subtract from a document's score. rank_bm25 patches
            this with an epsilon floor.

    lucene  idf = ln(1 + (N - df + 0.5) / (df + 0.5))
            The +1 inside the log keeps IDF strictly positive. What Lucene and Elasticsearch
            actually ship.

`lucene` is the default because a negative contribution for a common term is a genuine
misbehaviour on short news text, where a title of six words can easily be half stopwords.
`okapi` exists so the implementation can be checked score-for-score against `rank_bm25`,
which implements exactly that variant — see tests/test_bm25.py.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import defaultdict

from src.lexical.index import InvertedIndex

DEFAULT_K1 = 1.2
DEFAULT_B = 0.75
RANK_BM25_EPSILON = 0.25   # rank_bm25's floor for negative okapi IDFs


def _tf_of(postings: list[tuple[int, int]], doc_id: int) -> int:
    """Term frequency of one document, by binary search over its postings list.

    Postings are maintained sorted by doc_id (see InvertedIndex), so this is O(log n) rather
    than the O(n) linear scan the first version used. That mattered: scoring a candidate list
    touches every query term for every candidate, and a linear scan there made in-impression
    re-ranking scale with corpus size instead of with candidate count.
    """
    i = bisect_left(postings, (doc_id, 0))
    if i < len(postings) and postings[i][0] == doc_id:
        return postings[i][1]
    return 0


class BM25:
    """BM25 scorer bound to one index.

    IDF is precomputed per term at construction: it depends only on corpus statistics, so
    recomputing it inside the scoring loop would repeat a logarithm for every query term on
    every query for no benefit.
    """

    def __init__(self, index: InvertedIndex, k1: float = DEFAULT_K1, b: float = DEFAULT_B,
                 idf_variant: str = "lucene") -> None:
        if idf_variant not in ("lucene", "okapi"):
            raise ValueError(f"unknown idf_variant {idf_variant!r}")
        self.index = index
        self.k1 = k1
        self.b = b
        self.idf_variant = idf_variant
        self.idf = self._compute_idf()

    def _compute_idf(self) -> dict[str, float]:
        n = self.index.n_docs
        idf: dict[str, float] = {}
        for term, postings in self.index.postings.items():
            df = len(postings)
            if self.idf_variant == "lucene":
                idf[term] = math.log(1 + (n - df + 0.5) / (df + 0.5))
            else:
                idf[term] = math.log(n - df + 0.5) - math.log(df + 0.5)

        if self.idf_variant == "okapi":
            # rank_bm25 replaces every negative IDF with eps * mean(ALL IDFs), and the mean
            # includes the negative values themselves. Averaging only the positives -- the
            # intuitive reading -- gives a floor of the wrong sign and magnitude; the
            # cross-check against rank_bm25 caught exactly that.
            mean_idf = sum(idf.values()) / len(idf) if idf else 0.0
            floor = RANK_BM25_EPSILON * mean_idf
            idf = {t: (v if v > 0 else floor) for t, v in idf.items()}
        return idf

    def score_document(self, query_tokens: list[str], doc_id: int) -> float:
        """Score one document. Used by the hand-computed oracle; retrieval uses `search`."""
        length = self.index.doc_lengths[doc_id]
        norm = self.k1 * (1 - self.b + self.b * length / self.index.avg_doc_length)
        total = 0.0
        for term in query_tokens:
            postings = self.index.postings.get(term)
            if not postings:
                continue
            tf = _tf_of(postings, doc_id)
            if tf:
                total += self.idf[term] * (tf * (self.k1 + 1)) / (tf + norm)
        return total

    def search(self, query_tokens: list[str], top_k: int = 100) -> list[tuple[str, float]]:
        """Term-at-a-time retrieval: walk each query term's postings, accumulate, then rank.

        This is why the inverted index exists. Only documents that contain at least one
        query term are ever touched — on a 121K-article corpus a typical query reaches a few
        thousand, not all of them. Scoring every document would be ~30x more work for an
        identical answer.

        Repeated query terms are collapsed first, so a history query that mentions "election"
        five times costs one postings traversal rather than five.
        """
        query_tf: dict[str, int] = defaultdict(int)
        for tok in query_tokens:
            query_tf[tok] += 1

        avgdl = self.index.avg_doc_length
        if avgdl == 0:
            return []

        scores: dict[int, float] = defaultdict(float)
        for term, q_count in query_tf.items():
            postings = self.index.postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for doc_id, tf in postings:
                length = self.index.doc_lengths[doc_id]
                norm = self.k1 * (1 - self.b + self.b * length / avgdl)
                scores[doc_id] += q_count * idf * (tf * (self.k1 + 1)) / (tf + norm)

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return [(self.index.article_id_of(d), s) for d, s in ranked]

    def score_candidates(self, query_tokens: list[str], candidate_ids: list[str]) -> list[float]:
        """Score a fixed candidate list — mode (b), in-impression re-ranking (SPEC.md §1).

        Returns one score per candidate in the given order, 0.0 for candidates absent from
        the index. Same scorer as `search`, different harness around it.
        """
        query_tf: dict[str, int] = defaultdict(int)
        for tok in query_tokens:
            query_tf[tok] += 1

        forward = self.index.forward
        avgdl = self.index.avg_doc_length
        out = []
        for article_id in candidate_ids:
            doc_id = self.index.doc_id_of(article_id)
            if doc_id is None:
                out.append(0.0)
                continue
            total = 0.0
            length = self.index.doc_lengths[doc_id]
            norm = self.k1 * (1 - self.b + self.b * length / avgdl)

            if forward is not None:
                # Walk the document's own terms -- far fewer than the query's -- and keep only
                # those the query also mentions. Same arithmetic, same result, ~60x less work.
                doc_terms = forward[doc_id]
                if len(doc_terms) < len(query_tf):
                    for term, tf in doc_terms.items():
                        q_count = query_tf.get(term)
                        if q_count:
                            total += q_count * self.idf[term] * (tf * (self.k1 + 1)) / (tf + norm)
                    out.append(total)
                    continue
                for term, q_count in query_tf.items():
                    tf = doc_terms.get(term)
                    if tf:
                        total += q_count * self.idf[term] * (tf * (self.k1 + 1)) / (tf + norm)
                out.append(total)
                continue

            for term, q_count in query_tf.items():
                postings = self.index.postings.get(term)
                if not postings:
                    continue
                tf = _tf_of(postings, doc_id)
                if tf:
                    total += q_count * self.idf[term] * (tf * (self.k1 + 1)) / (tf + norm)
            out.append(total)
        return out
