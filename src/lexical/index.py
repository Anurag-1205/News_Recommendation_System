"""Inverted index over article text.

The classic structure: term -> postings list of (doc, term-frequency), plus the two
statistics BM25 needs at scoring time — per-document length and the corpus average.

Documents are addressed by a dense integer `doc_id` rather than by their string article id.
Postings then store ints instead of repeated strings, which is what keeps a 121K-article
index in tens of MB rather than hundreds, and lets scoring accumulate into a flat array.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class InvertedIndex:
    """Term -> [(doc_id, tf), ...], with document lengths alongside.

    Postings are kept sorted by doc_id. Nothing here needs the order yet, but it is the
    property that makes intersection and skip-pointer traversal possible later, and it
    costs nothing to maintain at build time.
    """

    postings: dict[str, list[tuple[int, int]]] = field(default_factory=lambda: defaultdict(list))
    # Article ids are whatever the dataset uses: MIND's "N12345" strings, EB-NeRD's int32.
    # Both are hashable and are kept in their native type -- converting EB-NeRD's 13.5M
    # impressions' worth of candidate ids to strings would cost more than it buys.
    doc_ids: list = field(default_factory=list)               # doc_id -> article id
    doc_lengths: list[int] = field(default_factory=list)      # doc_id -> token count
    _total_length: int = 0                                    # sum(doc_lengths), kept by add()
    _id_to_doc: dict = field(default_factory=dict)            # article id -> doc_id

    # Optional forward index: doc_id -> {term: tf}. The inverted index answers "which
    # documents contain this term", which is what corpus-wide search needs. Re-ranking asks
    # the opposite question -- "what does THIS document contain" -- and answering it by
    # searching postings is backwards. With ~110 query terms against ~37 candidates, the
    # inverted route costs 110*37 binary searches; the forward route costs 37 documents *
    # ~28 terms of dict lookup, roughly 60x fewer operations in the inner loop.
    # Off by default: it roughly doubles index memory, which matters at 2 GB free.
    forward: list[dict[str, int]] | None = None

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    @property
    def avg_doc_length(self) -> float:
        """Mean document length; 0.0 for an empty index rather than a ZeroDivisionError.

        A running total, not `sum(self.doc_lengths)`: BM25 reads this once per scored
        impression, and summing 125k lengths each time was 37 % of the EB-NeRD test pass.
        """
        return self._total_length / len(self.doc_lengths) if self.doc_lengths else 0.0

    def doc_frequency(self, term: str) -> int:
        """In how many documents does this term appear at least once?"""
        return len(self.postings.get(term, ()))

    def add(self, article_id, tokens: list[str]) -> int:
        """Index one document. Returns its doc_id.

        Term frequencies are counted locally first so each (term, doc) pair appends exactly
        one posting, keeping the sorted-by-doc_id invariant without a later sort pass.
        """
        if article_id in self._id_to_doc:
            raise ValueError(f"duplicate article_id {article_id!r}")
        doc_id = len(self.doc_ids)
        self._id_to_doc[article_id] = doc_id
        self.doc_ids.append(article_id)
        self.doc_lengths.append(len(tokens))
        self._total_length += len(tokens)

        tf: dict[str, int] = defaultdict(int)
        for tok in tokens:
            tf[tok] += 1
        for term, count in tf.items():
            self.postings[term].append((doc_id, count))
        if self.forward is not None:
            self.forward.append(dict(tf))
        return doc_id

    def enable_forward_index(self) -> None:
        """Turn on the forward index. Must be called before any document is added.

        Refuses afterwards rather than silently building a forward index that is missing its
        first N documents -- a partial forward index would score those documents as empty,
        which looks like a ranking problem rather than a bug.
        """
        if self.doc_ids:
            raise RuntimeError("enable_forward_index() must be called before adding documents")
        self.forward = []

    def doc_id_of(self, article_id) -> int | None:
        return self._id_to_doc.get(article_id)

    def article_id_of(self, doc_id: int):
        return self.doc_ids[doc_id]

    def stats(self) -> dict:
        return {
            "documents": self.n_docs,
            "unique_terms": len(self.postings),
            "postings": sum(len(p) for p in self.postings.values()),
            "avg_doc_length": round(self.avg_doc_length, 2),
        }


def build_index(documents, lang: str | None = "en", forward: bool = False) -> InvertedIndex:
    """Build an index from an iterable of (article_id, *text_fields).

    Tokenisation happens here rather than in the caller so that the index and every query
    against it are guaranteed to use identical analysis — a mismatch between index-time and
    query-time tokenisation is a silent recall killer.
    """
    from src.lexical.tokenize import tokenize_fields

    idx = InvertedIndex()
    if forward:
        idx.enable_forward_index()
    for article_id, *fields in documents:
        idx.add(article_id, tokenize_fields(*fields, lang=lang))
    return idx
