"""Query construction from click history, and recall@K evaluation (A1 Q2.2-Q2.4).

The query is the concatenated text of the user's most recently clicked articles. `n_recent`
is a genuine hyper-parameter, not a constant: a short query is sharply on-topic but narrow,
a long one covers more of the user's interests while drifting toward a generic profile.
Q2 asks for the ablation, so `n_recent` is threaded through rather than fixed.

Recency matters because news decays in days (RESULTS.md: MIND's train/test gap is up to 8
days). MIND stores history oldest-first, so the *last* n entries are the most recent.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.lexical.bm25 import BM25
from src.lexical.tokenize import tokenize_fields


def build_query(history_ids: list[str], article_text: dict[str, tuple], n_recent: int = 5,
                lang: str = "en") -> list[str]:
    """Tokens for the n most recently clicked articles. Empty for a cold-start user."""
    if not history_ids:
        return []
    tokens: list[str] = []
    for article_id in history_ids[-n_recent:]:
        fields = article_text.get(article_id)
        if fields:
            tokens.extend(tokenize_fields(*fields, lang=lang))
    return tokens


@dataclass
class RecallResult:
    """recall@K over impressions, plus the denominators that make it interpretable."""

    k: int
    hits: int
    relevant: int
    impressions_scored: int
    impressions_skipped: int

    @property
    def recall(self) -> float:
        return self.hits / self.relevant if self.relevant else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> tuple[int, int]:
    """Hits within the top k, and the number of relevant items. Returns (hits, relevant)."""
    return len(set(retrieved[:k]) & relevant), len(relevant)


def evaluate_recall(bm25: BM25, impressions, article_text: dict[str, tuple],
                    ks: tuple[int, ...] = (50, 100, 200), n_recent: int = 5,
                    lang: str = "en", limit: int | None = None) -> dict[int, RecallResult]:
    """Mode (a): retrieve from the whole corpus, measure recall@K (SPEC.md §1).

    Cold-start impressions — no history, so no query — are **skipped rather than scored 0**.
    Counting them as misses would conflate "the retriever failed" with "there was nothing to
    retrieve from", and the skipped count is reported so the denominator is never hidden.

    Retrieval runs once at max(ks) and each K slices that list, rather than re-querying per
    K: the top-50 of a top-200 retrieval is exactly the top-50.
    """
    top = max(ks)
    acc = {k: RecallResult(k, 0, 0, 0, 0) for k in ks}

    for i, (history_ids, clicked) in enumerate(impressions):
        if limit is not None and i >= limit:
            break
        relevant = set(clicked)
        if not history_ids or not relevant:
            for k in ks:
                acc[k].impressions_skipped += 1
            continue

        query = build_query(history_ids, article_text, n_recent=n_recent, lang=lang)
        if not query:
            for k in ks:
                acc[k].impressions_skipped += 1
            continue

        retrieved = [article_id for article_id, _ in bm25.search(query, top_k=top)]
        for k in ks:
            hits, n_rel = recall_at_k(retrieved, relevant, k)
            acc[k].hits += hits
            acc[k].relevant += n_rel
            acc[k].impressions_scored += 1
    return acc
