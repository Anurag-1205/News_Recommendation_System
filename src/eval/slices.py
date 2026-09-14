"""Evaluation slices for Q5 (SPEC.md §17): cold vs warm users, head vs tail articles.

Both thresholds are A1's, kept so A2's slices are comparable with A1's (SPEC.md §10, decision 6):
a user is **cold** with at most 5 history clicks, and an article is **head** if it sits in the top
popularity quintile of the training clicks. The definitions live here, not in a script, because a
slice is a claim about who the metric describes and it has to be testable on its own.
"""

from __future__ import annotations

COLD_MAX_HISTORY = 5      # <= this many history clicks at t -> cold-start user
HEAD_FRACTION = 0.2       # top quintile of clicked articles -> head


def head_articles(click_counts: dict, fraction: float = HEAD_FRACTION) -> set:
    """The most-clicked `fraction` of the articles that received at least one training click.

    Articles with no training click are **not** head: they are the tail the metric is about.
    Ties are broken by article id so the set is deterministic, and at least one article is head
    whenever anything was clicked.
    """
    if not click_counts or fraction <= 0:
        return set()
    ranked = sorted(click_counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
    n_head = max(1, round(len(ranked) * fraction))
    return {article for article, _ in ranked[:n_head]}


def is_cold(history_len: int, threshold: int = COLD_MAX_HISTORY) -> bool:
    """A user with `threshold` or fewer history clicks is cold; the boundary value is cold."""
    return history_len <= threshold
