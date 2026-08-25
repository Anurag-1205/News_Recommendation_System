"""Oracles for the EB-NeRD reader (SPEC.md §5).

The label conversion is the load-bearing piece: EB-NeRD gives the *set* of clicked article
ids where MIND gives a parallel 0/1 mask, and silently getting that wrong yields a
well-formed submission that scores like noise.
"""

import polars as pl
import pytest

from src.lexical.tokenize import tokenize
from src.pipeline.ebnerd import labels_from_clicked


class TestLabelsFromClicked:
    def test_marks_only_clicked_candidates(self):
        assert labels_from_clicked([10, 20, 30], [20]) == [0, 1, 0]

    def test_alignment_is_positional(self):
        """The mask must line up with `candidates` order, not with sorted ids."""
        assert labels_from_clicked([30, 10, 20], [30]) == [1, 0, 0]

    def test_multiple_clicks(self):
        assert labels_from_clicked([1, 2, 3, 4], [4, 1]) == [1, 0, 0, 1]

    def test_no_clicks_gives_all_zero(self):
        assert labels_from_clicked([1, 2, 3], []) == [0, 0, 0]

    def test_clicked_id_absent_from_candidates_is_ignored(self):
        """A click on an article not shown must not shift the mask or raise."""
        assert labels_from_clicked([1, 2], [99]) == [0, 0]

    def test_length_always_matches_candidates(self):
        for cands, clicked in ([[1], [1]], [[1, 2, 3], [2, 3]], [[], []], [[5, 6], []]):
            assert len(labels_from_clicked(cands, clicked)) == len(cands)

    def test_duplicate_candidate_ids_both_marked(self):
        """EB-NeRD impressions can repeat an article; both positions must be labelled."""
        assert labels_from_clicked([7, 7, 8], [7]) == [1, 1, 0]


class TestDanishTokenisation:
    def test_danish_letters_survive(self):
        assert tokenize("Røde Kors på Fyn", lang="da") == ["røde", "kors", "fyn"]

    def test_danish_stopwords_removed(self):
        assert tokenize("det er en test af noget", lang="da") == ["test"]

    def test_english_stoplist_would_not_strip_danish_function_words(self):
        """Why `lang` is threaded through: the English list leaves Danish stopwords in."""
        assert "det" in tokenize("det er en test", lang="en")
        assert "det" not in tokenize("det er en test", lang="da")


def test_recent_history_truncates_to_the_tail(tmp_path):
    """Only the most recent n ids are kept, and they stay oldest-first."""
    from src.pipeline.ebnerd import recent_history

    p = tmp_path / "history.parquet"
    pl.DataFrame({"user_id": [1, 2],
                  "article_id_fixed": [[10, 11, 12, 13, 14, 15], [20]]}).write_parquet(p)
    out = recent_history(p, n_recent=3)
    assert out[1] == [13, 14, 15]      # tail, order preserved
    assert out[2] == [20]              # shorter than n is kept whole


def test_article_text_maps_id_to_title_and_subtitle(tmp_path):
    from src.pipeline.ebnerd import article_text

    p = tmp_path / "articles.parquet"
    pl.DataFrame({"article_id": [1, 2], "title": ["A", "B"],
                  "subtitle": ["sa", "sb"], "body": ["x", "y"]}).write_parquet(p)
    assert article_text(p) == {1: ("A", "sa"), 2: ("B", "sb")}
