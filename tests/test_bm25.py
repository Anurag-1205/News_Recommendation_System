"""Oracles for the inverted index and BM25 (SPEC.md §5).

Two independent checks, because either alone is weak:

1. **Hand-computed scores** on a 5-document toy corpus. Catches a wrong formula that is
   nonetheless self-consistent.
2. **Exact agreement with `rank_bm25`** in the `okapi` variant. Catches a formula that is
   right in isolation but mis-wired to the index.

`rank_bm25` is the oracle here and never the implementation (SPEC.md §5).
"""

import math

import pytest
from rank_bm25 import BM25Okapi

from src.lexical.bm25 import BM25
from src.lexical.index import InvertedIndex, build_index
from src.lexical.tokenize import tokenize, tokenize_fields

# Five short documents, pre-tokenised so the arithmetic below is unambiguous.
TOY = [
    ("D0", ["cat", "cat", "dog"]),
    ("D1", ["cat", "bird"]),
    ("D2", ["dog", "dog", "dog", "bird"]),
    ("D3", ["fish"]),
    ("D4", ["cat", "dog", "bird", "fish"]),
]
# lengths 3,2,4,1,4  -> total 14, avgdl 2.8
# df: cat 3 (D0,D1,D4), dog 3 (D0,D2,D4), bird 3 (D1,D2,D4), fish 2 (D3,D4)


@pytest.fixture
def toy_index() -> InvertedIndex:
    idx = InvertedIndex()
    for article_id, tokens in TOY:
        idx.add(article_id, tokens)
    return idx


class TestInvertedIndex:
    def test_document_count_and_lengths(self, toy_index):
        assert toy_index.n_docs == 5
        assert toy_index.doc_lengths == [3, 2, 4, 1, 4]

    def test_average_length_hand_computed(self, toy_index):
        assert toy_index.avg_doc_length == pytest.approx(14 / 5)

    def test_document_frequencies_hand_counted(self, toy_index):
        assert toy_index.doc_frequency("cat") == 3
        assert toy_index.doc_frequency("dog") == 3
        assert toy_index.doc_frequency("fish") == 2
        assert toy_index.doc_frequency("absent") == 0

    def test_postings_carry_term_frequency(self, toy_index):
        assert toy_index.postings["cat"] == [(0, 2), (1, 1), (4, 1)]
        assert toy_index.postings["dog"] == [(0, 1), (2, 3), (4, 1)]

    def test_postings_sorted_by_doc_id(self, toy_index):
        for postings in toy_index.postings.values():
            assert [d for d, _ in postings] == sorted(d for d, _ in postings)

    def test_duplicate_article_id_rejected(self, toy_index):
        with pytest.raises(ValueError, match="duplicate"):
            toy_index.add("D0", ["x"])

    def test_empty_index_has_no_zero_division(self):
        assert InvertedIndex().avg_doc_length == 0.0


class TestBM25HandComputed:
    def test_single_term_score_worked_by_hand(self, toy_index):
        """Query "fish" against D3, k1=1.2, b=0.75, lucene IDF.

        N=5, df(fish)=2  -> idf = ln(1 + (5-2+0.5)/(2+0.5)) = ln(1 + 1.4) = ln(2.4)
        |D3|=1, avgdl=2.8
        norm = 1.2 * (1 - 0.75 + 0.75*1/2.8) = 1.2 * (0.25 + 0.267857) = 0.6214285
        tf=1 -> score = ln(2.4) * (1*2.2)/(1 + 0.6214285)
        """
        bm = BM25(toy_index, k1=1.2, b=0.75, idf_variant="lucene")
        idf = math.log(2.4)
        norm = 1.2 * (1 - 0.75 + 0.75 * 1 / 2.8)
        expected = idf * 2.2 / (1 + norm)
        assert bm.score_document(["fish"], 3) == pytest.approx(expected, abs=1e-12)
        # idf = ln(2.4) = 0.87547; norm = 0.62143; score = 0.87547 * 2.2/1.62143 = 1.18787.
        # The first version asserted 0.8752 here -- the IDF, not the score.
        assert expected == pytest.approx(1.1879, abs=1e-4)

    def test_term_frequency_saturates(self, toy_index):
        """D2 has dog x3, D0 has dog x1. Three occurrences must score more, but far less
        than three times more — that saturation is the whole point of k1."""
        bm = BM25(toy_index, idf_variant="lucene")
        s_three = bm.score_document(["dog"], 2)
        s_one = bm.score_document(["dog"], 0)
        assert s_three > s_one
        assert s_three < 3 * s_one

    def test_length_normalisation_penalises_longer_documents(self, toy_index):
        """D1 (len 2) and D4 (len 4) each contain cat once. The shorter must score higher."""
        bm = BM25(toy_index, idf_variant="lucene")
        assert bm.score_document(["cat"], 1) > bm.score_document(["cat"], 4)

    def test_b_zero_disables_length_normalisation(self, toy_index):
        """With b=0 the two equal-tf documents above must score identically."""
        bm = BM25(toy_index, b=0.0, idf_variant="lucene")
        assert bm.score_document(["cat"], 1) == pytest.approx(bm.score_document(["cat"], 4))

    def test_absent_term_contributes_nothing(self, toy_index):
        bm = BM25(toy_index)
        assert bm.score_document(["absent"], 0) == 0.0

    def test_lucene_idf_never_negative(self, toy_index):
        bm = BM25(toy_index, idf_variant="lucene")
        assert all(v > 0 for v in bm.idf.values())

    def test_okapi_idf_goes_negative_on_common_terms(self):
        """The behaviour that motivates defaulting to lucene, pinned by a test.

        'common' appears in 3 of 4 documents: ln((4-3+0.5)/(3+0.5)) = ln(0.4286) < 0.
        """
        idx = InvertedIndex()
        for i, toks in enumerate([["common", "a"], ["common", "b"], ["common", "c"], ["d"]]):
            idx.add(f"D{i}", toks)
        raw = math.log(4 - 3 + 0.5) - math.log(3 + 0.5)
        assert raw < 0
        assert BM25(idx, idf_variant="lucene").idf["common"] > 0

    def test_unknown_variant_rejected(self, toy_index):
        with pytest.raises(ValueError, match="unknown idf_variant"):
            BM25(toy_index, idf_variant="bm42")


class TestAgainstRankBM25:
    """Exact score agreement with rank_bm25 in its own variant."""

    @pytest.fixture
    def pair(self, toy_index):
        corpus = [tokens for _, tokens in TOY]
        return BM25(toy_index, k1=1.5, b=0.75, idf_variant="okapi"), BM25Okapi(corpus, k1=1.5, b=0.75)

    @pytest.mark.parametrize("query", [["cat"], ["dog"], ["fish"], ["cat", "dog"],
                                       ["bird", "fish"], ["cat", "cat", "dog"], ["absent"]])
    def test_scores_match_exactly(self, pair, query):
        ours, theirs = pair
        expected = theirs.get_scores(query)
        got = [ours.score_candidates(query, [d])[0] for d, _ in TOY]
        assert got == pytest.approx(list(expected), abs=1e-10)

    def test_idf_matches_rank_bm25(self, pair):
        ours, theirs = pair
        for term in ("cat", "dog", "bird", "fish"):
            assert ours.idf[term] == pytest.approx(theirs.idf[term], abs=1e-12)

    def test_ranking_agrees_on_a_larger_random_corpus(self):
        """Score-for-score on 200 synthetic documents, not just the 5-doc toy."""
        import random

        rng = random.Random(0)
        vocab = [f"w{i}" for i in range(50)]
        corpus = [[rng.choice(vocab) for _ in range(rng.randint(3, 30))] for _ in range(200)]
        idx = InvertedIndex()
        for i, toks in enumerate(corpus):
            idx.add(f"D{i}", toks)
        ours = BM25(idx, k1=1.5, b=0.75, idf_variant="okapi")
        theirs = BM25Okapi(corpus, k1=1.5, b=0.75)

        query = [rng.choice(vocab) for _ in range(4)]
        got = ours.score_candidates(query, [f"D{i}" for i in range(200)])
        assert got == pytest.approx(list(theirs.get_scores(query)), abs=1e-9)


class TestSearch:
    def test_search_ranks_by_score(self, toy_index):
        bm = BM25(toy_index, idf_variant="lucene")
        results = bm.search(["dog"], top_k=5)
        assert results[0][0] == "D2"                       # dog x3
        assert [s for _, s in results] == sorted((s for _, s in results), reverse=True)

    def test_search_only_returns_documents_containing_a_query_term(self, toy_index):
        """D3 holds only 'fish' and must not appear for a 'cat' query."""
        assert "D3" not in [d for d, _ in BM25(toy_index).search(["cat"], top_k=10)]

    def test_search_respects_top_k(self, toy_index):
        assert len(BM25(toy_index).search(["cat", "dog", "bird"], top_k=2)) == 2

    def test_search_agrees_with_score_candidates(self, toy_index):
        """The two harnesses share a scorer, so they must not disagree (SPEC.md §1)."""
        bm = BM25(toy_index, idf_variant="lucene")
        query = ["cat", "dog"]
        via_search = dict(bm.search(query, top_k=5))
        ids = [d for d, _ in TOY]
        via_candidates = dict(zip(ids, bm.score_candidates(query, ids)))
        for article_id, score in via_search.items():
            assert via_candidates[article_id] == pytest.approx(score, abs=1e-12)

    def test_empty_index_returns_nothing(self):
        assert BM25(InvertedIndex()).search(["cat"]) == []


class TestTokenizer:
    def test_lowercases_and_splits(self):
        assert tokenize("Dog BITES Man", lang=None) == ["dog", "bites", "man"]

    def test_drops_english_stopwords(self):
        assert tokenize("the cat and the dog", lang="en") == ["cat", "dog"]

    def test_keeps_danish_characters(self):
        """The check that matters for EB-NeRD: æ, ø, å must survive tokenisation."""
        assert tokenize("Røde Kors på Fyn", lang=None) == ["røde", "kors", "på", "fyn"]

    def test_danish_stopwords_not_applied_to_english(self):
        """'der' and 'som' are Danish stopwords but ordinary English-adjacent tokens."""
        assert "der" in tokenize("der som", lang="en")
        assert tokenize("der som", lang="da") == []

    def test_drops_single_characters(self):
        assert tokenize("a b cd", lang=None) == ["cd"]

    def test_handles_none_and_empty(self):
        assert tokenize(None) == []
        assert tokenize("") == []

    def test_tokenize_fields_concatenates(self):
        assert tokenize_fields("Cat news", "About a dog", lang="en") == ["cat", "news", "about", "dog"]

    def test_tokenize_fields_skips_null_abstract(self):
        """MIND abstracts are frequently null — that must not break indexing."""
        assert tokenize_fields("Cat news", None, lang="en") == ["cat", "news"]


def test_build_index_uses_same_tokenizer_as_queries():
    """Index-time and query-time analysis must agree or recall dies silently."""
    idx = build_index([("A1", "The Cat", "A dog story")], lang="en")
    assert idx.doc_frequency("cat") == 1
    assert idx.doc_frequency("the") == 0        # stopword removed at index time
    assert BM25(idx).score_candidates(tokenize("cat", lang="en"), ["A1"])[0] > 0


class TestForwardIndex:
    """The forward index is an optimisation, so the property that matters is that it changes
    nothing. Any score difference is a bug, not a tolerance."""

    def _both(self, docs):
        inv = InvertedIndex()
        fwd = InvertedIndex()
        fwd.enable_forward_index()
        for article_id, tokens in docs:
            inv.add(article_id, tokens)
            fwd.add(article_id, tokens)
        return BM25(inv, idf_variant="lucene"), BM25(fwd, idf_variant="lucene")

    def test_scores_identical_to_inverted_path(self):
        a, b = self._both(TOY)
        ids = [d for d, _ in TOY]
        for query in (["cat"], ["dog", "bird"], ["cat", "cat", "fish"], ["absent"], []):
            assert a.score_candidates(query, ids) == pytest.approx(
                b.score_candidates(query, ids), abs=1e-12
            ), f"forward index diverged on {query!r}"

    def test_identical_on_random_corpus(self):
        """Covers both branches: documents shorter and longer than the query."""
        import random

        rng = random.Random(7)
        vocab = [f"w{i}" for i in range(40)]
        docs = [(f"D{i}", [rng.choice(vocab) for _ in range(rng.randint(2, 60))])
                for i in range(150)]
        a, b = self._both(docs)
        ids = [d for d, _ in docs]
        for _ in range(20):
            query = [rng.choice(vocab) for _ in range(rng.randint(1, 50))]
            assert a.score_candidates(query, ids) == pytest.approx(
                b.score_candidates(query, ids), abs=1e-12
            )

    def test_forward_index_matches_rank_bm25_too(self):
        """The external oracle must still hold through the optimised path."""
        corpus = [tokens for _, tokens in TOY]
        idx = InvertedIndex()
        idx.enable_forward_index()
        for article_id, tokens in TOY:
            idx.add(article_id, tokens)
        ours = BM25(idx, k1=1.5, b=0.75, idf_variant="okapi")
        theirs = BM25Okapi(corpus, k1=1.5, b=0.75)
        for query in (["cat"], ["dog"], ["cat", "dog"]):
            got = ours.score_candidates(query, [d for d, _ in TOY])
            assert got == pytest.approx(list(theirs.get_scores(query)), abs=1e-10)

    def test_enabling_after_adding_documents_is_refused(self):
        """A forward index missing its first documents would score them as empty."""
        idx = InvertedIndex()
        idx.add("D0", ["cat"])
        with pytest.raises(RuntimeError, match="before adding documents"):
            idx.enable_forward_index()

    def test_forward_index_is_populated_for_every_document(self):
        idx = InvertedIndex()
        idx.enable_forward_index()
        for article_id, tokens in TOY:
            idx.add(article_id, tokens)
        assert len(idx.forward) == idx.n_docs
        assert idx.forward[0] == {"cat": 2, "dog": 1}
