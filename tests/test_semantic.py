"""Oracles for the semantic retriever (SPEC.md §5).

Embeddings have no hand-computable "right answer" the way BM25 does, so the checks are
structural and behavioural: normalisation invariants, pooling arithmetic worked by hand, and
the property that an approximate index must agree with the exact one on easy cases.
"""

import numpy as np
import pytest

from src.semantic.ann import ANNIndex
from src.semantic.embeddings import compute_lsa, l2_normalise
from src.semantic.user_vector import build_user_vector, mean_pool, recency_pool


class TestNormalisation:
    def test_rows_become_unit_length(self):
        out = l2_normalise(np.array([[3.0, 4.0], [1.0, 0.0]]))
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0)

    def test_hand_computed(self):
        """[3,4] has norm 5, so it normalises to [0.6, 0.8]."""
        assert np.allclose(l2_normalise(np.array([[3.0, 4.0]])), [[0.6, 0.8]])

    def test_zero_row_stays_zero_not_nan(self):
        """An article whose text tokenises to nothing must not poison every similarity."""
        out = l2_normalise(np.array([[0.0, 0.0], [1.0, 0.0]]))
        assert np.all(np.isfinite(out))
        assert np.allclose(out[0], [0.0, 0.0])

    def test_output_is_float32(self):
        assert l2_normalise(np.array([[1.0, 2.0]], dtype=np.float64)).dtype == np.float32


class TestPooling:
    def test_mean_pool_hand_computed(self):
        """Mean of [1,0] and [0,1] is [0.5,0.5], normalised to [0.707,0.707]."""
        out = mean_pool(np.array([[1.0, 0.0], [0.0, 1.0]]))
        assert np.allclose(out, [2 ** -0.5, 2 ** -0.5], atol=1e-6)

    def test_mean_pool_is_order_independent(self):
        v = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        assert np.allclose(mean_pool(v), mean_pool(v[::-1]), atol=1e-6)

    def test_recency_pool_favours_the_last_entry(self):
        """History is oldest-first, so index -1 is the newest click and dominates."""
        v = np.array([[1.0, 0.0], [0.0, 1.0]])          # old=[1,0], new=[0,1]
        out = recency_pool(v, tau=1.0)
        assert out[1] > out[0], "recency pooling must weight the most recent click higher"

    def test_recency_pool_order_matters_unlike_mean(self):
        v = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert not np.allclose(recency_pool(v, tau=1.0), recency_pool(v[::-1], tau=1.0))

    def test_large_tau_approaches_mean_pool(self):
        """With decay effectively switched off the two poolings must coincide."""
        v = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        assert np.allclose(recency_pool(v, tau=1e6), mean_pool(v), atol=1e-4)

    def test_outputs_are_unit_length(self):
        v = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert np.linalg.norm(mean_pool(v)) == pytest.approx(1.0, abs=1e-6)
        assert np.linalg.norm(recency_pool(v)) == pytest.approx(1.0, abs=1e-6)

    def test_empty_history_gives_empty_vector(self):
        assert mean_pool(np.zeros((0, 4))).size == 0


class TestBuildUserVector:
    def _fixture(self):
        matrix = l2_normalise(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
        return {"A": 0, "B": 1, "C": 2}, matrix

    def test_uses_only_resolvable_history(self):
        id_to_row, matrix = self._fixture()
        both = build_user_vector(["A", "B"], id_to_row, matrix)
        with_unknown = build_user_vector(["A", "B", "ZZZ"], id_to_row, matrix)
        assert np.allclose(both, with_unknown)

    def test_returns_none_for_cold_start(self):
        """None, not a zero vector — the caller must handle 'cannot represent this user'."""
        id_to_row, matrix = self._fixture()
        assert build_user_vector([], id_to_row, matrix) is None
        assert build_user_vector(["unknown"], id_to_row, matrix) is None

    def test_pooling_mode_is_respected(self):
        id_to_row, matrix = self._fixture()
        m = build_user_vector(["A", "B"], id_to_row, matrix, pooling="mean")
        r = build_user_vector(["A", "B"], id_to_row, matrix, pooling="recency", tau=0.5)
        assert not np.allclose(m, r)


class TestANNIndex:
    def _index(self, kind="flat"):
        ids = [f"A{i}" for i in range(200)]
        rng = np.random.default_rng(0)
        matrix = l2_normalise(rng.normal(size=(200, 16)))
        return ANNIndex(ids, matrix, kind=kind), matrix

    def test_query_retrieves_itself_first(self):
        """A vector's nearest neighbour under cosine similarity is itself."""
        idx, matrix = self._index()
        assert idx.search(matrix[7], top_k=1)[0][0] == "A7"

    def test_results_are_sorted_by_similarity(self):
        idx, matrix = self._index()
        scores = [s for _, s in idx.search(matrix[3], top_k=10)]
        assert scores == sorted(scores, reverse=True)

    def test_respects_top_k(self):
        idx, matrix = self._index()
        assert len(idx.search(matrix[0], top_k=5)) == 5

    def test_score_candidates_matches_direct_cosine(self):
        idx, matrix = self._index()
        query = matrix[11]
        got = idx.score_candidates(query, ["A0", "A11", "A5"])
        assert got == pytest.approx([float(matrix[0] @ query), 1.0, float(matrix[5] @ query)], abs=1e-5)

    def test_unknown_candidates_score_zero(self):
        idx, matrix = self._index()
        assert idx.score_candidates(matrix[0], ["nope", "A0"]) == pytest.approx([0.0, 1.0], abs=1e-5)

    def test_empty_query_scores_zero_not_crash(self):
        idx, _ = self._index()
        assert idx.score_candidates(None, ["A0", "A1"]) == [0.0, 0.0]
        assert idx.search(None) == []

    def test_ivf_returns_results_and_mostly_agrees_with_flat(self):
        """IVF is approximate, so exact agreement is not guaranteed and must not be asserted.

        What must hold is that it returns the requested number of results and agrees with the
        exact index on most queries. Demanding perfect agreement would make this test fail
        intermittently on a legitimate approximation.
        """
        flat, matrix = self._index("flat")
        ivf, _ = self._index("ivf")
        agree = 0
        for row in range(0, 200, 10):
            got = ivf.search(matrix[row], top_k=1)
            assert len(got) == 1
            if got[0][0] == flat.search(matrix[row], top_k=1)[0][0]:
                agree += 1
        assert agree >= 15, f"IVF agreed with flat on only {agree}/20 exact-match queries"

    def test_mismatched_ids_and_vectors_rejected(self):
        with pytest.raises(ValueError, match="ids but"):
            ANNIndex(["a", "b"], np.zeros((3, 4), dtype=np.float32))

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError, match="unknown index kind"):
            ANNIndex(["a"], np.zeros((1, 4), dtype=np.float32), kind="hnsw-ish")


class TestLSA:
    def test_similar_documents_land_close(self):
        """The property that makes it 'semantic': shared vocabulary implies proximity."""
        texts = ["dog cat pet animal", "cat dog animal pet", "election vote parliament",
                 "vote parliament election", "cooking recipe food", "recipe food cooking"]
        matrix, _, _ = compute_lsa(texts, n_components=3, min_df=1)
        assert matrix[0] @ matrix[1] > matrix[0] @ matrix[2]
        assert matrix[2] @ matrix[3] > matrix[2] @ matrix[4]

    def test_output_rows_are_unit_length(self):
        matrix, _, _ = compute_lsa(["alpha beta gamma", "beta gamma delta",
                                   "gamma delta epsilon", "xray yankee zulu"],
                                  n_components=2, min_df=1)
        assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)

    def test_components_clamped_to_matrix_rank(self):
        """Asking for more components than the corpus supports must not raise."""
        matrix, _, _ = compute_lsa(["alpha beta", "beta gamma", "gamma delta"],
                                  n_components=999, min_df=1)
        assert matrix.shape[0] == 3 and matrix.shape[1] >= 1

    def test_deterministic_for_fixed_seed(self):
        texts = ["alpha beta gamma", "beta gamma delta",
                 "gamma delta epsilon", "delta epsilon zeta"]
        m1, _, _ = compute_lsa(texts, n_components=2, seed=7, min_df=1)
        m2, _, _ = compute_lsa(texts, n_components=2, seed=7, min_df=1)
        assert np.allclose(m1, m2)
