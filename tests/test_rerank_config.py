"""Oracle for the locked Phase 2 configuration (CONTEXT.md C-018, SPEC.md §12.1).

The configuration is the single source of truth the submission runs will read, so its invariants
are asserted here rather than trusted: nothing unsafe or test-absent, dwell out of the EB-NeRD
model, and MIND exactly A1 v4's feature set under the pointwise objective.
"""

import numpy as np

from src.features.behavioural import ABSENT_FROM_TEST_FILE, SERVING_OK, UNSAFE_FEATURES
from src.rerank.common import model_features


def _final():
    from src.rerank.config import FINAL
    return FINAL


def test_both_datasets_are_configured_with_a_known_objective():
    final = _final()
    assert set(final) == {"ebnerd", "mind"}
    assert final["ebnerd"]["objective"] == "lambdarank"
    assert final["mind"]["objective"] == "pointwise"


def test_no_final_feature_is_unsafe_or_absent_from_the_test_file():
    for dataset, cfg in _final().items():
        assert model_features(cfg["features"]) == cfg["features"], dataset
        assert not set(cfg["features"]) & (set(UNSAFE_FEATURES) | ABSENT_FROM_TEST_FILE), dataset


def test_ebnerd_keeps_the_category_profile_and_drops_dwell():
    f = _final()["ebnerd"]["features"]
    assert {"recency_weighted_profile", "category_match"} <= set(f)
    assert not {"hist_read_time_mean", "hist_scroll_mean"} & set(f)
    assert len(f) == 11 and len(set(f)) == 11


def test_mind_is_exactly_a1_v4():
    assert _final()["mind"]["features"] == ["bm25", "semantic", "pop_total", "ctr_total", "n_candidates",
                                            "cat_affinity", "history_len"]


def test_every_phase1_feature_in_the_final_sets_is_registered():
    phase1 = [f for cfg in _final().values() for f in cfg["features"] if f in SERVING_OK]
    assert all(SERVING_OK[f] for f in phase1)


def test_fit_final_dispatches_on_objective():
    from src.rerank.common import fit_final, predict_scores
    rng = np.random.default_rng(0)
    X, y, groups = rng.normal(size=(200, 2)), np.tile([1, 0, 0, 0], 50).astype(np.int8), np.full(50, 4)
    for objective in ("lambdarank", "pointwise"):
        s = predict_scores(fit_final(objective, X, y, groups), X)
        assert s.shape == (200,) and np.isfinite(s).all(), objective
