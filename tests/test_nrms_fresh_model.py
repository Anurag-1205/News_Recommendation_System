"""Model-level oracles for the freshness variant (SPEC.md §15.4). They need TensorFlow and the
benchmark on sys.path, so they skip on the laptop and run inside the Kaggle kernels before
training (`pytest tests/test_nrms_fresh_model.py`)."""
import sys
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
for p in ("external/ebnerd-benchmark/src", "/kaggle/working/ebnerd-benchmark/src"):
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)
ebrec = pytest.importorskip("ebrec")

from ebrec.models.newsrec.model_config import hparams_nrms          # noqa: E402
from ebrec.models.newsrec.nrms import NRMSModel                     # noqa: E402
from src.baselines.nrms_fresh_ebrec import FRESH_DIM, NRMSFreshModel  # noqa: E402

VOCAB, DIM, HIST, TITLE = 50, 16, 4, 6


def _hparams():
    hp = hparams_nrms
    hp.history_size, hp.title_size, hp.head_num, hp.head_dim, hp.attention_hidden_dim = HIST, TITLE, 2, 4, 8
    hp.newsencoder_units_per_layer = None
    return hp


def _inputs(rng, batch=3, n=5):
    his = rng.integers(1, VOCAB, size=(batch, HIST, TITLE)).astype("int32")
    pred = rng.integers(1, VOCAB, size=(batch, n, TITLE)).astype("int32")
    fresh = np.stack([rng.standard_normal((batch, n)), rng.integers(0, 2, (batch, n))], -1).astype("float32")
    return his, pred, fresh


def test_additive_identity_g_zero_reproduces_baseline():
    """With g ≡ 0 and the same encoder weights, the variant is the baseline exactly (§15.1)."""
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((VOCAB, DIM)).astype("float32")
    base = NRMSModel(hparams=_hparams(), word2vec_embedding=emb, seed=1)
    var = NRMSFreshModel(hparams=_hparams(), word2vec_embedding=emb, seed=1)
    var.copy_encoders_from(base); var.zero_g()
    his, pred, fresh = _inputs(rng)
    np.testing.assert_allclose(var.model.predict([his, pred, fresh], verbose=0),
                               base.model.predict([his, pred], verbose=0), atol=1e-6)
    np.testing.assert_allclose(var.scorer.predict([his, pred[:, :1], fresh[:, :1]], verbose=0),
                               base.scorer.predict([his, pred[:, :1]], verbose=0), atol=1e-6)


def test_freshness_term_moves_scores_and_is_per_candidate():
    """With g ≠ 0, changing one candidate's freshness changes that candidate's logit only."""
    rng = np.random.default_rng(1)
    emb = rng.standard_normal((VOCAB, DIM)).astype("float32")
    var = NRMSFreshModel(hparams=_hparams(), word2vec_embedding=emb, seed=1)
    his, pred, fresh = _inputs(rng, batch=1, n=4)
    p0 = var.scorer.predict([his, pred[:, 2:3], fresh[:, 2:3]], verbose=0)
    fresh2 = fresh.copy(); fresh2[0, 2, 0] += 3.0
    p1 = var.scorer.predict([his, pred[:, 2:3], fresh2[:, 2:3]], verbose=0)
    assert not np.allclose(p0, p1)
    # a different candidate's input is untouched by that change
    q0 = var.scorer.predict([his, pred[:, 1:2], fresh[:, 1:2]], verbose=0)
    q1 = var.scorer.predict([his, pred[:, 1:2], fresh2[:, 1:2]], verbose=0)
    np.testing.assert_allclose(q0, q1)


def test_loader_shapes_and_mask():
    import polars as pl
    from src.baselines.nrms_fresh_ebrec import NRMSFreshLoader
    beh = pl.DataFrame({
        "user_id": [1, 2], "article_id_fixed": [[10, 11, 0, 0], [12, 0, 0, 0]],
        "article_ids_inview": [[20, 21, 22], [21, 23, 24]], "labels": [[1, 0, 0], [0, 1, 0]],
        "fresh_inview": [[[0.1, 0.0], [0.2, 0.0], [0.0, 1.0]], [[0.3, 0.0], [0.4, 0.0], [0.5, 0.0]]],
    })
    mapping = {a: np.zeros(TITLE, dtype="int32") + a for a in (10, 11, 12, 20, 21, 22, 23, 24)}
    dl = NRMSFreshLoader(behaviors=beh, article_dict=mapping, unknown_representation="zeros",
                         history_column="article_id_fixed", eval_mode=False, batch_size=2)
    (his, pred, fresh), y = dl[0]
    assert his.shape == (2, 4, TITLE) and pred.shape == (2, 3, TITLE) and fresh.shape == (2, 3, FRESH_DIM)
    assert fresh[0, 2].tolist() == [0.0, 1.0]
    ev = NRMSFreshLoader(behaviors=beh, article_dict=mapping, unknown_representation="zeros",
                         history_column="article_id_fixed", eval_mode=True, batch_size=2)
    (his, pred, fresh), y = ev[0]
    assert pred.shape[0] == 6 and fresh.shape == (6, 1, FRESH_DIM)
    masked = NRMSFreshLoader(behaviors=beh, article_dict=mapping, unknown_representation="zeros",
                             history_column="article_id_fixed", eval_mode=True, batch_size=2, mask=True)
    (_, _, fm), _ = masked[0]
    assert (fm.reshape(-1, FRESH_DIM) == np.array([0.0, 1.0], dtype="float32")).all()
