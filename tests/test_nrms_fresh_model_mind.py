"""Model-level oracles for the MIND (recommenders) freshness variant (SPEC.md §15.4). Need
tf-keras (TF_USE_LEGACY_KERAS=1), the recommenders package on sys.path and the MIND utils
(`NRMS_YAML`, `NRMS_EMB`, `NRMS_WDICT`, `NRMS_UDICT` env vars) — so they skip on the laptop and
run inside the Kaggle kernel before training."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
for p in ("external/recommenders", "/kaggle/working/recommenders"):
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)
rec = pytest.importorskip("recommenders")
if not all(os.environ.get(k) for k in ("NRMS_YAML", "NRMS_EMB", "NRMS_WDICT", "NRMS_UDICT")):
    pytest.skip("MIND utils env vars not set", allow_module_level=True)

from recommenders.models.newsrec.newsrec_utils import prepare_hparams   # noqa: E402
from recommenders.models.newsrec.models.nrms import NRMSModel            # noqa: E402
from recommenders.models.newsrec.io.mind_iterator import MINDIterator    # noqa: E402
from src.baselines.nrms_fresh_rec import FRESH_DIM, MINDFreshIterator, NRMSFreshModel  # noqa: E402


def _hparams():
    return prepare_hparams(os.environ["NRMS_YAML"], wordEmb_file=os.environ["NRMS_EMB"], wordDict_file=os.environ["NRMS_WDICT"],
                           userDict_file=os.environ["NRMS_UDICT"], batch_size=4, epochs=1, show_step=10**9)


def _inputs(rng, hp, batch=3):
    vocab = 1000
    his = rng.integers(1, vocab, size=(batch, hp.his_size, hp.title_size)).astype("int32")
    pred = rng.integers(1, vocab, size=(batch, hp.npratio + 1, hp.title_size)).astype("int32")
    fresh = np.stack([rng.standard_normal((batch, hp.npratio + 1)), rng.integers(0, 2, (batch, hp.npratio + 1))], -1).astype("float32")
    return his, pred, fresh


def test_additive_identity_g_zero_reproduces_baseline():
    hp = _hparams()
    base = NRMSModel(hp, MINDIterator, seed=42)
    var = NRMSFreshModel(hp, MINDFreshIterator, seed=42)
    var.copy_encoders_from(base); var.zero_g()
    his, pred, fresh = _inputs(np.random.default_rng(0), hp)
    np.testing.assert_allclose(var.model.predict([his, pred, fresh]), base.model.predict([his, pred]), atol=1e-6)
    np.testing.assert_allclose(var.scorer.predict([his, pred[:, :1], fresh[:, :1]]), base.scorer.predict([his, pred[:, :1]]), atol=1e-6)


def test_freshness_term_is_per_candidate():
    hp = _hparams()
    var = NRMSFreshModel(hp, MINDFreshIterator, seed=42)
    his, pred, fresh = _inputs(np.random.default_rng(1), hp, batch=1)
    p0 = var.model.predict([his, pred, fresh])
    fresh2 = fresh.copy(); fresh2[0, 2, 0] += 3.0
    p1 = var.model.predict([his, pred, fresh2])
    assert not np.allclose(p0, p1)
    g = var.g_values(np.array([[0.0, 1.0], [0.0, 1.0]], dtype="float32"))
    assert g.shape == (2,) and g[0] == g[1]


def test_iterator_yields_fresh_batch_and_mask(tmp_path):
    """A 2-line behaviors file: the fresh batch lines up with the sampled candidates."""
    hp = _hparams()
    news = tmp_path / "news.tsv"; beh = tmp_path / "behaviors.tsv"
    news.write_text("\n".join(f"N{i}\tsports\tfootball\ttitle {i} words here\tabs\turl\t[]\t[]" for i in range(1, 8)) + "\n")
    beh.write_text("1\tU1\t11/15/2019 12:00:00 PM\tN1 N2\tN3-1 N4-0 N5-0 N6-0 N7-0\n"
                   "2\tU2\t11/15/2019 12:05:00 PM\tN2\tN4-0 N5-1 N6-0\n")
    it = MINDFreshIterator(hp, npratio=hp.npratio, col_spliter="\t")
    it.fresh_by_row = [{f"N{i}": (0.1 * i, 0.0) for i in range(3, 8)}, {"N4": (0.4, 0.0), "N5": (0.5, 0.0)}]   # N6 unknown in row 2
    batches = list(it.load_data_from_file(str(news), str(beh)))
    b = batches[0]
    assert b["candidate_fresh_batch"].shape == (2, hp.npratio + 1, FRESH_DIM)
    # first sample of each line is its positive: row 0 → N3 (0.3), row 1 → N5 (0.5); order may be shuffled
    firsts = sorted(b["candidate_fresh_batch"][:, 0, 0].round(3).tolist())
    assert firsts == [0.3, 0.5]
    it.mask = True
    bm = next(it.load_data_from_file(str(news), str(beh)))
    assert (bm["candidate_fresh_batch"].reshape(-1, FRESH_DIM) == np.array([0.0, 1.0], dtype="float32")).all()
