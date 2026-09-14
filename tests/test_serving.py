"""Oracles for the per-request serving path (SPEC.md §16.3).

The serving path must produce exactly the features the batch path trains on — A1's submission 3
lost 0.09 AUC to a train/serve divergence of this kind — so the central test is parity with
`scripts.rerank_ebnerd_a2.build` on real impressions. The toy tests cover the request contract
and the no-future-leak rule without data or the embedding asset.
"""
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.serving.request import Request, features_for, retrieve
from src.serving.state import SessionStore, UserStore

T0 = datetime(2023, 5, 24, 12, 0)
H = timedelta(hours=1)


# ---- toy: the session store and the user store obey strict < t ---------------------------------------

def test_session_store_position_is_pre_t_count():
    beh = pl.DataFrame({"imp_row": pl.Series([0, 1, 2, 3], dtype=pl.UInt32), "user_id": [1, 1, 1, 2],
                        "session_id": [5, 5, 5, 9], "t": [T0, T0 + H, T0 + H, T0]})
    store = SessionStore.from_behaviors(beh)
    assert store.position(0) == 1 and store.position(1) == 2 and store.position(2) == 2   # tie at t: not prior
    assert store.position(3) == 1


def test_user_store_last_n_and_log_strictly_before_t():
    hist = pl.DataFrame({"user_id": [7, 7, 7], "ts": [T0 - 3 * H, T0 - 1 * H, T0 + 1 * H],
                         "article_id": [10, 11, 12], "category": ["a", "b", "a"]})
    store = UserStore.from_click_log(hist, n_recent=2)
    assert store.recent(7) == [11, 12]                     # last n ids, as recent_history gives them
    log = store.log(7)
    assert log["ts"].to_list() == sorted(log["ts"].to_list())
    # the profile functions filter < t themselves; the store hands over the whole log
    assert log.height == 3


# ---- real data: parity with the batch path --------------------------------------------------------

EMB = Path("data/interim/ebnerd/Ekstra_Bladet_word2vec/document_vector.parquet")
needs = pytest.mark.skipif(not (EMB.exists() and Path("data/interim/ebnerd/ebnerd_small/validation/behaviors.parquet").exists()),
                           reason="needs make data and the word2vec asset (P4 U0)")


@pytest.fixture(scope="module")
def ebnerd_state():
    from src.serving.state import ServingState
    return ServingState.build("ebnerd")


@needs
def test_parity_with_batch_features_and_scores(ebnerd_state):
    """200 seeded validation impressions: per-request rows == batch rows (config.FINAL features)."""
    from scripts.rerank_ebnerd_a2 import build, seeded_sample
    from src.rerank.config import FINAL
    from src.rerank.ebnerd import SMALL, load_articles, load_behaviors
    from src.serving.models import load_or_fit
    from src.serving.request import serve
    st = ebnerd_state
    val = load_behaviors(SMALL / "validation/behaviors.parquet")
    sample = seeded_sample(val.height, 200, 1)
    batch, _ = build(val, SMALL / "validation/history.parquet", sample, load_articles(), st.stage1)
    feats = FINAL["ebnerd"]["features"]
    model = load_or_fit("ebnerd", st)
    sbeh = val.filter(pl.col("imp_row").is_in(sample))
    for row in sbeh.sort("imp_row").iter_rows(named=True):
        req = Request(user_id=row["user_id"], t=row["t"], candidates=list(row["candidates"]), imp_row=row["imp_row"])
        ref = batch.filter(pl.col("imp_row") == row["imp_row"]).sort("cand_position")
        X_ref = ref.select(feats).to_numpy().astype(float)
        for cache in (True, False):          # the served (cached-profile) path and the row-function path
            resp = serve(st, model, req, framing="a", profile_cache=cache)
            assert resp.candidates == ref["article_id"].to_list()
            np.testing.assert_allclose(resp.X, X_ref, atol=1e-9, equal_nan=True, err_msg=f"{row['imp_row']} cache={cache}")
    # scores: the batch path's predict on the same matrix
    from src.rerank.common import matrix, predict_scores
    ref_scores = predict_scores(model, matrix(batch.sort("imp_row", "cand_position"), feats))
    got = np.concatenate([serve(st, model, Request(r["user_id"], r["t"], list(r["candidates"]), r["imp_row"]), framing="a").scores
                          for r in sbeh.sort("imp_row").iter_rows(named=True)])
    np.testing.assert_allclose(got, ref_scores, atol=1e-9)


@needs
def test_retrieval_set_is_the_union_of_both_generators(ebnerd_state):
    st = ebnerd_state
    from src.rerank.ebnerd import SMALL, load_behaviors
    row = load_behaviors(SMALL / "validation/behaviors.parquet", limit=50).row(7, named=True)
    req = Request(user_id=row["user_id"], t=row["t"], candidates=None, imp_row=row["imp_row"])
    ids, q, uv = retrieve(st, req, k=100)
    bm = [a for a, _ in st.stage1.bm.search(q, top_k=100)]
    ann = [a for a, _ in st.stage1.ann.search(uv, top_k=100)] if uv is not None else []
    assert ids == bm + [a for a in ann if a not in set(bm)]
    assert 0 < len(ids) <= 200 and len(set(ids)) == len(ids)


@needs
def test_no_future_leak_in_counts_and_freshness(ebnerd_state):
    """Moving t earlier than an article's clicks removes them from pop/ctr; a publish time at or
    after t makes freshness NaN."""
    st = ebnerd_state
    aid = next(iter(st.stage1.rc._clicks))
    ts = st.stage1.rc._clicks[aid]
    t_after, t_before = ts[-1] + timedelta(seconds=1), ts[0]
    f_after = features_for(st, Request(0, t_after, [aid], None), [aid], q=[], uv=None, session_pos=1)
    f_before = features_for(st, Request(0, t_before, [aid], None), [aid], q=[], uv=None, session_pos=1)
    i = st.features.index("pop_total")
    assert f_after[0][i] == len(ts) and f_before[0][i] == 0
    j = st.features.index("freshness_hours")
    pub = st.article_time.get(aid)
    if pub is not None:
        f_at = features_for(st, Request(0, pub, [aid], None), [aid], q=[], uv=None, session_pos=1)
        assert math.isnan(f_at[0][j])


@needs
def test_memory_report_names_every_component(ebnerd_state):
    mem = ebnerd_state.memory
    for k in ("articles", "bm25_index", "ann_index", "counts", "user_store", "session_store"):
        assert k in mem and mem[k]["ram_bytes"] > 0, k
    assert mem["ann_index"]["ram_bytes"] == ebnerd_state.stage1.ann.matrix.nbytes
