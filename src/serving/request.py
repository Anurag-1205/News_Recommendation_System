"""One request through the two-stage system, with per-stage timing (SPEC.md §16.1).

    serve(state, model, Request(user_id, t, candidates, imp_row), framing="a"|"b", k=100)

Stage 1 (framing (b)): BM25 over the last-5-click query ∪ flat-ANN over the mean-pooled user
vector, K each, union in BM25-then-ANN order. Framing (a): the impression's own candidates.
Stage 2: every `config.FINAL` feature from the same function the batch path uses, then
`predict_scores`. `features_for` is the per-request twin of `Stage1.add` + `add_phase1_features`
+ `candidate_frame` (scripts/rerank_ebnerd_a2.py, src/rerank/ebnerd.py); the parity test holds
the two together.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from src.features.behavioural import category_match, decay_weights, recency_weighted_profile
from src.lexical.retrieval import build_query
from src.rerank.common import predict_scores
from src.semantic.user_vector import build_user_vector

H_INF = timedelta.max            # h = ∞ (C-014)
N_RECENT = 5                     # A1's query / user-vector length


@dataclass
class Request:
    user_id: int
    t: datetime
    candidates: list | None      # framing (a): the impression's list; framing (b): None
    imp_row: int | None          # key into the session store (None -> position 1)


@dataclass
class Response:
    candidates: list
    scores: np.ndarray
    X: np.ndarray
    timings: dict = field(default_factory=dict)   # seconds per stage


def _query(state, req: Request):
    hist = state.users.recent(req.user_id)
    q = build_query(hist, state.stage1.text, n_recent=N_RECENT, lang=_lang(state))
    uv = build_user_vector(hist, state.stage1.ann.id_to_row, state.stage1.ann.matrix, pooling="mean")
    return hist, q, uv


def _lang(state):
    return "da" if state.dataset == "ebnerd" else "en"


def retrieve(state, req: Request, k: int, timings: dict | None = None):
    """Framing (b): top-K from each generator, union in BM25-then-ANN order. Sub-timers for the
    two scans go into `timings` (they scale differently: postings vs ntotal × d)."""
    t0 = time.perf_counter()
    _, q, uv = _query(state, req)
    t1 = time.perf_counter()
    ids = [a for a, _ in state.stage1.bm.search(q, top_k=k)]
    t2 = time.perf_counter()
    seen = set(ids)
    if uv is not None:
        ids += [a for a, _ in state.stage1.ann.search(uv, top_k=k) if a not in seen]
    t3 = time.perf_counter()
    if timings is not None:
        timings.update(query=t1 - t0, bm25_search=t2 - t1, ann_search=t3 - t2, query_tokens=len(q))
    return ids, q, uv


def _profile_masses(state, req: Request):
    """The user's decayed click mass per category at t, computed ONCE per request. Both category
    features are functions of these masses (§11.1–11.3): P(c) = m_c / Σm, cos = m_c / ||m||₂.
    The per-candidate row functions recompute `decay_weights` for every candidate; the bench's
    first run measured that at ~2 ms per candidate, so the cached form is the served one. Parity
    with the batch path is tested either way."""
    w = decay_weights(state.users.log(req.user_id), req.user_id, req.t, H_INF, untimed_ts=state.untimed_ts)
    if w.height == 0:
        return None
    g = w.group_by("category", maintain_order=True).agg(pl_sum=pl.col("weight").sum())
    masses = dict(zip(g["category"].to_list(), g["pl_sum"].to_list()))
    total = float(sum(masses.values()))
    norm = math.sqrt(sum(v * v for v in masses.values()))
    return masses, total, norm


def features_for(state, req: Request, cands: list, *, q, uv, session_pos: int, profile_cache: bool = True) -> list[list[float]]:
    """One row per candidate, columns in `state.features` order — the batch path's definitions."""
    st = state.stage1
    hist = state.users.recent(req.user_id)
    log = state.users.log(req.user_id)
    t = req.t
    pm = _profile_masses(state, req) if profile_cache else None
    bm25 = st.bm.score_candidates(q, cands)
    sem = st.ann.score_candidates(uv, cands)
    n = len(cands)
    rows = []
    for pos, (c, b, s) in enumerate(zip(cands, bm25, sem), start=1):
        cat = state.article_cat.get(c)
        vals = {
            "bm25": b, "semantic": s,
            "pop_total": st.rc.clicks_before(c, t), "ctr_total": st.rc.ctr_before(c, t),
            "freshness_hours": _freshness(state, c, t),
            "n_candidates": n, "history_len": len(hist),
            "cand_position": pos, "session_pos": session_pos,
        }
        if "recency_weighted_profile" in state.features or "category_match" in state.features:
            if profile_cache:
                if pm is None:
                    vals["recency_weighted_profile"] = vals["category_match"] = math.nan
                else:
                    masses, total, norm = pm
                    m = masses.get(cat, 0.0)
                    vals["recency_weighted_profile"], vals["category_match"] = m / total, m / norm
            else:
                vals["recency_weighted_profile"] = recency_weighted_profile(log, req.user_id, cat, t, H_INF, untimed_ts=state.untimed_ts)
                vals["category_match"] = category_match(log, req.user_id, cat, t, H_INF, untimed_ts=state.untimed_ts)
        if "cat_affinity" in state.features:
            vals["cat_affinity"] = state.cat_affinity(req.user_id, cat)
        rows.append([float(vals[f]) for f in state.features])
    return rows


def _freshness(state, article_id, t: datetime) -> float:
    """§11.8: hours since the article was first known, strictly before t; NaN otherwise."""
    fk = state.article_time.get(article_id)
    if fk is None or not (fk < t):
        return math.nan
    return (t - fk).total_seconds() / 3600.0


def serve(state, model, req: Request, *, framing: str = "a", k: int = 100, profile_cache: bool = True) -> Response:
    tm = {}
    t0 = time.perf_counter()
    if framing == "b":
        cands, q, uv = retrieve(state, req, k, timings=tm)
    else:
        cands = list(req.candidates)
        _, q, uv = _query(state, req)
    tm["retrieve"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    session_pos = state.sessions.position(req.imp_row) if req.imp_row is not None else 1
    X = np.asarray(features_for(state, req, cands, q=q, uv=uv, session_pos=session_pos, profile_cache=profile_cache), dtype=float).reshape(len(cands), len(state.features))
    tm["features"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    scores = predict_scores(model, X) if len(cands) else np.zeros(0)
    tm["score"] = time.perf_counter() - t2
    tm["total"] = time.perf_counter() - t0
    return Response(cands, np.asarray(scores, dtype=float), X, tm)
