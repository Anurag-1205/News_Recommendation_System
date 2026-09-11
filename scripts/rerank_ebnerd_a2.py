#!/usr/bin/env python3
"""A2 Q2 on EB-NeRD: stage-1 scores (A1) + Phase 1 behavioural features -> GBDT reranker.

    PYTHONPATH=. .venv/bin/python -u scripts/rerank_ebnerd_a2.py [--fit-impressions 100000]
                                         [--eval-impressions 100000] [--smoke-test-file 5000]

Protocol: **fit on the ebnerd_small train week, evaluate on the validation week** — the shipped
temporal split. A1's EB-NeRD reranker instead split the validation file 70/30 by *row order*;
behaviors.parquet is not sorted by time, so that was not a temporal split (CONTEXT.md C-015).
Impressions are seeded random samples of each split (sampling, not splitting).

Ranking rows, all scored on the same validation impressions (framing (a) of PLAN D1: re-rank the
impression's own candidates, which is what both leaderboards score):

  stage 1   BM25 alone, word2vec cosine alone    — A1's two candidate generators, in-impression
  GBDT      base: A1's skew-resistant feature set (windowed popularity dropped, as A1 v4 did)
  GBDT      A2:   base + Phase 1 safe features, h = inf (P1-D2, CONTEXT.md C-014)
  GBDT      A2, h = 72 h — the same, testing the half-life choice inside the full model
  lambdarank base and A2 — the same two feature sets under LightGBM lambdarank, one query per
            impression (D2, CONTEXT.md C-016); if A2 drops against base under the fixed rule in
            `drop_reasons`, each Phase 1 family is removed in turn and the model refitted

Every feature list passes through `model_features`, which removes UNSAFE_FEATURES and
ABSENT_FROM_TEST_FILE; the script asserts they are gone. Popularity counts come from train
events only, so validation — like the test file — sees counts that stop at the split boundary.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from src.features.behavioural import ABSENT_FROM_TEST_FILE, UNSAFE_FEATURES
from src.features.rolling import RollingCounts
from src.lexical.bm25 import BM25
from src.lexical.index import InvertedIndex
from src.lexical.retrieval import build_query
from src.lexical.tokenize import tokenize_fields
from src.pipeline.ebnerd import LANG, article_text, recent_history
from src.rerank.common import (SEED, category_profile_features, conditional_ablation, fit_final, fit_gbdt,
                               fit_lambdarank, group_sizes, matrix, model_features, predict_scores, report)
from src.rerank.config import FINAL
from src.rerank.ebnerd import (ARTICLES, SMALL, TEST, add_phase1_features, candidate_frame,
                               load_articles, load_behaviors, load_history)
from src.semantic.ann import ANNIndex
from src.semantic.embeddings import load_provided
from src.semantic.user_vector import build_user_vector

EMB = Path("data/interim/ebnerd/Ekstra_Bladet_word2vec/document_vector.parquet")
OUT = Path("data/processed")
N_RECENT = 5                                   # A1's stage-1 query / user vector length
H = timedelta.max                              # h = inf, chosen in C-014
H_ABLATION = timedelta(hours=72)

BASE = ["bm25", "semantic", "pop_total", "ctr_total", "freshness_hours", "n_candidates", "history_len"]
PHASE1_SAFE = ["recency_weighted_profile", "category_match", "cand_position", "session_pos",
               "hist_read_time_mean", "hist_scroll_mean"]
COMPUTED_BUT_BANNED = ["n_prior_clicks_in_session", "session_len", "cur_read_time", "cur_scroll_percentage"]
# The Phase 1 additions, grouped as they were built, for the leave-one-family-out ablation.
PHASE1_FAMILIES = {"category profile": ["recency_weighted_profile", "category_match"],
                   "list position": ["cand_position"], "session": ["session_pos"],
                   "dwell": ["hist_read_time_mean", "hist_scroll_mean"]}
assert sorted(sum(PHASE1_FAMILIES.values(), [])) == sorted(PHASE1_SAFE)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class Stage1:
    """A1's candidate generators and popularity counts, unchanged."""

    def __init__(self, train_beh: pl.DataFrame):
        self.text = article_text(ARTICLES)
        idx = InvertedIndex(); idx.enable_forward_index()
        for aid, (ti, sub) in self.text.items():
            idx.add(aid, tokenize_fields(ti, sub, lang=LANG))
        self.bm = BM25(idx, idf_variant="lucene")
        ids, emb = load_provided(EMB)
        self.ann = ANNIndex(ids, emb, kind="flat")
        self.rc = RollingCounts()                  # train events only: point-in-time for train rows,
        for ts, cands, clicked in zip(train_beh["t"], train_beh["candidates"], train_beh["clicked"]):
            cl = set(clicked.to_list())            # frozen at the boundary for validation / test rows
            for c in cands.to_list():
                self.rc.add_view(c, ts)
                if c in cl:
                    self.rc.add_click(c, ts)
        self.rc.seal()
        log(f"  stage 1: {idx.n_docs:,} articles indexed, word2vec {emb.shape}, counts over {self.rc.n_articles:,} articles")

    def add(self, long: pl.DataFrame, sample_beh: pl.DataFrame, hist5: dict) -> pl.DataFrame:
        """bm25, semantic, pop_total, ctr_total, history_len per candidate row. `sample_beh` is in
        imp_row order and `long` is sorted by (imp_row, cand_position), so they align by position."""
        cols = {k: [] for k in ("bm25", "semantic", "pop_total", "ctr_total", "history_len")}
        for uid, ts, cd in zip(sample_beh["user_id"], sample_beh["t"], sample_beh["candidates"]):
            cands = cd.to_list()
            if not cands:
                continue
            hist = hist5.get(uid, [])
            q = build_query(hist, self.text, n_recent=N_RECENT, lang=LANG)
            uv = build_user_vector(hist, self.ann.id_to_row, self.ann.matrix, pooling="mean")
            cols["bm25"] += self.bm.score_candidates(q, cands)
            cols["semantic"] += self.ann.score_candidates(uv, cands)
            cols["pop_total"] += [self.rc.clicks_before(c, ts) for c in cands]
            cols["ctr_total"] += [self.rc.ctr_before(c, ts) for c in cands]
            cols["history_len"] += [len(hist)] * len(cands)
        assert len(cols["bm25"]) == long.height, "stage-1 scores misaligned with the candidate frame"
        return long.with_columns(*(pl.Series(k, v, dtype=pl.Float64) for k, v in cols.items()))


def seeded_sample(n_total: int, n: int, seed: int) -> np.ndarray:
    """Sorted row numbers of a seeded random sample: the evaluation impressions are *sampled*
    from a split, never split out of it. The same (n_total, n, seed) always gives the same rows."""
    return np.sort(np.random.default_rng(seed).choice(n_total, size=min(n, n_total), replace=False))


def build(beh_full, history_path, sample, articles, stage1, limit_note=""):
    """Candidate frame + every feature for the impressions whose `imp_row` is in `sample`."""
    sbeh = beh_full.filter(pl.col("imp_row").is_in(sample))
    history = load_history(history_path, articles, users=sbeh["user_id"])
    t0 = time.perf_counter()
    long = add_phase1_features(candidate_frame(sbeh, articles), beh_full, history, articles, H)
    hist5 = recent_history(history_path, N_RECENT, users=set(sbeh["user_id"].to_list()))
    long = stage1.add(long, sbeh, hist5)
    log(f"  {sbeh.height:,} impressions{limit_note} -> {long.height:,} rows, {len(long.columns)} columns "
        f"({time.perf_counter() - t0:.0f}s); history {history.height:,} clicks")
    return long, history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-impressions", type=int, default=100_000)
    ap.add_argument("--eval-impressions", type=int, default=100_000)
    ap.add_argument("--smoke-test-file", type=int, default=5_000, help="0 to skip")
    args = ap.parse_args()

    a2 = model_features(BASE + PHASE1_SAFE + COMPUTED_BUT_BANNED)
    assert a2 == BASE + PHASE1_SAFE, a2
    assert not set(a2) & (set(UNSAFE_FEATURES) | ABSENT_FROM_TEST_FILE)
    log(f"A2 model features ({len(a2)}): {a2}")
    log(f"excluded by model_features: {COMPUTED_BUT_BANNED}")

    articles = load_articles()
    train_beh = load_behaviors(SMALL / "train/behaviors.parquet")
    val_beh = load_behaviors(SMALL / "validation/behaviors.parquet")
    log(f"train {train_beh.height:,} impressions ({train_beh['t'].min()} .. {train_beh['t'].max()}); "
        f"validation {val_beh.height:,} ({val_beh['t'].min()} .. {val_beh['t'].max()})")
    stage1 = Stage1(train_beh)

    log("building the train (fit) frame")
    tr, tr_hist = build(train_beh, SMALL / "train/history.parquet",
                        seeded_sample(train_beh.height, args.fit_impressions, SEED), articles, stage1)
    log("building the validation (evaluation) frame")
    va, va_hist = build(val_beh, SMALL / "validation/history.parquet",
                        seeded_sample(val_beh.height, args.eval_impressions, SEED + 1), articles, stage1)
    del train_beh, val_beh
    fresh_nan = va["freshness_hours"].is_nan().sum()
    log(f"freshness NaN on validation (published_time >= t or unknown): {fresh_nan:,} of {va.height:,} "
        f"({fresh_nan / va.height:.2%})")

    y_tr = tr["label"].to_numpy()
    models = {}
    for name, cols in (("gbdt_base", BASE), ("gbdt_a2", a2)):
        t0 = time.perf_counter()
        models[name] = fit_gbdt(matrix(tr, cols), y_tr)
        va = va.with_columns(pl.Series(name, models[name].predict_proba(matrix(va, cols))[:, 1]))
        log(f"  fitted {name} on {tr['imp_row'].n_unique():,} impressions, {len(cols)} features "
            f"({time.perf_counter() - t0:.0f}s)")

    log(f"half-life ablation: recompute the category features with h = 72 h")
    prof = ["recency_weighted_profile", "category_match"]
    tr72 = tr.drop(prof).join(category_profile_features(tr, tr_hist, H_ABLATION), on=["imp_row", "cand_position"])
    va72 = va.drop(prof).join(category_profile_features(va, va_hist, H_ABLATION), on=["imp_row", "cand_position"])
    m72 = fit_gbdt(matrix(tr72.sort("imp_row", "cand_position"), a2), y_tr)
    va = va.with_columns(pl.Series("gbdt_a2_h72", m72.predict_proba(matrix(va72.sort("imp_row", "cand_position"), a2))[:, 1]))
    del tr72, va72

    log("LightGBM lambdarank (D2): one query per impression")
    groups = group_sizes(tr)
    log(f"  {len(groups):,} queries, mean {groups.mean():.1f} candidates, max {groups.max()}")
    for name, cols in (("lr_base", BASE), ("lr_a2", a2)):
        t0 = time.perf_counter()
        models[name] = fit_lambdarank(matrix(tr, cols), y_tr, groups)
        va = va.with_columns(pl.Series(name, models[name].predict(matrix(va, cols))))
        log(f"  fitted {name}, {len(cols)} features ({time.perf_counter() - t0:.0f}s)")

    final = FINAL["ebnerd"]                      # the locked configuration (C-018)
    assert final["features"] == [f for f in a2 if f not in PHASE1_FAMILIES["dwell"]], \
        "src/rerank/config.py drifted from the measured model (A2 without the dwell family)"
    models["final"] = fit_final(final["objective"], matrix(tr, final["features"]), y_tr, groups)
    va = va.with_columns(pl.Series("final", predict_scores(models["final"], matrix(va, final["features"]))))
    log(f"  fitted FINAL: {final['objective']}, {len(final['features'])} features (config.FINAL['ebnerd'])")

    rows = {"bm25": "stage 1: BM25 only", "semantic": "stage 1: word2vec only",
            "gbdt_base": "pointwise GBDT: A1 base", "gbdt_a2": "pointwise GBDT: A2, h=inf",
            "gbdt_a2_h72": "pointwise GBDT: A2, h=72h (ablation)",
            "lr_base": "lambdarank: A1 base", "lr_a2": "lambdarank: A2 (base + Phase 1)",
            "final": "FINAL: lambdarank, A2 without dwell"}
    results = {"eval_impressions": int(va["imp_row"].n_unique()), "eval_rows": va.height,
               "fit_impressions": int(tr["imp_row"].n_unique()), "features_a2": a2, "features_base": BASE}
    log(f"validation metrics, {results['eval_impressions']:,} impressions (bootstrap 95% CI, 1000 resamples)")
    per_imp = report(va, rows, [("gbdt_base", "gbdt_a2"), ("bm25", "gbdt_a2"), ("semantic", "gbdt_a2"),
                                ("gbdt_a2", "gbdt_a2_h72"), ("lr_base", "lr_a2"), ("gbdt_a2", "lr_a2"),
                                ("gbdt_base", "lr_base"), ("bm25", "lr_a2"), ("lr_base", "final"),
                                ("bm25", "final")], log, results)
    conditional_ablation(tr, va, a2, PHASE1_FAMILIES, per_imp, "lr_base", "lr_a2", log, results)

    from sklearn.inspection import permutation_importance
    sub = va.head(60_000)
    imp = permutation_importance(models["gbdt_a2"], matrix(sub, a2), sub["label"].to_numpy(),
                                 n_repeats=3, random_state=SEED, scoring="roc_auc")
    ranked = sorted(zip(a2, imp.importances_mean), key=lambda kv: -kv[1])
    results["permutation_importance_a2"] = {k: float(v) for k, v in ranked}
    log("permutation importance, A2 model (pooled ROC-AUC drop, first 60k validation rows):")
    for k, v in ranked:
        log(f"    {k:26s} {v:+.4f}")

    if args.smoke_test_file:
        log(f"smoke: the unlabelled Codabench test file, first {args.smoke_test_file:,} impressions")
        test_beh = load_behaviors(TEST / "test/behaviors.parquet", limit=args.smoke_test_file)
        assert "clicked" not in test_beh.columns
        tl, _ = build(test_beh, TEST / "test/history.parquet", np.arange(test_beh.height), articles, stage1,
                      limit_note=" (test file)")
        assert "label" not in tl.columns and "n_prior_clicks_in_session" not in tl.columns
        p = predict_scores(models["final"], matrix(tl, final["features"]))
        assert np.isfinite(p).all() and len(p) == tl.height
        results["smoke_test_file"] = {"impressions": test_beh.height, "rows": tl.height,
                                      "absent_columns_confirmed": ["label", "n_prior_clicks_in_session"]}
        log(f"  scored {tl.height:,} test rows; no label, no click count, all predictions finite")

    (OUT / "rerank_ebnerd_a2.json").write_text(json.dumps(results, indent=2))
    log(f"done -> {OUT / 'rerank_ebnerd_a2.json'}")


if __name__ == "__main__":
    main()
