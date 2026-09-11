#!/usr/bin/env python3
"""A2 Q2 on MIND: stage-1 scores (A1) + Phase 1 features -> GBDT reranker.

    PYTHONPATH=. .venv/bin/python -u scripts/rerank_mind_a2.py [--fit-impressions 80000] [--smoke-test-file 5000]

Protocol: fit on MINDsmall_train (9-14 Nov), evaluate on **all** of MINDsmall_dev (15 Nov) — the
shipped temporal split. Popularity counts come from train events only, so dev sees counts that
stop at the split boundary, as the test file would.

Ranking rows on the same dev impressions (PLAN D1 framing (a), in-impression):
  stage 1   BM25 alone; MiniLM cosine alone   — A1's candidate generators
  GBDT      base: A1 v4's feature set (MiniLM, skew-resistant counts, cat_affinity)
  GBDT      A2:   base + category_match, cand_position, freshness_hours (first-seen)
  lambdarank base and A2 — the same feature sets under LightGBM lambdarank, one query per
            impression (D2, CONTEXT.md C-016), plus the conditional leave-one-family-out ablation

`recency_weighted_profile` is *not* added: on MIND its history has no click times, so it equals
A1's `cat_affinity` exactly (C-008) — the script asserts this instead of adding a duplicate column.
MIND has no session or dwell data, so those EB-NeRD features do not exist here.

Text index and category table are built from train + dev news only, so no test-period article
statistics reach the dev evaluation.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from src.features.rolling import RollingCounts
from src.lexical.bm25 import BM25
from src.lexical.index import InvertedIndex
from src.lexical.retrieval import build_query
from src.lexical.tokenize import tokenize_fields
from src.pipeline.mind import read_news, split_dir
from src.rerank.common import (SEED, conditional_ablation, fit_final, fit_gbdt, fit_lambdarank, group_sizes,
                               matrix, model_features, predict_scores, report)
from src.rerank.config import FINAL
from src.rerank.mind import (ROOT, add_phase1_features, candidate_frame, first_sightings, load_behaviors,
                             load_categories)
from src.semantic.ann import ANNIndex
from src.semantic.user_vector import build_user_vector

OUT = Path("data/processed")
N_RECENT = 5
H = timedelta.max                        # irrelevant on MIND (C-008); kept equal to EB-NeRD's choice
TRAIN, DEV, TEST = "MINDsmall_train", "MINDsmall_dev", "MINDlarge_test"

BASE = ["bm25", "semantic", "pop_total", "ctr_total", "n_candidates", "cat_affinity", "history_len"]
PHASE1_NEW = ["category_match", "cand_position", "freshness_hours"]
PHASE1_FAMILIES = {"category match": ["category_match"], "list position": ["cand_position"],
                   "freshness": ["freshness_hours"]}
assert sorted(sum(PHASE1_FAMILIES.values(), [])) == sorted(PHASE1_NEW)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class Stage1:
    """A1 v4's candidate generators, counts and category share, unchanged in definition."""

    def __init__(self, news_splits: list[str], train_beh: pl.DataFrame):
        self.text, self.cat = {}, {}
        for n in news_splits:
            news = read_news(split_dir(ROOT, n))
            for a, ti, ab, c in zip(news["news_id"], news["title"], news["abstract"], news["category"]):
                if a not in self.text:
                    self.text[a], self.cat[a] = (ti, ab), c
        idx = InvertedIndex(); idx.enable_forward_index()
        for a, fields in self.text.items():
            idx.add(a, tokenize_fields(*fields, lang="en"))
        self.bm = BM25(idx, idf_variant="lucene")
        z = np.load(OUT / "mind_minilm.npz", allow_pickle=True)
        self.ann = ANNIndex(list(z["ids"]), z["matrix"].astype(np.float32), kind="flat")
        del z
        self.rc = RollingCounts()
        for ts, cands, labels in zip(train_beh["t"], train_beh["candidates"], train_beh["labels"]):
            for c, l in zip(cands.to_list(), labels.to_list()):
                self.rc.add_view(c, ts)
                if l == 1:
                    self.rc.add_click(c, ts)
        self.rc.seal()
        log(f"  stage 1: {idx.n_docs:,} articles indexed, MiniLM {self.ann.matrix.shape}, counts over {self.rc.n_articles:,}")

    def add(self, long: pl.DataFrame, sample_beh: pl.DataFrame) -> pl.DataFrame:
        cols = {k: [] for k in ("bm25", "semantic", "pop_total", "ctr_total", "cat_affinity", "history_len")}
        for ts, h, cd in zip(sample_beh["t"], sample_beh["history_ids"], sample_beh["candidates"]):
            hist, cands = h.to_list(), cd.to_list()
            if not cands:
                continue
            q = build_query(hist, self.text, n_recent=N_RECENT)
            uv = build_user_vector(hist, self.ann.id_to_row, self.ann.matrix, pooling="mean")
            cats = [self.cat[a] for a in hist if a in self.cat]
            share = {}
            for c in cats:
                share[c] = share.get(c, 0) + 1
            cn = len(cats) or 1
            cols["bm25"] += self.bm.score_candidates(q, cands)
            cols["semantic"] += self.ann.score_candidates(uv, cands)
            cols["pop_total"] += [self.rc.clicks_before(c, ts) for c in cands]
            cols["ctr_total"] += [self.rc.ctr_before(c, ts) for c in cands]
            cols["cat_affinity"] += [share.get(self.cat.get(c), 0) / cn for c in cands]
            cols["history_len"] += [len(hist)] * len(cands)
        assert len(cols["bm25"]) == long.height, "stage-1 scores misaligned with the candidate frame"
        return long.with_columns(*(pl.Series(k, v, dtype=pl.Float64) for k, v in cols.items()))


def build(beh, split_name, sample, categories, sightings, stage1, note=""):
    sbeh = beh.filter(pl.col("imp_row").is_in(sample))
    t0 = time.perf_counter()
    long = add_phase1_features(candidate_frame(sbeh, categories), beh, split_name, categories, sightings, H)
    long = stage1.add(long, sbeh)
    log(f"  {sbeh.height:,} impressions{note} -> {long.height:,} rows ({time.perf_counter() - t0:.0f}s)")
    return long


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-impressions", type=int, default=80_000)
    ap.add_argument("--smoke-test-file", type=int, default=5_000, help="0 to skip")
    args = ap.parse_args()

    a2 = model_features(BASE + PHASE1_NEW)
    assert a2 == BASE + PHASE1_NEW
    log(f"A2 model features ({len(a2)}): {a2}")

    train, dev = load_behaviors(TRAIN), load_behaviors(DEV)
    log(f"train {train.height:,} impressions ({train['t'].min()} .. {train['t'].max()}); "
        f"dev {dev.height:,} ({dev['t'].min()} .. {dev['t'].max()})")
    categories = load_categories([TRAIN, DEV])
    sightings = first_sightings([train, dev])        # strict ts < t keeps dev sightings out of train rows
    stage1 = Stage1([TRAIN, DEV], train)

    rng = np.random.default_rng(SEED)
    fit_sample = np.sort(rng.choice(train.height, size=min(args.fit_impressions, train.height), replace=False))
    log("building the train (fit) frame")
    tr = build(train, TRAIN, fit_sample, categories, sightings, stage1)
    log("building the dev (evaluation) frame — every dev impression")
    va = build(dev, DEV, np.arange(dev.height), categories, sightings, stage1)

    has_hist = va.filter(pl.col("history_len") > 0)
    diff = (has_hist["recency_weighted_profile"] - has_hist["cat_affinity"]).abs().max()
    log(f"check: |recency_weighted_profile - cat_affinity| max over {has_hist.height:,} rows with history = {diff:.2e}")
    assert diff < 1e-9, "the A2 profile should reproduce A1's cat_affinity on MIND (C-008)"
    fresh_nan = va["freshness_hours"].is_nan().sum()
    log(f"freshness NaN on dev (never seen before t): {fresh_nan:,} of {va.height:,} ({fresh_nan / va.height:.2%})")

    y_tr = tr["label"].to_numpy()
    models = {}
    for name, cols in (("gbdt_base", BASE), ("gbdt_a2", a2)):
        t0 = time.perf_counter()
        models[name] = fit_gbdt(matrix(tr, cols), y_tr)
        va = va.with_columns(pl.Series(name, models[name].predict_proba(matrix(va, cols))[:, 1]))
        log(f"  fitted {name} ({len(cols)} features, {time.perf_counter() - t0:.0f}s)")

    log("LightGBM lambdarank (D2): one query per impression")
    groups = group_sizes(tr)
    log(f"  {len(groups):,} queries, mean {groups.mean():.1f} candidates, max {groups.max()}")
    for name, cols in (("lr_base", BASE), ("lr_a2", a2)):
        t0 = time.perf_counter()
        models[name] = fit_lambdarank(matrix(tr, cols), y_tr, groups)
        va = va.with_columns(pl.Series(name, models[name].predict(matrix(va, cols))))
        log(f"  fitted {name}, {len(cols)} features ({time.perf_counter() - t0:.0f}s)")

    final = FINAL["mind"]                        # the locked configuration (C-018)
    assert final["objective"] == "pointwise" and final["features"] == BASE, \
        "src/rerank/config.py drifted from the measured model (pointwise, A1 v4 base)"
    models["final"] = fit_final(final["objective"], matrix(tr, final["features"]), y_tr, groups)
    va = va.with_columns(pl.Series("final", predict_scores(models["final"], matrix(va, final["features"]))))
    assert (va["final"] == va["gbdt_base"]).all(), "the final model must reproduce the measured base fit exactly"
    log(f"  fitted FINAL: {final['objective']}, {len(final['features'])} features; identical to gbdt_base")

    rows = {"bm25": "stage 1: BM25 only", "semantic": "stage 1: MiniLM only",
            "gbdt_base": "pointwise GBDT: A1 v4 base", "gbdt_a2": "pointwise GBDT: A2",
            "lr_base": "lambdarank: A1 v4 base", "lr_a2": "lambdarank: A2 (base + Phase 1)",
            "final": "FINAL: pointwise, A1 v4 base"}
    results = {"eval_impressions": int(va["imp_row"].n_unique()), "eval_rows": va.height,
               "fit_impressions": int(tr["imp_row"].n_unique()), "features_a2": a2, "features_base": BASE,
               "profile_vs_cat_affinity_max_abs_diff": float(diff)}
    log(f"dev metrics, {results['eval_impressions']:,} impressions (bootstrap 95% CI, 1000 resamples)")
    per_imp = report(va, rows, [("gbdt_base", "gbdt_a2"), ("bm25", "gbdt_a2"), ("semantic", "gbdt_a2"),
                                ("lr_base", "lr_a2"), ("gbdt_a2", "lr_a2"), ("gbdt_base", "lr_base"),
                                ("semantic", "lr_a2"), ("semantic", "final")], log, results)
    conditional_ablation(tr, va, a2, PHASE1_FAMILIES, per_imp, "lr_base", "lr_a2", log, results)

    from sklearn.inspection import permutation_importance
    sub = va.head(60_000)
    imp = permutation_importance(models["gbdt_a2"], matrix(sub, a2), sub["label"].to_numpy(),
                                 n_repeats=3, random_state=SEED, scoring="roc_auc")
    ranked = sorted(zip(a2, imp.importances_mean), key=lambda kv: -kv[1])
    results["permutation_importance_a2"] = {k: float(v) for k, v in ranked}
    log("permutation importance, A2 model (pooled ROC-AUC drop, first 60k dev rows):")
    for k, v in ranked:
        log(f"    {k:18s} {v:+.4f}")

    if args.smoke_test_file:
        log(f"smoke: the unlabelled MINDlarge_test file, first {args.smoke_test_file:,} impressions")
        test = load_behaviors(TEST, labelled=False, limit=args.smoke_test_file)
        assert "labels" not in test.columns
        test_cats = load_categories([TRAIN, DEV, TEST])
        tl = build(test, TEST, np.arange(test.height), test_cats, first_sightings([train, dev, test]), stage1,
                   note=" (test file)")
        assert "label" not in tl.columns
        p = predict_scores(models["final"], matrix(tl, final["features"]))
        assert np.isfinite(p).all() and len(p) == tl.height
        results["smoke_test_file"] = {"impressions": test.height, "rows": tl.height}
        log(f"  scored {tl.height:,} test rows; no label column, all predictions finite")

    (OUT / "rerank_mind_a2.json").write_text(json.dumps(results, indent=2))
    log(f"done -> {OUT / 'rerank_mind_a2.json'}")


if __name__ == "__main__":
    main()
