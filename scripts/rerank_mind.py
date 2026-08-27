#!/usr/bin/env python3
"""Feature-based learned reranker on MIND, with the frozen-vs-rolling ablation (Q9).

    PYTHONPATH=. .venv/bin/python -u scripts/rerank_mind.py

Trains on `MINDsmall_train`, evaluates on `MINDsmall_dev` (strictly later in time, so the
split is temporal by construction). Features, per candidate:

    rolling popularity   clicks strictly before this impression, total / 24h / 1h windows
    rolling CTR          smoothed clicks/views strictly before, total / 24h
    bm25                 lexical score of candidate vs history text
    semantic             cosine of candidate vs mean-pooled history embedding
    slate_size           number of candidates in the impression
    cat_affinity         share of the user's history in this candidate's category
    history_len          number of prior clicks

The frozen variant replaces every rolling count with a whole-window total, which both leaks
and goes stale. Reporting both is the Q9 with/without-serving-time-features ablation, and
the gap between them is the measurement.

`HistGradientBoostingClassifier` is used rather than LightGBM: it is the same family of
histogram-based GBDT, already present in scikit-learn, and needs no download on a link that
has been the binding constraint all week (SPEC.md §11).
"""
from __future__ import annotations

import json, time
from datetime import timedelta
from pathlib import Path

import numpy as np

from src.eval.bootstrap import bootstrap_metrics, compare
from src.eval.metrics import per_impression_metrics
from src.features.rolling import RollingCounts
from src.lexical.bm25 import BM25
from src.lexical.index import InvertedIndex
from src.lexical.retrieval import build_query
from src.lexical.tokenize import tokenize_fields
from src.pipeline.mind import (read_news, scan_behaviors, split_dir, with_history,
                               with_labelled_candidates, with_parsed_time)
from src.semantic.ann import ANNIndex
from src.semantic.embeddings import compute_lsa
from src.semantic.user_vector import build_user_vector

ROOT, OUT = Path("data/interim/mind"), Path("data/processed")
N_RECENT, SEED = 5, 0
FEATURES = ["pop_total", "pop_24h", "pop_1h", "ctr_total", "ctr_24h",
            "bm25", "semantic", "slate_size", "cat_affinity", "history_len"]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load(split_d):
    return with_history(with_labelled_candidates(with_parsed_time(scan_behaviors(split_d)))).select(
        "ts", "history_ids", "candidates", "labels").collect()


def build_counts(df) -> RollingCounts:
    """Event stream from labelled impressions. Views and clicks both carry the impression time."""
    rc = RollingCounts()
    for ts, cands, labels in zip(df["ts"], df["candidates"], df["labels"]):
        for c, l in zip(cands.to_list(), labels.to_list()):
            rc.add_view(c, ts)
            if l == 1:
                rc.add_click(c, ts)
    rc.seal()
    return rc


def featurise(df, rc, bm, text, ann, matrix, news_cat, frozen: bool, far_future):
    """Rows of feature vectors plus per-impression grouping. `frozen=True` is the leaky variant."""
    X, y, groups = [], [], []
    W24, W1 = timedelta(hours=24), timedelta(hours=1)
    for ts, hist_s, cands_s, labels_s in zip(df["ts"], df["history_ids"], df["candidates"], df["labels"]):
        hist, cands, labels = hist_s.to_list(), cands_s.to_list(), labels_s.to_list()
        if not any(labels):
            continue
        # Frozen: every count evaluated at a far-future time, i.e. the whole window at once.
        at = far_future if frozen else ts
        q = build_query(hist, text, n_recent=N_RECENT)
        bm_scores = bm.score_candidates(q, cands)
        uv = build_user_vector(hist, ann.id_to_row, matrix, pooling="mean")
        sem_scores = ann.score_candidates(uv, cands)

        hist_cats = [news_cat.get(h) for h in hist if h in news_cat]
        cat_n = len(hist_cats) or 1
        cat_share = {}
        for c in hist_cats:
            cat_share[c] = cat_share.get(c, 0) + 1

        n = len(cands)
        for i, c in enumerate(cands):
            X.append([
                rc.clicks_before(c, at), rc.clicks_before(c, at, W24), rc.clicks_before(c, at, W1),
                rc.ctr_before(c, at), rc.ctr_before(c, at, W24),
                bm_scores[i], sem_scores[i], n,
                cat_share.get(news_cat.get(c), 0) / cat_n, len(hist),
            ])
            y.append(labels[i])
        groups.append(n)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int8), groups


def regroup(scores, groups):
    out, i = [], 0
    for n in groups:
        out.append(scores[i:i + n]); i += n
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train_d, dev_d = split_dir(ROOT, "MINDsmall_train"), split_dir(ROOT, "MINDsmall_dev")

    log("corpus + indices")
    text, order = {}, []
    for d in (train_d, dev_d):
        n = read_news(d)
        for a, ti, ab in zip(n["news_id"], n["title"], n["abstract"]):
            if a not in text:
                text[a] = (ti, ab); order.append(a)
    news_cat = {}
    for d in (train_d, dev_d):
        n = read_news(d)
        for a, c in zip(n["news_id"], n["category"]):
            news_cat.setdefault(a, c)

    idx = InvertedIndex(); idx.enable_forward_index()
    for a in order:
        idx.add(a, tokenize_fields(*text[a], lang="en"))
    bm = BM25(idx, idf_variant="lucene")
    matrix, _, _ = compute_lsa([f"{text[a][0] or ''} {text[a][1] or ''}" for a in order], n_components=128, seed=SEED)
    ann = ANNIndex(order, matrix, kind="flat")
    log(f"  {idx.n_docs:,} articles indexed")

    train, dev = load(train_d), load(dev_d)
    log(f"  train {train.height:,}  dev {dev.height:,}")
    log(f"  train ts {train['ts'].min()} .. {train['ts'].max()}")
    log(f"  dev   ts {dev['ts'].min()} .. {dev['ts'].max()}  (strictly later — temporal split)")

    # Counts are built from TRAIN ONLY. Including dev events would let a dev impression's
    # own slate inform its features, which is leakage of a different kind from the temporal one.
    rc = build_counts(train)
    log(f"  rolling counts over {rc.n_articles:,} articles")
    far_future = dev["ts"].max() + timedelta(days=3650)

    from sklearn.ensemble import HistGradientBoostingClassifier

    results, cis = {}, {}
    for variant, frozen in (("rolling", False), ("frozen", True)):
        log(f"featurising ({variant})")
        t0 = time.perf_counter()
        Xtr, ytr, _ = featurise(train, rc, bm, text, ann, matrix, news_cat, frozen, far_future)
        Xdv, ydv, gdv = featurise(dev, rc, bm, text, ann, matrix, news_cat, frozen, far_future)
        log(f"  train {Xtr.shape}  dev {Xdv.shape}  ({time.perf_counter()-t0:.0f}s)")

        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                               max_leaf_nodes=31, random_state=SEED)
        model.fit(Xtr, ytr)
        scores = model.predict_proba(Xdv)[:, 1]
        rows = list(zip(regroup(ydv.tolist(), gdv), regroup(scores.tolist(), gdv)))
        cis[variant] = bootstrap_metrics(per_impression_metrics(rows), iterations=1000, seed=SEED)
        results[variant] = {k: {"mean": v.mean, "lo": v.lo, "hi": v.hi, "n": v.n}
                            for k, v in cis[variant].items()}
        log(f"  {variant:8s} AUC {cis[variant]['auc']}  nDCG@10 {cis[variant]['ndcg@10']}")

        if variant == "rolling":
            from sklearn.inspection import permutation_importance
            sub = min(30000, len(Xdv))
            imp = permutation_importance(model, Xdv[:sub], ydv[:sub], n_repeats=3,
                                         random_state=SEED, scoring="roc_auc")
            ranked = sorted(zip(FEATURES, imp.importances_mean), key=lambda kv: -kv[1])
            results["feature_importance"] = {k: float(v) for k, v in ranked}
            log("  permutation importance (AUC drop when shuffled):")
            for k, v in ranked:
                log(f"    {k:14s} {v:+.4f}")

    log(f"  {compare(cis['frozen']['auc'], cis['rolling']['auc'], 'frozen', 'rolling')}")
    results["frozen_minus_rolling_auc"] = cis["frozen"]["auc"].mean - cis["rolling"]["auc"].mean
    (OUT / "rerank_mind.json").write_text(json.dumps(results, indent=2))
    log("done")


if __name__ == "__main__":
    main()
