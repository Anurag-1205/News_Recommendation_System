#!/usr/bin/env python3
"""Train the point-in-time reranker on MIND and generate the Codabench submission.

    PYTHONPATH=. .venv/bin/python -u scripts/submit_mind_v3.py

Also runs a leave-one-out check on `slate_size`. Permutation importance ranked it first by a
wide margin, which is suspicious: slate_size is **constant within an impression**, so it
cannot separate candidates inside one. Shuffling it globally injects variation that does not
exist in the real feature, which inflates its apparent importance. The honest test is to drop
it and refit; only that says whether it earns its place through interactions with other
features.
"""
from __future__ import annotations

import json, time
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from src.eval.bootstrap import bootstrap_metrics, compare
from src.eval.metrics import per_impression_metrics
from src.eval.submission import format_line, ranks_from_scores, validate_file, zip_submission
from src.features.rolling import RollingCounts
from src.lexical.bm25 import BM25
from src.lexical.index import InvertedIndex
from src.lexical.retrieval import build_query
from src.lexical.tokenize import tokenize_fields
from src.pipeline.mind import (read_news, scan_behaviors, split_dir, with_history,
                               with_labelled_candidates, with_parsed_time,
                               with_unlabelled_candidates)
from src.semantic.ann import ANNIndex
from src.semantic.embeddings import compute_lsa
from src.semantic.user_vector import build_user_vector

ROOT, OUT = Path("data/interim/mind"), Path("data/processed")
N_RECENT, SEED, BATCH = 5, 0, 200_000
FEATURES = ["pop_total", "pop_24h", "pop_1h", "ctr_total", "ctr_24h",
            "bm25", "semantic", "slate_size", "cat_affinity", "history_len"]
W24, W1 = timedelta(hours=24), timedelta(hours=1)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def row_features(rc, at, cands, hist, bm_scores, sem_scores, news_cat):
    hist_cats = [news_cat.get(h) for h in hist if h in news_cat]
    cat_n = len(hist_cats) or 1
    share = {}
    for c in hist_cats:
        share[c] = share.get(c, 0) + 1
    n = len(cands)
    return [[rc.clicks_before(c, at), rc.clicks_before(c, at, W24), rc.clicks_before(c, at, W1),
             rc.ctr_before(c, at), rc.ctr_before(c, at, W24),
             bm_scores[i], sem_scores[i], n,
             share.get(news_cat.get(c), 0) / cat_n, len(hist)]
            for i, c in enumerate(cands)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train_d, dev_d = split_dir(ROOT, "MINDsmall_train"), split_dir(ROOT, "MINDsmall_dev")
    test_d = split_dir(ROOT, "MINDlarge_test")

    log("corpus + indices (train + dev + test news)")
    text, order, news_cat = {}, [], {}
    for d in (train_d, dev_d, test_d):
        n = read_news(d)
        for a, ti, ab, c in zip(n["news_id"], n["title"], n["abstract"], n["category"]):
            if a not in text:
                text[a] = (ti, ab); order.append(a); news_cat[a] = c
    idx = InvertedIndex(); idx.enable_forward_index()
    for a in order:
        idx.add(a, tokenize_fields(*text[a], lang="en"))
    bm = BM25(idx, idf_variant="lucene")
    matrix, _, _ = compute_lsa([f"{text[a][0] or ''} {text[a][1] or ''}" for a in order],
                               n_components=128, seed=SEED)
    ann = ANNIndex(order, matrix, kind="flat")
    log(f"  {idx.n_docs:,} articles")

    def load_labelled(d):
        return with_history(with_labelled_candidates(with_parsed_time(scan_behaviors(d)))).select(
            "ts", "history_ids", "candidates", "labels").collect()

    train, dev = load_labelled(train_d), load_labelled(dev_d)
    rc = RollingCounts()
    for ts, cands, labels in zip(train["ts"], train["candidates"], train["labels"]):
        for c, l in zip(cands.to_list(), labels.to_list()):
            rc.add_view(c, ts)
            if l == 1:
                rc.add_click(c, ts)
    rc.seal()
    log(f"  rolling counts over {rc.n_articles:,} articles (TRAIN only)")

    def featurise(df):
        X, y, groups = [], [], []
        for ts, h, cd, lb in zip(df["ts"], df["history_ids"], df["candidates"], df["labels"]):
            hist, cands, labels = h.to_list(), cd.to_list(), lb.to_list()
            if not any(labels):
                continue
            q = build_query(hist, text, n_recent=N_RECENT)
            uv = build_user_vector(hist, ann.id_to_row, matrix, pooling="mean")
            X.extend(row_features(rc, ts, cands, hist,
                                  bm.score_candidates(q, cands), ann.score_candidates(uv, cands), news_cat))
            y.extend(labels); groups.append(len(cands))
        return np.asarray(X, np.float32), np.asarray(y, np.int8), groups

    log("featurising train + dev")
    t0 = time.perf_counter()
    Xtr, ytr, _ = featurise(train)
    Xdv, ydv, gdv = featurise(dev)
    log(f"  {Xtr.shape} / {Xdv.shape}  ({time.perf_counter()-t0:.0f}s)")

    from sklearn.ensemble import HistGradientBoostingClassifier

    def fit_eval(cols, label):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                           max_leaf_nodes=31, random_state=SEED)
        m.fit(Xtr[:, cols], ytr)
        s = m.predict_proba(Xdv[:, cols])[:, 1]
        rows, i = [], 0
        for n in gdv:
            rows.append((ydv[i:i + n].tolist(), s[i:i + n].tolist())); i += n
        ci = bootstrap_metrics(per_impression_metrics(rows), iterations=1000, seed=SEED)
        log(f"  {label:22s} AUC {ci['auc']}  nDCG@10 {ci['ndcg@10']}")
        return m, ci

    all_cols = list(range(len(FEATURES)))
    no_slate = [i for i in all_cols if FEATURES[i] != "slate_size"]
    log("leave-one-out check on slate_size")
    model, ci_all = fit_eval(all_cols, "all features")
    _, ci_drop = fit_eval(no_slate, "without slate_size")
    log(f"  {compare(ci_all['auc'], ci_drop['auc'], 'with_slate', 'without_slate')}")

    keep = all_cols if ci_all["auc"].mean >= ci_drop["auc"].mean else no_slate
    if keep is no_slate:
        model, ci_all = fit_eval(no_slate, "refit without slate_size")
    log(f"  shipping {len(keep)} features")

    log("scoring MINDlarge_test")
    lf = with_history(with_unlabelled_candidates(with_parsed_time(scan_behaviors(test_d))))
    total = lf.select(pl.len()).collect().item()
    # Test impressions are later than every training event, so a rolling count as-of the test
    # timestamp sees the entire training window -- correct, and the same thing the model saw.
    out_txt = OUT / "prediction.txt"
    written, t0 = 0, time.perf_counter()
    with out_txt.open("w") as fh:
        for start in range(0, total, BATCH):
            b = lf.slice(start, min(BATCH, total - start)).select(
                "impression_id", "ts", "history_ids", "candidates").collect()
            for imp, ts, h, cd in zip(b["impression_id"], b["ts"], b["history_ids"], b["candidates"]):
                hist, cands = h.to_list(), cd.to_list()
                q = build_query(hist, text, n_recent=N_RECENT)
                uv = build_user_vector(hist, ann.id_to_row, matrix, pooling="mean")
                feats = np.asarray(row_features(rc, ts, cands, hist,
                                                bm.score_candidates(q, cands),
                                                ann.score_candidates(uv, cands), news_cat), np.float32)
                s = model.predict_proba(feats[:, keep])[:, 1]
                fh.write(format_line(imp, ranks_from_scores(s.tolist())) + "\n")
                written += 1
            log(f"  {written:,}/{total:,}  ({time.perf_counter()-t0:.0f}s)")
            del b
    stats = validate_file(out_txt)
    assert stats["lines"] == total, f"wrote {stats['lines']} for {total}"
    z = zip_submission(out_txt, OUT / "mind_prediction_v3.zip")
    log(f"  {stats} -> {z} ({z.stat().st_size/1e6:.1f} MB)")
    (OUT / "mind_v3.json").write_text(json.dumps(
        {"features": [FEATURES[i] for i in keep],
         "dev": {k: {"mean": v.mean, "lo": v.lo, "hi": v.hi} for k, v in ci_all.items()},
         "validation": stats}, indent=2))
    log("done")


if __name__ == "__main__":
    main()
