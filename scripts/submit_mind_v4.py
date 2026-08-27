#!/usr/bin/env python3
"""MIND submission v4: MiniLM semantics, skew-resistant feature set.

    PYTHONPATH=. .venv/bin/python -u scripts/submit_mind_v4.py

Two changes from v3, each justified by a measurement rather than a hunch.

**1. MiniLM replaces LSA.** Under the gap-aware protocol (`scripts/gap_aware_mind.py`), which
evaluates two to three days past the last counted event so that it resembles the test split,
MiniLM scores AUC 0.6447 [0.6425, 0.6466] against LSA's 0.6050 [0.6028, 0.6073] -- disjoint.
LSA retained only 12.2% of the variance of the term-document matrix; a 384-dimensional
sentence encoder is a materially better representation of the same text.

**2. The time-windowed count features are dropped.** `pop_24h`, `pop_1h` and `ctr_24h` are
zero for 100% of test candidates, because training events end 14 Nov and the test split runs
16-22 Nov. On the dev-adjacent protocol they varied and the trees split on them, which is what
produced v3's -0.090 offset between dev and the leaderboard. Under the gap-aware protocol they
are constant and the model ignores them anyway -- dropping them measured identically there.
They are removed regardless, because this script fits on data adjacent to its own counts, which
is precisely the condition that recreates the skew.

Counts are built from train **and** dev events. Dev precedes test, so this is point-in-time
sound, and it narrows the gap between the last counted event and the test window.
"""
from __future__ import annotations

import json, os, time
from pathlib import Path

import numpy as np
import polars as pl

from src.eval.bootstrap import bootstrap_metrics
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
from src.semantic.user_vector import build_user_vector

ROOT, OUT = Path("data/interim/mind"), Path("data/processed")
N_RECENT, SEED, CHUNK = 5, 0, 20_000
FEATURES = ["pop_total", "ctr_total", "bm25", "semantic",
            "slate_size", "cat_affinity", "history_len"]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train_d, dev_d = split_dir(ROOT, "MINDsmall_train"), split_dir(ROOT, "MINDsmall_dev")
    test_d = split_dir(ROOT, "MINDlarge_test")

    log("article text, categories, indices")
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

    z = np.load(OUT / "mind_minilm.npz", allow_pickle=True)
    ann = ANNIndex(list(z["ids"]), z["matrix"].astype(np.float32), kind="flat")
    matrix = ann.matrix
    log(f"  {len(order):,} articles, MiniLM {matrix.shape}")
    del z

    def load(d):
        return with_history(with_labelled_candidates(with_parsed_time(scan_behaviors(d)))).select(
            "ts", "history_ids", "candidates", "labels").collect()

    train, dev = load(train_d), load(dev_d)
    rc = RollingCounts()
    for df in (train, dev):                    # dev precedes test: point-in-time sound
        for ts, cands, labels in zip(df["ts"], df["candidates"], df["labels"]):
            for c, l in zip(cands.to_list(), labels.to_list()):
                rc.add_view(c, ts)
                if l == 1:
                    rc.add_click(c, ts)
    rc.seal()
    log(f"  counts from train+dev over {rc.n_articles:,} articles")

    def row_feats(ts, cands, hist):
        q = build_query(hist, text, n_recent=N_RECENT)
        bmv = bm.score_candidates(q, cands)
        uv = build_user_vector(hist, ann.id_to_row, matrix, pooling="mean")
        sev = ann.score_candidates(uv, cands)
        cats = [news_cat.get(a) for a in hist if a in news_cat]
        cn = len(cats) or 1
        share = {}
        for c in cats:
            share[c] = share.get(c, 0) + 1
        n = len(cands)
        return [[rc.clicks_before(c, ts), rc.ctr_before(c, ts), bmv[i], sev[i],
                 n, share.get(news_cat.get(c), 0) / cn, len(hist)]
                for i, c in enumerate(cands)]

    log("featurising train + dev for the ranker")
    X, y, groups = [], [], []
    t0 = time.perf_counter()
    for df in (train, dev):
        for ts, h, cd, lb in zip(df["ts"], df["history_ids"], df["candidates"], df["labels"]):
            hist, cands, labels = h.to_list(), cd.to_list(), lb.to_list()
            if not any(labels):
                continue
            X.extend(row_feats(ts, cands, hist)); y.extend(labels); groups.append(len(cands))
    X = np.asarray(X, np.float32); y = np.asarray(y, np.int8)
    log(f"  {X.shape}  ({time.perf_counter()-t0:.0f}s)")
    del train, dev

    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                           max_leaf_nodes=31, random_state=SEED)
    model.fit(X, y)
    log(f"  fitted on {len(groups):,} impressions")
    # In-sample only: the honest out-of-sample estimate is the gap-aware protocol, and this
    # number is recorded to confirm the fit ran, not as evidence of generalisation.
    s = model.predict_proba(X)[:, 1]
    rows, i = [], 0
    for n in groups:
        rows.append((y[i:i + n].tolist(), s[i:i + n].tolist())); i += n
    ci = bootstrap_metrics(per_impression_metrics(rows), iterations=200, seed=SEED)
    log(f"  in-sample AUC {ci['auc']} (not a generalisation estimate)")
    del X, y, s, rows

    log("scoring MINDlarge_test")
    lf = with_history(with_unlabelled_candidates(with_parsed_time(scan_behaviors(test_d))))
    total = lf.select(pl.len()).collect().item()
    out_txt = OUT / "prediction.txt"
    written, t0 = 0, time.perf_counter()
    with out_txt.open("w") as fh:
        for start in range(0, total, CHUNK):
            b = lf.slice(start, min(CHUNK, total - start)).select(
                "impression_id", "ts", "history_ids", "candidates").collect()
            feats, meta = [], []
            for imp, ts, h, cd in zip(b["impression_id"], b["ts"], b["history_ids"], b["candidates"]):
                cands = cd.to_list()
                feats.extend(row_feats(ts, cands, h.to_list()))
                meta.append((imp, len(cands)))
            probs = model.predict_proba(np.asarray(feats, np.float32))[:, 1]
            i = 0
            for imp, n in meta:
                fh.write(format_line(imp, ranks_from_scores(probs[i:i + n].tolist())) + "\n")
                i += n; written += 1
            fh.flush(); os.fsync(fh.fileno())      # durable against this machine's kills
            if (written // CHUNK) % 10 == 0:
                log(f"  {written:,}/{total:,}  ({time.perf_counter()-t0:.0f}s)")
            del feats, probs, b
    log(f"  wrote {written:,}")

    stats = validate_file(out_txt)
    assert stats["lines"] == total, f"wrote {stats['lines']} for {total}"
    zp = zip_submission(out_txt, OUT / "mind_prediction_v4.zip")
    log(f"  {stats} -> {zp} ({zp.stat().st_size/1e6:.1f} MB)")
    (OUT / "mind_v4.json").write_text(json.dumps(
        {"features": FEATURES, "semantic": "all-MiniLM-L6-v2",
         "counts_from": "train+dev", "validation": stats}, indent=2))
    log("done")


if __name__ == "__main__":
    main()
