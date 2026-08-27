#!/usr/bin/env python3
"""EB-NeRD point-in-time reranker: does rolling popularity flip the anti-predictive finding?

    PYTHONPATH=. .venv/bin/python -u scripts/rerank_ebnerd.py [--submit]

RESULTS.md records that *frozen* popularity on EB-NeRD scores AUC 0.4429 — below chance — and
attributes it to popularity bias in the logging policy. That measurement used a single count
per article over the whole training window. This script re-tests the same claim with counts
computed strictly before each impression, which is the only honest way to ask whether the
effect is about popularity itself or about staleness.

EB-NeRD supports two features MIND cannot:
  * **article age at impression** — `published_time` exists, so recency is computable
  * **provided word2vec embeddings** — publisher-trained on Danish news

Prediction is batched: features for a whole row group are stacked and passed to
`predict_proba` once. Calling it per impression, as the MIND script does, spends most of its
time in per-call overhead — at 13.5M impressions that is the difference between hours and
most of a day.
"""
from __future__ import annotations

import argparse, json, time
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
from src.pipeline.ebnerd import LANG, article_text, iter_row_groups, labels_from_clicked, n_rows, recent_history
from src.semantic.ann import ANNIndex
from src.semantic.embeddings import load_provided
from src.semantic.user_vector import build_user_vector

SMALL = Path("data/interim/ebnerd/ebnerd_small")
TEST = Path("data/interim/ebnerd/ebnerd_testset/ebnerd_testset")
EMB = Path("data/interim/ebnerd/Ekstra_Bladet_word2vec/document_vector.parquet")
OUT = Path("data/processed")
N_RECENT, SEED = 5, 0
FEATURES = ["pop_total", "pop_24h", "pop_1h", "ctr_total", "ctr_24h",
            "age_hours", "bm25", "semantic", "slate_size", "history_len"]
W24, W1 = timedelta(hours=24), timedelta(hours=1)
FAR_FUTURE = None  # set at runtime


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def featurise_impression(rc, at, ts, cands, hist, bm_scores, sem_scores, published):
    n = len(cands)
    rows = []
    for i, c in enumerate(cands):
        pub = published.get(c)
        age = (ts - pub).total_seconds() / 3600.0 if pub is not None else -1.0
        rows.append([rc.clicks_before(c, at), rc.clicks_before(c, at, W24), rc.clicks_before(c, at, W1),
                     rc.ctr_before(c, at), rc.ctr_before(c, at, W24),
                     age, bm_scores[i], sem_scores[i], n, len(hist)])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true", help="also score the 13.5M test set")
    ap.add_argument("--limit", type=int, default=120_000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    log("article text, publish times, indices")
    text = article_text(TEST / "articles.parquet")
    pub_df = pl.read_parquet(TEST / "articles.parquet", columns=["article_id", "published_time"])
    published = dict(zip(pub_df["article_id"].to_list(), pub_df["published_time"].to_list()))
    idx = InvertedIndex(); idx.enable_forward_index()
    for aid, (ti, sub) in text.items():
        idx.add(aid, tokenize_fields(ti, sub, lang=LANG))
    bm = BM25(idx, idf_variant="lucene")
    ids, matrix = load_provided(EMB)
    ann = ANNIndex(ids, matrix, kind="flat")
    log(f"  {idx.n_docs:,} articles, embeddings {matrix.shape} (provided word2vec)")

    log("training events from ebnerd_small train")
    train = (pl.scan_parquet(SMALL / "train/behaviors.parquet")
             .select("user_id", pl.col("impression_time").alias("ts"),
                     "article_ids_inview", "article_ids_clicked").collect())
    rc = RollingCounts()
    for ts, cands, clicked in zip(train["ts"], train["article_ids_inview"], train["article_ids_clicked"]):
        cl = set(clicked.to_list())
        for c in cands.to_list():
            rc.add_view(c, ts)
            if c in cl:
                rc.add_click(c, ts)
    rc.seal()
    log(f"  {train.height:,} impressions -> counts over {rc.n_articles:,} articles")
    global FAR_FUTURE
    FAR_FUTURE = train["ts"].max() + timedelta(days=3650)

    hist_val = recent_history(SMALL / "validation/history.parquet", N_RECENT)
    val = (pl.scan_parquet(SMALL / "validation/behaviors.parquet")
           .select("user_id", pl.col("impression_time").alias("ts"),
                   "article_ids_inview", "article_ids_clicked")
           .head(args.limit).collect())
    log(f"  validation {val.height:,}")

    def build(frozen: bool):
        X, y, groups = [], [], []
        for uid, ts, cd, ck in zip(val["user_id"], val["ts"], val["article_ids_inview"], val["article_ids_clicked"]):
            cands = cd.to_list()
            labels = labels_from_clicked(cands, ck.to_list())
            if not any(labels):
                continue
            hist = hist_val.get(uid, [])
            q = build_query(hist, text, n_recent=N_RECENT, lang=LANG)
            uv = build_user_vector(hist, ann.id_to_row, matrix, pooling="mean")
            X.extend(featurise_impression(rc, FAR_FUTURE if frozen else ts, ts, cands, hist,
                                          bm.score_candidates(q, cands),
                                          ann.score_candidates(uv, cands), published))
            y.extend(labels); groups.append(len(cands))
        return np.asarray(X, np.float32), np.asarray(y, np.int8), groups

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    results, cis = {}, {}
    for variant, frozen in (("rolling", False), ("frozen", True)):
        log(f"featurising ({variant})")
        t0 = time.perf_counter()
        X, y, groups = build(frozen)
        log(f"  {X.shape}  ({time.perf_counter()-t0:.0f}s)")
        # Impression-level split so no impression straddles fit and evaluation.
        cut = int(len(groups) * 0.7)
        n_fit = sum(groups[:cut])
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                               max_leaf_nodes=31, random_state=SEED)
        model.fit(X[:n_fit], y[:n_fit])
        s = model.predict_proba(X[n_fit:])[:, 1]
        rows, i = [], 0
        for n in groups[cut:]:
            rows.append((y[n_fit + i:n_fit + i + n].tolist(), s[i:i + n].tolist())); i += n
        cis[variant] = bootstrap_metrics(per_impression_metrics(rows), iterations=1000, seed=SEED)
        results[variant] = {k: {"mean": v.mean, "lo": v.lo, "hi": v.hi, "n": v.n} for k, v in cis[variant].items()}
        log(f"  {variant:8s} AUC {cis[variant]['auc']}  nDCG@10 {cis[variant]['ndcg@10']}")
        if variant == "rolling":
            from sklearn.inspection import permutation_importance
            sub = min(40000, len(X) - n_fit)
            imp = permutation_importance(model, X[n_fit:n_fit + sub], y[n_fit:n_fit + sub],
                                         n_repeats=3, random_state=SEED, scoring="roc_auc")
            ranked = sorted(zip(FEATURES, imp.importances_mean), key=lambda kv: -kv[1])
            results["feature_importance"] = {k: float(v) for k, v in ranked}
            log("  permutation importance:")
            for k, v in ranked:
                log(f"    {k:12s} {v:+.4f}")
            final = model

    log(f"  {compare(cis['rolling']['auc'], cis['frozen']['auc'], 'rolling', 'frozen')}")
    results["rolling_minus_frozen_auc"] = cis["rolling"]["auc"].mean - cis["frozen"]["auc"].mean
    (OUT / "rerank_ebnerd.json").write_text(json.dumps(results, indent=2))

    if not args.submit:
        log("done (validation only; pass --submit to score the test set)")
        return

    log("scoring the 13.5M test set (batched prediction)")
    hist_test = recent_history(TEST / "test/history.parquet", N_RECENT)
    beh = TEST / "test/behaviors.parquet"
    total = n_rows(beh)
    out_txt = OUT / "predictions.txt"
    written, t0 = 0, time.perf_counter()
    with out_txt.open("w") as fh:
        for group in iter_row_groups(beh, ["impression_id", "impression_time", "user_id", "article_ids_inview"]):
            feats, meta = [], []
            for imp, ts, uid, cd in zip(group["impression_id"], group["impression_time"],
                                        group["user_id"], group["article_ids_inview"]):
                cands = cd.to_list()
                hist = hist_test.get(uid, [])
                q = build_query(hist, text, n_recent=N_RECENT, lang=LANG)
                uv = build_user_vector(hist, ann.id_to_row, matrix, pooling="mean")
                feats.extend(featurise_impression(rc, ts, ts, cands, hist,
                                                  bm.score_candidates(q, cands),
                                                  ann.score_candidates(uv, cands), published))
                meta.append((imp, len(cands)))
            probs = final.predict_proba(np.asarray(feats, np.float32))[:, 1]
            i = 0
            for imp, n in meta:
                fh.write(format_line(imp, ranks_from_scores(probs[i:i + n].tolist())) + "\n")
                i += n; written += 1
            log(f"  {written:,}/{total:,}  ({time.perf_counter()-t0:.0f}s)")
            del feats, probs, group
    # EB-NeRD repeats impression_id 0 across its 200,000 beyond-accuracy rows.
    stats = validate_file(out_txt, allow_duplicate_ids=True)
    assert stats["lines"] == total, f"wrote {stats['lines']} for {total}"
    z = zip_submission(out_txt, OUT / "ebnerd_predictions_v2.zip")
    log(f"  {stats} -> {z} ({z.stat().st_size/1e6:.1f} MB)")
    log("done")


if __name__ == "__main__":
    main()
