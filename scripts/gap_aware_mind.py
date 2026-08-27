#!/usr/bin/env python3
"""Choose the MIND v4 model under a validation that mimics the test set's temporal gap.

    PYTHONPATH=. .venv/bin/python -u scripts/gap_aware_mind.py

**Why this exists.** Submission v3 scored 0.6454 on the standard dev split and 0.5554 on the
leaderboard — an offset of -0.090 against -0.03 for the two previous submissions. The cause is
measured: training click events end 14 Nov, dev is 15 Nov, and the test split is 16-22 Nov. A
24-hour lookback from a dev impression still reaches live training data; the same lookback from
any test impression reaches a period with no events at all. `pop_24h`, `pop_1h` and `ctr_24h`
are zero for **100%** of test candidates and for only 58%, 99.7% and 58% of dev candidates.

So dev cannot detect this class of failure: it sits adjacent to training, and the test set never
does. This script rebuilds the validation so that it can.

**The gap-aware protocol.** Rolling counts are fitted on 9-12 Nov only. The ranker is then
trained on 14 Nov impressions and evaluated on 15 Nov impressions -- two and three days past the
last counted event respectively. Both sides therefore see the same dead-feature regime the test
split imposes, and a model that depends on it is penalised here rather than on the leaderboard.

Four configurations are compared, crossing the feature set against the semantic representation,
so the two proposed changes are separated rather than confounded.
"""
from __future__ import annotations

import json, time
from datetime import datetime, timedelta
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
W24, W1 = timedelta(hours=24), timedelta(hours=1)

COUNTS_END = datetime(2019, 11, 13)     # counts use events strictly before this
FIT_DAY    = datetime(2019, 11, 14)     # ranker trains here  (1-2 days past counts)
EVAL_DAY   = datetime(2019, 11, 15)     # ranker evaluated here (2-3 days past counts)

V3 = ["pop_total", "pop_24h", "pop_1h", "ctr_total", "ctr_24h",
      "bm25", "semantic", "slate_size", "cat_affinity", "history_len"]
V4 = ["pop_total", "ctr_total", "bm25", "semantic", "slate_size", "cat_affinity", "history_len"]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load(split_d):
    return with_history(with_labelled_candidates(with_parsed_time(scan_behaviors(split_d)))).select(
        "ts", "history_ids", "candidates", "labels").collect()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train_d, dev_d = split_dir(ROOT, "MINDsmall_train"), split_dir(ROOT, "MINDsmall_dev")

    log("article text and category map")
    text, order, news_cat = {}, [], {}
    for d in (train_d, dev_d, split_dir(ROOT, "MINDlarge_test")):
        n = read_news(d)
        for a, ti, ab, c in zip(n["news_id"], n["title"], n["abstract"], n["category"]):
            if a not in text:
                text[a] = (ti, ab); order.append(a); news_cat[a] = c
    log(f"  {len(order):,} articles")

    log("lexical index")
    idx = InvertedIndex(); idx.enable_forward_index()
    for a in order:
        idx.add(a, tokenize_fields(*text[a], lang="en"))
    bm = BM25(idx, idf_variant="lucene")

    log("semantic representations")
    reps = {}
    lsa, _, svd = compute_lsa([f"{text[a][0] or ''} {text[a][1] or ''}" for a in order],
                              n_components=128, seed=SEED)
    reps["lsa"] = (ANNIndex(order, lsa, kind="flat"), lsa)
    log(f"  lsa       {lsa.shape} explained_variance={svd.explained_variance_ratio_.sum():.3f}")
    cache = OUT / "mind_minilm.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        m_ids, m_mat = list(z["ids"]), z["matrix"].astype(np.float32)
        reps["minilm"] = (ANNIndex(m_ids, m_mat, kind="flat"), m_mat)
        log(f"  minilm    {m_mat.shape} (cached)")
    else:
        log("  minilm    NOT CACHED — run scripts/encode_mind_minilm.py first")

    log(f"rolling counts from events strictly before {COUNTS_END}")
    train = load(train_d)
    rc = RollingCounts()
    kept = 0
    for ts, cands, labels in zip(train["ts"], train["candidates"], train["labels"]):
        if ts >= COUNTS_END:
            continue
        kept += 1
        for c, l in zip(cands.to_list(), labels.to_list()):
            rc.add_view(c, ts)
            if l == 1:
                rc.add_click(c, ts)
    rc.seal()
    log(f"  {kept:,} impressions counted, {rc.n_articles:,} articles")

    fit_rows = train.filter((train["ts"] >= FIT_DAY) & (train["ts"] < FIT_DAY + timedelta(days=1)))
    dev = load(dev_d)
    eval_rows = dev.filter((dev["ts"] >= EVAL_DAY) & (dev["ts"] < EVAL_DAY + timedelta(days=1)))
    log(f"  fit on {fit_rows.height:,} impressions of {FIT_DAY.date()}")
    log(f"  eval on {eval_rows.height:,} impressions of {EVAL_DAY.date()}")

    def featurise(df, ann, matrix):
        X, y, groups = [], [], []
        for ts, h, cd, lb in zip(df["ts"], df["history_ids"], df["candidates"], df["labels"]):
            hist, cands, labels = h.to_list(), cd.to_list(), lb.to_list()
            if not any(labels):
                continue
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
            for i, c in enumerate(cands):
                X.append([rc.clicks_before(c, ts), rc.clicks_before(c, ts, W24), rc.clicks_before(c, ts, W1),
                          rc.ctr_before(c, ts), rc.ctr_before(c, ts, W24),
                          bmv[i], sev[i], n, share.get(news_cat.get(c), 0) / cn, len(hist)])
                y.append(labels[i])
            groups.append(n)
        return np.asarray(X, np.float32), np.asarray(y, np.int8), groups

    from sklearn.ensemble import HistGradientBoostingClassifier
    results, cis = {}, {}
    for rep_name, (ann, matrix) in reps.items():
        log(f"featurising with {rep_name}")
        Xf, yf, _ = featurise(fit_rows, ann, matrix)
        Xe, ye, ge = featurise(eval_rows, ann, matrix)
        log(f"  fit {Xf.shape}  eval {Xe.shape}")
        for fs_name, fs in (("v3", V3), ("v4", V4)):
            cols = [V3.index(f) for f in fs]
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                               max_leaf_nodes=31, random_state=SEED)
            m.fit(Xf[:, cols], yf)
            s = m.predict_proba(Xe[:, cols])[:, 1]
            rows, i = [], 0
            for n in ge:
                rows.append((ye[i:i + n].tolist(), s[i:i + n].tolist())); i += n
            key = f"{fs_name}+{rep_name}"
            cis[key] = bootstrap_metrics(per_impression_metrics(rows), iterations=1000, seed=SEED)
            results[key] = {k: {"mean": v.mean, "lo": v.lo, "hi": v.hi, "n": v.n}
                            for k, v in cis[key].items()}
            log(f"  {key:14s} AUC {cis[key]['auc']}  nDCG@10 {cis[key]['ndcg@10']}")
        del Xf, yf, Xe, ye

    best = max(results, key=lambda k: results[k]["auc"]["mean"])
    log(f"best under the gap-aware protocol: {best}")
    baseline = "v3+lsa"                       # the configuration already submitted
    if baseline in cis and best != baseline:
        log(f"  {compare(cis[best]['auc'], cis[baseline]['auc'], best, baseline)}")
    results["_best"] = best
    results["_protocol"] = {"counts_end": str(COUNTS_END), "fit_day": str(FIT_DAY),
                            "eval_day": str(EVAL_DAY)}
    (OUT / "gap_aware_mind.json").write_text(json.dumps(results, indent=2))
    log("done")


if __name__ == "__main__":
    main()
