#!/usr/bin/env python3
"""Q3 + Q3.5: semantic retrieval on MIND, and lexical vs semantic by slice.

    PYTHONPATH=. .venv/bin/python -u scripts/eval_semantic_mind.py [--limit N]

MIND ships no article embeddings, so vectors come from TF-IDF + truncated SVD (see
src/semantic/embeddings.py for why that, and not a sentence-transformer, on this machine).

Reports:
  * recall@K, K in {50,100,200}, mean- vs recency-pooled user vectors (Q3.3, Q3.4)
  * in-impression AUC/MRR/nDCG with bootstrap CIs (Q4)
  * the same, sliced by user history length and by article popularity (Q3.5, Q4.3)
  * beyond-accuracy: diversity, novelty, coverage (Q4.2)
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np

from src.baselines.popularity import click_counts, score_map
from src.eval.beyond_accuracy import CoverageTracker, NoveltyModel, intra_list_diversity, top_k_items
from src.eval.bootstrap import bootstrap_metrics, compare
from src.eval.metrics import per_impression_metrics
from src.lexical.bm25 import BM25
from src.lexical.index import InvertedIndex
from src.lexical.retrieval import build_query, recall_at_k
from src.lexical.tokenize import tokenize_fields
from src.pipeline.mind import (read_news, scan_behaviors, split_dir, with_history,
                               with_labelled_candidates)
from src.semantic.ann import ANNIndex
from src.semantic.embeddings import compute_lsa
from src.semantic.user_vector import build_user_vector

ROOT, OUT = Path("data/interim/mind"), Path("data/processed")
N_RECENT, COLD_START_MAX = 5, 5      # <=5 history clicks counts as cold-start
KS = (50, 100, 200)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--recall-limit", type=int, default=3000)
    ap.add_argument("--components", type=int, default=128)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    train_d, dev_d = split_dir(ROOT, "MINDsmall_train"), split_dir(ROOT, "MINDsmall_dev")

    log("loading article text")
    text, order = {}, []
    for d in (train_d, dev_d):
        n = read_news(d)
        for a, ti, ab in zip(n["news_id"], n["title"], n["abstract"]):
            if a not in text:
                text[a] = (ti, ab); order.append(a)
    log(f"  {len(order):,} articles")

    log("lexical index")
    t0 = time.perf_counter()
    idx = InvertedIndex(); idx.enable_forward_index()
    for a in order:
        idx.add(a, tokenize_fields(*text[a], lang="en"))
    lex_build = time.perf_counter() - t0
    bm = BM25(idx, idf_variant="lucene")
    log(f"  {idx.stats()}  {lex_build:.1f}s")

    log(f"semantic embeddings (TF-IDF -> SVD, {args.components}d)")
    t0 = time.perf_counter()
    corpus = [f"{text[a][0] or ''} {text[a][1] or ''}" for a in order]
    matrix, _, svd = compute_lsa(corpus, n_components=args.components)
    emb_build = time.perf_counter() - t0
    log(f"  {matrix.shape}  explained variance {svd.explained_variance_ratio_.sum():.3f}  {emb_build:.1f}s")

    t0 = time.perf_counter()
    ann = ANNIndex(order, matrix, kind="flat")
    ann_build = time.perf_counter() - t0
    log(f"  flat ANN built in {ann_build:.1f}s")

    pop = score_map(click_counts(with_labelled_candidates(scan_behaviors(train_d))))
    novelty = NoveltyModel(pop)
    pop_rank = sorted(pop, key=pop.get, reverse=True)
    head = set(pop_rank[: max(1, len(pop_rank) // 5)])       # top quintile by clicks
    cat = {a: (text[a][0] or "")[:0] for a in order}          # placeholder, replaced below
    news_cat = {}
    for d in (train_d, dev_d):
        n = read_news(d)
        for a, c in zip(n["news_id"], n["category"]):
            news_cat.setdefault(a, c)

    log("loading dev impressions")
    dev = with_history(with_labelled_candidates(scan_behaviors(dev_d))).select(
        "history_ids", "candidates", "labels").collect()
    log(f"  {dev.height:,}")

    # ---- Q3.4 recall@K, pooling ablation -------------------------------------------------
    log(f"recall@K over {len(order):,} articles (limit={args.recall_limit:,})")
    recall = {}
    for pooling in ("mean", "recency"):
        acc = {k: [0, 0] for k in KS}
        skipped = scored = 0
        t0 = time.perf_counter()
        for i, (h, cand, lab) in enumerate(zip(dev["history_ids"], dev["candidates"], dev["labels"])):
            if i >= args.recall_limit:
                break
            hist = h.to_list()
            relevant = {c for c, l in zip(cand.to_list(), lab.to_list()) if l == 1}
            uv = build_user_vector(hist, ann.id_to_row, matrix, pooling=pooling)
            if uv is None or not relevant:
                skipped += 1; continue
            got = [a for a, _ in ann.search(uv, top_k=max(KS))]
            for k in KS:
                hits, n_rel = recall_at_k(got, relevant, k)
                acc[k][0] += hits; acc[k][1] += n_rel
            scored += 1
        recall[pooling] = {"recall": {k: round(v[0] / v[1], 5) if v[1] else 0.0 for k, v in acc.items()},
                           "scored": scored, "skipped": skipped,
                           "seconds": round(time.perf_counter() - t0, 1)}
        log(f"  {pooling:8s} {recall[pooling]['recall']}  scored={scored:,} ({recall[pooling]['seconds']:.0f}s)")

    # ---- Q4 + Q3.5 in-impression, sliced --------------------------------------------------
    log(f"in-impression re-ranking, sliced (limit={args.limit:,})")
    slices = ("all", "cold_start", "warm", "head", "tail")
    models = ("bm25", "semantic")
    per_imp = {m: {s: {k: [] for k in ("auc", "mrr", "ndcg@5", "ndcg@10")} for s in slices} for m in models}
    ba = {m: {"div": [], "nov": [], "cov": CoverageTracker(len(order))} for m in models}

    t0 = time.perf_counter()
    for i, (h, cand, lab) in enumerate(zip(dev["history_ids"], dev["candidates"], dev["labels"])):
        if i >= args.limit:
            break
        hist, cands, labels = h.to_list(), cand.to_list(), lab.to_list()
        if not any(labels):
            continue
        clicked = [c for c, l in zip(cands, labels) if l == 1]
        row_slices = ["all", "cold_start" if len(hist) <= COLD_START_MAX else "warm"]
        row_slices.append("head" if any(c in head for c in clicked) else "tail")

        scores = {
            "bm25": bm.score_candidates(build_query(hist, text, n_recent=N_RECENT), cands),
            "semantic": ann.score_candidates(
                build_user_vector(hist, ann.id_to_row, matrix, pooling="mean"), cands),
        }
        for m, s in scores.items():
            one = per_impression_metrics([(labels, s)])
            for sl in row_slices:
                for k in one:
                    per_imp[m][sl][k].append(one[k][0])
            top = top_k_items(cands, s, 10)
            ba[m]["div"].append(intra_list_diversity([news_cat.get(a, "?") for a in top]))
            ba[m]["nov"].append(novelty.mean(top))
            ba[m]["cov"].observe(top)
    log(f"  scored in {time.perf_counter()-t0:.0f}s")

    results = {"index": {"lexical": idx.stats(), "lexical_build_s": round(lex_build, 1),
                         "embedding_dim": int(matrix.shape[1]),
                         "explained_variance": float(svd.explained_variance_ratio_.sum()),
                         "embedding_build_s": round(emb_build, 1),
                         "ann_build_s": round(ann_build, 1)},
               "recall": recall, "sliced": {}, "beyond_accuracy": {}}

    for m in models:
        results["sliced"][m] = {}
        for sl in slices:
            if not per_imp[m][sl]["auc"]:
                continue
            cis = bootstrap_metrics(per_imp[m][sl], iterations=1000, seed=0)
            results["sliced"][m][sl] = {k: {"mean": v.mean, "lo": v.lo, "hi": v.hi, "n": v.n}
                                        for k, v in cis.items()}
        results["beyond_accuracy"][m] = {
            "intra_list_diversity@10": float(np.mean(ba[m]["div"])),
            "novelty@10_bits": float(np.mean(ba[m]["nov"])),
            "coverage@10": ba[m]["cov"].coverage,
            "gini@10": ba[m]["cov"].gini(),
        }

    log("lexical vs semantic by slice (AUC)")
    for sl in slices:
        a = results["sliced"]["bm25"].get(sl); b = results["sliced"]["semantic"].get(sl)
        if not a or not b:
            continue
        from src.eval.bootstrap import CI
        ca = CI(a["auc"]["mean"], a["auc"]["lo"], a["auc"]["hi"], a["auc"]["n"], 1000)
        cb = CI(b["auc"]["mean"], b["auc"]["lo"], b["auc"]["hi"], b["auc"]["n"], 1000)
        log(f"  {sl:11s} {compare(ca, cb, 'bm25', 'semantic')}")

    for m in models:
        log(f"  {m:9s} beyond-accuracy {results['beyond_accuracy'][m]}")
    (OUT / "semantic_mind.json").write_text(json.dumps(results, indent=2))
    log("done")


if __name__ == "__main__":
    main()
