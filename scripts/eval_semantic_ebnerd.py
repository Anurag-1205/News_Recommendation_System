#!/usr/bin/env python3
"""Q3 + Q3.5: semantic retrieval on EB-NeRD, and lexical vs semantic by slice.

    PYTHONPATH=. .venv/bin/python -u scripts/eval_semantic_ebnerd.py [--limit N]

The EB-NeRD counterpart of scripts/eval_semantic_mind.py. The one structural difference is
the embedding source: MIND ships no article vectors so they are computed (TF-IDF -> SVD),
while EB-NeRD *provides* them -- Ekstra Bladet's own word2vec, 300-d, covering 100% of the
20,738-article small bundle (verified before this ran). Using the publisher's vectors is
what the brief suggests and removes a multi-hour encoding job from the critical path.

Reports:
  * recall@K, K in {50,100,200}, mean- vs recency-pooled user vectors (Q3.3, Q3.4)
  * in-impression AUC/MRR/nDCG with bootstrap CIs (Q4)
  * the same, sliced by user history length and article popularity (Q3.5, Q4.3)
  * beyond-accuracy: diversity, novelty, coverage (Q4.2)
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import polars as pl

from src.eval.beyond_accuracy import CoverageTracker, NoveltyModel, intra_list_diversity, top_k_items
from src.eval.bootstrap import CI, bootstrap_metrics, compare
from src.eval.metrics import per_impression_metrics
from src.lexical.bm25 import BM25
from src.lexical.index import InvertedIndex
from src.lexical.retrieval import build_query, recall_at_k
from src.lexical.tokenize import tokenize_fields
from src.pipeline.ebnerd import (LANG, article_text, labels_from_clicked, recent_history,
                                 scan_behaviors)
from src.semantic.ann import ANNIndex
from src.semantic.embeddings import load_provided
from src.semantic.user_vector import build_user_vector

ROOT = Path("data/interim/ebnerd/ebnerd_small")
W2V = Path("data/interim/ebnerd/Ekstra_Bladet_word2vec/document_vector.parquet")
OUT = Path("data/processed")
N_RECENT, COLD_START_MAX = 5, 5      # <=5 history clicks counts as cold-start, as on MIND
KS = (50, 100, 200)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20000, help="impressions for the sliced eval")
    ap.add_argument("--recall-limit", type=int, default=3000, help="impressions for recall@K")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    log("loading article text + categories")
    text = article_text(ROOT / "articles.parquet")
    cats = pl.read_parquet(ROOT / "articles.parquet", columns=["article_id", "category_str"])
    news_cat = dict(zip(cats["article_id"].to_list(), cats["category_str"].to_list()))
    order = list(text.keys())
    log(f"  {len(order):,} articles, {len(set(news_cat.values())):,} categories")

    log("lexical index (Danish)")
    t0 = time.perf_counter()
    idx = InvertedIndex(); idx.enable_forward_index()
    for a in order:
        idx.add(a, tokenize_fields(*text[a], lang=LANG))
    lex_build = time.perf_counter() - t0
    bm = BM25(idx, idf_variant="lucene")
    log(f"  {idx.stats()}  {lex_build:.1f}s")

    log("semantic embeddings — provided Ekstra Bladet word2vec")
    t0 = time.perf_counter()
    all_ids, all_mat = load_provided(W2V, vector_column="document_vector")
    keep = {a: i for i, a in enumerate(all_ids)}
    rows = [keep[a] for a in order if a in keep]
    emb_ids = [a for a in order if a in keep]
    matrix = all_mat[rows]
    emb_build = time.perf_counter() - t0
    covered = len(emb_ids) / len(order)
    log(f"  {matrix.shape}  coverage {covered:.1%} of the corpus  {emb_build:.1f}s")
    del all_mat, all_ids, keep

    t0 = time.perf_counter()
    ann = ANNIndex(emb_ids, matrix, kind="flat")
    ann_build = time.perf_counter() - t0
    log(f"  flat ANN built in {ann_build:.1f}s")

    log("popularity from train (for head/tail slice + novelty)")
    counts = (scan_behaviors(ROOT / "train/behaviors.parquet")
              .select(pl.col("article_ids_clicked").alias("article_id"))
              .explode("article_id").drop_nulls()
              .group_by("article_id").agg(pl.len().alias("c")).collect())
    pop = dict(zip(counts["article_id"].to_list(), counts["c"].to_list()))
    novelty = NoveltyModel(pop)
    pop_rank = sorted(pop, key=pop.get, reverse=True)
    head = set(pop_rank[: max(1, len(pop_rank) // 5)])       # top quintile by clicks
    log(f"  {len(pop):,} distinct clicked articles")

    log("loading validation impressions + history")
    val = scan_behaviors(ROOT / "validation/behaviors.parquet").select(
        "user_id", "article_ids_inview", "article_ids_clicked").collect()
    hist = recent_history(ROOT / "validation/history.parquet", n_recent=20)
    log(f"  {val.height:,} impressions, {len(hist):,} users with history")

    # ---- Q3.4 recall@K, pooling ablation ---------------------------------------------------
    log(f"recall@K over {len(emb_ids):,} articles (limit={args.recall_limit:,})")
    recall = {}
    for pooling in ("mean", "recency"):
        acc = {k: [0, 0] for k in KS}
        skipped = scored = 0
        t0 = time.perf_counter()
        for i, (uid, cand, clicked) in enumerate(
                zip(val["user_id"], val["article_ids_inview"], val["article_ids_clicked"])):
            if i >= args.recall_limit:
                break
            h = hist.get(uid, [])
            relevant = set(clicked.to_list())
            uv = build_user_vector(h, ann.id_to_row, matrix, pooling=pooling)
            if uv is None or not relevant:
                skipped += 1; continue
            got = [a for a, _ in ann.search(uv, top_k=max(KS))]
            for k in KS:
                hits, n_rel = recall_at_k(got, relevant, k)
                acc[k][0] += hits; acc[k][1] += n_rel
            scored += 1
        recall[pooling] = {"recall": {k: round(v[0] / v[1], 5) if v[1] else 0.0
                                      for k, v in acc.items()},
                           "scored": scored, "skipped": skipped,
                           "seconds": round(time.perf_counter() - t0, 1)}
        log(f"  {pooling:8s} {recall[pooling]['recall']}  scored={scored:,} "
            f"({recall[pooling]['seconds']:.0f}s)")

    # ---- Q4 + Q3.5 in-impression, sliced ---------------------------------------------------
    log(f"in-impression re-ranking, sliced (limit={args.limit:,})")
    slices = ("all", "cold_start", "warm", "head", "tail")
    models = ("bm25", "semantic")
    per_imp = {m: {s: {k: [] for k in ("auc", "mrr", "ndcg@5", "ndcg@10")} for s in slices}
               for m in models}
    ba = {m: {"div": [], "nov": [], "cov": CoverageTracker(len(order))} for m in models}

    t0 = time.perf_counter()
    for i, (uid, cand, clicked) in enumerate(
            zip(val["user_id"], val["article_ids_inview"], val["article_ids_clicked"])):
        if i >= args.limit:
            break
        cands = cand.to_list()
        labels = labels_from_clicked(cands, clicked.to_list())
        if not any(labels):
            continue
        h = hist.get(uid, [])
        clicked_ids = [c for c, l in zip(cands, labels) if l == 1]
        row_slices = ["all", "cold_start" if len(h) <= COLD_START_MAX else "warm"]
        row_slices.append("head" if any(c in head for c in clicked_ids) else "tail")

        uv = build_user_vector(h, ann.id_to_row, matrix, pooling="mean")
        scores = {
            "bm25": bm.score_candidates(build_query(h, text, n_recent=N_RECENT, lang=LANG), cands),
            "semantic": ann.score_candidates(uv, cands),
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
                         "embedding_source": "Ekstra_Bladet_word2vec (provided)",
                         "embedding_coverage": round(covered, 4),
                         "embedding_load_s": round(emb_build, 1),
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
        ca = CI(a["auc"]["mean"], a["auc"]["lo"], a["auc"]["hi"], a["auc"]["n"], 1000)
        cb = CI(b["auc"]["mean"], b["auc"]["lo"], b["auc"]["hi"], b["auc"]["n"], 1000)
        log(f"  {sl:11s} n={a['auc']['n']:>6,}  {compare(ca, cb, 'bm25', 'semantic')}")

    for m in models:
        log(f"  {m:9s} beyond-accuracy {results['beyond_accuracy'][m]}")
    (OUT / "semantic_ebnerd.json").write_text(json.dumps(results, indent=2))
    log("done")


if __name__ == "__main__":
    main()
