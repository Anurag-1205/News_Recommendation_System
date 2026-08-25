#!/usr/bin/env python3
"""Q2: BM25 on MIND — index build, recall@K, and the n_recent ablation.

    PYTHONPATH=. .venv/bin/python scripts/eval_bm25_mind.py [--limit N]

Reports recall@K for K in {50,100,200} over the whole 51K-article corpus (mode (a)), and
AUC/MRR/nDCG for in-impression re-ranking (mode (b)) on MINDsmall_dev.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import polars as pl

from src.eval.metrics import evaluate_impressions
from src.lexical.bm25 import BM25
from src.lexical.index import InvertedIndex
from src.lexical.retrieval import build_query, evaluate_recall
from src.lexical.tokenize import tokenize_fields
from src.pipeline.mind import (read_news, scan_behaviors, split_dir,
                               with_history, with_labelled_candidates)

ROOT = Path("data/interim/mind")
OUT = Path("data/processed")


def load_corpus(*dirs) -> tuple[InvertedIndex, dict[str, tuple]]:
    """Index the union of the news files. Later duplicates are skipped, not re-added."""
    text: dict[str, tuple] = {}
    for d in dirs:
        news = read_news(d)
        for nid, title, abstract in zip(news["news_id"], news["title"], news["abstract"]):
            text.setdefault(nid, (title, abstract))
    t0 = time.perf_counter()
    idx = InvertedIndex()
    for nid, (title, abstract) in text.items():
        idx.add(nid, tokenize_fields(title, abstract, lang="en"))
    return idx, text, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2000, help="impressions to evaluate")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    train_d, dev_d = split_dir(ROOT, "MINDsmall_train"), split_dir(ROOT, "MINDsmall_dev")

    print("[1/4] building inverted index over title+abstract")
    idx, text, build_s = load_corpus(train_d, dev_d)
    print(f"  {idx.stats()}   build {build_s:.1f}s")

    print("[2/4] loading dev impressions")
    dev = with_history(with_labelled_candidates(scan_behaviors(dev_d))).select(
        "history_ids", "candidates", "labels").collect()
    print(f"  {dev.height:,} impressions")

    bm = BM25(idx, k1=1.2, b=0.75, idf_variant="lucene")

    print(f"[3/4] recall@K over the {idx.n_docs:,}-article corpus, limit={args.limit:,}")
    ablation = {}
    for n_recent in (1, 5, 20):
        rows = ((h.to_list(), [c for c, l in zip(cand.to_list(), lab.to_list()) if l == 1])
                for h, cand, lab in zip(dev["history_ids"], dev["candidates"], dev["labels"]))
        t0 = time.perf_counter()
        res = evaluate_recall(bm, rows, text, ks=(50, 100, 200), n_recent=n_recent,
                              limit=args.limit)
        el = time.perf_counter() - t0
        r = {k: round(v.recall, 4) for k, v in res.items()}
        any_k = next(iter(res.values()))
        ablation[n_recent] = {"recall": r, "scored": any_k.impressions_scored,
                              "skipped": any_k.impressions_skipped, "seconds": round(el, 1)}
        print(f"  n_recent={n_recent:>2}  recall@50={r[50]:.4f}  @100={r[100]:.4f}  "
              f"@200={r[200]:.4f}   scored={any_k.impressions_scored:,} "
              f"skipped={any_k.impressions_skipped:,}  ({el:.0f}s)")

    print("[4/4] in-impression re-ranking on dev (mode b)")
    def rerank_rows():
        for h, cand, lab in zip(dev["history_ids"], dev["candidates"], dev["labels"]):
            q = build_query(h.to_list(), text, n_recent=5)
            yield lab.to_list(), bm.score_candidates(q, cand.to_list())
    t0 = time.perf_counter()
    m = evaluate_impressions(rerank_rows())
    print(f"  {json.dumps({k: (round(v,4) if isinstance(v,float) else v) for k,v in m.items()})}"
          f"  ({time.perf_counter()-t0:.0f}s)")

    (OUT / "bm25_mind_metrics.json").write_text(json.dumps(
        {"index": idx.stats(), "index_build_seconds": round(build_s, 1),
         "recall_ablation": ablation, "rerank_dev": m}, indent=2))


if __name__ == "__main__":
    main()
