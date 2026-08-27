#!/usr/bin/env python3
"""Q2: BM25 on EB-NeRD -- index build, recall@K, and the n_recent ablation.

    PYTHONPATH=. .venv/bin/python scripts/eval_bm25_ebnerd.py [--limit N]

Mirrors scripts/eval_bm25_mind.py (SPEC.md Sec1: one scorer, two harnesses) over
ebnerd_small's Danish corpus instead of MIND's English one -- same ablation, same two
modes. Reports recall@K for K in {50,100,200} over the whole article corpus (mode a), and
AUC/MRR/nDCG for in-impression re-ranking (mode b) on ebnerd_small validation.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

from src.eval.metrics import evaluate_impressions
from src.lexical.bm25 import BM25
from src.lexical.index import InvertedIndex
from src.lexical.retrieval import build_query, evaluate_recall
from src.lexical.tokenize import tokenize_fields
from src.pipeline.ebnerd import LANG, article_text, labels_from_clicked, recent_history, scan_behaviors

ROOT = Path("data/interim/ebnerd/ebnerd_small")
OUT = Path("data/processed")
N_RECENT_MAX = 20   # history loaded once at this depth; the ablation slices it per query


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20000, help="impressions to evaluate")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("[1/4] building inverted index over title+subtitle (Danish)")
    text = article_text(ROOT / "articles.parquet")
    t0 = time.perf_counter()
    idx = InvertedIndex()
    idx.enable_forward_index()
    for aid, (title, subtitle) in text.items():
        idx.add(aid, tokenize_fields(title, subtitle, lang=LANG))
    build_s = time.perf_counter() - t0
    print(f"  {idx.stats()}   build {build_s:.1f}s")

    print("[2/4] loading validation impressions + history")
    val = scan_behaviors(ROOT / "validation/behaviors.parquet").select(
        "user_id", "article_ids_inview", "article_ids_clicked").collect()
    print(f"  {val.height:,} impressions")
    hist = recent_history(ROOT / "validation/history.parquet", n_recent=N_RECENT_MAX)
    print(f"  {len(hist):,} users with history")

    bm = BM25(idx, k1=1.2, b=0.75, idf_variant="lucene")

    print(f"[3/4] recall@K over the {idx.n_docs:,}-article corpus, limit={args.limit:,}")
    ablation = {}
    for n_recent in (1, 5, 20):
        rows = ((hist.get(uid, []), clicked.to_list())
                for uid, clicked in zip(val["user_id"], val["article_ids_clicked"]))
        t0 = time.perf_counter()
        res = evaluate_recall(bm, rows, text, ks=(50, 100, 200), n_recent=n_recent,
                              lang=LANG, limit=args.limit)
        el = time.perf_counter() - t0
        r = {k: round(v.recall, 5) for k, v in res.items()}
        any_k = next(iter(res.values()))
        ablation[n_recent] = {"recall": r, "scored": any_k.impressions_scored,
                              "skipped": any_k.impressions_skipped, "seconds": round(el, 1)}
        print(f"  n_recent={n_recent:>2}  recall@50={r[50]:.5f}  @100={r[100]:.5f}  "
              f"@200={r[200]:.5f}   scored={any_k.impressions_scored:,} "
              f"skipped={any_k.impressions_skipped:,}  ({el:.0f}s)")

    print("[4/4] in-impression re-ranking on validation (mode b)")
    def rerank_rows():
        for uid, cand, clicked in zip(val["user_id"], val["article_ids_inview"], val["article_ids_clicked"]):
            cands = cand.to_list()
            labels = labels_from_clicked(cands, clicked.to_list())
            if not any(labels):
                continue
            q = build_query(hist.get(uid, []), text, n_recent=5, lang=LANG)
            yield labels, bm.score_candidates(q, cands)
    t0 = time.perf_counter()
    m = evaluate_impressions(rerank_rows())
    print(f"  {json.dumps({k: (round(v,4) if isinstance(v,float) else v) for k,v in m.items()})}"
          f"  ({time.perf_counter()-t0:.0f}s)")

    (OUT / "bm25_ebnerd_metrics.json").write_text(json.dumps(
        {"index": idx.stats(), "index_build_seconds": round(build_s, 1),
         "recall_ablation": ablation, "rerank_val": m}, indent=2))


if __name__ == "__main__":
    main()
