#!/usr/bin/env python3
"""Q9: metrics with and without features unavailable at serving time.

    PYTHONPATH=. .venv/bin/python -u scripts/ablation_serving_time.py

EB-NeRD hands us this ablation directly. `next_read_time` and `next_scroll_percentage`
describe the user's *next* impression. They exist in train and validation and are **absent
from the test set by construction** -- the organisers removed them because they cannot exist
when a recommendation is actually served.

So they are the perfect probe: a model using them looks excellent offline and is unusable in
production. Reporting both rows is what Q9 asks for, and the gap between them measures how
much an offline number can be inflated by a feature nobody could have at request time.

The honest model is the one that ships. The leaky row exists to be *disclosed*, not used.
"""
from __future__ import annotations

import json, time
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from src.eval.bootstrap import bootstrap_metrics, compare
from src.eval.metrics import per_impression_metrics
from src.lexical.bm25 import BM25
from src.lexical.index import InvertedIndex
from src.lexical.retrieval import build_query
from src.lexical.tokenize import tokenize_fields
from src.pipeline.ebnerd import LANG, article_text, labels_from_clicked, recent_history

SMALL = Path("data/interim/ebnerd/ebnerd_small")
TEST = Path("data/interim/ebnerd/ebnerd_testset/ebnerd_testset")
OUT = Path("data/processed")
N_RECENT = 5
LIMIT = 60_000


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    log("confirming the columns really are absent from test")
    val_cols = {f.name for f in pq.ParquetFile(SMALL / "validation/behaviors.parquet").schema_arrow}
    test_cols = {f.name for f in pq.ParquetFile(TEST / "test/behaviors.parquet").schema_arrow}
    unavailable = sorted(val_cols - test_cols)
    log(f"  in train/val but NOT in test: {unavailable}")

    log("index")
    text = article_text(TEST / "articles.parquet")
    idx = InvertedIndex(); idx.enable_forward_index()
    for aid, (title, subtitle) in text.items():
        idx.add(aid, tokenize_fields(title, subtitle, lang=LANG))
    bm = BM25(idx, idf_variant="lucene")

    hist = recent_history(SMALL / "validation/history.parquet", N_RECENT)
    val = (pl.scan_parquet(SMALL / "validation/behaviors.parquet")
           .select("user_id", "article_ids_inview", "article_ids_clicked",
                   "next_read_time", "next_scroll_percentage")
           .head(LIMIT).collect())
    log(f"  {val.height:,} validation impressions")

    # `next_*` are per-impression scalars, not per-candidate, so a model cannot use them to
    # separate candidates directly. The realistic leak is the one an incautious feature
    # pipeline actually creates: joining the *clicked* article's own next_read_time back onto
    # its candidate row. That is what is reconstructed here, to size the inflation honestly.
    variants = {"honest_bm25": [], "leaky_next_read_time": []}
    scored, t0 = 0, time.perf_counter()
    for uid, cand, clicked, nrt, nsp in zip(
            val["user_id"], val["article_ids_inview"], val["article_ids_clicked"],
            val["next_read_time"], val["next_scroll_percentage"]):
        cands = cand.to_list()
        labels = labels_from_clicked(cands, clicked.to_list())
        if not any(labels):
            continue
        base = bm.score_candidates(build_query(hist.get(uid, []), text, n_recent=N_RECENT, lang=LANG), cands)
        variants["honest_bm25"].append((labels, base))
        signal = float(nrt or 0.0) + float(nsp or 0.0)
        variants["leaky_next_read_time"].append(
            (labels, [b + (signal if l == 1 else 0.0) for b, l in zip(base, labels)]))
        scored += 1
    log(f"  scored {scored:,} in {time.perf_counter()-t0:.0f}s")

    results, cis = {}, {}
    for name, rows in variants.items():
        cis[name] = bootstrap_metrics(per_impression_metrics(rows), iterations=1000, seed=0)
        results[name] = {k: {"mean": v.mean, "lo": v.lo, "hi": v.hi, "n": v.n}
                         for k, v in cis[name].items()}
        log(f"  {name:22s} AUC {cis[name]['auc']}  nDCG@10 {cis[name]['ndcg@10']}")

    log(f"  {compare(cis['leaky_next_read_time']['auc'], cis['honest_bm25']['auc'], 'leaky', 'honest')}")
    inflation = cis["leaky_next_read_time"]["auc"].mean - cis["honest_bm25"]["auc"].mean
    log(f"  AUC inflation from serving-unavailable features: +{inflation:.4f}")

    (OUT / "ablation_serving_time.json").write_text(json.dumps(
        {"unavailable_at_serving": unavailable, "variants": results,
         "auc_inflation": inflation, "impressions": scored}, indent=2))
    log("done")


if __name__ == "__main__":
    main()
