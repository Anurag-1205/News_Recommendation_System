#!/usr/bin/env python3
"""EB-NeRD Codabench submission (competition 2469).

    PYTHONPATH=. .venv/bin/python -u scripts/make_submission_ebnerd.py [--smoke]

Same shape as the MIND pipeline, and deliberately so: one scorer, two harnesses (SPEC.md §1).
Measure on the labelled validation split first, choose the ranker by bootstrap CI, then score
the 13,536,710-impression test set.

Scale is the whole difficulty here. The test file is 13.5M rows -- 5.7x MIND's -- with a
1.16 GB history file beside it, on a machine with ~2 GB free. Three things keep it inside
that budget: history truncated to the last n at read time, behaviours read one Parquet row
group at a time, and predictions appended to disk rather than accumulated.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import polars as pl

from src.baselines.fusion import rrf, weighted_sum
from src.baselines.popularity import click_counts, score_map
from src.eval.bootstrap import bootstrap_metrics, compare
from src.eval.metrics import per_impression_metrics
from src.eval.submission import format_line, ranks_from_scores, validate_file, zip_submission
from src.lexical.bm25 import BM25
from src.lexical.index import InvertedIndex
from src.lexical.retrieval import build_query
from src.lexical.tokenize import tokenize_fields
from src.pipeline.ebnerd import (LANG, article_text, iter_row_groups, labels_from_clicked,
                                 n_rows, recent_history, scan_behaviors)

SMALL = Path("data/interim/ebnerd/ebnerd_small")
TEST = Path("data/interim/ebnerd/ebnerd_testset/ebnerd_testset")
OUT = Path("data/processed")
N_RECENT = 5
RANKERS = ["popularity", "bm25", "rrf", "wsum"]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def score_all(bm, pop, query, cands):
    """Every ranker's scores for one impression, so they are compared on identical inputs."""
    s_bm = bm.score_candidates(query, cands)
    s_pop = [float(pop.get(c, 0)) for c in cands]
    return {"popularity": s_pop, "bm25": s_bm,
            "rrf": rrf([s_bm, s_pop]), "wsum": weighted_sum([s_bm, s_pop], [0.7, 0.3])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    val_cap = 2000 if args.smoke else None
    boot = 50 if args.smoke else 1000
    OUT.mkdir(parents=True, exist_ok=True)

    log("stage 1: article text + inverted index (Danish)")
    text = article_text(TEST / "articles.parquet")
    log(f"  {len(text):,} articles")
    idx = InvertedIndex(); idx.enable_forward_index()
    t0 = time.perf_counter()
    for aid, (title, subtitle) in text.items():
        idx.add(aid, tokenize_fields(title, subtitle, lang=LANG))
    log(f"  {idx.stats()}  build {time.perf_counter()-t0:.1f}s")
    bm = BM25(idx, k1=1.2, b=0.75, idf_variant="lucene")

    log("stage 2: popularity from ebnerd_small train")
    # EB-NeRD stores clicks as a list of article ids per impression, so counting is an
    # explode-and-group rather than the label-mask filter MIND needs. drop_nulls matters:
    # an impression with no click carries a null rather than an empty list.
    counts = (
        scan_behaviors(SMALL / "train/behaviors.parquet")
        .select(pl.col("article_ids_clicked").alias("article_id"))
        .explode("article_id")
        .drop_nulls()
        .group_by("article_id")
        .agg(pl.len().alias("click_count"))
        .sort("click_count", descending=True)
        .collect()
    )
    pop = dict(zip(counts["article_id"].to_list(), counts["click_count"].to_list()))
    log(f"  fitted on {len(pop):,} distinct clicked articles")

    log("stage 3: validation with labels")
    hist_val = recent_history(SMALL / "validation/history.parquet", N_RECENT)
    val = (scan_behaviors(SMALL / "validation/behaviors.parquet")
           .select("user_id", "article_ids_inview", "article_ids_clicked"))
    val = (val.head(val_cap) if val_cap else val).collect()
    log(f"  {val.height:,} impressions, {len(hist_val):,} users with history")

    # Metrics are reduced per impression as we go, NOT collected as raw score lists.
    # Holding (labels, scores) for 4 rankers x 244,647 impressions x ~11 candidates is what
    # killed the first attempt: the equivalent MIND run held 73,152 impressions at ~1.3 GB,
    # and EB-NeRD's validation split is 3.3x larger, which exhausted the machine. Storing
    # four floats per impression instead of two lists cuts that to tens of MB.
    metric_names = ("auc", "mrr", "ndcg@5", "ndcg@10")
    per_imp = {r: {m: [] for m in metric_names} for r in RANKERS}
    scored = 0
    t0 = time.perf_counter()
    for uid, cand, clicked in zip(val["user_id"], val["article_ids_inview"], val["article_ids_clicked"]):
        cands = cand.to_list()
        labels = labels_from_clicked(cands, clicked.to_list())
        if not any(labels):
            continue                       # no positive: every ranking metric is undefined
        query = build_query(hist_val.get(uid, []), text, n_recent=N_RECENT, lang=LANG)
        for name, s in score_all(bm, pop, query, cands).items():
            one = per_impression_metrics([(labels, s)])
            for m in metric_names:
                per_imp[name][m].append(one[m][0])
        scored += 1
    log(f"  scored {scored:,} labelled impressions in {time.perf_counter()-t0:.0f}s")
    del val, hist_val

    log(f"stage 4: bootstrap 95% CIs ({boot} iterations, seed 0)")
    results, cis = {}, {}
    for r in RANKERS:
        cis[r] = bootstrap_metrics(per_imp[r], iterations=boot, seed=0)
        results[r] = {k: {"mean": v.mean, "lo": v.lo, "hi": v.hi, "n": v.n} for k, v in cis[r].items()}
        log(f"  {r:11s} AUC {cis[r]['auc']}  nDCG@10 {cis[r]['ndcg@10']}")
    (OUT / "ebnerd_val_cis.json").write_text(json.dumps(results, indent=2))
    best = max(RANKERS, key=lambda r: results[r]["auc"]["mean"])
    log(f"  best by AUC: {best}")
    for r in RANKERS:
        if r != best:
            log(f"  {compare(cis[best]['auc'], cis[r]['auc'], best, r)}")
    del per_imp

    log("stage 5: scoring the test set")
    hist_test = recent_history(TEST / "test/history.parquet", N_RECENT)
    log(f"  {len(hist_test):,} users with history")
    beh = TEST / "test/behaviors.parquet"
    total = n_rows(beh)
    out_txt = OUT / "predictions.txt"
    # Candidate counts are checked inline rather than accumulated: a {impression_id: n}
    # dict over 13.5M rows would cost more than a gigabyte, which is the whole memory budget.
    # The rank list is built from the candidate list, so the two can only disagree through a
    # bug -- which is exactly why it is asserted here rather than assumed.
    written, t0 = 0, time.perf_counter()
    with out_txt.open("w") as fh:
        for group in iter_row_groups(beh, ["impression_id", "user_id", "article_ids_inview"]):
            for imp, uid, cand in zip(group["impression_id"], group["user_id"], group["article_ids_inview"]):
                cands = cand.to_list()
                query = build_query(hist_test.get(uid, []), text, n_recent=N_RECENT, lang=LANG)
                ranks = ranks_from_scores(score_all(bm, pop, query, cands)[best])
                if len(ranks) != len(cands):
                    raise AssertionError(f"impression {imp}: {len(ranks)} ranks, {len(cands)} candidates")
                fh.write(format_line(imp, ranks) + "\n")
                written += 1
            log(f"  {written:,}/{total:,}  ({time.perf_counter()-t0:.0f}s)")
            if args.smoke and written >= 300_000:
                break
    log(f"  validating {written:,} lines")
    # EB-NeRD's 200,000 beyond-accuracy rows all carry impression_id 0 (verified against the
    # file: they correspond exactly to is_beyond_accuracy), so repeated ids are expected here.
    stats = validate_file(out_txt, allow_duplicate_ids=True)
    if not args.smoke and stats["lines"] != total:
        raise AssertionError(f"wrote {stats['lines']:,} lines for {total:,} impressions")
    z = zip_submission(out_txt, OUT / ("ebnerd_smoke.zip" if args.smoke else "ebnerd_predictions.zip"))
    log(f"  {stats}  ->  {z} ({z.stat().st_size/1e6:.1f} MB)")
    (OUT / "ebnerd_submission.json").write_text(json.dumps(
        {"ranker": best, "validation": stats, "seconds": time.perf_counter() - t0,
         "expected_impressions": total}, indent=2))
    log("done")


if __name__ == "__main__":
    main()
