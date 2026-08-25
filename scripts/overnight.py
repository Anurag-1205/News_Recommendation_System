#!/usr/bin/env python3
"""Overnight run: compare rankers on MIND dev with bootstrap CIs, then build submission 2.

    PYTHONPATH=. .venv/bin/python -u scripts/overnight.py

Stages, in order, so a crash late still leaves the earlier results on disk:

  1. Index the full MIND corpus (train + dev + test news) with a forward index.
  2. Score every dev impression under four rankers: popularity, BM25, and two fusions.
  3. Bootstrap 95% CIs over impressions; the winner is chosen by CI, and "beats" is only
     used when the intervals are disjoint.
  4. Generate the MIND test submission from whichever ranker actually won.
  5. Larger recall@K ablation than the 2,000-impression run in RESULTS.md.

Every stage writes its own JSON so nothing has to be recomputed to be read back.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

from src.baselines.fusion import rrf, weighted_sum
from src.baselines.popularity import click_counts, score_map
from src.eval.bootstrap import bootstrap_metrics, compare
from src.eval.metrics import per_impression_metrics
from src.eval.submission import format_line, ranks_from_scores, validate_file, zip_submission
from src.lexical.bm25 import BM25
from src.lexical.index import InvertedIndex
from src.lexical.retrieval import build_query, evaluate_recall
from src.lexical.tokenize import tokenize_fields
from src.pipeline.mind import (read_news, scan_behaviors, split_dir, with_history,
                               with_labelled_candidates, with_unlabelled_candidates)

ROOT, OUT = Path("data/interim/mind"), Path("data/processed")
N_RECENT, BATCH = 5, 200_000


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end pass to prove every stage runs before the real one")
    args = ap.parse_args()
    dev_cap = 300 if args.smoke else None
    test_cap = 500 if args.smoke else None
    boot_iters = 50 if args.smoke else 1000
    recall_limit = 30 if args.smoke else 20000
    OUT.mkdir(parents=True, exist_ok=True)
    train_d = split_dir(ROOT, "MINDsmall_train")
    dev_d = split_dir(ROOT, "MINDsmall_dev")
    test_d = split_dir(ROOT, "MINDlarge_test")

    log("stage 1: indexing full corpus (train + dev + test news)")
    text: dict[str, tuple] = {}
    for d in (train_d, dev_d, test_d):
        n = read_news(d)
        for a, ti, ab in zip(n["news_id"], n["title"], n["abstract"]):
            text.setdefault(a, (ti, ab))
    idx = InvertedIndex(); idx.enable_forward_index()
    t0 = time.perf_counter()
    for nid, (ti, ab) in text.items():
        idx.add(nid, tokenize_fields(ti, ab, lang="en"))
    build_s = time.perf_counter() - t0
    log(f"  {idx.stats()}  build {build_s:.1f}s")

    bm = BM25(idx, k1=1.2, b=0.75, idf_variant="lucene")
    pop = score_map(click_counts(with_labelled_candidates(scan_behaviors(train_d))))
    log(f"  popularity fitted on {len(pop):,} articles")

    log("stage 2: scoring dev under four rankers")
    _dev_lf = with_history(with_labelled_candidates(scan_behaviors(dev_d))).select(
        "history_ids", "candidates", "labels")
    dev = (_dev_lf.head(dev_cap) if dev_cap else _dev_lf).collect()
    rankers = ["popularity", "bm25", "rrf", "wsum"]
    collected = {r: [] for r in rankers}
    t0 = time.perf_counter()
    for i, (h, cand, lab) in enumerate(zip(dev["history_ids"], dev["candidates"], dev["labels"])):
        cands, labels = cand.to_list(), lab.to_list()
        q = build_query(h.to_list(), text, n_recent=N_RECENT)
        s_pop = [float(pop.get(c, 0)) for c in cands]
        s_bm = bm.score_candidates(q, cands)
        collected["popularity"].append((labels, s_pop))
        collected["bm25"].append((labels, s_bm))
        collected["rrf"].append((labels, rrf([s_bm, s_pop])))
        collected["wsum"].append((labels, weighted_sum([s_bm, s_pop], [0.7, 0.3])))
        if (i + 1) % 20000 == 0:
            log(f"  {i+1:,}/{dev.height:,}  ({time.perf_counter()-t0:.0f}s)")
    log(f"  scored {dev.height:,} impressions in {time.perf_counter()-t0:.0f}s")

    log("stage 3: bootstrap 95% CIs (1000 iterations, seed 0)")
    results = {}
    for r in rankers:
        cis = bootstrap_metrics(per_impression_metrics(collected[r]), iterations=boot_iters, seed=0)
        results[r] = {k: {"mean": v.mean, "lo": v.lo, "hi": v.hi, "n": v.n} for k, v in cis.items()}
        log(f"  {r:11s} AUC {cis['auc']}  nDCG@10 {cis['ndcg@10']}")
    (OUT / "overnight_dev_cis.json").write_text(json.dumps(results, indent=2))

    best = max(rankers, key=lambda r: results[r]["auc"]["mean"])
    log(f"  best by AUC: {best}")
    for r in rankers:
        if r != best:
            a = bootstrap_metrics(per_impression_metrics(collected[best]), iterations=boot_iters, seed=0)["auc"]
            b = bootstrap_metrics(per_impression_metrics(collected[r]), iterations=boot_iters, seed=0)["auc"]
            log(f"  {compare(a, b, best, r)}")
    del collected

    log(f"stage 4: generating MIND submission from '{best}'")
    lf = with_history(with_unlabelled_candidates(scan_behaviors(test_d)))
    total = lf.select(__import__("polars").len()).collect().item()
    if test_cap:
        total = min(total, test_cap)
    out_txt = OUT / "prediction.txt"
    lengths, written, t0 = {}, 0, time.perf_counter()
    with out_txt.open("w") as fh:
        for start in range(0, total, BATCH):
            # Clamp the final slice: with a capped total, slice(start, BATCH) would happily
            # fetch a full batch past the cap. Harmless on the real 2.37M run, wrong on any
            # capped one -- which is exactly the run used to check the code.
            take = min(BATCH, total - start)
            batch = lf.slice(start, take).select("impression_id", "history_ids", "candidates").collect()
            for imp, h, cand in zip(batch["impression_id"], batch["history_ids"], batch["candidates"]):
                cands = cand.to_list()
                s_bm = bm.score_candidates(build_query(h.to_list(), text, n_recent=N_RECENT), cands)
                s_pop = [float(pop.get(c, 0)) for c in cands]
                scores = {"popularity": s_pop, "bm25": s_bm,
                          "rrf": rrf([s_bm, s_pop]),
                          "wsum": weighted_sum([s_bm, s_pop], [0.7, 0.3])}[best]
                fh.write(format_line(imp, ranks_from_scores(scores)) + "\n")
                lengths[imp] = len(cands)
                written += 1
            log(f"  {written:,}/{total:,}  ({time.perf_counter()-t0:.0f}s)")
            del batch
    log(f"  validating {written:,} lines")
    stats = validate_file(out_txt, expected_lengths=lengths)
    z = zip_submission(out_txt, OUT / ("smoke.zip" if args.smoke else "mind_prediction_v2.zip"))
    log(f"  {stats}  ->  {z} ({z.stat().st_size/1e6:.1f} MB)")
    (OUT / "overnight_submission.json").write_text(json.dumps(
        {"ranker": best, "validation": stats, "seconds": time.perf_counter() - t0}, indent=2))
    del lengths

    log(f"stage 5: recall@K ablation, {recall_limit:,} impressions")
    recall = {}
    for n_recent in (1, 5, 20):
        rows = ((h.to_list(), [c for c, l in zip(cd.to_list(), lb.to_list()) if l == 1])
                for h, cd, lb in zip(dev["history_ids"], dev["candidates"], dev["labels"]))
        t0 = time.perf_counter()
        res = evaluate_recall(bm, rows, text, ks=(50, 100, 200), n_recent=n_recent, limit=recall_limit)
        any_k = next(iter(res.values()))
        recall[n_recent] = {"recall": {k: round(v.recall, 5) for k, v in res.items()},
                            "scored": any_k.impressions_scored,
                            "skipped": any_k.impressions_skipped,
                            "seconds": round(time.perf_counter() - t0, 1)}
        log(f"  n_recent={n_recent:>2} {recall[n_recent]['recall']}  ({recall[n_recent]['seconds']:.0f}s)")
        (OUT / "overnight_recall.json").write_text(json.dumps(recall, indent=2))
    log("done")


if __name__ == "__main__":
    main()
