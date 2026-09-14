#!/usr/bin/env python3
"""A2 Q5: the extended evaluation harness (SPEC.md §17).

    make eval SCORES=data/scores/ebnerd/validation/reranker_final.parquet [K=10] [JSON=out.json]

Reads a scores file written to the §13.3 contract, joins the labels **from its split** (never from
the file), and reports, for the whole split and for each required slice:

  * AUC, MRR, nDCG@5, nDCG@10, each with a bootstrap 95% CI over impressions;
  * the beyond-accuracy trio over each impression's top-k: intra-list diversity, novelty
    (self-information against the training click distribution), catalogue coverage and its Gini.

Slices (SPEC.md §17, definitions in `src/eval/slices.py`): cold (<= 5 history clicks) vs warm
users, and head vs tail impressions — head when a clicked article of that impression is in the top
popularity quintile of the *training* clicks, so "head" never uses the evaluation split's own
labels to define itself beyond the click being scored.
"""
from __future__ import annotations

import argparse, json, time
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

from src.eval.beyond_accuracy import CoverageTracker, NoveltyModel, intra_list_diversity
from src.eval.bootstrap import bootstrap_ci
from src.eval.paired import read_scores, split_labels
from src.eval.slices import COLD_MAX_HISTORY, head_articles, is_cold
from src.rerank.common import METRICS, per_impression

METRIC_LABEL = {"auc": "AUC", "mrr": "MRR", "ndcg@5": "nDCG@5", "ndcg@10": "nDCG@10"}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def context(dataset: str, split: str) -> tuple[pl.DataFrame, dict, dict]:
    """(imp_row, history_len) for the split, the training click counts, and article -> category."""
    if dataset == "ebnerd":
        from src.rerank.ebnerd import SMALL, load_articles, load_behaviors
        beh = load_behaviors(Path("data/interim/ebnerd") / split / "behaviors.parquet")
        hist = (pl.scan_parquet(Path("data/interim/ebnerd") / split / "history.parquet")
                .select("user_id", history_len=pl.col("article_id_fixed").list.len()).collect())
        ctx = (beh.select("imp_row", "user_id").join(hist, on="user_id", how="left")
               .with_columns(pl.col("history_len").fill_null(0)).select("imp_row", "history_len"))
        train = load_behaviors(SMALL / "train/behaviors.parquet")
        counts = Counter(train.select("clicked").explode("clicked", empty_as_null=False)["clicked"].drop_nulls().to_list())
        articles = load_articles()
        cats = dict(zip(articles["article_id"].to_list(), articles["category"].to_list()))
    else:
        from src.rerank.mind import load_behaviors, load_categories
        beh = load_behaviors(split)
        ctx = beh.select("imp_row", history_len=pl.col("history_ids").list.len())
        train = load_behaviors("MINDsmall_train")
        clicked = (train.select("candidates", "labels").explode("candidates", "labels")
                   .filter(pl.col("labels") == 1))
        counts = Counter(clicked["candidates"].to_list())
        c = load_categories(["MINDsmall_train", split])
        cats = dict(zip(c["article_id"].to_list(), c["category"].to_list()))
    return ctx, dict(counts), cats


def beyond_accuracy(joined: pl.DataFrame, k: int, cats: dict, counts: dict, catalogue: int):
    """Per-impression diversity and novelty over the top-k, plus one coverage tracker."""
    novelty = NoveltyModel(counts)
    cover = CoverageTracker(catalogue)
    top = (joined.sort("imp_row", "score", "cand_position", descending=[False, True, False])
           .group_by("imp_row", maintain_order=True).head(k)
           .group_by("imp_row", maintain_order=True).agg(pl.col("article_id")))
    div, nov = [], []
    for items in top["article_id"].to_list():
        div.append(intra_list_diversity([cats.get(a) for a in items]))
        nov.append(novelty.mean(items))
        cover.observe(items)
    return np.array(div), np.array(nov), cover


def row(name: str, n: int, per_imp: dict, mask: np.ndarray | None, div, nov, cover) -> dict:
    sel = (lambda v: v) if mask is None else (lambda v: v[mask])
    out = {"slice": name, "impressions": int(n)}
    for m in METRICS:
        c = bootstrap_ci(sel(per_imp[m]), iterations=1000, seed=0)
        out[m] = {"mean": c.mean, "lo": c.lo, "hi": c.hi, "n": c.n}
    out["diversity"] = float(np.mean(sel(div)))
    out["novelty"] = float(np.mean(sel(nov)))
    if cover is not None:
        out["coverage"] = cover.coverage
        out["gini"] = cover.gini()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--json")
    args = ap.parse_args()

    frame, man = read_scores(Path(args.scores))
    dataset, split = man["dataset"], man["split"]
    log(f"{man['system']} · {dataset} · {split} · {frame.height:,} rows, "
        f"{frame['imp_row'].n_unique():,} impressions (framing {man['framing']})")
    labels = split_labels(dataset, split)
    joined = frame.join(labels, on=["imp_row", "cand_position"], how="inner").sort("imp_row", "cand_position")
    if joined.height != frame.height:
        raise SystemExit(f"label join lost rows: {frame.height:,} -> {joined.height:,}")

    per_imp = per_impression(joined, "score")
    imp_rows = joined["imp_row"].unique(maintain_order=True)
    ctx, counts, cats = context(dataset, split)
    catalogue = frame["article_id"].n_unique()
    log(f"training clicks over {len(counts):,} articles; catalogue in this split {catalogue:,}")

    div, nov, cover = beyond_accuracy(joined, args.k, cats, counts, catalogue)

    hist_len = (pl.DataFrame({"imp_row": imp_rows}).join(ctx, on="imp_row", how="left")
                .with_columns(pl.col("history_len").fill_null(0))["history_len"].to_numpy())
    cold = np.array([is_cold(int(h)) for h in hist_len])
    head_set = head_articles(counts)
    clicked_head = (joined.filter(pl.col("label") == 1)
                    .group_by("imp_row").agg(head=pl.col("article_id").is_in(list(head_set)).any()))
    head = (pl.DataFrame({"imp_row": imp_rows}).join(clicked_head, on="imp_row", how="left")
            .with_columns(pl.col("head").fill_null(False))["head"].to_numpy())
    log(f"slices: cold {cold.sum():,} / warm {(~cold).sum():,} (<= {COLD_MAX_HISTORY} history clicks); "
        f"head {head.sum():,} / tail {(~head).sum():,} (top {len(head_set):,} articles by training clicks)")

    rows = [row("all", len(imp_rows), per_imp, None, div, nov, cover)]
    for name, mask in (("cold", cold), ("warm", ~cold), ("head", head), ("tail", ~head)):
        sub = joined.filter(pl.col("imp_row").is_in(imp_rows.filter(pl.Series(mask)).implode()))
        _, _, sub_cover = beyond_accuracy(sub, args.k, cats, counts, catalogue)
        rows.append(row(name, int(mask.sum()), per_imp, mask, div, nov, sub_cover))

    head_line = f"| slice | impressions | " + " | ".join(METRIC_LABEL[m] for m in METRICS) + " | diversity | novelty | coverage | Gini |"
    print(f"\n**{man['system']} · {dataset} · {split}** — top-{args.k} for the beyond-accuracy trio, "
          f"bootstrap 95% CI over impressions (1,000 resamples, seed 0)\n")
    print(head_line); print("|" + "---|" * (len(METRICS) + 5))
    for r in rows:
        cells = " | ".join(f"{r[m]['mean']:.4f} [{r[m]['lo']:.4f}, {r[m]['hi']:.4f}]" for m in METRICS)
        print(f"| {r['slice']} | {r['impressions']:,} | {cells} | {r['diversity']:.4f} | "
              f"{r['novelty']:.3f} | {r['coverage']:.4f} | {r['gini']:.4f} |")

    out = {"scores": str(args.scores), "manifest": man, "k": args.k, "catalogue": int(catalogue),
           "cold_max_history": COLD_MAX_HISTORY, "head_articles": len(head_set), "rows": rows}
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nrecord: {args.json}")


if __name__ == "__main__":
    main()
