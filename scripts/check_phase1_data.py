#!/usr/bin/env python3
"""Print every data fact that the Phase 1 feature definitions rest on (SPEC.md §11).

    PYTHONPATH=. .venv/bin/python scripts/check_phase1_data.py

Each section backs a specific decision in CONTEXT.md, so a teammate can re-derive it instead of
trusting it. Runs on the laptop: full-split aggregates are streamed, and anything that must explode
per-candidate rows is capped at the first SAMPLE impressions (an unsampled explode of EB-NeRD train
was killed by the OOM killer on a 7 GB machine).
"""

from __future__ import annotations

import polars as pl

from src.pipeline.mind import scan_behaviors, split_dir, with_parsed_time

MIND = "data/interim/mind"
EB = "data/interim/ebnerd"
EB_SPLITS = {"small/train": f"{EB}/ebnerd_small/train", "small/validation": f"{EB}/ebnerd_small/validation",
             "test": f"{EB}/ebnerd_testset/ebnerd_testset/test"}
SAMPLE = 50_000


def eb(split: str, name: str = "behaviors") -> pl.LazyFrame:
    return pl.scan_parquet(f"{EB_SPLITS[split]}/{name}.parquet")


def ctr_by_position_quintile(long: pl.LazyFrame) -> str:
    q = (long.with_columns(qn=(pl.col("pos") * 5 // pl.col("n")).cast(pl.Int8))
         .group_by("qn").agg(ctr=pl.col("label").mean()).sort("qn").collect())
    return "  ".join(f"q{d}:{c:.4f}" for d, c in zip(q["qn"], q["ctr"]))


def mind_history_and_windows() -> None:                               # C-008: P1-D1
    print("\n## MIND — split windows and whether history is a frozen snapshot")
    frames = {}
    for name in ("MINDsmall_train", "MINDsmall_dev", "MINDlarge_test"):
        df = (with_parsed_time(scan_behaviors(split_dir(MIND, name)))
              .select("user_id", "ts", pl.col("history").fill_null("")).collect())
        frames[name] = df
        per_user = df.group_by("user_id").agg(n=pl.len(), variants=pl.col("history").n_unique())
        multi = per_user.filter(pl.col("n") > 1)
        print(f"{name:16s} first {df['ts'].min()}  last {df['ts'].max()}  | repeat users {multi.height:,}, "
              f"history varies within split for {multi.filter(pl.col('variants') > 1).height:,}")
    a = frames["MINDsmall_train"].group_by("user_id").agg(pl.col("history").first())
    b = frames["MINDsmall_dev"].group_by("user_id").agg(pl.col("history").first())
    j = a.join(b, on="user_id", suffix="_dev")
    print(f"users in both small train and dev: {j.height:,}; identical history: {(j['history'] == j['history_dev']).sum():,}")


def candidates_per_impression() -> None:                              # C-009: cost model
    print("\n## Mean candidates per impression")
    for name in ("MINDsmall_train", "MINDsmall_dev", "MINDlarge_test"):
        n = (scan_behaviors(split_dir(MIND, name))
             .select(pl.col("impressions").str.count_matches(" ").add(1).mean()).collect().item())
        print(f"MIND {name:16s} {n:.1f}")
    n = eb("small/validation").select(pl.col("article_ids_inview").list.len().mean()).collect().item()
    print(f"EB-NeRD small/validation {n:.1f}")


def ebnerd_columns() -> None:                                         # C-013: test-file availability
    print("\n## EB-NeRD behaviors columns — what the Codabench test file lacks")
    train = set(eb("small/train").collect_schema().names())
    test = set(eb("test").collect_schema().names())
    print("in train, absent from test:", sorted(train - test))
    print("in test, absent from train:", sorted(test - train))


def list_position() -> None:                                          # C-013: cand_position
    print("\n## List position — sorted by id? click rate by position quintile?")
    for split in EB_SPLITS:
        r = (eb(split).select(l=pl.col("article_ids_inview")).filter(pl.col("l").list.len() > 1)
             .select(asc=pl.col("l").list.eval(pl.element().diff().drop_nulls() >= 0).list.all().mean(),
                     desc=pl.col("l").list.eval(pl.element().diff().drop_nulls() <= 0).list.all().mean())
             .collect(engine="streaming").row(0))
        print(f"EB-NeRD {split:17s} inview sorted ascending {r[0]:.3f}, descending {r[1]:.3f}")
    for split in ("small/train", "small/validation"):
        long = (eb(split).head(SAMPLE).select("article_ids_inview", "article_ids_clicked")
                .with_columns(n=pl.col("article_ids_inview").list.len(),
                              pos=pl.int_ranges(pl.col("article_ids_inview").list.len()))
                .explode("article_ids_inview", "pos", empty_as_null=False)
                .with_columns(label=pl.col("article_ids_inview").is_in(pl.col("article_ids_clicked")).cast(pl.Float64)))
        print(f"EB-NeRD {split:17s} (first {SAMPLE:,}) click rate by quintile: {ctr_by_position_quintile(long)}")
    for name in ("MINDsmall_train", "MINDsmall_dev"):
        long = (scan_behaviors(split_dir(MIND, name)).head(SAMPLE).select(c=pl.col("impressions").str.split(" "))
                .with_columns(n=pl.col("c").list.len(), pos=pl.int_ranges(pl.col("c").list.len()))
                .explode("c", "pos", empty_as_null=False)
                .with_columns(label=pl.col("c").str.ends_with("-1").cast(pl.Float64)))
        print(f"MIND {name:16s} (first {SAMPLE:,}) click rate by quintile: {ctr_by_position_quintile(long)}")


def sessions_and_dwell() -> None:                                     # C-013: session key, nulls
    print("\n## EB-NeRD sessions, clicks and dwell nulls")
    for split in ("small/train", "test"):
        lf = eb(split)
        by_sid = (lf.group_by("session_id").agg(n=pl.len(), users=pl.col("user_id").n_unique())
                  .select(max_len=pl.col("n").max(), multi_user_sessions=(pl.col("users") > 1).sum())
                  .collect(engine="streaming").row(0))
        by_key = (lf.group_by("user_id", "session_id")
                  .agg(n=pl.len().cast(pl.Int64), tied=(pl.len() - pl.col("impression_time").n_unique()))
                  .select(sessions=pl.len(), max_len=pl.col("n").max(), p99=pl.col("n").quantile(0.99),
                          pairs=(pl.col("n") ** 2).sum(), with_tied_t=(pl.col("tied") > 0).sum())
                  .collect(engine="streaming").row(0))
        nulls = lf.select(n=pl.len(), scroll_null=pl.col("scroll_percentage").null_count()).collect().row(0)
        print(f"{split:12s} keyed by session_id alone: max {by_sid[0]:,} impressions, {by_sid[1]} multi-user session(s)")
        print(f"{'':12s} keyed by (user_id, session_id): {by_key[0]:,} sessions, max {by_key[1]}, p99 {by_key[2]:.0f}, "
              f"self-join pairs {by_key[3]:,}, sessions with tied timestamps {by_key[4]}")
        print(f"{'':12s} scroll_percentage null {nulls[1]:,} / {nulls[0]:,} = {nulls[1] / nulls[0]:.1%}")
    test = eb("test")
    ba = test.filter(pl.col("is_beyond_accuracy")).select(
        rows=pl.len(), sessions=pl.col("session_id").unique().len(),
        id0=(pl.col("impression_id") == 0).all()).collect().row(0)
    print(f"test is_beyond_accuracy rows: {ba[0]:,}, distinct session_ids {ba[1]}, all impression_id == 0: {ba[2]}")
    clicks = eb("small/train").select(no_click=(pl.col("article_ids_clicked").list.len() == 0).sum(),
                                     mean=pl.col("article_ids_clicked").list.len().mean()).collect().row(0)
    print(f"small/train impressions with no click: {clicks[0]}, mean clicks per impression {clicks[1]:.3f}")
    h = eb("small/train", "history").select(
        clicks=pl.col("read_time_fixed").list.len().sum(),
        read_null=pl.col("read_time_fixed").list.eval(pl.element().is_null().sum()).list.first().sum(),
        scroll_null=pl.col("scroll_percentage_fixed").list.eval(pl.element().is_null().sum()).list.first().sum()
    ).collect().row(0)
    print(f"small/train history clicks {h[0]:,}: read_time null {h[1]:,}, scroll null {h[2]:,} = {h[2] / h[0]:.1%}")


if __name__ == "__main__":
    mind_history_and_windows()
    candidates_per_impression()
    ebnerd_columns()
    list_position()
    sessions_and_dwell()
