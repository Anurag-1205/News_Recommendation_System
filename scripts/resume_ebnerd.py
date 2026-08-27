#!/usr/bin/env python3
"""Resume an interrupted EB-NeRD scoring pass by completing the missing row groups.

    PYTHONPATH=. .venv/bin/python -u scripts/resume_ebnerd.py

The 13.5M-impression pass was killed partway with no traceback. Rather than redo three hours
of work already on disk, this finishes only what is missing.

**How "missing" is determined without holding 13M ids in memory.** Predictions are written in
Parquet row-group order, one line per impression, append-only. So the number of lines already
written maps directly onto a whole number of completed row groups — arithmetic over
`row_group(i).num_rows`, no set membership required.

The first version of this script did build a `set` of every written impression id. At 13.27M
boxed Python integers that is several hundred megabytes, and it exhausted the machine before
scoring a single missing row — the same object-per-row failure documented twice already in
`AI_USAGE.md`. The lesson did not transfer the first time it was needed.
"""
from __future__ import annotations

import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from src.eval.submission import format_line, ranks_from_scores, validate_file, zip_submission
from src.features.rolling import RollingCounts
from src.lexical.bm25 import BM25
from src.lexical.index import InvertedIndex
from src.lexical.retrieval import build_query
from src.lexical.tokenize import tokenize_fields
from src.pipeline.ebnerd import LANG, article_text, labels_from_clicked, recent_history
from src.semantic.ann import ANNIndex
from src.semantic.embeddings import load_provided
from src.semantic.user_vector import build_user_vector

SMALL = Path("data/interim/ebnerd/ebnerd_small")
TEST = Path("data/interim/ebnerd/ebnerd_testset/ebnerd_testset")
EMB = Path("data/interim/ebnerd/Ekstra_Bladet_word2vec/document_vector.parquet")
BEH = TEST / "test/behaviors.parquet"
OUT = Path("data/processed")
N_RECENT, SEED, LIMIT = 5, 0, 120_000
W24, W1 = timedelta(hours=24), timedelta(hours=1)
COLS = ["impression_id", "impression_time", "user_id", "article_ids_inview"]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def features(rc, ts, cands, hist, bm_scores, sem_scores, published):
    n = len(cands)
    out = []
    for i, c in enumerate(cands):
        pub = published.get(c)
        age = (ts - pub).total_seconds() / 3600.0 if pub is not None else -1.0
        out.append([rc.clicks_before(c, ts), rc.clicks_before(c, ts, W24), rc.clicks_before(c, ts, W1),
                    rc.ctr_before(c, ts), rc.ctr_before(c, ts, W24),
                    age, bm_scores[i], sem_scores[i], n, len(hist)])
    return out


def main() -> None:
    out_txt = OUT / "predictions.txt"
    if not out_txt.exists():
        raise SystemExit("no predictions.txt to resume")

    pf = pq.ParquetFile(BEH)
    sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)]
    total = sum(sizes)

    with out_txt.open("rb") as fh:                       # count lines without loading the file
        written = sum(buf.count(b"\n") for buf in iter(lambda: fh.read(1 << 20), b""))

    # Locate the resume point as (row group, offset within it). Lines are written in row-group
    # order, one per impression, append-only -- so N lines written means the first N
    # impressions are done, wherever that falls. Resuming mid-group matters because this
    # machine kills the process repeatedly; losing only the current chunk rather than the
    # whole group is what lets it finish at all.
    cumulative, first_missing, offset = 0, len(sizes), 0
    for i, n in enumerate(sizes):
        if written < cumulative + n:
            first_missing, offset = i, written - cumulative
            break
        cumulative += n
    if written == total:
        log(f"{written:,} lines already complete")
        first_missing = len(sizes)
    else:
        log(f"{written:,}/{total:,} written — resuming at row group {first_missing} "
            f"of {len(sizes)}, offset {offset:,} within it")

    if first_missing < len(sizes):
        log("rebuilding indices and model (seeded, identical to the original run)")
        text = article_text(TEST / "articles.parquet")
        pub_df = pl.read_parquet(TEST / "articles.parquet", columns=["article_id", "published_time"])
        published = dict(zip(pub_df["article_id"].to_list(), pub_df["published_time"].to_list()))
        del pub_df
        idx = InvertedIndex(); idx.enable_forward_index()
        for aid, (ti, sub) in text.items():
            idx.add(aid, tokenize_fields(ti, sub, lang=LANG))
        bm = BM25(idx, idf_variant="lucene")
        ids, matrix = load_provided(EMB)
        ann = ANNIndex(ids, matrix, kind="flat")

        train = (pl.scan_parquet(SMALL / "train/behaviors.parquet")
                 .select("user_id", pl.col("impression_time").alias("ts"),
                         "article_ids_inview", "article_ids_clicked").collect())
        rc = RollingCounts()
        for ts, cands, clicked in zip(train["ts"], train["article_ids_inview"], train["article_ids_clicked"]):
            cl = set(clicked.to_list())
            for c in cands.to_list():
                rc.add_view(c, ts)
                if c in cl:
                    rc.add_click(c, ts)
        rc.seal()
        del train

        hist_val = recent_history(SMALL / "validation/history.parquet", N_RECENT)
        val = (pl.scan_parquet(SMALL / "validation/behaviors.parquet")
               .select("user_id", pl.col("impression_time").alias("ts"),
                       "article_ids_inview", "article_ids_clicked").head(LIMIT).collect())
        X, y = [], []
        for uid, ts, cd, ck in zip(val["user_id"], val["ts"], val["article_ids_inview"], val["article_ids_clicked"]):
            cands = cd.to_list()
            labels = labels_from_clicked(cands, ck.to_list())
            if not any(labels):
                continue
            hist = hist_val.get(uid, [])
            q = build_query(hist, text, n_recent=N_RECENT, lang=LANG)
            uv = build_user_vector(hist, ann.id_to_row, matrix, pooling="mean")
            X.extend(features(rc, ts, cands, hist, bm.score_candidates(q, cands),
                              ann.score_candidates(uv, cands), published))
            y.extend(labels)
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                               max_leaf_nodes=31, random_state=SEED)
        model.fit(np.asarray(X, np.float32), np.asarray(y, np.int8))
        log(f"  model refit on {len(X):,} rows")
        del X, y, val, hist_val

        # Only the users appearing in the row groups still to be scored. Reading history for
        # all 807,677 users was the memory spike that killed two earlier resume attempts --
        # it happens after the model is fitted, so everything else is already resident.
        needed = set()
        for gi in range(first_missing, len(sizes)):
            needed.update(pl.from_arrow(pf.read_row_group(gi, columns=["user_id"]))["user_id"].to_list())
        log(f"  {len(needed):,} distinct users in the remaining groups (of 807,677)")
        hist_test = recent_history(TEST / "test/history.parquet", N_RECENT, users=needed)
        log(f"  history loaded for {len(hist_test):,} of them")
        del needed
        # Score in small chunks and fsync after each. A whole row group is 265K impressions;
        # at this machine's failure rate that is too much to lose, and holding a group's worth
        # of feature rows is itself part of the memory pressure that causes the failure.
        CHUNK = 20_000
        appended, t0 = 0, time.perf_counter()
        with out_txt.open("a") as fh:
            for gi in range(first_missing, len(sizes)):
                group = pl.from_arrow(pf.read_row_group(gi, columns=COLS))
                start = offset if gi == first_missing else 0
                offset = 0
                rows = group.slice(start, group.height - start)
                del group
                for cs in range(0, rows.height, CHUNK):
                    chunk = rows.slice(cs, CHUNK)
                    feats, meta = [], []
                    for imp, ts, uid, cd in zip(chunk["impression_id"], chunk["impression_time"],
                                                chunk["user_id"], chunk["article_ids_inview"]):
                        cands = cd.to_list()
                        hist = hist_test.get(uid, [])
                        q = build_query(hist, text, n_recent=N_RECENT, lang=LANG)
                        uv = build_user_vector(hist, ann.id_to_row, matrix, pooling="mean")
                        feats.extend(features(rc, ts, cands, hist, bm.score_candidates(q, cands),
                                              ann.score_candidates(uv, cands), published))
                        meta.append((imp, len(cands)))
                    probs = model.predict_proba(np.asarray(feats, np.float32))[:, 1]
                    i = 0
                    for imp, n in meta:
                        fh.write(format_line(imp, ranks_from_scores(probs[i:i + n].tolist())) + "\n")
                        i += n; appended += 1
                    fh.flush(); os.fsync(fh.fileno())   # durable: a kill now loses nothing
                    log(f"  group {gi} +{appended:,}  ({time.perf_counter()-t0:.0f}s)")
                    del feats, probs, chunk, meta
                del rows
        log(f"appended {appended:,} impressions")

    log("validating the complete file")
    # EB-NeRD repeats impression_id 0 across its 200,000 beyond-accuracy rows.
    stats = validate_file(out_txt, allow_duplicate_ids=True)
    if stats["lines"] != total:
        raise SystemExit(f"still short: {stats['lines']:,} of {total:,}")
    z = zip_submission(out_txt, OUT / "ebnerd_predictions_v2.zip")
    log(f"{stats} -> {z} ({z.stat().st_size/1e6:.1f} MB)")
    log("done")


if __name__ == "__main__":
    main()
