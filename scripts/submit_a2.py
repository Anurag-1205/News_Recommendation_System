#!/usr/bin/env python3
"""Score a Codabench test file with `config.FINAL` in resumable chunks and write the submission (Q5).

    PYTHONPATH=. python -u scripts/submit_a2.py --dataset mind   [--chunk 100000] [--limit N] [--out data/submissions]
    PYTHONPATH=. python -u scripts/submit_a2.py --dataset ebnerd [--chunk 250000] [--limit N] [--out data/submissions]

Designed for a Kaggle kernel with a session limit (C-004, PLAN P5.4): the A1 EB-NeRD pass took
more than three hours and was killed once, so nothing here restarts from zero.

1. **Fit once.** `src.serving.models.load_or_fit` fits `config.FINAL` on the same seeded sample as
   every measured run and caches it under `data/processed/models/`; it also re-checks the fit
   against the Q2 AUC, so a wrong fit is caught before 13.5M impressions are scored with it.
2. **Score in chunks of `imp_row`.** Each chunk builds its candidate frame with the *same* `build`
   the measured runs use (nothing re-implemented), scores it, converts scores to 1-based ranks in
   the candidate list's own order (`ranks_from_scores`, the only sanctioned way), and writes its
   lines to `<out>/<dataset>/chunks/chunk_NNNNN.txt`. The file is written to a `.part` name and
   renamed, so a chunk file that exists is complete.
3. **Resume = skip existing chunk files.** A killed run relaunched with the same arguments continues
   from the first missing chunk. A ledger line per chunk carries rows, impressions and seconds.
4. **Assemble + validate + zip.** Chunks are concatenated in order, every line is parsed by
   `validate_file` against the test file's own ids and slate lengths (EB-NeRD allows the 200,000
   repeated `impression_id 0` beyond-accuracy rows, MIND does not), and the .txt is zipped at the
   archive root, which is what Codabench expects.

Output: `<out>/<dataset>/predictions.txt`, `<out>/<dataset>/<dataset>_reranker_final.zip`, and a
`manifest.json` with counts, chunking, model, features, commit and command.
"""
from __future__ import annotations

import argparse, json, os, subprocess, time
from pathlib import Path

import numpy as np
import polars as pl

from src.eval.submission import format_line, ranks_from_scores, validate_file, zip_submission
from src.rerank.common import matrix, predict_scores
from src.rerank.config import FINAL

T0 = time.time()


def log(m): print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s] {m}", flush=True)


# ---- per-dataset plumbing --------------------------------------------------------------------------

class EbnerdTest:
    """The EB-NeRD test file, loaded once; chunk frames come from the measured `build`."""
    name = "ebnerd"

    def __init__(self, state, limit):
        from scripts.rerank_ebnerd_a2 import build
        from src.rerank.ebnerd import TEST, load_articles, load_behaviors
        self.build, self.TEST = build, TEST
        self.articles = load_articles()
        self.beh = load_behaviors(TEST / "test/behaviors.parquet", limit=limit)
        self.stage1 = state.stage1
        # slate lengths + ids in file order, for the writer's validation and the line ids
        self.imp_ids = self.beh["impression_id"].to_numpy()
        self.slate_len = self.beh["candidates"].list.len().to_numpy()

    @property
    def n(self): return self.beh.height

    def frame(self, rows: np.ndarray) -> pl.DataFrame:
        long, _ = self.build(self.beh, self.TEST / "test/history.parquet", rows, self.articles, self.stage1,
                             limit_note=" (test chunk)")
        return long

    allow_duplicate_ids = True   # 200,000 beyond-accuracy rows share impression_id 0 (SPEC §6)


class MindTest:
    name = "mind"

    def __init__(self, state, limit):
        from scripts.rerank_mind_a2 import DEV, TEST, TRAIN, build
        from src.rerank.mind import first_sightings, load_behaviors, load_categories
        self.build, self.TEST = build, TEST
        train, dev = load_behaviors(TRAIN), load_behaviors(DEV)
        self.beh = load_behaviors(TEST, labelled=False, limit=limit)
        assert "labels" not in self.beh.columns
        # categories and first-seen times must cover the test file's own articles (the smoke path did the same)
        self.categories = load_categories([TRAIN, DEV, TEST])
        self.sightings = first_sightings([train, dev, self.beh])
        self.stage1 = state.stage1
        self.imp_ids = self.beh["impression_id"].to_numpy()
        self.slate_len = self.beh["candidates"].list.len().to_numpy()

    @property
    def n(self): return self.beh.height

    def frame(self, rows: np.ndarray) -> pl.DataFrame:
        return self.build(self.beh, self.TEST, rows, self.categories, self.sightings, self.stage1, note=" (test chunk)")

    allow_duplicate_ids = False


# ---- chunk scoring --------------------------------------------------------------------------------------

def score_chunk(ds, model, feats, rows: np.ndarray, path: Path) -> dict:
    t0 = time.perf_counter()
    long = ds.frame(rows)
    assert "label" not in long.columns, "a test frame must carry no label"
    scores = predict_scores(model, matrix(long, feats))
    assert np.isfinite(scores).all(), "non-finite score in a test chunk"
    long = long.select("imp_row", "cand_position").with_columns(score=pl.Series(scores, dtype=pl.Float64)).sort("imp_row", "cand_position")
    g = long.group_by("imp_row", maintain_order=True).agg(pl.col("score"), pl.col("cand_position"))
    # every impression in `rows` must be present, in order, with a full slate
    got = g["imp_row"].to_numpy()
    assert np.array_equal(got, rows), f"chunk lost impressions: {len(rows) - len(got)} missing"
    tmp = path.with_suffix(".part")
    with open(tmp, "w") as fh:
        for imp_row, sc, cp in zip(got, g["score"].to_list(), g["cand_position"].to_list()):
            assert cp == list(range(1, len(cp) + 1)), f"imp_row {imp_row}: candidate positions not 1..N"
            assert len(sc) == ds.slate_len[imp_row], f"imp_row {imp_row}: {len(sc)} scores for a {ds.slate_len[imp_row]}-slate"
            fh.write(format_line(int(ds.imp_ids[imp_row]), ranks_from_scores(sc)) + "\n")
    os.replace(tmp, path)
    return {"impressions": int(len(rows)), "rows": int(long.height), "seconds": round(time.perf_counter() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("ebnerd", "mind"), required=True)
    ap.add_argument("--chunk", type=int, default=None, help="impressions per chunk (default: ebnerd 250k, mind 100k)")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N impressions (smoke)")
    ap.add_argument("--out", default="data/submissions")
    ap.add_argument("--zip-name", default=None)
    args = ap.parse_args()
    chunk = args.chunk or {"ebnerd": 250_000, "mind": 100_000}[args.dataset]
    cfg = FINAL[args.dataset]
    feats = cfg["features"]
    out = Path(args.out) / args.dataset
    chunks_dir = out / "chunks"; chunks_dir.mkdir(parents=True, exist_ok=True)
    log(f"config.FINAL[{args.dataset}]: {cfg['objective']}, {len(feats)} features: {feats}")

    from src.serving.models import load_or_fit
    from src.serving.state import ServingState
    state = ServingState.build(args.dataset)
    log("serving state built (stage-1 indexes + stores)")
    model = load_or_fit(args.dataset, state)
    log("model ready (fitted or loaded from data/processed/models/)")

    ds = (EbnerdTest if args.dataset == "ebnerd" else MindTest)(state, args.limit)
    n = ds.n
    n_chunks = (n + chunk - 1) // chunk
    log(f"test file: {n:,} impressions, {int(ds.slate_len.sum()):,} candidate rows, {n_chunks} chunks of {chunk:,}")

    ledger_path = out / "chunks.jsonl"
    done, skipped = 0, 0
    for i in range(n_chunks):
        path = chunks_dir / f"chunk_{i:05d}.txt"
        if path.exists():
            skipped += 1
            continue
        rows = np.arange(i * chunk, min((i + 1) * chunk, n), dtype=np.int64)
        info = score_chunk(ds, model, feats, rows, path)
        with open(ledger_path, "a") as fh:
            fh.write(json.dumps({"chunk": i, **info, "at": time.strftime("%H:%M:%S")}) + "\n")
        done += 1
        log(f"chunk {i + 1}/{n_chunks}: {info['impressions']:,} impressions, {info['rows']:,} rows, {info['seconds']}s")
    log(f"chunks: {done} scored now, {skipped} already present")

    # ---- assemble, validate, zip ----
    pred = out / "predictions.txt"
    with open(pred, "w") as w:
        for i in range(n_chunks):
            with open(chunks_dir / f"chunk_{i:05d}.txt") as r:
                w.write(r.read())
    expected_lengths = None if ds.allow_duplicate_ids else {int(a): int(b) for a, b in zip(ds.imp_ids, ds.slate_len)}
    rep = validate_file(pred, expected_ids=[int(x) for x in ds.imp_ids], expected_lengths=expected_lengths,
                        allow_duplicate_ids=ds.allow_duplicate_ids)
    log(f"validated: {rep}")
    zip_name = args.zip_name or f"{args.dataset}_reranker_final.zip"
    zp = zip_submission(pred, out / zip_name)
    manifest = {
        "dataset": args.dataset, "system": "reranker_final", "config": "src/rerank/config.FINAL (C-018)",
        "model": cfg["objective"], "features": feats, "impressions": int(n), "candidate_rows": int(ds.slate_len.sum()),
        "chunk": chunk, "n_chunks": n_chunks, "limit": args.limit, "validation": rep,
        "predictions_bytes": pred.stat().st_size, "zip": str(zp), "zip_bytes": zp.stat().st_size,
        "repo_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "command": " ".join(["PYTHONPATH=.", "python", "-u", "scripts/submit_a2.py"] + [a for a in os.sys.argv[1:]]),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "total_seconds": round(time.time() - T0),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"RESULT PASS dataset={args.dataset} impressions={n} zip={zp} bytes={zp.stat().st_size} total_seconds={manifest['total_seconds']}")


if __name__ == "__main__":
    main()
