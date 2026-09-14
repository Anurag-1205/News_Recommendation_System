#!/usr/bin/env python3
"""Q4 serving benchmark (SPEC.md §16): memory, per-request latency, throughput, cost.

    make bench DATASET=ebnerd            # = taskset -c 0 ... scripts/bench.py --dataset ebnerd
    PYTHONPATH=. .venv/bin/python scripts/bench.py --dataset ebnerd [--n 1000] [--warmup 100] [--seed 0]
                                                   [--qps 100 1000] [--price 0.0425] [--price-source URL]

Protocol: build the serving state (memory per component), fit-or-load config.FINAL, sample n
validation impressions (seed), run warm-up + n requests for framing (a), (b) K=100, (b) K=200 on
ONE core (LightGBM n_jobs=1, OpenMP/BLAS threads 1), record per-stage p50/p95/p99/mean, peak RSS,
the batch-path time for the same impressions, and the cost table. Writes
data/processed/bench_<dataset>.json and prints the RESULTS.md tables.
"""
import argparse, json, os, platform, subprocess, sys, time
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np
import polars as pl


def pct(xs, p): return float(np.percentile(np.asarray(xs) * 1000.0, p))


def summarise(times: list[dict]) -> dict:
    out = {}
    for stage in ("retrieve", "query", "bm25_search", "ann_search", "features", "score", "total"):
        xs = [t[stage] for t in times if stage in t]
        if xs:
            out[stage] = {"p50_ms": pct(xs, 50), "p95_ms": pct(xs, 95), "p99_ms": pct(xs, 99), "mean_ms": float(np.mean(xs) * 1000)}
    if any("query_tokens" in t for t in times):
        out["query_tokens_mean"] = float(np.mean([t["query_tokens"] for t in times if "query_tokens" in t]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ebnerd", "mind"])
    ap.add_argument("--n", type=int, default=1000); ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qps", type=float, nargs="+", default=[100.0, 1000.0])
    ap.add_argument("--price", type=float, default=0.0425, help="USD per vCPU-hour")
    ap.add_argument("--price-source", default="AWS EC2 on-demand c7i.large (2 vCPU) $0.085/h, us-east-1, https://aws.amazon.com/ec2/pricing/on-demand/ (read 2026-09-14)")
    args = ap.parse_args()
    import faiss
    faiss.omp_set_num_threads(1)
    from src.serving.cost import cost_per_1k
    from src.serving.models import load_or_fit
    from src.serving.request import Request, serve
    from src.serving.state import ServingState, peak_rss_bytes

    rec = {"dataset": args.dataset, "command": " ".join(sys.argv), "seed": args.seed, "n": args.n, "warmup": args.warmup,
           "hardware": {"cpu": platform.processor() or subprocess.run("lscpu | grep 'Model name' | sed 's/.*: *//'", shell=True, capture_output=True, text=True).stdout.strip(),
                        "affinity": sorted(os.sched_getaffinity(0)), "python": platform.python_version(), "machine": platform.node()},
           "repo_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "written_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print(f"affinity {rec['hardware']['affinity']}  cpu {rec['hardware']['cpu']}")

    t0 = time.perf_counter()
    st = ServingState.build(args.dataset)
    rec["build_seconds"] = {**st.build_seconds, "total": time.perf_counter() - t0}
    rec["memory"] = st.memory
    print(f"state built in {rec['build_seconds']['total']:.0f}s; RSS {st.memory['process_rss_after_build']['ram_bytes']/1e9:.2f} GB")
    model = load_or_fit(args.dataset, st)
    if type(model).__name__ == "Booster":                     # LightGBM: one core at predict time
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(Path("data/processed/models") / f"{args.dataset}_final.txt"), params={"num_threads": 1})

    # requests: seeded sample of validation impressions
    if args.dataset == "ebnerd":
        from src.rerank.ebnerd import SMALL, load_behaviors
        val = load_behaviors(SMALL / "validation/behaviors.parquet")
    else:
        from src.rerank.mind import load_behaviors
        val = load_behaviors("MINDsmall_dev")
    rows = np.sort(np.random.default_rng(args.seed).choice(val.height, size=args.n + args.warmup, replace=False))
    sample = val.filter(pl.col("imp_row").is_in(rows)).sort("imp_row")
    reqs = [Request(r["user_id"], r["t"], list(r["candidates"]), r["imp_row"], history=list(r["history_ids"]) if "history_ids" in sample.columns else None)
            for r in sample.iter_rows(named=True)]
    warm, meas = reqs[:args.warmup], reqs[args.warmup:]
    rec["latency"] = {}
    for label, framing, k, cache in (("a", "a", 0, True), ("b_k100", "b", 100, True), ("b_k200", "b", 200, True),
                                     ("a_naive", "a", 0, False)):     # naive = profile recomputed per candidate (the first measurement)
        for r in warm:
            serve(st, model, r, framing=framing, k=k, profile_cache=cache)
        times, ncand = [], []
        t1 = time.perf_counter()
        for r in meas:
            resp = serve(st, model, r, framing=framing, k=k, profile_cache=cache)
            times.append(resp.timings); ncand.append(len(resp.candidates))
        wall = time.perf_counter() - t1
        rec["latency"][label] = {**summarise(times), "n_requests": len(meas), "mean_candidates": float(np.mean(ncand)),
                                 "wall_s": wall, "single_core_qps": len(meas) / wall}
        L = rec["latency"][label]
        sub = f" (bm25 {L['bm25_search']['p99_ms']:.1f} / ann {L['ann_search']['p99_ms']:.1f}, {L.get('query_tokens_mean', 0):.0f} query tokens)" if "bm25_search" in L else ""
        print(f"{label:7s} total p50 {L['total']['p50_ms']:.1f} p95 {L['total']['p95_ms']:.1f} p99 {L['total']['p99_ms']:.1f} ms | "
              f"retrieve p99 {L['retrieve']['p99_ms']:.1f}{sub} features p99 {L['features']['p99_ms']:.1f} score p99 {L['score']['p99_ms']:.1f} | "
              f"{L['mean_candidates']:.0f} cands | {L['single_core_qps']:.1f} req/s on one core")
    rec["peak_rss_bytes"] = peak_rss_bytes()

    # the batch path on the same impressions (framing (a)), for the overhead comparison
    from src.rerank.common import matrix, predict_scores
    from src.rerank.config import FINAL
    t2 = time.perf_counter()
    if args.dataset == "ebnerd":
        from scripts.rerank_ebnerd_a2 import build
        from src.rerank.ebnerd import load_articles
        long, _ = build(val, SMALL / "validation/history.parquet", np.array([r.imp_row for r in meas]), load_articles(), st.stage1)
    else:
        from scripts.rerank_mind_a2 import DEV, TRAIN, build
        from src.rerank.mind import first_sightings, load_categories
        train = load_behaviors(TRAIN)
        long = build(val, DEV, np.array([r.imp_row for r in meas]), load_categories([TRAIN, DEV]), first_sightings([train, val]), st.stage1)
    predict_scores(model, matrix(long, FINAL[args.dataset]["features"]))
    rec["batch_path"] = {"wall_s": time.perf_counter() - t2, "per_request_ms": (time.perf_counter() - t2) / len(meas) * 1000}
    print(f"batch path: {rec['batch_path']['wall_s']:.1f}s for {len(meas)} impressions = {rec['batch_path']['per_request_ms']:.2f} ms/request")

    # cost at the target QPS, from the shipped framing (a) and the literal Q4.2 framing (b, K=100)
    rec["cost"] = {"price_per_vcpu_hour": args.price, "price_source": args.price_source, "rho": 0.5, "sla_s": 0.1, "rows": []}
    for label in ("a", "b_k100"):
        L = rec["latency"][label]
        for qps in args.qps:
            e = cost_per_1k(L["total"]["mean_ms"] / 1000, L["total"]["p99_ms"] / 1000, qps, args.price)
            rec["cost"]["rows"].append({"framing": label, **e.__dict__})
            print(f"cost {label:7s} @ {qps:6.0f} QPS: {e.cores} cores, SLA {'holds' if e.sla_holds else 'FAILS'} (p99 {L['total']['p99_ms']:.1f} ms), ${e.cost_per_1k_queries:.5f} per 1k queries")
    out = Path(f"data/processed/bench_{args.dataset}.json"); out.write_text(json.dumps(rec, indent=2, default=str))
    print("record:", out)


if __name__ == "__main__":
    main()
