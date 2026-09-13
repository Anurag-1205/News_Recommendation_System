#!/usr/bin/env python3
"""Paired bootstrap comparison of two scores files (SPEC.md §14; the Q3.4 judge).

    make paired A=data/scores/ebnerd/validation/nrms.parquet B=data/scores/ebnerd/validation/reranker_final.parquet
    PYTHONPATH=. .venv/bin/python scripts/paired_compare.py A.parquet B.parquet [--iterations 1000] [--seed 0] [--json out.json]

Prints the RESULTS.md table (Δ = B − A with its paired 95% CI and the verdict per metric) and
writes a JSON record carrying both manifests, the counts, iterations, seed and this command, so
the claim can be reproduced from the record alone. Labels come from the split named in the
manifests, never from the files.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

from src.eval.paired import paired_compare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, help="write the full record here")
    args = ap.parse_args()
    t = time.time()
    rep = paired_compare(Path(args.a), Path(args.b), iterations=args.iterations, seed=args.seed)
    print(rep.markdown())
    print(f"({time.time() - t:.0f} s)")
    if args.json:
        rec = rep.to_json()
        rec["command"] = " ".join(sys.argv)
        rec["repo_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        rec["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rec, indent=2))
        print("record:", args.json)


if __name__ == "__main__":
    main()
