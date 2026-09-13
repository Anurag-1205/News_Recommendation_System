#!/usr/bin/env python3
"""Archive a Kaggle kernel's log and append one row to scripts/kaggle/RUN_LEDGER.md.

    .venv/bin/python scripts/kaggle/ledger.py <kernel-slug> <tag> [--note "..."]
    e.g.  .venv/bin/python scripts/kaggle/ledger.py a2-nrms-mind-smoke v3

Fetches `kaggle kernels logs aayushpandey18602/<slug>`, saves it to
data/logs/kaggle/<slug>_<tag>.log (gitignored: logs are large), and extracts what the kernels
print on purpose — KNOBS, repo/benchmark commits, STAGE timings, HISTORY, TRAIN_SECONDS,
METRICS_*, EVAL_*, RESULT, or the last Traceback line — into a ledger row that is committed.
The ledger is the place to look before re-running anything: every number, its commit and its
cost, in one table (CLAUDE.md rule 3). Runs from the second account (`--account alt`) carry the
account name in the kernel column; runs from the main account carry none.
"""
import argparse, ast, json, re, subprocess, sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "scripts/kaggle/RUN_LEDGER.md"
LOGS = ROOT / "data/logs/kaggle"
USER = "aayushpandey18602"


def events(raw: str):
    i = raw.find("[")
    if i < 0:
        return []
    return json.loads(raw[i:])


def extract(ev):
    out = {"stages": {}, "metrics": {}, "outcome": "?", "error": None}
    last_line = None
    for e in ev:
        for line in e.get("data", "").replace("\r", "\n").split("\n"):
            line = line.strip()
            if not line:
                continue
            last_line = line
            for key in ("KNOBS", "HISTORY", "METRICS_OURS", "METRICS_CI", "METRICS_BENCH", "METRICS_PKG", "METRICS", "EVAL_BEFORE", "EVAL_AFTER"):
                if line.startswith(key + " "):
                    payload = line[len(key) + 1:]
                    try:
                        out["metrics"][key] = json.loads(payload)
                    except json.JSONDecodeError:
                        try:
                            out["metrics"][key] = ast.literal_eval(payload)   # older kernels printed a Python dict
                        except (ValueError, SyntaxError):
                            out["metrics"][key] = payload[:200]
            m = re.match(r"STAGE (\S+) ok (\d+)s", line)
            if m:
                out["stages"][m.group(1)] = int(m.group(2))
            if m := re.search(r"repo commit ([0-9a-f]{40})", line):
                out["repo_commit"] = m.group(1)[:7]
            if m := re.match(r"TRAIN_SECONDS (\d+)", line):
                out["train_seconds"] = int(m.group(1))
            if m := re.match(r"STEP_TIME (.*)", line):
                out["step_time"] = m.group(1)
            if line.startswith("DEPARTURE"):
                out["departure"] = line
            if line.startswith("RESULT "):
                out["outcome"] = line
            if line.startswith("MIXED_PRECISION_FAILED") or line.startswith("ALIGN "):
                out.setdefault("notes", []).append(line[:120])
    if out["outcome"] == "?":
        err = [l.strip() for e in ev for l in e.get("data", "").split("\n")
               if re.match(r"([\w.]+\.)?\w+(Error|Exception|Interrupted)\b", l.strip()) and "Traceback" not in l]
        out["outcome"] = "ERROR"; out["error"] = (err[-1] if err else last_line or "")[:200]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug"); ap.add_argument("tag"); ap.add_argument("--note", default="")
    ap.add_argument("--no-fetch", action="store_true", help="use the already-saved log")
    ap.add_argument("--account", default="main", choices=["main", "alt"],
                    help="which Kaggle account ran it: main = aayushpandey18602, alt = ~/.kaggle/alt (C-027)")
    a = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    import os
    env = dict(os.environ)
    user = USER
    if a.account == "alt":
        env["KAGGLE_CONFIG_DIR"] = os.path.expanduser("~/.kaggle/alt")
        user = json.load(open(os.path.expanduser("~/.kaggle/alt/kaggle.json")))["username"]
    log = LOGS / (f"{a.slug}_{a.tag}.log" if a.account == "main" else f"{a.slug}_{a.tag}_{user}.log")
    if not a.no_fetch:
        raw = subprocess.run([str(ROOT / ".venv/bin/kaggle"), "kernels", "logs", f"{user}/{a.slug}"], capture_output=True, text=True, env=env).stdout
        log.write_text(raw)
    x = extract(events(log.read_text()))
    total = max(x["stages"].values()) if x["stages"] else None
    m = x["metrics"]
    key = (m.get("METRICS_OURS") or m.get("EVAL_AFTER") or m.get("METRICS") or {})
    key_s = ", ".join(f"{k}={v:.4f}" for k, v in key.items() if isinstance(v, (int, float))) if isinstance(key, dict) else ""
    hist = m.get("HISTORY", {})
    if isinstance(hist, dict) and "val_auc" in hist:
        key_s = f"val_auc(holdout)={max(hist['val_auc']):.4f}; " + key_s
    ci = m.get("METRICS_CI", {})
    ci_s = f"auc 95% CI [{ci['auc'][1]:.4f}, {ci['auc'][2]:.4f}]" if isinstance(ci, dict) and "auc" in ci else ""
    knobs = m.get("KNOBS", {})
    mode = ", ".join(f"{k}={v}" for k, v in knobs.items() if k in ("DEMO_CHECK", "DATASPLIT", "EPOCHS", "TRANSFORMER")) if isinstance(knobs, dict) else ""
    stages = " · ".join(f"{k} {v}s" for k, v in x["stages"].items())
    outcome = x["outcome"] if x["outcome"] != "ERROR" else f"ERROR: {x['error']}"
    notes = "; ".join(x.get("notes", []) + ([x["departure"]] if x.get("departure") else []) + ([x["step_time"]] if x.get("step_time") else []) + ([a.note] if a.note else []))
    acct = "" if a.account == "main" else f" ({user})"
    row = (f"| {datetime.now():%Y-%m-%d %H:%M} | `{a.slug}` {a.tag}{acct} | {x.get('repo_commit', '—')} | {mode} | "
           f"{outcome[:160]} | {key_s} {ci_s} | {x.get('train_seconds', '—')} | {total if total is not None else '—'} | "
           f"{stages} | {notes} | `{log.relative_to(ROOT)}` |\n")
    if not LEDGER.exists():
        LEDGER.write_text("# Kaggle run ledger\n\nOne row per kernel run, appended by `scripts/kaggle/ledger.py` (never edited by hand). "
                          "Times in seconds; logs under `data/logs/kaggle/` (gitignored). Quota: `kaggle quota`.\n\n"
                          "| logged | kernel · version | repo | mode | outcome | key metrics | train s | total s | stages | notes | log |\n"
                          "|---|---|---|---|---|---|---|---|---|---|---|\n")
    LEDGER.open("a").write(row)
    print(row, end="")


if __name__ == "__main__":
    main()
