"""The paired bootstrap harness: two scores files in, Δ per metric with a 95% CI out (SPEC.md §14).

This is the judge for every Q3 claim. "Beats" is written only when the paired interval excludes
zero, and never inferred from point estimates (A1's `compare` rule, carried over).

Inputs are scores files in the P0.8 contract (SPEC §13.3): one row per (impression, candidate),
keyed by `imp_row` and 1-based `cand_position`, with a sibling `.json` manifest naming the
dataset, split, system and framing. Labels are never in the files; they are joined from the
split, so a file cannot carry its own answers.

Pairing rule: the two systems are compared on the impressions they **both** scored. A
sample-vs-full comparison (the reranker's seeded 100k sample vs NRMS on every impression) is
allowed; a comparison whose common set is under half of either side is refused, because that is
a wrong-split, not a sample.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from src.eval.bootstrap import CI, bootstrap_ci, paired_delta
from src.eval.metrics import per_impression_metrics
from src.rerank.common import impression_rows

METRICS = ("auc", "mrr", "ndcg@5", "ndcg@10")
COLUMNS = ["imp_row", "impression_id", "article_id", "cand_position", "score"]
MANIFEST_KEYS = ("dataset", "split", "system", "framing")
MIN_OVERLAP = 0.5


# ---- files -----------------------------------------------------------------------------------------

def manifest_path(path: Path) -> Path:
    return Path(path).with_suffix(".json")


def write_scores(frame: pl.DataFrame, manifest: dict, path: Path) -> Path:
    """Write a scores file and its manifest. Validates the same contract `read_scores` checks."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    _validate(frame, manifest, path)
    frame.write_parquet(path)
    manifest_path(path).write_text(json.dumps(manifest, indent=2))
    return path


def read_scores(path: Path) -> tuple[pl.DataFrame, dict]:
    path = Path(path)
    if not manifest_path(path).exists():
        raise ValueError(f"{path}: no manifest {manifest_path(path).name}")
    frame = pl.read_parquet(path)
    manifest = json.loads(manifest_path(path).read_text())
    _validate(frame, manifest, path)
    return frame, manifest


def _validate(frame: pl.DataFrame, manifest: dict, path: Path) -> None:
    if frame.columns != COLUMNS:
        raise ValueError(f"{path}: columns {frame.columns} != {COLUMNS}")
    if frame.schema["imp_row"] != pl.UInt32 or frame.schema["cand_position"] != pl.Int64 or frame.schema["score"] != pl.Float64:
        raise ValueError(f"{path}: dtypes {dict(frame.schema)} violate SPEC §13.3")
    if frame.null_count().sum_horizontal().item() != 0:
        raise ValueError(f"{path}: null values")
    keys = frame.select("imp_row", "cand_position")
    if not keys.equals(keys.sort("imp_row", "cand_position")):
        raise ValueError(f"{path}: not sorted by (imp_row, cand_position)")
    if keys.n_unique() != frame.height:
        raise ValueError(f"{path}: duplicate (imp_row, cand_position)")
    missing = [k for k in MANIFEST_KEYS if k not in manifest]
    if missing:
        raise ValueError(f"{path}: manifest lacks {missing}")


def check_compatible(a: dict, b: dict) -> None:
    """Refuse to pair files that are not on the same dataset, split and framing (PLAN.md §2)."""
    for k in ("dataset", "split", "framing"):
        if a.get(k) != b.get(k):
            raise ValueError(f"incompatible {k}: {a.get(k)!r} vs {b.get(k)!r}")


# ---- labels ----------------------------------------------------------------------------------------

def split_labels(dataset: str, split: str) -> pl.DataFrame:
    """(imp_row, cand_position, label) for a split, from the split file itself.

    `split` is the manifest's value: EB-NeRD "ebnerd_small/validation" (a directory under
    data/interim/ebnerd), MIND "MINDsmall_dev" (a name for src.rerank.mind.load_behaviors)."""
    if dataset == "ebnerd":
        from src.baselines.nrms_data import ebnerd_behaviors
        beh = ebnerd_behaviors(Path("data/interim/ebnerd") / split, history_size=1)
        lab = beh.select("imp_row", "labels")
    elif dataset == "mind":
        from src.rerank.mind import load_behaviors
        lab = load_behaviors(split).select("imp_row", "labels")
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    return (lab.explode("labels", empty_as_null=False)
            .with_columns(cand_position=pl.int_range(1, pl.len() + 1).over("imp_row").cast(pl.Int64),
                          label=pl.col("labels").cast(pl.Int8))
            .select("imp_row", "cand_position", "label"))


# ---- the comparison ------------------------------------------------------------------------------------

@dataclass
class Report:
    a: str
    b: str
    manifest_a: dict
    manifest_b: dict
    n_a: int
    n_b: int
    n_common: int
    iterations: int
    seed: int
    system_a: dict[str, CI] = field(default_factory=dict)   # each system's own mean, unpaired CI
    system_b: dict[str, CI] = field(default_factory=dict)
    delta: dict[str, CI] = field(default_factory=dict)      # paired b − a

    def verdict(self, metric: str) -> str:
        d = self.delta[metric]
        if d.lo > 0:
            return f"{self.b} beats {self.a}"
        if d.hi < 0:
            return f"{self.a} beats {self.b}"
        return "no significant difference"

    def markdown(self) -> str:
        head = (f"Paired bootstrap, {self.manifest_a['dataset']} {self.manifest_a['split']}, framing "
                f"{self.manifest_a['framing']}: **{self.b}** − **{self.a}** on {self.n_common:,} common impressions "
                f"(A scored {self.n_a:,}, B {self.n_b:,}); {self.iterations} resamples, seed {self.seed}.\n\n"
                f"| metric | {self.a} | {self.b} | Δ (B − A), paired 95% CI | verdict |\n|---|---|---|---|---|\n")
        rows = "".join(f"| {m} | {self.system_a[m]} | {self.system_b[m]} | {self.delta[m].mean:+.4f} "
                       f"[{self.delta[m].lo:+.4f}, {self.delta[m].hi:+.4f}] | {self.verdict(m)} |\n" for m in METRICS)
        return head + rows

    def to_json(self) -> dict:
        ci = lambda c: {"mean": c.mean, "lo": c.lo, "hi": c.hi, "n": c.n, "iterations": c.iterations}
        return {"a": self.a, "b": self.b, "manifest_a": self.manifest_a, "manifest_b": self.manifest_b,
                "n_a": self.n_a, "n_b": self.n_b, "n_common": self.n_common, "iterations": self.iterations,
                "seed": self.seed, "system_a": {m: ci(c) for m, c in self.system_a.items()},
                "system_b": {m: ci(c) for m, c in self.system_b.items()},
                "delta_b_minus_a": {m: ci(c) for m, c in self.delta.items()},
                "verdict": {m: self.verdict(m) for m in METRICS}}


def _per_impression(frame: pl.DataFrame, labels: pl.DataFrame) -> tuple[pl.Series, dict[str, np.ndarray]]:
    joined = frame.join(labels, on=["imp_row", "cand_position"], how="left")
    if joined["label"].null_count():
        raise ValueError(f"{joined['label'].null_count()} score rows have no label in the split")
    rows = impression_rows(joined, "score")
    imp = joined["imp_row"].unique(maintain_order=True)
    return imp, {k: np.asarray(v, float) for k, v in per_impression_metrics(rows).items()}


def paired_compare(path_a: Path, path_b: Path, *, labels: pl.DataFrame | None = None,
                   iterations: int = 1000, seed: int = 0) -> Report:
    """Compare system B against system A on the impressions both scored."""
    fa, ma = read_scores(path_a)
    fb, mb = read_scores(path_b)
    check_compatible(ma, mb)
    if labels is None:
        labels = split_labels(ma["dataset"], ma["split"])
    common = fa.select("imp_row").unique().join(fb.select("imp_row").unique(), on="imp_row", how="inner")["imp_row"]
    n_a, n_b, n_c = fa["imp_row"].n_unique(), fb["imp_row"].n_unique(), common.len()
    if n_c < MIN_OVERLAP * min(n_a, n_b):
        raise ValueError(f"overlap too small: {n_c} common of {n_a} and {n_b} impressions — wrong split?")
    fa = fa.filter(pl.col("imp_row").is_in(common.implode())).sort("imp_row", "cand_position")
    fb = fb.filter(pl.col("imp_row").is_in(common.implode())).sort("imp_row", "cand_position")
    if not fa.select("imp_row", "cand_position").equals(fb.select("imp_row", "cand_position")):
        raise ValueError("the two files disagree on the candidates of a common impression")
    imp_a, pa = _per_impression(fa, labels)
    imp_b, pb = _per_impression(fb, labels)
    assert imp_a.equals(imp_b)
    rep = Report(ma["system"], mb["system"], ma, mb, n_a, n_b, n_c, iterations, seed)
    for m in METRICS:
        rep.system_a[m] = bootstrap_ci(pa[m], iterations=iterations, seed=seed)
        rep.system_b[m] = bootstrap_ci(pb[m], iterations=iterations, seed=seed)
        rep.delta[m] = paired_delta(pa[m], pb[m], iterations=iterations, seed=seed)
    return rep
