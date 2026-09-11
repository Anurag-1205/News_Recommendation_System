"""Shared pieces of the two-stage reranker (A2 Q2, SPEC.md §12).

* `model_features` — the one place that decides which columns a model may train on. It always
  drops `UNSAFE_FEATURES`, and for any model that will score the Codabench test file it also drops
  `ABSENT_FROM_TEST_FILE`. Scripts call this rather than listing exclusions themselves.
* `evaluate` / `per_impression` — a scored candidate frame -> A1's per-impression metrics and
  bootstrap CIs, unchanged, so A2 numbers are comparable with A1's.
* `paired_delta` — a minimal paired bootstrap on per-impression differences. **Provisional**: the
  team's paired-bootstrap harness is Aayush's P3.4a and supersedes this (CONTEXT.md C-015).
* `fit_gbdt` — A1's HistGradientBoostingClassifier with A1's hyperparameters, for comparability.
* `fit_lambdarank` — the D2 model (CONTEXT.md C-016): LightGBM `lambdarank`, one query per
  impression, so the loss is about the *order of candidates within an impression* rather than
  about each row's click probability in a pool of rows from all impressions.
* `drop_reasons` / `leave_one_family_out` — the conditional ablation, with its trigger fixed in code.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from src.eval.bootstrap import CI, bootstrap_metrics
from src.eval.metrics import per_impression_metrics
from src.features.behavioural import (ABSENT_FROM_TEST_FILE, UNSAFE_FEATURES, category_match_batch,
                                      recency_profile_batch)

SEED = 0
GBDT_PARAMS = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, random_state=SEED)
# Same capacity as GBDT_PARAMS (300 trees, lr 0.08, 31 leaves), so pointwise vs listwise compares
# objectives, not model size. deterministic + force_row_wise + fixed threads -> identical refits.
LAMBDARANK_PARAMS = dict(objective="lambdarank", n_estimators=300, learning_rate=0.08, num_leaves=31,
                         random_state=SEED, deterministic=True, force_row_wise=True, n_jobs=4, verbose=-1)


def model_features(candidates: list[str], *, for_submission: bool = True) -> list[str]:
    """The subset of `candidates` a model may use, in the given order.

    Unsafe features are never allowed: they cannot exist at request time (Q9). With
    `for_submission=True` — the default, because every model we ship scores the test file — the
    features the Codabench test file cannot supply are dropped too.
    """
    banned = set(UNSAFE_FEATURES) | (ABSENT_FROM_TEST_FILE if for_submission else set())
    return [c for c in candidates if c not in banned]


def impression_rows(frame: pl.DataFrame, score_col: str, *, key: str = "imp_row",
                    label_col: str = "label") -> list[tuple[list, list]]:
    """(labels, scores) per impression, impressions in first-appearance order, candidates in row
    order. A NaN or null score becomes 0.0: a missing feature carries no ranking information, and
    NaN would otherwise sort unpredictably inside the metric."""
    g = (frame.select(key, label_col, pl.col(score_col).cast(pl.Float64).fill_nan(0.0).fill_null(0.0).alias("_s"))
         .group_by(key, maintain_order=True).agg(pl.col(label_col), pl.col("_s")))
    return list(zip(g[label_col].to_list(), g["_s"].to_list()))


def per_impression(frame: pl.DataFrame, score_col: str) -> dict[str, np.ndarray]:
    """A1's per-impression AUC / MRR / nDCG@5 / nDCG@10 for one score column."""
    return {k: np.asarray(v, float) for k, v in per_impression_metrics(impression_rows(frame, score_col)).items()}


def evaluate(per_imp: dict[str, np.ndarray], *, iterations: int = 1000, seed: int = SEED) -> dict[str, CI]:
    """Bootstrap 95% CI per metric, resampling impressions (A1's `bootstrap_metrics`)."""
    return bootstrap_metrics({k: v.tolist() for k, v in per_imp.items()}, iterations=iterations, seed=seed)


def paired_delta(a, b, *, iterations: int = 1000, confidence: float = 0.95, seed: int = SEED,
                 block: int = 100) -> CI:
    """Paired bootstrap of mean(b - a) over impressions.

    Both systems are scored on the same impressions, so resampling the per-impression
    *differences* cancels the impression-to-impression variance they share. That makes it far
    tighter than comparing two independent CIs. Impressions where either value is NaN (AUC with a
    single class) are dropped. Resampled in blocks to bound memory.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError(f"paired inputs differ in length: {a.shape} vs {b.shape}")
    d = (b - a)[~(np.isnan(a) | np.isnan(b))]
    n = len(d)
    if n == 0:
        return CI(float("nan"), float("nan"), float("nan"), 0, iterations)
    rng = np.random.default_rng(seed)
    means = np.concatenate([d[rng.integers(0, n, size=(min(block, iterations - s), n))].mean(axis=1)
                            for s in range(0, iterations, block)])
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return CI(float(d.mean()), float(lo), float(hi), n, iterations)


def by_impression_chunks(frame: pl.DataFrame, chunk: int):
    """Split a candidate frame into pieces of at most `chunk` impressions, order kept."""
    rows = frame["imp_row"].unique(maintain_order=True)
    for start in range(0, len(rows), chunk):
        yield frame.filter(pl.col("imp_row").is_in(rows[start:start + chunk].implode()))


def category_profile_features(long: pl.DataFrame, log: pl.DataFrame, half_life: timedelta, *,
                              untimed_ts: datetime | None = None, chunk: int = 10_000) -> pl.DataFrame:
    """`recency_weighted_profile` and `category_match` for every candidate row of `long`
    (`imp_row`, `cand_position`, `user_id`, `t`, `candidate_category`), from a click `log`
    (`user_id`, `category`, `ts`). Chunked by impressions to cap the pair table (SPEC.md §11.2).
    Clicks on articles with no known category are dropped: they cannot be placed in a profile."""
    log = log.select("user_id", "category", "ts").drop_nulls("category")
    parts = []
    for piece in by_impression_chunks(long, chunk):
        req = piece.select("imp_row", "cand_position", "user_id", "t", "candidate_category")
        rp = recency_profile_batch(log, req, half_life, untimed_ts=untimed_ts)
        cm = category_match_batch(log, req, half_life, untimed_ts=untimed_ts)
        parts.append(rp.select("imp_row", "cand_position", "recency_weighted_profile")
                     .with_columns(cm["category_match"]))
    return pl.concat(parts)


def matrix(frame: pl.DataFrame, cols: list[str]) -> np.ndarray:
    """Feature columns as a float32 matrix; nulls become NaN, which both models handle natively."""
    return frame.select(pl.col(cols).cast(pl.Float64)).to_numpy().astype(np.float32)


def fit_gbdt(X: np.ndarray, y: np.ndarray):
    """A1's reranker model: pointwise HistGradientBoostingClassifier, NaN handled natively."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(**GBDT_PARAMS).fit(X, y)


def group_sizes(frame: pl.DataFrame, key: str = "imp_row") -> np.ndarray:
    """Rows per impression, in frame order — LightGBM's `group` argument. LightGBM reads groups as
    consecutive row counts, so every impression's rows must be contiguous; an impression split
    into two runs would silently become two queries, and is rejected instead."""
    runs = frame.select(pl.col(key).rle()).unnest(key)
    if runs["value"].n_unique() != runs.height:
        raise ValueError(f"rows of some {key} are not contiguous; sort the frame by {key} first")
    return runs["len"].to_numpy()


def fit_lambdarank(X: np.ndarray, y: np.ndarray, groups: np.ndarray, **overrides):
    """The D2 reranker: LightGBM lambdarank, one query per impression (`groups` from `group_sizes`).
    `overrides` replace entries of LAMBDARANK_PARAMS or add LightGBM parameters, e.g.
    `lambdarank_truncation_level` (LightGBM's documented default is 30).

    The loss is built from pairs of candidates *inside* one impression, weighted by how much
    swapping them changes nDCG. A feature that is constant within an impression (slate size,
    history length) cannot order such a pair, so it can only help through interactions — unlike
    the pointwise loss, which rewards any feature that separates impressions from each other.
    """
    from lightgbm import LGBMRanker
    if int(np.sum(groups)) != len(y):
        raise ValueError(f"group sizes cover {int(np.sum(groups))} rows, the data has {len(y)}")
    return LGBMRanker(**{**LAMBDARANK_PARAMS, **overrides}).fit(X, y, group=groups)


def fit_final(objective: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """Fit the model a `src/rerank/config.FINAL` entry names: 'lambdarank' or 'pointwise'."""
    if objective == "lambdarank":
        return fit_lambdarank(X, y, groups)
    if objective == "pointwise":
        return fit_gbdt(X, y)
    raise ValueError(f"unknown objective {objective!r}")


def predict_scores(model, X: np.ndarray) -> np.ndarray:
    """Ranking scores from either model: the click probability for the pointwise classifier, the
    raw ranker score for lambdarank. Only the order within an impression matters to the metrics."""
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)


def drop_reasons(deltas: dict[str, CI]) -> list[str]:
    """Why a paired (A2 - base) comparison counts as a performance drop; empty if it does not.
    The rule is fixed here, before the numbers are seen: the AUC point estimate is below zero,
    or any metric's whole 95% CI is below zero."""
    reasons = []
    if "auc" in deltas and deltas["auc"].mean < 0:
        reasons.append("auc point estimate < 0")
    reasons += [f"{m} CI entirely below 0" for m, d in deltas.items() if d.hi < 0]
    return reasons


METRICS = ("auc", "mrr", "ndcg@5", "ndcg@10")


def _ci_dict(c: CI) -> dict:
    return {"mean": c.mean, "lo": c.lo, "hi": c.hi, "n": c.n}


def report(va: pl.DataFrame, rows: dict[str, str], pairs: list[tuple[str, str]], log, results: dict
           ) -> dict[str, dict[str, np.ndarray]]:
    """Per-impression metrics + bootstrap CI for each score column in `rows`, then a paired
    bootstrap for each (a, b) in `pairs`, reported as b - a. Logs a table and fills `results`."""
    per_imp = {}
    for col, label in rows.items():
        per_imp[col] = per_impression(va, col)
        ci = evaluate(per_imp[col])
        results[col] = {m: _ci_dict(c) for m, c in ci.items()}
        log(f"  {label:36s} AUC {ci['auc']}  MRR {ci['mrr']}  nDCG@5 {ci['ndcg@5']}  nDCG@10 {ci['ndcg@10']}")
    log("paired differences, b - a, same impressions:")
    for a, b in pairs:
        for m in METRICS:
            d = paired_delta(per_imp[a][m], per_imp[b][m])
            results[f"delta_{b}_minus_{a}_{m}"] = _ci_dict(d)
            log(f"  {b:12s} - {a:12s} {m:8s} {d.mean:+.4f} [{d.lo:+.4f}, {d.hi:+.4f}]"
                f"  {'excludes 0' if d.lo > 0 or d.hi < 0 else 'includes 0'}")
    return per_imp


def conditional_ablation(tr: pl.DataFrame, va: pl.DataFrame, features: list[str],
                         families: dict[str, list[str]], per_imp: dict, base: str, full: str, log,
                         results: dict) -> None:
    """If `full` (base + Phase 1) drops against `base` by `drop_reasons`, refit without each
    Phase 1 family and report, per family, (without - with) and (without - base). A positive
    (without - with) whose CI excludes 0 marks that family as costing performance."""
    trigger = {m: paired_delta(per_imp[base][m], per_imp[full][m]) for m in METRICS}
    reasons = drop_reasons(trigger)
    results["ablation_trigger"] = reasons
    if not reasons:
        log(f"no drop for {full} vs {base} under the fixed rule: leave-one-family-out not run")
        return
    log(f"drop for {full} vs {base} ({'; '.join(reasons)}): leave-one-family-out")
    for family, pim in leave_one_family_out(tr, va, features, families).items():
        entry = results.setdefault("ablation", {}).setdefault(family, {"removed": families[family]})
        for m in METRICS:
            vs_full = paired_delta(per_imp[full][m], pim[m])
            vs_base = paired_delta(per_imp[base][m], pim[m])
            entry[m] = {"without_minus_with": _ci_dict(vs_full), "without_minus_base": _ci_dict(vs_base)}
            log(f"  without {family:16s} {m:8s} vs full {vs_full.mean:+.4f} [{vs_full.lo:+.4f}, {vs_full.hi:+.4f}]"
                f"   vs base {vs_base.mean:+.4f} [{vs_base.lo:+.4f}, {vs_base.hi:+.4f}]")


def leave_one_family_out(tr: pl.DataFrame, va: pl.DataFrame, features: list[str],
                         families: dict[str, list[str]]) -> dict[str, dict[str, np.ndarray]]:
    """Refit lambdarank once per family with that family's columns removed; return each refit's
    per-impression metrics on `va`. Both frames must be sorted by (imp_row, cand_position)."""
    y, groups, out = tr["label"].to_numpy(), group_sizes(tr), {}
    for family, cols in families.items():
        kept = [f for f in features if f not in cols]
        model = fit_lambdarank(matrix(tr, kept), y, groups)
        out[family] = per_impression(va.with_columns(pl.Series("_s", model.predict(matrix(va, kept)))), "_s")
    return out
