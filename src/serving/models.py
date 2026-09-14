"""The served model: `config.FINAL` fitted exactly as RESULTS.md Q2 measured it, then cached
(SPEC.md §16.3). Reruns load the cache; the first run prints the fit time and the AUC on the
same 100k evaluation sample, which must equal Q2's locked number."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl

MODELS = Path("data/processed/models")


def load_or_fit(dataset: str, state, *, force: bool = False):
    MODELS.mkdir(parents=True, exist_ok=True)
    return {"ebnerd": _ebnerd, "mind": _mind}[dataset](state, force)


def _ebnerd(state, force):
    import lightgbm as lgb
    from scripts.rerank_ebnerd_a2 import build, seeded_sample
    from src.rerank.common import SEED, evaluate, group_sizes, fit_final, matrix, per_impression, predict_scores
    from src.rerank.config import FINAL
    from src.rerank.ebnerd import SMALL, load_articles, load_behaviors
    path = MODELS / "ebnerd_final.txt"
    feats = FINAL["ebnerd"]["features"]
    if path.exists() and not force:
        return lgb.Booster(model_file=str(path))
    t0 = time.perf_counter()
    articles = load_articles()
    train_beh = load_behaviors(SMALL / "train/behaviors.parquet")
    val_beh = load_behaviors(SMALL / "validation/behaviors.parquet")
    tr, _ = build(train_beh, SMALL / "train/history.parquet", seeded_sample(train_beh.height, 100_000, SEED), articles, state.stage1)
    model = fit_final(FINAL["ebnerd"]["objective"], matrix(tr, feats), tr["label"].to_numpy(), group_sizes(tr))
    fit_s = time.perf_counter() - t0
    va, _ = build(val_beh, SMALL / "validation/history.parquet", seeded_sample(val_beh.height, 100_000, SEED + 1), articles, state.stage1)
    va = va.with_columns(pl.Series("final", predict_scores(model, matrix(va, feats))))
    ci = evaluate(per_impression(va, "final"))
    booster = model.booster_ if hasattr(model, "booster_") else model
    booster.save_model(str(path))
    (MODELS / "ebnerd_final.json").write_text(json.dumps({"fit_seconds": round(fit_s), "fit_impressions": 100_000, "eval_impressions": 100_000,
                                                          "auc": ci["auc"].mean, "auc_ci": [ci["auc"].lo, ci["auc"].hi], "features": feats}, indent=2))
    print(f"FINAL ebnerd fitted in {fit_s:.0f}s; eval AUC {ci}  (RESULTS.md Q2 locked: 0.6728)")
    return lgb.Booster(model_file=str(path))


def _mind(state, force):
    import joblib
    from scripts.rerank_mind_a2 import DEV, TRAIN, build
    from src.rerank.common import SEED, evaluate, group_sizes, fit_final, matrix, per_impression, predict_scores
    from src.rerank.config import FINAL
    from src.rerank.mind import first_sightings, load_behaviors, load_categories
    path = MODELS / "mind_final.joblib"
    feats = FINAL["mind"]["features"]
    if path.exists() and not force:
        return joblib.load(path)
    t0 = time.perf_counter()
    train, dev = load_behaviors(TRAIN), load_behaviors(DEV)
    categories = load_categories([TRAIN, DEV])
    sightings = first_sightings([train, dev])
    fit_sample = np.sort(np.random.default_rng(SEED).choice(train.height, size=min(80_000, train.height), replace=False))
    tr = build(train, TRAIN, fit_sample, categories, sightings, state.stage1)
    model = fit_final(FINAL["mind"]["objective"], matrix(tr, feats), tr["label"].to_numpy(), group_sizes(tr))
    fit_s = time.perf_counter() - t0
    va = build(dev, DEV, np.arange(dev.height), categories, sightings, state.stage1)
    va = va.with_columns(pl.Series("final", predict_scores(model, matrix(va, feats))))
    ci = evaluate(per_impression(va, "final"))
    joblib.dump(model, path)
    (MODELS / "mind_final.json").write_text(json.dumps({"fit_seconds": round(fit_s), "fit_impressions": 80_000, "eval_impressions": int(dev.height),
                                                        "auc": ci["auc"].mean, "auc_ci": [ci["auc"].lo, ci["auc"].hi], "features": feats}, indent=2))
    print(f"FINAL mind fitted in {fit_s:.0f}s; eval AUC {ci}  (RESULTS.md Q2 locked: 0.6747)")
    return joblib.load(path)
