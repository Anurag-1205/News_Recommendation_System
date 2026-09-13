"""P3.2 · NRMS + freshness term on EB-NeRD (SPEC.md §15): the P3.1 baseline kernel with the variant
plugged in. Same recipe, seeds, epochs, data and sampling as the baseline run (ledger a2-nrms-ebnerd
v4); the only change is `score = user·news + g(freshness)`. Writes two scores files: `nrms_fresh`
(ablation row 2) and `nrms_fresh_masked` (row 3: the same weights scored with the term switched off).

Push:  .venv/bin/kaggle kernels push -p scripts/kaggle/nrms_ebnerd_fresh --accelerator NvidiaTeslaT4
Logs:  .venv/bin/python scripts/kaggle/ledger.py a2-nrms-ebnerd-fresh vN
Files: .venv/bin/kaggle kernels output aayushpandey18602/a2-nrms-ebnerd-fresh -p data/scores/_kaggle/ebnerd_fresh --page-size 200 --file-pattern 'out/scores/.*'

Modes (constants below, edited before each push; the log prints them):
  DEMO_CHECK=True   demo split, 1 epoch, xlm-roberta-base. Pushed twice: the two logs must show
                    the same val_auc to every digit (determinism oracle, SPEC §13.4 (1)).
  DEMO_CHECK=False  the published recipe on ebnerd_small (5 epochs, xlm-roberta-large, bs 32,
                    history 20, npratio 4, title 30, 20x20 heads, attn 200, Adam 1e-4, dropout 0.2).

Departures from examples/reproducibility_scripts/ebnerd_nrms.py, each printed:
  * fit on `train` only (validation is the evaluation split, SPEC §13.2); the last training day
    is the internal early-stopping holdout, as in the published script;
  * the word-embedding matrix is passed to NRMSModel (the published script drops it, C-021);
  * op determinism on, seeds fixed (C-022 measured ±0.01 without it);
  * polars pinned to our venv's 1.43.2 so src/ runs on the version it is tested on; the ebrec
    shim (C-021) is installed regardless;
  * xlm-roberta-large is timed on the first 200 steps; if a step exceeds LARGE_MAX_STEP_S the run
    rebuilds with -base and says so (cost is measured, not assumed).
Before training the kernel runs `pytest tests/test_nrms_fresh_model.py` (the additive identity
g ≡ 0 ⇒ baseline, per-candidate locality, loader shapes and mask) and stops if it fails.
Every stage prints "STAGE <name> ok <seconds>".
"""
import gc, glob, json, os, subprocess, sys, time
import datetime as dt
from pathlib import Path

# ---- knobs -----------------------------------------------------------------------------------------
DEMO_CHECK = True
DATASPLIT = "ebnerd_demo" if DEMO_CHECK else "ebnerd_small"
EPOCHS = 1 if DEMO_CHECK else 5
TRANSFORMER = "FacebookAI/xlm-roberta-base" if DEMO_CHECK else "FacebookAI/xlm-roberta-large"
LARGE_MAX_STEP_S = 0.40          # ≈ 2× the measured -base step (0.19 s) on a T4
SEED_SAMPLING, SEED_MODEL = 123, 42
HISTORY_SIZE, NPRATIO, MAX_TITLE_LENGTH, BS = 20, 4, 30, 32
SCORE_CHUNK = 25_000
BENCH_COMMIT = "5164e2ce7c92b99cbcb853d5f804cc95f0232b2f"
REPO_URL, REPO_REF = "https://github.com/Anurag-1205/News_Recommendation_System", "a2-click-logs"
S3 = "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com"
WORK = Path("/kaggle/working"); DATA = WORK / "ebnerd_data"; OUT = WORK / "out"

T0 = time.time()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def stage(name):
    print(f"STAGE {name} ok {time.time() - T0:.0f}s", flush=True)


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


print("KNOBS", json.dumps({k: globals()[k] for k in ["DEMO_CHECK", "DATASPLIT", "EPOCHS", "TRANSFORMER",
      "SEED_SAMPLING", "SEED_MODEL", "HISTORY_SIZE", "NPRATIO", "MAX_TITLE_LENGTH", "BS"]}))

# ---- 1. data + code ---------------------------------------------------------------------------------
sh(f"pip install -q polars==1.43.2")
sh(f"wget -q {S3}/{DATASPLIT}.zip -O {WORK}/{DATASPLIT}.zip && mkdir -p {DATA}/{DATASPLIT} "
   f"&& unzip -q -o {WORK}/{DATASPLIT}.zip -d {DATA}/{DATASPLIT} && rm {WORK}/{DATASPLIT}.zip")
sh(f"git clone -q https://github.com/jppol-ai/ebnerd-benchmark {WORK}/ebnerd-benchmark "
   f"&& cd {WORK}/ebnerd-benchmark && git checkout -q {BENCH_COMMIT} && git rev-parse HEAD")
sh(f"git clone -q --branch {REPO_REF} {REPO_URL} {WORK}/repo && cd {WORK}/repo && git rev-parse HEAD")
REPO_COMMIT = subprocess.check_output(["git", "-C", str(WORK / "repo"), "rev-parse", "HEAD"], text=True).strip()
sys.path[:0] = [str(WORK / "ebnerd-benchmark" / "src"), str(WORK / "repo")]
root = next(Path(p).parent for p in glob.glob(f"{DATA}/{DATASPLIT}/**/articles.parquet", recursive=True))
print("dataset root", root, "repo commit", REPO_COMMIT)
stage("fetch")

import numpy as np, polars as pl, tensorflow as tf, transformers
print("tf", tf.__version__, "keras", tf.keras.__version__, "polars", pl.__version__, "numpy", np.__version__)
tf.config.experimental.enable_op_determinism()
tf.keras.utils.set_random_seed(SEED_MODEL)
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)

from ebrec.utils._constants import (DEFAULT_HISTORY_ARTICLE_ID_COL, DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_IMPRESSION_TIMESTAMP_COL, DEFAULT_SUBTITLE_COL, DEFAULT_TITLE_COL,
    DEFAULT_BODY_COL)
from ebrec.utils._behaviors import create_binary_labels_column, sampling_strategy_wu2019
from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore
from ebrec.utils._articles import convert_text2encoding_with_transformers, create_article_id_to_value_mapping
from ebrec.utils._polars import concat_str_columns
from ebrec.utils._nlp import get_transformers_word_embeddings
from ebrec.models.newsrec.dataloader import NRMSDataLoader, NRMSDataLoaderPretransform
from ebrec.models.newsrec.model_config import hparams_nrms
from ebrec.models.newsrec import NRMSModel
from src.baselines import ebrec_compat; ebrec_compat.install()
from ebrec.utils._behaviors import add_prediction_scores          # after install(): the shim
from src.baselines.nrms_data import ebnerd_behaviors, scores_frame
from src.eval.metrics import per_impression_metrics
from src.rerank.common import evaluate as bootstrap_evaluate, impression_rows
from src.baselines.nrms_fresh_features import FreshStats, as_lists, fit_stats, fresh_from_frames
from src.baselines.nrms_fresh_ebrec import NRMSFreshLoader, NRMSFreshModel
stage("imports")

# ---- 1b. the model-level oracles, on this image, before any training -----------------------------
sh(f"set -o pipefail; cd {WORK}/repo && PYTHONPATH=. python -m pytest tests/test_nrms_fresh_model.py -q -p no:cacheprovider 2>&1 | tail -3")
stage("oracle")

# ---- 2. frames --------------------------------------------------------------------------------------
train_full = ebnerd_behaviors(root / "train", history_size=HISTORY_SIZE)
val_full = ebnerd_behaviors(root / "validation", history_size=HISTORY_SIZE)
print("impressions train", train_full.height, "validation", val_full.height,
      "val slate len mean/max", val_full["article_ids_inview"].list.len().mean(), val_full["article_ids_inview"].list.len().max())

first_known = pl.read_parquet(root / "articles.parquet", columns=["article_id", "published_time"]).select("article_id", ts=pl.col("published_time"))
STATS = fit_stats(fresh_from_frames(train_full.select("imp_row", t=pl.col("impression_time"), article_ids_inview=pl.col("article_ids_inview")), first_known, FreshStats(0.0, 1.0)))
print("FRESH_STATS", json.dumps(STATS.to_dict()), "(fitted on the full train split's candidates)")


def attach_fresh(beh, mask_note=""):
    """fresh_inview aligned with article_ids_inview, computed on THIS frame (after any sampling)."""
    f = fresh_from_frames(beh.select("imp_row", t=pl.col("impression_time"), article_ids_inview=pl.col("article_ids_inview")), first_known, STATS)
    lists = as_lists(beh.select("imp_row", "article_ids_inview"), f)
    print(f"fresh{mask_note}: {f.height} candidates, unknown {100 * f['unknown'].mean():.3f}%, x mean {f.filter(pl.col('unknown') == 0)['x'].mean():.3f} sd {f.filter(pl.col('unknown') == 0)['x'].std():.3f}")
    return beh.join(lists, on="imp_row", how="left", maintain_order="left")


df = (train_full.drop("labels")
      .pipe(sampling_strategy_wu2019, npratio=NPRATIO, shuffle=True, with_replacement=True, seed=SEED_SAMPLING)
      .pipe(create_binary_labels_column))
last_dt = df[DEFAULT_IMPRESSION_TIMESTAMP_COL].dt.date().max()
df_train = attach_fresh(df.filter(pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL).dt.date() < last_dt), " train(sampled)")
df_hold = attach_fresh(df.filter(pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL).dt.date() >= last_dt), " holdout(sampled)")
val_full = attach_fresh(val_full, " validation(full slates)")
print("wu2019 samples: fit", df_train.height, "internal holdout (last train day", last_dt, ")", df_hold.height)
del df, train_full; gc.collect()
stage("frames")

# ---- 3. articles + embeddings -----------------------------------------------------------------------
df_articles = pl.read_parquet(root / "articles.parquet")
tok = transformers.AutoTokenizer.from_pretrained(TRANSFORMER)
lm = transformers.AutoModel.from_pretrained(TRANSFORMER)
word2vec_embedding = get_transformers_word_embeddings(lm); del lm
df_articles, cat_col = concat_str_columns(df_articles, columns=[DEFAULT_TITLE_COL, DEFAULT_SUBTITLE_COL, DEFAULT_BODY_COL])
df_articles, token_col = convert_text2encoding_with_transformers(df_articles, tok, cat_col, max_length=MAX_TITLE_LENGTH)
article_mapping = create_article_id_to_value_mapping(df=df_articles, value_col=token_col)
print("articles", df_articles.height, "embedding matrix", word2vec_embedding.shape)
stage("articles")

# ---- 4. model ---------------------------------------------------------------------------------------
hparams_nrms.history_size, hparams_nrms.title_size = HISTORY_SIZE, MAX_TITLE_LENGTH
hparams_nrms.head_num, hparams_nrms.head_dim, hparams_nrms.attention_hidden_dim = 20, 20, 200
hparams_nrms.optimizer, hparams_nrms.loss, hparams_nrms.dropout, hparams_nrms.learning_rate = "adam", "cross_entropy_loss", 0.2, 1e-4


def build(emb):
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(SEED_MODEL)
    m = NRMSFreshModel(hparams=hparams_nrms, word2vec_embedding=emb, seed=SEED_MODEL)
    m.model.compile(optimizer=m.model.optimizer, loss=m.model.loss, metrics=[tf.keras.metrics.AUC(name="auc")])
    return m


def loader(beh, eval_mode, mask=False):
    return NRMSFreshLoader(behaviors=beh, article_dict=article_mapping, unknown_representation="zeros",
                           history_column=DEFAULT_HISTORY_ARTICLE_ID_COL, eval_mode=eval_mode, batch_size=BS, mask=mask)


model = build(word2vec_embedding)
print("params", model.model.count_params())
train_dl, hold_dl = loader(df_train, False), loader(df_hold, False)

if "large" in TRANSFORMER:
    # cost check on real batches before committing 5 epochs to it
    t = time.time(); n = min(200, len(train_dl))
    for i in range(n):
        model.model.train_on_batch(*train_dl[i])
    step = (time.time() - t) / n
    print(f"STEP_TIME {TRANSFORMER} {step:.3f} s/step over {n} steps (limit {LARGE_MAX_STEP_S})")
    if step > LARGE_MAX_STEP_S:
        TRANSFORMER = "FacebookAI/xlm-roberta-base"
        print("DEPARTURE: xlm-roberta-large too slow on a T4 at this budget; rebuilding with", TRANSFORMER)
        tok = transformers.AutoTokenizer.from_pretrained(TRANSFORMER); lm = transformers.AutoModel.from_pretrained(TRANSFORMER)
        word2vec_embedding = get_transformers_word_embeddings(lm); del lm
        df_articles = pl.read_parquet(root / "articles.parquet")
        df_articles, cat_col = concat_str_columns(df_articles, columns=[DEFAULT_TITLE_COL, DEFAULT_SUBTITLE_COL, DEFAULT_BODY_COL])
        df_articles, token_col = convert_text2encoding_with_transformers(df_articles, tok, cat_col, max_length=MAX_TITLE_LENGTH)
        article_mapping = create_article_id_to_value_mapping(df=df_articles, value_col=token_col)
        train_dl, hold_dl = loader(df_train, False), loader(df_hold, False)
    model = build(word2vec_embedding)      # fresh weights either way: the timing steps must not count as training
stage("build")

# ---- 5. train ---------------------------------------------------------------------------------------
WEIGHTS = OUT / "weights" / "nrms_fresh.weights.h5"; WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(filepath=str(WEIGHTS), monitor="val_auc", mode="max", save_best_only=True, save_weights_only=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.2, patience=2, min_lr=1e-6),
]
t = time.time()
hist = model.model.fit(train_dl, validation_data=hold_dl, epochs=EPOCHS, callbacks=callbacks, verbose=2)
train_seconds = time.time() - t
print("HISTORY", json.dumps({k: [float(x) for x in v] for k, v in hist.history.items()}))
print(f"TRAIN_SECONDS {train_seconds:.0f}  steps/epoch {len(train_dl)}  s/step {train_seconds / EPOCHS / len(train_dl):.3f}")
model.model.load_weights(str(WEIGHTS))
del train_dl, hold_dl, df_train, df_hold; gc.collect()
stage("train")

# ---- 6+7. score every validation impression (full slate) and evaluate: twice ----------------------
SYSTEMS = [("nrms_fresh", False), ("nrms_fresh_masked", True)]
labels = (val_full.select("imp_row", "labels").explode("labels")
          .with_columns(cand_position=pl.int_range(1, pl.len() + 1).over("imp_row").cast(pl.Int64)))
results = {}
for system, mask in SYSTEMS:
    CH = OUT / "chunks" / system; CH.mkdir(parents=True, exist_ok=True)
    t = time.time()
    for start in range(0, val_full.height, SCORE_CHUNK):
        f = CH / f"val_{start:07d}.parquet"
        if f.exists():
            print("chunk present, skipping", f.name); continue
        piece = val_full.slice(start, SCORE_CHUNK)
        scores = model.scorer.predict(loader(piece, True, mask=mask), verbose=0)
        piece = add_prediction_scores(piece, scores.tolist())
        piece.select("imp_row", "scores").write_parquet(f)
        print(f"[{system}] chunk {f.name} {piece.height} impressions {time.time() - t:.0f}s", flush=True)
        tf.keras.backend.clear_session(); gc.collect()
    score_seconds = time.time() - t
    nested = pl.concat([pl.read_parquet(f) for f in sorted(CH.glob("val_*.parquet"))]).sort("imp_row")
    assert nested["imp_row"].to_list() == val_full["imp_row"].to_list()
    sf = scores_frame(val_full.select("imp_row", "impression_id", "article_ids_inview"), nested["scores"].to_list())
    assert sf.height == int(val_full["article_ids_inview"].list.len().sum()) and sf["imp_row"].n_unique() == val_full.height
    SC = OUT / "scores" / "ebnerd" / "validation"; SC.mkdir(parents=True, exist_ok=True)
    sf.write_parquet(SC / f"{system}.parquet")
    manifest = {"dataset": "ebnerd", "split": f"{DATASPLIT}/validation", "system": system, "framing": "in-impression",
                "variant": "nrms + freshness term (SPEC §15)", "fresh_masked_at_inference": mask, "fresh_stats": STATS.to_dict(),
                "n_rows": sf.height, "n_impressions": val_full.height, "repo_commit": REPO_COMMIT, "benchmark_commit": BENCH_COMMIT,
                "seeds": {"sampling": SEED_SAMPLING, "model": SEED_MODEL}, "transformer": TRANSFORMER, "epochs": EPOCHS,
                "history_size": HISTORY_SIZE, "npratio": NPRATIO, "precision": "float32", "determinism": True,
                "train_seconds": round(train_seconds), "score_seconds": round(score_seconds),
                "command": "kaggle kernels push -p scripts/kaggle/nrms_ebnerd_fresh --accelerator NvidiaTeslaT4", "written_at": dt.datetime.utcnow().isoformat()}
    (SC / f"{system}.json").write_text(json.dumps(manifest, indent=2))
    print(f"SCORES[{system}]", sf.height, "rows", SC / f"{system}.parquet")
    joined = sf.join(labels, on=["imp_row", "cand_position"], how="left")
    assert joined["labels"].null_count() == 0
    per_imp = {k: np.asarray(v, float) for k, v in per_impression_metrics(impression_rows(joined, "score", label_col="labels")).items()}
    ci = bootstrap_evaluate(per_imp)
    ours = {k: float(np.nanmean(v)) for k, v in per_imp.items()}
    print(f"METRICS_OURS[{system}]", json.dumps(ours))
    print(f"METRICS_CI[{system}]", json.dumps({k: [c.mean, c.lo, c.hi] for k, c in ci.items()}))
    if not mask:
        bench = MetricEvaluator(labels=val_full["labels"].to_list(), predictions=nested["scores"].to_list(),
                                metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)]).evaluate().evaluations
        print("METRICS_BENCH", json.dumps(bench))
        for k in ("auc", "mrr", "ndcg@5", "ndcg@10"):
            d = abs(ours[k] - bench[k]); print(f"AGREE {k} |Δ|={d:.2e}"); assert d < 1e-6
        print("METRICS_OURS", json.dumps(ours)); print("METRICS_CI", json.dumps({k: [c.mean, c.lo, c.hi] for k, c in ci.items()}))
    results[system] = ours
    stage(f"score_eval[{system}]")
sh(f"rm -rf {OUT / 'chunks'}")
sh(f"rm -rf {CH} {DATA} {WORK}/ebnerd-benchmark {WORK}/repo")   # chunks were the resume mechanism; the
# concatenated file is the deliverable. Data and code copies are deleted so `kaggle kernels output`
# (paginated at 20 files) returns out/ first instead of 130 MB of inputs.
print(f"RESULT PASS transformer={TRANSFORMER} epochs={EPOCHS} val_auc_holdout={max(hist.history['val_auc']):.6f} "
      f"full_slate_auc={results['nrms_fresh']['auc']:.6f} masked_auc={results['nrms_fresh_masked']['auc']:.6f} total_seconds={time.time() - T0:.0f}")
