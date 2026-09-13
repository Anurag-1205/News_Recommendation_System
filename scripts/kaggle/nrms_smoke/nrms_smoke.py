"""P0 step 7: NRMS from ebnerd-benchmark runs end to end on EB-NeRD demo on a Kaggle T4.

Push:   .venv/bin/kaggle kernels push -p scripts/kaggle/nrms_smoke --accelerator NvidiaTeslaT4
Fetch:  .venv/bin/kaggle kernels output aayushpandey18602/a2-nrms-smoke -p <dir>

What it settles (PLAN.md P0.7, D3): the framework, whether the benchmark's code imports against
Kaggle's image (it pins polars 0.20.8 / numpy <1.26, the image has newer), whether Keras mixed
precision (the TF analogue of fp16 autocast) works with the NRMS layers, and how long one epoch
on demo takes. Mirrors examples/quick_start/nrms_ebnerd.py from the benchmark, with three
deliberate departures, each printed so the log is the record:
  * data pulled from the official S3 bucket inside the kernel (laptop uplink is ~17 KB/s);
  * xlm-roberta-base rather than -large (smoke test, not the reproduction);
  * 1 epoch, demo split;
  * src/baselines/ebrec_compat.py patched in after the imports (run v1 died in the benchmark's
    `map_list_article_id_to_value` on the image's polars 1.35; see that module's docstring).
    Until the shim is on GitHub the kernel carries an inline copy, written by scripts/kaggle/
    nrms_smoke/inline_shim.py from the module (base64), so the two cannot drift silently.
Every stage prints "STAGE <name> ok <seconds>" so a failure is located from the log alone.
"""
import os, subprocess, sys, time, glob
import datetime as dt

T0 = time.time()
BENCH_COMMIT = "5164e2ce7c92b99cbcb853d5f804cc95f0232b2f"   # pinned in CONTEXT.md
S3 = "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com"
WORK = "/kaggle/working"
DATA = f"{WORK}/ebnerd_data"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def stage(name):
    print(f"STAGE {name} ok {time.time() - T0:.0f}s", flush=True)


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


# ---- 1. data + code ---------------------------------------------------------------------------
sh(f"wget -q {S3}/ebnerd_demo.zip -O {WORK}/ebnerd_demo.zip")
sh(f"mkdir -p {DATA}/ebnerd_demo && unzip -q -o {WORK}/ebnerd_demo.zip -d {DATA}/ebnerd_demo")
sh(f"find {DATA} -maxdepth 3 | sort")
sh(f"git clone -q https://github.com/jppol-ai/ebnerd-benchmark {WORK}/ebnerd-benchmark"
   f" && cd {WORK}/ebnerd-benchmark && git checkout -q {BENCH_COMMIT} && git rev-parse HEAD")
sys.path.insert(0, f"{WORK}/ebnerd-benchmark/src")
stage("fetch")

# The zip may or may not carry a top-level folder; locate articles.parquet and go from there.
arts = glob.glob(f"{DATA}/ebnerd_demo/**/articles.parquet", recursive=True)
assert len(arts) == 1, arts
from pathlib import Path
PATH = Path(arts[0]).parent
print("dataset root", PATH)

# ---- 2. framework -------------------------------------------------------------------------------
import numpy as np, polars as pl, tensorflow as tf, transformers
print("tf", tf.__version__, "keras", tf.keras.__version__, "polars", pl.__version__,
      "numpy", np.__version__, "transformers", transformers.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("tf GPUs", [g.name for g in gpus])
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

from ebrec.utils._constants import (DEFAULT_HISTORY_ARTICLE_ID_COL, DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_IMPRESSION_ID_COL, DEFAULT_IMPRESSION_TIMESTAMP_COL,
    DEFAULT_SUBTITLE_COL, DEFAULT_TITLE_COL, DEFAULT_USER_COL)
from ebrec.utils._behaviors import (create_binary_labels_column, sampling_strategy_wu2019,
    truncate_history, ebnerd_from_path)
from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore
from ebrec.utils._articles import convert_text2encoding_with_transformers, create_article_id_to_value_mapping
from ebrec.utils._polars import concat_str_columns
from ebrec.utils._nlp import get_transformers_word_embeddings
from ebrec.models.newsrec.dataloader import NRMSDataLoader, NRMSDataLoaderPretransform
from ebrec.models.newsrec.model_config import hparams_nrms
from ebrec.models.newsrec import NRMSModel
stage("imports")

# ---- 2b. polars compat shim (inline copy of src/baselines/ebrec_compat.py) --------------------
EBREC_COMPAT_B64 = "IiIiQ29tcGF0aWJpbGl0eSBzaGltIHNvIGVibmVyZC1iZW5jaG1hcmsgKHBpbm5lZCB0byBwb2xhcnMgMC4yMC44KSBydW5zIG9uIHBvbGFycyA+PSAxLjAuCgpUaGUgYmVuY2htYXJrJ3MgYG1hcF9saXN0X2FydGljbGVfaWRfdG9fdmFsdWVgIChzcmMvZWJyZWMvdXRpbHMvX2FydGljbGVzX2JlaGF2aW9ycy5weSkgZG9lcwpgcGwuY29sKGNvbCkucmVwbGFjZShtYXBwaW5nLCBkZWZhdWx0PU5vbmUpYCB3aXRoIGEgZGljdCB3aG9zZSB2YWx1ZXMgYXJlICpsaXN0cyogKHRva2VuIGlkcyBwZXIKYXJ0aWNsZSkuIHBvbGFycyAwLjIwIGFjY2VwdGVkIHRoYXQuIHBvbGFycyAxLjM1LCB0aGUgdmVyc2lvbiBvbiBLYWdnbGUncyBHUFUgaW1hZ2UsIHJvdXRlcyBpdCB0bwpgcmVwbGFjZV9zdHJpY3RgIGFuZCByYWlzZXMgIm5vdCB5ZXQgaW1wbGVtZW50ZWQ6IE5lc3RlZCBvYmplY3QgdHlwZXMiIChzbW9rZSBydW4gdjEsIFJFU1VMVFMubWQKUDApOyBwb2xhcnMgMS40Mywgb3VyIHZlbnYsIGFjY2VwdHMgaXQgYWdhaW4uIFRoZSBzaGltIG1ha2VzIHRoZSBiYXNlbGluZSBpbmRlcGVuZGVudCBvZiB3aGljaApzaWRlIG9mIHRoYXQgbGluZSB0aGUgcnVudGltZSBmYWxscyBvbiAoQ09OVEVYVC5tZCBDLTAyMSkuCgpBIHNlY29uZCBmdW5jdGlvbiwgYGFkZF9wcmVkaWN0aW9uX3Njb3Jlc2AgKHNyYy9lYnJlYy91dGlscy9fYmVoYXZpb3JzLnB5KSwgZW5kcyB3aXRoCmAuZHJvcCgiX2dyb3VwYnlfaWQiKWAgb24gYSBmcmFtZSB0aGF0IG5ldmVyIGhhZCB0aGF0IGNvbHVtbjsgcG9sYXJzIDAuMjAgaWdub3JlZCBhIG1pc3NpbmcKY29sdW1uIGluIGBkcm9wYCwgcG9sYXJzIDEueCByYWlzZXMgQ29sdW1uTm90Rm91bmRFcnJvciAoc21va2UgcnVuIHYyKS4KCkJvdGggZnVuY3Rpb25zIGJlbG93IGtlZXAgdGhlIGJlbmNobWFyaydzIHNpZ25hdHVyZSBhbmQgc2VtYW50aWNzIGFuZCBhcmUgdmVyaWZpZWQgYWdhaW5zdCB0aGUKYmVuY2htYXJrJ3Mgb3duIGRvY3N0cmluZyBleGFtcGxlcyBpbiB0ZXN0cy90ZXN0X2VicmVjX2NvbXBhdC5weS4gYGluc3RhbGwoKWAgcGF0Y2hlcyB0aGVtIGludG8KZXZlcnkgbW9kdWxlIHRoYXQgaW1wb3J0ZWQgdGhlIG9yaWdpbmFscyBieSBuYW1lLgoiIiIKZnJvbSBfX2Z1dHVyZV9fIGltcG9ydCBhbm5vdGF0aW9ucwoKZnJvbSB0eXBpbmcgaW1wb3J0IEFueQoKaW1wb3J0IHBvbGFycyBhcyBwbAoKCmRlZiBfdW5pcXVlX25hbWUoZXhpc3Rpbmc6IGxpc3Rbc3RyXSwgYmFzZTogc3RyKSAtPiBzdHI6CiAgICBuYW1lLCBpID0gYmFzZSwgMAogICAgd2hpbGUgbmFtZSBpbiBleGlzdGluZzoKICAgICAgICBpICs9IDEKICAgICAgICBuYW1lID0gZiJ7YmFzZX1fe2l9IgogICAgcmV0dXJuIG5hbWUKCgpkZWYgbWFwX2xpc3RfYXJ0aWNsZV9pZF90b192YWx1ZSgKICAgIGJlaGF2aW9yczogcGwuRGF0YUZyYW1lLAogICAgYmVoYXZpb3JzX2NvbHVtbjogc3RyLAogICAgbWFwcGluZzogZGljdFtBbnksIEFueV0sCiAgICBkcm9wX251bGxzOiBib29sID0gRmFsc2UsCiAgICBmaWxsX251bGxzOiBBbnkgPSBOb25lLAopIC0+IHBsLkRhdGFGcmFtZToKICAgICIiIkRyb3AtaW4gZm9yIGVicmVjLnV0aWxzLl9hcnRpY2xlc19iZWhhdmlvcnMubWFwX2xpc3RfYXJ0aWNsZV9pZF90b192YWx1ZSBvbiBwb2xhcnMgPj0gMS4iIiIKICAgIHJvd19pZCA9IF91bmlxdWVfbmFtZShiZWhhdmlvcnMuY29sdW1ucywgIl9ncm91cGJ5X2lkIikKICAgIHZhbF9jb2wgPSBfdW5pcXVlX25hbWUoYmVoYXZpb3JzLmNvbHVtbnMgKyBbcm93X2lkXSwgIl9tYXBwZWQiKQogICAga2V5cyA9IGxpc3QobWFwcGluZy5rZXlzKCkpCiAgICB2YWx1ZXMgPSBbbGlzdCh2KSBpZiBpc2luc3RhbmNlKHYsIHBsLlNlcmllcykgZWxzZSB2IGZvciB2IGluIG1hcHBpbmcudmFsdWVzKCldCiAgICBtYXBfZGYgPSBwbC5EYXRhRnJhbWUoe2JlaGF2aW9yc19jb2x1bW46IHBsLlNlcmllcyhrZXlzKSwgdmFsX2NvbDogcGwuU2VyaWVzKHZhbHVlcyl9KQoKICAgIHdpdGhfaWQgPSBiZWhhdmlvcnMud2l0aF9yb3dfaW5kZXgocm93X2lkKQogICAgZXhwbG9kZWQgPSAoCiAgICAgICAgd2l0aF9pZC5zZWxlY3Qocm93X2lkLCBiZWhhdmlvcnNfY29sdW1uKQogICAgICAgIC5leHBsb2RlKGJlaGF2aW9yc19jb2x1bW4pCiAgICAgICAgLmpvaW4obWFwX2RmLCBvbj1iZWhhdmlvcnNfY29sdW1uLCBob3c9ImxlZnQiLCBtYWludGFpbl9vcmRlcj0ibGVmdCIpCiAgICAgICAgLmRyb3AoYmVoYXZpb3JzX2NvbHVtbikKICAgICAgICAucmVuYW1lKHt2YWxfY29sOiBiZWhhdmlvcnNfY29sdW1ufSkKICAgICkKICAgIGlmIGRyb3BfbnVsbHM6CiAgICAgICAgZXhwbG9kZWQgPSBleHBsb2RlZC5kcm9wX251bGxzKCkKICAgIGVsaWYgZmlsbF9udWxscyBpcyBub3QgTm9uZToKICAgICAgICBleHBsb2RlZCA9IGV4cGxvZGVkLndpdGhfY29sdW1ucyhwbC5jb2woYmVoYXZpb3JzX2NvbHVtbikuZmlsbF9udWxsKGZpbGxfbnVsbHMpKQogICAgYWdnID0gZXhwbG9kZWQuZ3JvdXBfYnkocm93X2lkLCBtYWludGFpbl9vcmRlcj1UcnVlKS5hZ2coYmVoYXZpb3JzX2NvbHVtbikKICAgIHJldHVybiB3aXRoX2lkLmRyb3AoYmVoYXZpb3JzX2NvbHVtbikuam9pbihhZ2csIG9uPXJvd19pZCwgaG93PSJsZWZ0IikuZHJvcChyb3dfaWQpCgoKZGVmIGFkZF9wcmVkaWN0aW9uX3Njb3JlcygKICAgIGRmOiBwbC5EYXRhRnJhbWUsCiAgICBzY29yZXMsCiAgICBpbnZpZXdfY29sOiBzdHIgPSAiYXJ0aWNsZV9pZHNfaW52aWV3IiwKICAgIHByZWRpY3Rpb25fc2NvcmVzX2NvbDogc3RyID0gInNjb3JlcyIsCikgLT4gcGwuRGF0YUZyYW1lOgogICAgIiIiRHJvcC1pbiBmb3IgZWJyZWMudXRpbHMuX2JlaGF2aW9ycy5hZGRfcHJlZGljdGlvbl9zY29yZXM6IGF0dGFjaCBvbmUgZmxhdCBzY29yZSBsaXN0IHRvIGEKICAgIGZyYW1lIG9mIHNsYXRlcywgcmUtbmVzdGVkIHRvIG1hdGNoIGBpbnZpZXdfY29sYC4gU2FtZSBib2R5IGFzIHRoZSBvcmlnaW5hbCBleGNlcHQgdGhlIGZpbmFsCiAgICBgLmRyb3BgIG9mIGEgY29sdW1uIGBkZmAgbmV2ZXIgaGFkLiIiIgogICAgcm93X2lkID0gX3VuaXF1ZV9uYW1lKGRmLmNvbHVtbnMsICJfZ3JvdXBieV9pZCIpCiAgICBuZXN0ZWQgPSAoCiAgICAgICAgZGYubGF6eSgpCiAgICAgICAgLnNlbGVjdChwbC5jb2woaW52aWV3X2NvbCkpCiAgICAgICAgLndpdGhfcm93X2luZGV4KHJvd19pZCkKICAgICAgICAuZXhwbG9kZShpbnZpZXdfY29sKQogICAgICAgIC53aXRoX2NvbHVtbnMocGwuU2VyaWVzKHByZWRpY3Rpb25fc2NvcmVzX2NvbCwgc2NvcmVzKS5leHBsb2RlKCkpCiAgICAgICAgLmdyb3VwX2J5KHJvd19pZCkKICAgICAgICAuYWdnKGludmlld19jb2wsIHByZWRpY3Rpb25fc2NvcmVzX2NvbCkKICAgICAgICAuc29ydChyb3dfaWQpCiAgICAgICAgLmNvbGxlY3QoKQogICAgKQogICAgcmV0dXJuIGRmLndpdGhfY29sdW1ucyhuZXN0ZWQuZ2V0X2NvbHVtbihwcmVkaWN0aW9uX3Njb3Jlc19jb2wpKQoKCmRlZiBpbnN0YWxsKCkgLT4gTm9uZToKICAgICIiIlJlcGxhY2UgdGhlIGJlbmNobWFyaydzIGZ1bmN0aW9ucyBldmVyeXdoZXJlIHRoZXkgd2VyZSBpbXBvcnRlZCBieSBuYW1lLiIiIgogICAgaW1wb3J0IGltcG9ydGxpYgoKICAgIHBhdGNoZXMgPSB7CiAgICAgICAgbWFwX2xpc3RfYXJ0aWNsZV9pZF90b192YWx1ZTogWwogICAgICAgICAgICAiZWJyZWMudXRpbHMuX2FydGljbGVzX2JlaGF2aW9ycyIsCiAgICAgICAgICAgICJlYnJlYy5tb2RlbHMubmV3c3JlYy5kYXRhbG9hZGVyIiwKICAgICAgICAgICAgImVicmVjLm1vZGVscy5mYXN0Zm9ybWVyLmRhdGFsb2FkZXIiLAogICAgICAgIF0sCiAgICAgICAgYWRkX3ByZWRpY3Rpb25fc2NvcmVzOiBbImVicmVjLnV0aWxzLl9iZWhhdmlvcnMiXSwKICAgIH0KICAgIGZvciBmbiwgbW9kdWxlcyBpbiBwYXRjaGVzLml0ZW1zKCk6CiAgICAgICAgZm9yIG5hbWUgaW4gbW9kdWxlczoKICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgbW9kID0gaW1wb3J0bGliLmltcG9ydF9tb2R1bGUobmFtZSkKICAgICAgICAgICAgZXhjZXB0IEltcG9ydEVycm9yOiAgIyBmYXN0Zm9ybWVyIHB1bGxzIHRvcmNoOyBhYnNlbnQgb24gc29tZSBpbWFnZXMKICAgICAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgICAgIHNldGF0dHIobW9kLCBmbi5fX25hbWVfXywgZm4pCg=="
EBREC_COMPAT_SRC = __import__("base64").b64decode(EBREC_COMPAT_B64).decode()
import types
ebrec_compat = types.ModuleType("ebrec_compat"); exec(EBREC_COMPAT_SRC, ebrec_compat.__dict__)
ebrec_compat.install()
print("ebrec_compat installed; shim sha", __import__("hashlib").sha256(EBREC_COMPAT_SRC.encode()).hexdigest()[:12])
stage("shim")


# ---- 3. data prep (as the reproducibility script: train ∪ validation, last day held out) --------
SEED, HISTORY_SIZE, NPRATIO, MAX_TITLE_LENGTH, BS = 123, 20, 4, 30, 32
COLUMNS = [DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL, DEFAULT_INVIEW_ARTICLES_COL,
           DEFAULT_CLICKED_ARTICLES_COL, DEFAULT_IMPRESSION_ID_COL, DEFAULT_IMPRESSION_TIMESTAMP_COL]
df = (pl.concat([ebnerd_from_path(PATH / "train", history_size=HISTORY_SIZE, padding=0),
                 ebnerd_from_path(PATH / "validation", history_size=HISTORY_SIZE, padding=0)])
      .select(COLUMNS)
      .pipe(sampling_strategy_wu2019, npratio=NPRATIO, shuffle=True, with_replacement=True, seed=SEED)
      .pipe(create_binary_labels_column))
last_dt = df[DEFAULT_IMPRESSION_TIMESTAMP_COL].dt.date().max() - dt.timedelta(days=1)
df_train = df.filter(pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL).dt.date() < last_dt)
df_val = df.filter(pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL).dt.date() >= last_dt)
print("impressions train", df_train.height, "val", df_val.height, "val day >=", last_dt)

df_articles = pl.read_parquet(PATH / "articles.parquet")
TRANSFORMER = "FacebookAI/xlm-roberta-base"
tok = transformers.AutoTokenizer.from_pretrained(TRANSFORMER)
lm = transformers.AutoModel.from_pretrained(TRANSFORMER)
word2vec_embedding = get_transformers_word_embeddings(lm)
print("word embedding matrix", word2vec_embedding.shape)
df_articles, cat_col = concat_str_columns(df_articles, columns=[DEFAULT_SUBTITLE_COL, DEFAULT_TITLE_COL])
df_articles, token_col = convert_text2encoding_with_transformers(df_articles, tok, cat_col, max_length=MAX_TITLE_LENGTH)
article_mapping = create_article_id_to_value_mapping(df=df_articles, value_col=token_col)
del lm
stage("data_prep")

# ---- 4. model: mixed_float16 first, float32 fallback -------------------------------------------
hparams_nrms.history_size = HISTORY_SIZE
hparams_nrms.title_size = MAX_TITLE_LENGTH


def build(policy):
    tf.keras.mixed_precision.set_global_policy(policy)
    m = NRMSModel(hparams=hparams_nrms, word2vec_embedding=word2vec_embedding, seed=42)
    m.model.compile(optimizer=m.model.optimizer, loss=m.model.loss, metrics=["AUC"])
    return m

# one real batch, built outside the try: a data-side failure must stop the run, not be mistaken
# for a precision failure (that is what happened in run v1)
probe = NRMSDataLoaderPretransform(behaviors=df_train.head(BS), article_dict=article_mapping,
    unknown_representation="zeros", history_column=DEFAULT_HISTORY_ARTICLE_ID_COL, eval_mode=False, batch_size=BS)
probe_batch = probe[0]
print("probe batch shapes", [x.shape for x in probe_batch[0]], probe_batch[1].shape)
stage("probe_batch")
policy_used = "mixed_float16"
try:
    model = build("mixed_float16")
    model.model.train_on_batch(*probe_batch)   # a dtype error surfaces here, not mid-epoch
except Exception as e:  # noqa: BLE001 — the point of the smoke test is to see what breaks
    print("MIXED_PRECISION_FAILED:", type(e).__name__, str(e)[:500])
    policy_used = "float32"
    tf.keras.backend.clear_session()
    model = build("float32")
print("precision policy", policy_used)
print("params", model.model.count_params())
stage("build")

# ---- 5. train one epoch ------------------------------------------------------------------------
train_dl = NRMSDataLoaderPretransform(behaviors=df_train, article_dict=article_mapping,
    unknown_representation="zeros", history_column=DEFAULT_HISTORY_ARTICLE_ID_COL, eval_mode=False, batch_size=BS)
val_dl = NRMSDataLoaderPretransform(behaviors=df_val, article_dict=article_mapping,
    unknown_representation="zeros", history_column=DEFAULT_HISTORY_ARTICLE_ID_COL, eval_mode=False, batch_size=BS)
t1 = time.time()
hist = model.model.fit(train_dl, validation_data=val_dl, epochs=1, verbose=2)
print(f"epoch_seconds {time.time() - t1:.0f} history {hist.history}")
stage("train")

# ---- 6. evaluate on the held-out day with the benchmark's own metrics ---------------------------
val_eval_dl = NRMSDataLoader(behaviors=df_val, article_dict=article_mapping,
    unknown_representation="zeros", history_column=DEFAULT_HISTORY_ARTICLE_ID_COL, eval_mode=True, batch_size=BS)
scores = model.scorer.predict(val_eval_dl)
from ebrec.utils._behaviors import add_prediction_scores
df_val = add_prediction_scores(df_val, scores.tolist())
metrics = MetricEvaluator(labels=df_val["labels"].to_list(), predictions=df_val["scores"].to_list(),
                          metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)])
print("METRICS", metrics.evaluate().evaluations)
stage("eval")
print(f"RESULT PASS policy={policy_used} total_seconds={time.time() - T0:.0f}")
