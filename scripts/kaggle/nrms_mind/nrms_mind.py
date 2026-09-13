"""P3.1 · NRMS baseline on MIND (recommenders, the brief's "MIND baseline"): train on
MINDsmall_train, score every MINDsmall_dev impression on its full slate, write the scores file,
evaluate (SPEC.md §13, CONTEXT.md C-024).

Push:  .venv/bin/kaggle kernels push -p scripts/kaggle/nrms_mind --accelerator NvidiaTeslaT4
Logs:  .venv/bin/kaggle kernels logs aayushpandey18602/a2-nrms-mind > data/logs/kaggle/<name>.log
Files: .venv/bin/kaggle kernels output aayushpandey18602/a2-nrms-mind -p data/scores/_kaggle/

Modes:
  DEMO_CHECK=True   MINDdemo (from huggingface.co/datasets/Recommenders/MIND, ungated), 1 epoch.
                    Pushed twice: identical group_auc to every digit = determinism oracle.
  DEMO_CHECK=False  the quick-start recipe on MINDsmall: 5 epochs, bs 32, history 50, npratio 4,
                    GloVe-300d from MINDsmall_utils.zip, title 30, 20x20 heads, attn 200, Adam 1e-4,
                    dropout 0.2 (nrms.yaml), seed 42. Data from the private dataset
                    aayushpandey18602/mind-small-official (official zips, hashes in the repo).
Departures from examples/00_quick_start/nrms_MIND.ipynb, each printed:
  * op determinism on (tf.config.experimental.enable_op_determinism) before the v1 session, and
    Python's `random` seeded (the package leaves it unseeded; it draws the negatives);
  * runs under tf-keras (TF_USE_LEGACY_KERAS=1) on TF 2.20, the package vendored --no-deps;
  * after training, run_fast_eval's per-impression predictions are written as the scores file
    (impr_index is the 0-based behaviors.tsv row = imp_row, proven in the U4 smoke), and the
    metrics are recomputed with our src/eval/metrics and required to match cal_metric's rounded
    values.
Every stage prints "STAGE <name> ok <seconds>".
"""
import hashlib, glob, json, os, subprocess, sys, time
import datetime as dt
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
DEMO_CHECK = True
EPOCHS = 1 if DEMO_CHECK else 5
SEED, BS = 42, 32
REC_COMMIT = "0bb4b3690941ffb668118e31ccaf8a7d19f8212a"
REPO_URL, REPO_REF = "https://github.com/Anurag-1205/News_Recommendation_System", "a2-click-logs"
HF = "https://huggingface.co/datasets/Recommenders/MIND/resolve/main"
WORK = Path("/kaggle/working"); DATA = WORK / "mind"; OUT = WORK / "out"
T0 = time.time()


def stage(name): print(f"STAGE {name} ok {time.time() - T0:.0f}s", flush=True)
def sh(cmd): print("+", cmd, flush=True); subprocess.run(cmd, shell=True, check=True)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


print("KNOBS", json.dumps({"DEMO_CHECK": DEMO_CHECK, "EPOCHS": EPOCHS, "SEED": SEED, "BS": BS}))

# ---- 1. environment + code -------------------------------------------------------------------------
sh("pip install -q 'tf-keras~=2.20.0' 'retrying>=1.3.4' polars==1.43.2")
sh(f"git clone -q https://github.com/recommenders-team/recommenders {WORK}/recommenders && cd {WORK}/recommenders && git checkout -q {REC_COMMIT} && git rev-parse HEAD")
sh(f"git clone -q --branch {REPO_REF} {REPO_URL} {WORK}/repo && cd {WORK}/repo && git rev-parse HEAD")
REPO_COMMIT = subprocess.check_output(["git", "-C", str(WORK / "repo"), "rev-parse", "HEAD"], text=True).strip()
sys.path[:0] = [str(WORK / "recommenders"), str(WORK / "repo")]
import numpy as np, polars as pl, tensorflow as tf
tf.config.experimental.enable_op_determinism()
print("tf", tf.__version__, "tf.keras", tf.keras.__name__, "polars", pl.__version__, "GPUs", [g.name for g in tf.config.list_physical_devices("GPU")])
from recommenders.models.newsrec.newsrec_utils import prepare_hparams
from recommenders.models.newsrec.models.nrms import NRMSModel
from recommenders.models.newsrec.io.mind_iterator import MINDIterator
from recommenders.models.deeprec.deeprec_utils import cal_metric
from src.baselines.nrms_data import scores_frame
from src.eval.metrics import per_impression_metrics
from src.rerank.common import evaluate as bootstrap_evaluate, impression_rows
stage("imports")

# ---- 2. data ----------------------------------------------------------------------------------------
if DEMO_CHECK:
    for name, sub in [("MINDdemo_train", "train"), ("MINDdemo_dev", "valid"), ("MINDdemo_utils", "utils")]:
        sh(f"mkdir -p {DATA}/{sub} && wget -q {HF}/{name}.zip -O {DATA}/{name}.zip && unzip -q -o {DATA}/{name}.zip -d {DATA}/{sub}")
        print(f"{name}.zip sha256", sha(DATA / f"{name}.zip"))
else:
    for name, sub in [("MINDsmall_train", "train"), ("MINDsmall_dev", "valid")]:
        hits = sorted(glob.glob(f"/kaggle/input/**/{name}/**/behaviors.tsv", recursive=True))
        assert hits, f"no behaviors.tsv for {name} under /kaggle/input"
        src = Path(hits[0]).parent
        for tsv in ("behaviors.tsv", "news.tsv"):
            print(f"{name}/{tsv} sha256", sha(src / tsv))
        sh(f"mkdir -p {DATA}/{sub} && cp {src}/behaviors.tsv {src}/news.tsv {DATA}/{sub}/")
    sh(f"mkdir -p {DATA}/utils && wget -q {HF}/MINDsmall_utils.zip -O {DATA}/MINDsmall_utils.zip && unzip -q -o {DATA}/MINDsmall_utils.zip -d {DATA}/utils")
    print("MINDsmall_utils.zip sha256", sha(DATA / "MINDsmall_utils.zip"))
f = {k: str(DATA / p) for k, p in {"train_news": "train/news.tsv", "train_beh": "train/behaviors.tsv",
     "valid_news": "valid/news.tsv", "valid_beh": "valid/behaviors.tsv", "emb": "utils/embedding.npy",
     "udict": "utils/uid2index.pkl", "wdict": "utils/word_dict.pkl", "yaml": "utils/nrms.yaml"}.items()}
stage("data")

# ---- 3. train ---------------------------------------------------------------------------------------
hparams = prepare_hparams(f["yaml"], wordEmb_file=f["emb"], wordDict_file=f["wdict"], userDict_file=f["udict"],
                          batch_size=BS, epochs=EPOCHS, show_step=100000)
print("HPARAMS", str(hparams)[:600])
model = NRMSModel(hparams, MINDIterator, seed=SEED)
print("params", model.model.count_params())
# The package seeds TF and numpy (base_model.__init__) but not Python's `random`, which
# `newsrec_utils.newsample` uses to draw the npratio negatives: demo twin runs v1/v2 differed
# (AUC 0.5805 vs 0.5784). Seeding it here makes the sampling sequence reproducible.
import random; random.seed(SEED); np.random.seed(SEED)
t = time.time()
model.fit(f["train_news"], f["train_beh"], f["valid_news"], f["valid_beh"])
train_seconds = time.time() - t
print(f"TRAIN_SECONDS {train_seconds:.0f} (epochs={EPOCHS}, incl. the per-epoch eval)")
WEIGHTS = OUT / "weights"; WEIGHTS.mkdir(parents=True, exist_ok=True)
model.model.save_weights(str(WEIGHTS / "nrms_ckpt"))
stage("train")

# ---- 4. score every dev impression, full slate ---------------------------------------------------------
t = time.time()
impr_idx, labels, preds = model.run_fast_eval(f["valid_news"], f["valid_beh"])
score_seconds = time.time() - t
rows = [line.rstrip("\n").split("\t") for line in open(f["valid_beh"])]
beh = pl.DataFrame({
    "imp_row": pl.Series(range(len(rows)), dtype=pl.UInt32),
    "impression_id": pl.Series([int(r[0]) for r in rows], dtype=pl.Int64),
    "article_ids_inview": pl.Series([[x.split("-")[0] for x in r[4].split()] for r in rows], dtype=pl.List(pl.Utf8)),
    "labels": pl.Series([[int(x.split("-")[1]) for x in r[4].split()] for r in rows], dtype=pl.List(pl.Int8)),
})
assert list(impr_idx) == list(range(beh.height)), "impr_index is not the file row"
assert [list(map(int, l)) for l in labels] == beh["labels"].to_list(), "run_fast_eval labels != file labels"
order = np.argsort(impr_idx)   # already sorted by the assert above; kept for safety
sf = scores_frame(beh.select("imp_row", "impression_id", "article_ids_inview"), [list(map(float, preds[i])) for i in order])
assert sf.height == int(beh["article_ids_inview"].list.len().sum()) and sf["imp_row"].n_unique() == beh.height
SC = OUT / "scores" / "mind" / ("MINDdemo_dev" if DEMO_CHECK else "MINDsmall_dev"); SC.mkdir(parents=True, exist_ok=True)
sf.write_parquet(SC / "nrms.parquet")
manifest = {"dataset": "mind", "split": "MINDdemo_dev" if DEMO_CHECK else "MINDsmall_dev", "system": "nrms", "framing": "in-impression",
            "n_rows": sf.height, "n_impressions": beh.height, "repo_commit": REPO_COMMIT, "recommenders_commit": REC_COMMIT,
            "seeds": {"model": SEED}, "epochs": EPOCHS, "history_size": hparams.his_size, "npratio": hparams.npratio,
            "word_embeddings": "GloVe-300d (MIND utils embedding.npy)", "precision": "float32", "determinism": True,
            "train_seconds": round(train_seconds), "score_seconds": round(score_seconds),
            "command": "kaggle kernels push -p scripts/kaggle/nrms_mind --accelerator NvidiaTeslaT4", "written_at": dt.datetime.utcnow().isoformat()}
(SC / "nrms.json").write_text(json.dumps(manifest, indent=2))
print("SCORES", sf.height, "rows", SC / "nrms.parquet")
stage("score")

# ---- 5. evaluate: ours (the reranker's code) vs the package's cal_metric ----------------------------
lab = (beh.select("imp_row", "labels").explode("labels")
       .with_columns(cand_position=pl.int_range(1, pl.len() + 1).over("imp_row").cast(pl.Int64)))
joined = sf.join(lab, on=["imp_row", "cand_position"], how="left")
assert joined["labels"].null_count() == 0
per_imp = {k: np.asarray(v, float) for k, v in per_impression_metrics(impression_rows(joined, "score", label_col="labels")).items()}
ci = bootstrap_evaluate(per_imp)
ours = {k: float(np.nanmean(v)) for k, v in per_imp.items()}
print("METRICS_OURS", json.dumps(ours))
print("METRICS_CI", json.dumps({k: [c.mean, c.lo, c.hi] for k, c in ci.items()}))
pkg = cal_metric(labels, preds, ["group_auc", "mean_mrr", "ndcg@5;10"])
print("METRICS_PKG", json.dumps(pkg))
for k_ours, k_pkg in [("auc", "group_auc"), ("mrr", "mean_mrr"), ("ndcg@5", "ndcg@5"), ("ndcg@10", "ndcg@10")]:
    d = abs(round(ours[k_ours], 4) - pkg[k_pkg]); print(f"AGREE {k_ours} |Δ|={d:.2e} (cal_metric rounds to 4 dp)")
    assert d <= 1e-4 + 1e-9, (k_ours, ours[k_ours], pkg[k_pkg])
stage("eval")
print(f"RESULT PASS epochs={EPOCHS} group_auc={pkg['group_auc']} full_slate_auc={ours['auc']:.6f} total_seconds={time.time() - T0:.0f}")
