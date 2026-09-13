"""P3.2 · NRMS + freshness term on MIND (SPEC.md §15): the P3.1 baseline kernel with the variant
plugged in — same recipe, seeds, epochs and data as ledger a2-nrms-mind v5; the only change is
`score = user·news + g(freshness)`. Writes `nrms_fresh` (ablation row 2) and `nrms_fresh_masked`
(row 3: same weights, term switched off at inference).

Push:  ./scripts/kaggle/alt_account.sh push scripts/kaggle/nrms_mind_fresh --accelerator NvidiaTeslaT4   (C-027: alt account)
Logs:  .venv/bin/python scripts/kaggle/ledger.py a2-nrms-mind-fresh vN --account alt
Files: KAGGLE_CONFIG_DIR=~/.kaggle/alt .venv/bin/kaggle kernels output aayushpandey602/a2-nrms-mind-fresh -p data/scores/_kaggle/mind_fresh --page-size 200 --file-pattern 'out/scores/.*'

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
Before training the kernel runs `pytest tests/test_nrms_fresh_model_mind.py` (additive identity,
per-candidate locality, iterator batch + mask) and stops if it fails.
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
from src.baselines.nrms_fresh_features import FreshStats, fit_stats, fresh_inputs
from src.baselines.nrms_fresh_rec import MINDFreshIterator, NRMSFreshModel
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
# the repo's loaders read data/interim/mind/<split>/<split>/*.tsv relative to the repo root
if not DEMO_CHECK:
    for name, sub in [("MINDsmall_train", "train"), ("MINDsmall_dev", "valid")]:
        sh(f"mkdir -p {WORK}/repo/data/interim/mind/{name}/{name} && cp {DATA}/{sub}/behaviors.tsv {DATA}/{sub}/news.tsv {WORK}/repo/data/interim/mind/{name}/{name}/")
os.environ.update(NRMS_YAML=f["yaml"], NRMS_EMB=f["emb"], NRMS_WDICT=f["wdict"], NRMS_UDICT=f["udict"])
sh(f"set -o pipefail; cd {WORK}/repo && PYTHONPATH=. TF_USE_LEGACY_KERAS=1 python -m pytest tests/test_nrms_fresh_model_mind.py -q -p no:cacheprovider 2>&1 | tail -3")
stage("data+oracle")

# ---- 2b. freshness table: the reranker's first-seen feature, standardised on the train split ----------
def fresh_by_row(feats, beh_frame):
    """list indexed by behaviors row → {news_id: (x, unknown)}, from an (imp_row, cand_position, x, unknown) frame."""
    ids = beh_frame.select("imp_row", "candidates").explode("candidates").with_columns(
        cand_position=pl.int_range(1, pl.len() + 1).over("imp_row").cast(pl.Int64))
    j = feats.join(ids, on=["imp_row", "cand_position"], how="left")
    out = [dict() for _ in range(beh_frame.height)]
    for r, nid, x, u in zip(j["imp_row"].to_list(), j["candidates"].to_list(), j["x"].to_list(), j["unknown"].to_list()):
        out[r][nid] = (float(x), float(u))
    return out

if DEMO_CHECK:
    # MINDdemo is not in the repo's layout; use per-split first-seen from the demo files themselves
    import os as _os; _os.chdir(WORK / "repo")
    from src.rerank.mind import first_sightings, load_behaviors, DATASET_START
    from src.baselines.nrms_fresh_features import fresh_from_frames
    for name, sub in [("MINDsmall_train", "train"), ("MINDsmall_dev", "valid")]:      # demo files under the small names
        sh(f"mkdir -p {WORK}/repo/data/interim/mind/{name}/{name} && cp {DATA}/{sub}/behaviors.tsv {DATA}/{sub}/news.tsv {WORK}/repo/data/interim/mind/{name}/{name}/")
else:
    import os as _os; _os.chdir(WORK / "repo")
    from src.rerank.mind import load_behaviors
f_tr = fresh_inputs("mind", "MINDsmall_train", None); STATS = fit_stats(f_tr)
f_tr = fresh_inputs("mind", "MINDsmall_train", STATS); f_ev = fresh_inputs("mind", "MINDsmall_dev", STATS)
print("FRESH_STATS", json.dumps(STATS.to_dict()), "| train unknown %.3f%% | dev unknown %.3f%%" % (100 * f_tr["unknown"].mean(), 100 * f_ev["unknown"].mean()))
FRESH_TRAIN = fresh_by_row(f_tr, load_behaviors("MINDsmall_train").select("imp_row", "candidates"))
FRESH_DEV = fresh_by_row(f_ev, load_behaviors("MINDsmall_dev").select("imp_row", "candidates"))
_os.chdir(WORK)
stage("fresh")

# ---- 3. train ---------------------------------------------------------------------------------------
hparams = prepare_hparams(f["yaml"], wordEmb_file=f["emb"], wordDict_file=f["wdict"], userDict_file=f["udict"],
                          batch_size=BS, epochs=EPOCHS, show_step=100000)
print("HPARAMS", str(hparams)[:600])
model = NRMSFreshModel(hparams, MINDFreshIterator, seed=SEED)
model.train_iterator.fresh_by_row = FRESH_TRAIN
model.test_iterator.fresh_by_row = FRESH_DEV
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
model.model.save_weights(str(WEIGHTS / "nrms_fresh_ckpt"))
stage("train")

# ---- 4+5. score every dev impression, full slate, and evaluate: twice (row 2, row 3 masked) -------------
rows = [line.rstrip("\n").split("\t") for line in open(f["valid_beh"])]
beh = pl.DataFrame({
    "imp_row": pl.Series(range(len(rows)), dtype=pl.UInt32),
    "impression_id": pl.Series([int(r[0]) for r in rows], dtype=pl.Int64),
    "article_ids_inview": pl.Series([[x.split("-")[0] for x in r[4].split()] for r in rows], dtype=pl.List(pl.Utf8)),
    "labels": pl.Series([[int(x.split("-")[1]) for x in r[4].split()] for r in rows], dtype=pl.List(pl.Int8)),
})
lab = (beh.select("imp_row", "labels").explode("labels")
       .with_columns(cand_position=pl.int_range(1, pl.len() + 1).over("imp_row").cast(pl.Int64)))
results = {}
for system, mask in [("nrms_fresh", False), ("nrms_fresh_masked", True)]:
    model.test_iterator.mask = mask
    t = time.time()
    impr_idx, labels, preds = model.run_fast_eval(f["valid_news"], f["valid_beh"])
    score_seconds = time.time() - t
    assert list(impr_idx) == list(range(beh.height)), "impr_index is not the file row"
    assert [list(map(int, l)) for l in labels] == beh["labels"].to_list()
    sf = scores_frame(beh.select("imp_row", "impression_id", "article_ids_inview"), [list(map(float, p)) for p in preds])
    assert sf.height == int(beh["article_ids_inview"].list.len().sum()) and sf["imp_row"].n_unique() == beh.height
    SC = OUT / "scores" / "mind" / ("MINDdemo_dev" if DEMO_CHECK else "MINDsmall_dev"); SC.mkdir(parents=True, exist_ok=True)
    sf.write_parquet(SC / f"{system}.parquet")
    manifest = {"dataset": "mind", "split": "MINDdemo_dev" if DEMO_CHECK else "MINDsmall_dev", "system": system, "framing": "in-impression",
                "variant": "nrms + freshness term (SPEC §15)", "fresh_masked_at_inference": mask, "fresh_stats": STATS.to_dict(),
                "n_rows": sf.height, "n_impressions": beh.height, "repo_commit": REPO_COMMIT, "recommenders_commit": REC_COMMIT,
                "seeds": {"model": SEED}, "epochs": EPOCHS, "history_size": hparams.his_size, "npratio": hparams.npratio,
                "word_embeddings": "GloVe-300d (MIND utils embedding.npy)", "precision": "float32", "determinism": True,
                "train_seconds": round(train_seconds), "score_seconds": round(score_seconds),
                "command": "./scripts/kaggle/alt_account.sh push scripts/kaggle/nrms_mind_fresh --accelerator NvidiaTeslaT4", "written_at": dt.datetime.utcnow().isoformat()}
    (SC / f"{system}.json").write_text(json.dumps(manifest, indent=2))
    print(f"SCORES[{system}]", sf.height, "rows", SC / f"{system}.parquet")
    joined = sf.join(lab, on=["imp_row", "cand_position"], how="left")
    assert joined["labels"].null_count() == 0
    per_imp = {k: np.asarray(v, float) for k, v in per_impression_metrics(impression_rows(joined, "score", label_col="labels")).items()}
    ci = bootstrap_evaluate(per_imp)
    ours = {k: float(np.nanmean(v)) for k, v in per_imp.items()}
    print(f"METRICS_OURS[{system}]", json.dumps(ours))
    print(f"METRICS_CI[{system}]", json.dumps({k: [c.mean, c.lo, c.hi] for k, c in ci.items()}))
    if not mask:
        pkg = cal_metric(labels, preds, ["group_auc", "mean_mrr", "ndcg@5;10"])
        print("METRICS_PKG", json.dumps(pkg))
        for k_ours, k_pkg in [("auc", "group_auc"), ("mrr", "mean_mrr"), ("ndcg@5", "ndcg@5"), ("ndcg@10", "ndcg@10")]:
            d = abs(round(ours[k_ours], 4) - pkg[k_pkg]); print(f"AGREE {k_ours} |Δ|={d:.2e}"); assert d <= 1e-4 + 1e-9
        print("METRICS_OURS", json.dumps(ours)); print("METRICS_CI", json.dumps({k: [c.mean, c.lo, c.hi] for k, c in ci.items()}))
    results[system] = (ours, pkg if not mask else None)
    stage(f"score_eval[{system}]")
sh(f"rm -rf {DATA} {WORK}/recommenders {WORK}/repo")   # keep only out/: `kaggle kernels output` pages at 20 files
print(f"RESULT PASS epochs={EPOCHS} group_auc={results['nrms_fresh'][1]['group_auc']} full_slate_auc={results['nrms_fresh'][0]['auc']:.6f} masked_auc={results['nrms_fresh_masked'][0]['auc']:.6f} total_seconds={time.time() - T0:.0f}")
