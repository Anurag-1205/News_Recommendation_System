"""P3.1 · U4: does the recommenders NRMS (the brief's "MIND baseline") run on Kaggle's image?

Push:  .venv/bin/kaggle kernels push -p scripts/kaggle/nrms_mind_smoke --accelerator NvidiaTeslaT4
Logs:  .venv/bin/kaggle kernels logs aayushpandey18602/a2-nrms-mind-smoke

Go/no-go for CONTEXT.md C-024: one epoch of the quick-start recipe on MINDsmall_train, evaluated
on MINDsmall_dev with the package's own run_eval, by Mon 14 Sep 20:00 IST; otherwise MIND falls
back to the EB-NeRD implementation. What this has to survive, each printed as it is settled:
  * recommenders 1.2.1 at 0bb4b36 is TF1 graph-mode Keras (tf.compat.v1, disable_eager_execution);
    Kaggle has TF 2.20 / Keras 3, so it runs under tf-keras with TF_USE_LEGACY_KERAS=1, vendored
    `--no-deps` (its pins: numpy<2, transformers<5, tensorflow<2.16 — none satisfiable here);
  * data: the official MINDsmall zips from the private dataset aayushpandey18602/mind-small-official
    (zip sha256 in scripts/kaggle/mind_small_dataset/SHA256SUMS; Kaggle serves them extracted); utils (GloVe embedding.npy,
    word_dict.pkl, uid2index.pkl, nrms.yaml) from huggingface.co/datasets/Recommenders/MIND;
  * the U5 alignment oracle: run_fast_eval's impr_index is the 0-based file row (= our imp_row)
    and each prediction list has the slate's length.
Every stage prints "STAGE <name> ok <seconds>".
"""
import hashlib, os, subprocess, sys, time, json
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"          # must precede the first tensorflow import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
T0 = time.time()
EPOCHS, SEED, BS = 1, 42, 32
REC_COMMIT = "0bb4b3690941ffb668118e31ccaf8a7d19f8212a"
HF = "https://huggingface.co/datasets/Recommenders/MIND/resolve/main"
WORK = Path("/kaggle/working"); DATA = WORK / "mind"; INPUT = Path("/kaggle/input/mind-small-official")


def stage(name): print(f"STAGE {name} ok {time.time() - T0:.0f}s", flush=True)
def sh(cmd): print("+", cmd, flush=True); subprocess.run(cmd, shell=True, check=True)


# ---- 1. environment ---------------------------------------------------------------------------------
sh("pip list 2>/dev/null | grep -iE '^(tensorflow|tf-keras|keras|numpy) '")
sh("pip install -q 'tf-keras~=2.20.0' 'retrying>=1.3.4'")
sh(f"git clone -q https://github.com/recommenders-team/recommenders {WORK}/recommenders && cd {WORK}/recommenders && git checkout -q {REC_COMMIT} && git rev-parse HEAD")
sys.path.insert(0, str(WORK / "recommenders"))
import tensorflow as tf
print("tf", tf.__version__, "legacy keras:", os.environ["TF_USE_LEGACY_KERAS"], "GPUs", [g.name for g in tf.config.list_physical_devices("GPU")])
print("tf.keras is", tf.keras.__name__, getattr(tf.keras, "__version__", "?"))
from recommenders.models.newsrec.newsrec_utils import prepare_hparams
from recommenders.models.newsrec.models.nrms import NRMSModel
from recommenders.models.newsrec.io.mind_iterator import MINDIterator
stage("imports")

# ---- 2. data ----------------------------------------------------------------------------------------
# Kaggle extracts uploaded archives: the dataset mounts as MINDsmall_train/MINDsmall_train/*.tsv
# (smoke run v1 died looking for the zips). The zip hashes are in the repo's SHA256SUMS; here
# the extracted TSVs are hashed so the log still pins what was trained on.
sh("find /kaggle/input -maxdepth 4 | sort | head -40")
import glob
for name, sub in [("MINDsmall_train", "train"), ("MINDsmall_dev", "valid")]:
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}/**/behaviors.tsv", recursive=True)) or \
           sorted(glob.glob(f"/kaggle/input/**/{name}*behaviors.tsv", recursive=True))
    assert hits, f"no behaviors.tsv for {name} under /kaggle/input"
    src = Path(hits[0]).parent
    print(name, "found at", src)
    for tsv in ("behaviors.tsv", "news.tsv"):
        print(f"{name}/{tsv} sha256", hashlib.sha256((src / tsv).read_bytes()).hexdigest())
    sh(f"mkdir -p {DATA}/{sub} && cp {src}/behaviors.tsv {src}/news.tsv {DATA}/{sub}/")
sh(f"mkdir -p {DATA}/utils && wget -q {HF}/MINDsmall_utils.zip -O {DATA}/MINDsmall_utils.zip && unzip -q -o {DATA}/MINDsmall_utils.zip -d {DATA}/utils")
print("MINDsmall_utils.zip sha256", hashlib.sha256((DATA / "MINDsmall_utils.zip").read_bytes()).hexdigest())
sh(f"ls -la {DATA}/utils {DATA}/train {DATA}/valid")
f = {k: str(DATA / p) for k, p in {"train_news": "train/news.tsv", "train_beh": "train/behaviors.tsv",
     "valid_news": "valid/news.tsv", "valid_beh": "valid/behaviors.tsv", "emb": "utils/embedding.npy",
     "udict": "utils/uid2index.pkl", "wdict": "utils/word_dict.pkl", "yaml": "utils/nrms.yaml"}.items()}
print(Path(f["yaml"]).read_text())
stage("data")

# ---- 3. model + one epoch ---------------------------------------------------------------------------
hparams = prepare_hparams(f["yaml"], wordEmb_file=f["emb"], wordDict_file=f["wdict"], userDict_file=f["udict"],
                          batch_size=BS, epochs=EPOCHS, show_step=500)
print("HPARAMS", hparams)
model = NRMSModel(hparams, MINDIterator, seed=SEED)
print("params", model.model.count_params())
print("EVAL_BEFORE", json.dumps(model.run_eval(f["valid_news"], f["valid_beh"])))
stage("build")
t = time.time()
model.fit(f["train_news"], f["train_beh"], f["valid_news"], f["valid_beh"])
print(f"TRAIN_SECONDS {time.time() - t:.0f} (epochs={EPOCHS}, incl. the per-epoch eval)")
res = model.run_eval(f["valid_news"], f["valid_beh"])
print("EVAL_AFTER", json.dumps(res))
stage("train")

# ---- 4. U5 alignment oracle: impr_index == file row, preds match slate lengths ----------------------
impr_idx, labels, preds = model.run_fast_eval(f["valid_news"], f["valid_beh"])
n_lines = sum(1 for _ in open(f["valid_beh"]))
slate_len = [len(line.rstrip("\n").split("\t")[-1].split()) for line in open(f["valid_beh"])]
print("run_fast_eval groups", len(impr_idx), "behaviors.tsv lines", n_lines)
assert list(impr_idx) == list(range(n_lines)), "impr_index is not the 0-based file row"
assert [len(p) for p in preds] == slate_len, "prediction lengths != slate lengths"
assert [len(l) for l in labels] == slate_len
print("ALIGN imp_row=impr_index and slate order confirmed on", n_lines, "dev impressions")
stage("align")
print(f"RESULT PASS group_auc={res['group_auc']} total_seconds={time.time() - T0:.0f}")
