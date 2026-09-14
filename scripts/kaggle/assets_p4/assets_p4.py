"""P4 · U0: the two stage-1 embedding assets the laptop cannot produce (SPEC.md §16, C-031).

    .venv/bin/kaggle kernels push -p scripts/kaggle/assets_p4          (CPU kernel, no GPU quota)
    .venv/bin/kaggle kernels output aayushpandey18602/a2-assets-p4 -p data/scores/_kaggle/assets_p4 --page-size 200 --file-pattern 'out/.*'

1. EB-NeRD: `Ekstra_Bladet_word2vec.zip` from the official S3 bucket → `document_vector.parquet`
   (125,541 articles × 300, the vectors A1/P2 use: `scripts/rerank_ebnerd_a2.EMB`).
2. MIND: `scripts/encode_mind_minilm.py` unchanged, on the official TSVs from the private
   dataset (train + dev + large_test news), → `mind_minilm.npz` (all-MiniLM-L6-v2, 384-d,
   L2-normalised) — the vectors `scripts/rerank_mind_a2.py` loads.
Prints row counts, dims, norms and sha256 so the ledger row pins what was produced.
"""
import glob, hashlib, subprocess, sys, time
from pathlib import Path

T0 = time.time()
WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
REPO_URL, REPO_REF = "https://github.com/Anurag-1205/News_Recommendation_System", "a2-click-logs"
S3 = "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com"


def stage(n): print(f"STAGE {n} ok {time.time() - T0:.0f}s", flush=True)
def sh(c): print("+", c, flush=True); subprocess.run(c, shell=True, check=True)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


sh("pip install -q polars==1.43.2")
sh(f"git clone -q --branch {REPO_REF} {REPO_URL} {WORK}/repo && cd {WORK}/repo && git rev-parse HEAD")
stage("fetch")

# ---- 1. EB-NeRD word2vec document vectors ----------------------------------------------------------
sh(f"wget -q {S3}/artifacts/Ekstra_Bladet_word2vec.zip -O {WORK}/w2v.zip && unzip -q -o {WORK}/w2v.zip -d {WORK}/w2v && find {WORK}/w2v -name '*.parquet'")
src = next(Path(p) for p in glob.glob(f"{WORK}/w2v/**/document_vector.parquet", recursive=True))
import numpy as np, polars as pl
w = pl.read_parquet(src)
vec_col = [c for c in w.columns if c != "article_id"][0]
print("word2vec:", w.height, "rows; columns", w.columns, "; dim", len(w[vec_col][0]))
(OUT / "ebnerd").mkdir(exist_ok=True); sh(f"cp {src} {OUT}/ebnerd/document_vector.parquet")
print("document_vector.parquet sha256", sha(OUT / "ebnerd" / "document_vector.parquet"), "zip sha256", sha(WORK / "w2v.zip"))
stage("ebnerd_w2v")

# ---- 2. MIND MiniLM vectors, via the repo's own encoder -------------------------------------------------
for name in ("MINDsmall_train", "MINDsmall_dev", "MINDlarge_test"):
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}/**/news.tsv", recursive=True))
    assert hits, f"no news.tsv for {name} under /kaggle/input"
    d = Path(hits[0]).parent
    sh(f"mkdir -p {WORK}/repo/data/interim/mind/{name}/{name} && cp {d}/news.tsv {d}/behaviors.tsv {WORK}/repo/data/interim/mind/{name}/{name}/")
sh(f"cd {WORK}/repo && PYTHONPATH=. python scripts/encode_mind_minilm.py")
z = np.load(WORK / "repo" / "data/processed/mind_minilm.npz", allow_pickle=True)
norms = np.linalg.norm(z["matrix"], axis=1)
print("mind_minilm:", z["matrix"].shape, "ids", len(z["ids"]), "norm min/max %.4f/%.4f" % (norms.min(), norms.max()))
(OUT / "mind").mkdir(exist_ok=True); sh(f"cp {WORK}/repo/data/processed/mind_minilm.npz {OUT}/mind/")
print("mind_minilm.npz sha256", sha(OUT / "mind" / "mind_minilm.npz"))
stage("mind_minilm")
sh(f"rm -rf {WORK}/repo {WORK}/w2v {WORK}/w2v.zip")
print(f"RESULT PASS total_seconds={time.time() - T0:.0f}")
