"""P5: score a Codabench test file with config.FINAL and write the submission zip (CONTEXT.md C-038).

One script for both datasets; DATASET is the only line that differs between the two kernel
folders (scripts/kaggle/submit_mind, scripts/kaggle/submit_ebnerd).

    .venv/bin/kaggle kernels push -p scripts/kaggle/submit_<dataset>      (CPU kernel, no GPU quota)
    .venv/bin/kaggle kernels output anuragkaushal183/a2-submit-<dataset> -p data/submissions/_kaggle/<dataset>

Steps: clone the repo at REPO_REF -> verify every mounted input file against the sha256 committed
in scripts/kaggle/*/SHA256SUMS (a silently wrong mount is the one failure that would produce a
valid-looking zip that scores like noise) -> symlink the mounts into the paths the loaders read
-> run scripts/submit_a2.py, which fits once, scores in resumable chunks and validates the file
-> copy predictions, zip, chunks and manifest to /kaggle/working/out.

Resume: a chunk file that exists is complete, so a relaunched kernel that finds a previous
version's `out/chunks/` restores it before running and continues from the first missing chunk.
"""
import glob, hashlib, os, subprocess, sys, time
from pathlib import Path

DATASET = "mind"
CHUNK = {"ebnerd": 250_000, "mind": 100_000}[DATASET]
T0 = time.time()
WORK = Path("/kaggle/working"); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
REPO_URL, REPO_REF = "https://github.com/Anurag-1205/News_Recommendation_System", "a2-click-logs"
INPUT = Path("/kaggle/input")


def stage(n): print(f"STAGE {n} ok {time.time() - T0:.0f}s", flush=True)
def sh(c, **kw): print("+", c, flush=True); return subprocess.run(c, shell=True, check=True, **kw)
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()


def find_one(pattern: str) -> Path:
    hits = sorted(glob.glob(str(INPUT / "**" / pattern), recursive=True))
    assert len(hits) >= 1, f"no {pattern} under /kaggle/input"
    return Path(hits[0])


sh("pip install -q polars==1.43.2 lightgbm==4.7.0 faiss-cpu==1.15.0 pyarrow==25.0.1")
sh(f"git clone -q --branch {REPO_REF} {REPO_URL} {WORK}/repo && cd {WORK}/repo && git rev-parse HEAD")
REPO = WORK / "repo"
stage("fetch")

# ---- 1. verify the mounts against the committed hashes, then link them where the loaders look -----
def verify_and_link(sums_file: Path, mount_root: Path, link_root: Path):
    n = 0
    for line in sums_file.read_text().splitlines():
        want, rel = line.split()
        src = mount_root / rel
        assert src.exists(), f"missing on mount: {src}"
        got = sha(src)
        assert got == want, f"sha256 mismatch for {rel}: mount {got[:12]} != committed {want[:12]}"
        dst = link_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink(): dst.unlink()
        os.symlink(src, dst)
        n += 1
    print(f"verified + linked {n} files from {mount_root} -> {link_root}", flush=True)

# datasets are mounted extracted; find the directory that contains the first SHA256SUMS entry
def mount_root_for(sums_file: Path) -> Path:
    first_rel = sums_file.read_text().split()[1]
    hits = sorted(p for p in glob.glob(str(INPUT / "**" / first_rel), recursive=True))   # full relative path, not basename
    assert len(hits) == 1, f"{first_rel}: expected exactly one match under /kaggle/input, got {hits}"
    return Path(hits[0]).parents[len(first_rel.split("/")) - 1]

interim = REPO / "data" / "interim" / DATASET
verify_and_link(REPO / f"scripts/kaggle/{DATASET}_test_dataset/SHA256SUMS", mount_root_for(REPO / f"scripts/kaggle/{DATASET}_test_dataset/SHA256SUMS"), interim)
# stage-1 embedding assets
assets_sums = REPO / "scripts/kaggle/assets_dataset/SHA256SUMS"
assets_root = mount_root_for(assets_sums)
for line in assets_sums.read_text().splitlines():
    want, rel = line.split()
    if not rel.startswith(DATASET + "/"): continue
    src = assets_root / rel
    assert sha(src) == want, f"sha256 mismatch for asset {rel}"
    dst = (REPO / "data/interim/ebnerd/Ekstra_Bladet_word2vec/document_vector.parquet" if DATASET == "ebnerd"
           else REPO / "data/processed/mind_minilm.npz")
    dst.parent.mkdir(parents=True, exist_ok=True); os.symlink(src, dst)
    print(f"verified + linked asset {rel} -> {dst.relative_to(REPO)}", flush=True)
stage("verify_inputs")

# ---- 2. resume: restore chunk files from a previous version of this kernel's output, if present ----
prev = sorted(glob.glob(str(INPUT / "**" / "out" / "chunks" / "chunk_*.txt"), recursive=True))
sub_out = REPO / "data/submissions" / DATASET / "chunks"
sub_out.mkdir(parents=True, exist_ok=True)
for p in prev:
    sh(f"cp {p} {sub_out}/")
print(f"restored {len(prev)} chunk files from a previous run", flush=True)
stage("resume")

# ---- 3. score + validate + zip ------------------------------------------------------------------------
sh(f"cd {REPO} && PYTHONPATH=. python -u scripts/submit_a2.py --dataset {DATASET} --chunk {CHUNK}")
stage("score")

# ---- 4. publish outputs ---------------------------------------------------------------------------------
sh(f"cp -r {REPO}/data/submissions/{DATASET} {OUT}/")
z = next(Path(OUT / DATASET).glob("*.zip"))
print(f"zip sha256 {sha(z)} bytes {z.stat().st_size}", flush=True)
print((OUT / DATASET / "manifest.json").read_text(), flush=True)
sh(f"rm -rf {REPO}")
print(f"RESULT PASS dataset={DATASET} zip={z.name} total_seconds={time.time() - T0:.0f}", flush=True)
