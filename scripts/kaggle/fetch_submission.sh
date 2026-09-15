#!/usr/bin/env bash
# Download a finished a2-submit-<dataset> kernel's output, verify the zip against the sha256 the
# kernel printed, re-validate the predictions file locally against the test file, and ledger the run.
#
#   ./scripts/kaggle/fetch_submission.sh <ebnerd|mind> <vN>
set -euo pipefail
cd "$(dirname "$0")/../.."
D=${1:?dataset}; V=${2:?version tag, e.g. v1}
USER=$(python -c "import json;print(json.load(open('$HOME/.kaggle/kaggle.json'))['username'])")
DEST=data/submissions/_kaggle/$D/$V
mkdir -p "$DEST" data/logs/kaggle
.venv/bin/kaggle kernels output "$USER/a2-submit-$D" -p "$DEST" --page-size 400 >/dev/null
echo "downloaded to $DEST:"; find "$DEST" -maxdepth 3 -type f \( -name '*.zip' -o -name 'manifest.json' -o -name 'predictions.txt' \) -printf "  %10s  %p\n"
ZIP=$(find "$DEST" -name "${D}_reranker_final.zip" | head -1); test -n "$ZIP" || { echo "no zip in output"; exit 1; }
# the kernel printed "zip sha256 <hex> bytes <n>"; the ledger tool saves the log, so fetch it first
.venv/bin/python scripts/kaggle/ledger.py "a2-submit-$D" "$V" --account anurag --note "P5 submission run" >/dev/null
LOG=data/logs/kaggle/a2-submit-${D}_${V}_${USER}.log
WANT=$(tr -d '\\' < "$LOG" | grep -o 'zip sha256 [0-9a-f]\{64\}' | tail -1 | awk '{print $3}')
GOT=$(sha256sum "$ZIP" | awk '{print $1}')
echo "zip sha256: kernel $WANT"; echo "            local  $GOT"
test "$WANT" = "$GOT" && echo "ZIP HASH MATCHES the kernel's" || { echo "ZIP HASH MISMATCH"; exit 1; }
echo "--- local re-validation against the test file ---"
PYTHONPATH=. .venv/bin/python - "$D" "$ZIP" <<'PY'
import sys, zipfile, tempfile, pathlib, json
from src.eval.submission import validate_file
d, zp = sys.argv[1], pathlib.Path(sys.argv[2])
with zipfile.ZipFile(zp) as z:
    names = z.namelist(); assert names == ["predictions.txt"], f"zip must hold predictions.txt at root, got {names}"
    tmp = pathlib.Path(tempfile.mkdtemp()) / "predictions.txt"; tmp.write_bytes(z.read("predictions.txt"))
if d == "mind":
    from src.rerank.mind import load_behaviors
    beh = load_behaviors("MINDlarge_test", labelled=False)
    rep = validate_file(tmp, expected_ids=beh["impression_id"].to_list(),
                        expected_lengths=dict(zip(beh["impression_id"].to_list(), beh["candidates"].list.len().to_list())))
else:
    from src.rerank.ebnerd import TEST, load_behaviors
    beh = load_behaviors(TEST / "test/behaviors.parquet")
    rep = validate_file(tmp, expected_ids=beh["impression_id"].to_list(), allow_duplicate_ids=True)
print("validate_file:", json.dumps(rep), "| test impressions:", beh.height)
assert rep["lines"] == beh.height, f"line count {rep['lines']} != test impressions {beh.height}"
print("LOCAL VALIDATION PASS")
PY
tail -1 scripts/kaggle/RUN_LEDGER.md | cut -c1-200
