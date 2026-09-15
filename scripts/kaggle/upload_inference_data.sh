#!/usr/bin/env bash
# Upload the files the locked reranker needs for test-set inference as two private Kaggle datasets
# on Anurag's account (CONTEXT.md C-036, P0 step 5 for the test splits).
#
#   ./scripts/kaggle/upload_inference_data.sh ebnerd
#   ./scripts/kaggle/upload_inference_data.sh mind
#
# Each dataset is one zip whose member paths are exactly the relative paths src/rerank/{ebnerd,mind}.py
# opens under data/interim/, so a kernel unzips it and points the code at the mount with no rewriting.
# Only the files that are actually read are included: no entity/relation embeddings, no __MACOSX,
# no .DS_Store. Idempotent: `create` the first time, `version` afterwards. Nothing under data/ is
# modified and nothing large enters git -- the zip is built in a temp dir and deleted on exit.
set -euo pipefail
cd "$(dirname "$0")/../.."

DATASET=${1:?usage: upload_inference_data.sh <ebnerd|mind>}
STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT

case "$DATASET" in
  ebnerd)
    META=scripts/kaggle/ebnerd_test_dataset
    BASE=data/interim/ebnerd
    FILES=(
      ebnerd_small/articles.parquet
      ebnerd_small/train/behaviors.parquet
      ebnerd_small/train/history.parquet
      ebnerd_small/validation/behaviors.parquet
      ebnerd_small/validation/history.parquet
      ebnerd_testset/ebnerd_testset/articles.parquet
      ebnerd_testset/ebnerd_testset/test/behaviors.parquet
      ebnerd_testset/ebnerd_testset/test/history.parquet
    )
    ZIP=a2_ebnerd_inference.zip
    ;;
  mind)
    META=scripts/kaggle/mind_test_dataset
    BASE=data/interim/mind
    FILES=(
      MINDsmall_train/MINDsmall_train/behaviors.tsv
      MINDsmall_train/MINDsmall_train/news.tsv
      MINDsmall_dev/MINDsmall_dev/behaviors.tsv
      MINDsmall_dev/MINDsmall_dev/news.tsv
      MINDlarge_test/MINDlarge_test/behaviors.tsv
      MINDlarge_test/MINDlarge_test/news.tsv
    )
    ZIP=a2_mind_inference.zip
    ;;
  *) echo "unknown dataset: $DATASET" >&2; exit 2 ;;
esac

for f in "${FILES[@]}"; do test -f "$BASE/$f" || { echo "missing: $BASE/$f" >&2; exit 1; }; done

# sha256 of every member, recorded in git so the mount can be verified against this machine.
( cd "$BASE" && sha256sum "${FILES[@]}" ) > "$META/SHA256SUMS"
cat "$META/SHA256SUMS"

# -0: the parquets are already compressed and the TSVs are ~1.5 GB; storing is far faster and
# Kaggle does not charge for the difference.
echo "building $ZIP ..."
( cd "$BASE" && zip -0 -q -X "$STAGE/$ZIP" "${FILES[@]}" )
ls -la "$STAGE/$ZIP"

cp "$META/dataset-metadata.json" "$STAGE/"
ID=$(python -c "import json,sys;print(json.load(open('$META/dataset-metadata.json'))['id'])")
if .venv/bin/kaggle datasets status "$ID" >/dev/null 2>&1; then
  .venv/bin/kaggle datasets version -p "$STAGE" -m "refresh $(date -Is)" --dir-mode skip
else
  .venv/bin/kaggle datasets create -p "$STAGE" --dir-mode skip
fi
