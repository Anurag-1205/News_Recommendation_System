#!/usr/bin/env bash
# Upload the official MIND-small zips as a private Kaggle dataset (CONTEXT.md C-024, P0 step 5 for
# MIND-small). Idempotent: `create` the first time, `version` afterwards. Files are staged in a
# temp dir next to the metadata so nothing under data/ is touched and nothing large enters git.
#
#   ./scripts/kaggle/upload_mind_small.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
META=scripts/kaggle/mind_small_dataset
STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
cp "$META/dataset-metadata.json" "$STAGE/"
for z in MINDsmall_train.zip MINDsmall_dev.zip; do ln -s "$PWD/data/raw/mind/$z" "$STAGE/$z"; done
( cd data/raw/mind && sha256sum MINDsmall_train.zip MINDsmall_dev.zip ) > "$META/SHA256SUMS"
cat "$META/SHA256SUMS"
if .venv/bin/kaggle datasets status aayushpandey18602/mind-small-official >/dev/null 2>&1; then
  .venv/bin/kaggle datasets version -p "$STAGE" -m "refresh $(date -Is)" --dir-mode skip
else
  .venv/bin/kaggle datasets create -p "$STAGE" --dir-mode skip
fi
