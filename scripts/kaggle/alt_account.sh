#!/usr/bin/env bash
# Run a committed kernel folder from the second Kaggle account (CONTEXT.md C-027).
#
#   ./scripts/kaggle/alt_account.sh push   scripts/kaggle/gpu_check [--accelerator NvidiaTeslaT4]
#   ./scripts/kaggle/alt_account.sh status a2-gpu-check
#   ./scripts/kaggle/alt_account.sh logs   a2-gpu-check          # raw log to stdout
#
# The committed kernel-metadata.json files name the main account in their "id"; a copy with the
# alt username is staged in a temp dir so nothing under scripts/ changes. The token lives in
# ~/.kaggle/alt/kaggle.json (mode 600) and is selected per call with KAGGLE_CONFIG_DIR.
set -euo pipefail
cd "$(dirname "$0")/../.."
export KAGGLE_CONFIG_DIR="$HOME/.kaggle/alt"
ALT=$(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.kaggle/alt/kaggle.json')))['username'])")
MAIN=aayushpandey18602
cmd=$1; shift
case "$cmd" in
  push)
    src=$1; shift
    stage=$(mktemp -d); trap 'rm -rf "$stage"' EXIT
    cp "$src"/*.py "$src"/*.json "$stage"/          # files only: py_compile leaves a __pycache__ dir
    sed -i "s|\"id\": \"$MAIN/|\"id\": \"$ALT/|" "$stage/kernel-metadata.json"
    grep '"id"' "$stage/kernel-metadata.json"
    .venv/bin/kaggle kernels push -p "$stage" "$@"
    ;;
  status) .venv/bin/kaggle kernels status "$ALT/$1" ;;
  logs)   .venv/bin/kaggle kernels logs "$ALT/$1" ;;
  quota)  .venv/bin/kaggle quota ;;
  *) echo "usage: $0 {push <kernel-dir> [push args]|status <slug>|logs <slug>|quota}" >&2; exit 2 ;;
esac
