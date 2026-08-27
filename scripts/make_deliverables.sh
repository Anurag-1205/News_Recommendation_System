#!/usr/bin/env bash
# Assemble the submission bundles.
#
#   report/A1_report.zip    -> Moodle: design note, specification, results, AI usage log,
#                              measured metrics and run logs, leaderboard screenshots
#   report/A1_prompts.zip   -> submitted separately: the prompt record (A1 Q7.4)
#
# Source code is delivered through GitHub Classroom and is not duplicated here.
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE=report/deliverables
rm -rf "$STAGE"; mkdir -p "$STAGE"

# --- graded documents -------------------------------------------------------
cp report/design_note.md SPEC.md RESULTS.md AI_USAGE.md README.md "$STAGE/"

# --- the measured evidence those documents cite -----------------------------
mkdir -p "$STAGE/metrics" "$STAGE/logs"
for f in data/processed/*.json; do [ -f "$f" ] && cp "$f" "$STAGE/metrics/"; done
for f in data/processed/*.log;  do [ -f "$f" ] && cp "$f" "$STAGE/logs/";  done

# --- leaderboard screenshots (A1 Q7.3) --------------------------------------
mkdir -p "$STAGE/screenshots"
shopt -s nullglob
shots=(report/*.png report/*.jpg)
if (( ${#shots[@]} )); then
  cp "${shots[@]}" "$STAGE/screenshots/"
else
  echo "NOTE: no leaderboard screenshots found in report/ — required by A1 Q7.3" >&2
fi
shopt -u nullglob

rm -f report/A1_report.zip report/A1_prompts.zip
( cd "$STAGE" && zip -qr ../A1_report.zip . )
( zip -qr report/A1_prompts.zip prompts )

echo "report/A1_report.zip   $(du -h report/A1_report.zip  | cut -f1)  -> Moodle"
echo "report/A1_prompts.zip  $(du -h report/A1_prompts.zip | cut -f1)  -> submitted separately"
