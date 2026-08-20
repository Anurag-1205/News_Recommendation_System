#!/usr/bin/env bash
# Raw-data fetcher. Downloads only; extraction is `make data`'s job (A1 Q1: rebuild from raw).
#
#   ./scripts/fetch_data.sh small   ~0.10 GB  EB-NeRD demo+small          -> needed by P1, run now
#   ./scripts/fetch_data.sh large   ~5.10 GB  EB-NeRD large/test/embeds   -> needed by P5 (25 Aug)
#   ./scripts/fetch_data.sh mind    gated     MIND via HuggingFace        -> needs `hf auth login`
#
# All wget calls use -c (resume): a dropped connection overnight costs the current chunk,
# not the whole 3 GB. Downloads are sequential on purpose — one stream already saturates
# the link, and parallel streams make resume semantics harder to reason about.
set -euo pipefail

S3=https://ebnerd-dataset.s3.eu-west-1.amazonaws.com
EB=data/raw/ebnerd
MIND=data/raw/mind
mkdir -p "$EB" "$MIND"

# Measured on wifi@iiith, 20 Aug 2026: 4-13 KB/s with 50% packet loss and 300-1200ms RTT.
# On a link that lossy, a plain `wget` dies on the first stall and loses hours. These flags make
# it grind instead: retry forever, resume from the byte offset, and treat a 20s stall as a drop
# rather than hanging on a dead socket.
WGET_RESILIENT=(
  --continue                 # resume from bytes already on disk
  --tries=inf                # never give up
  --retry-connrefused        # a refused connection is transient here, not fatal
  --timeout=20               # a socket quiet for 20s is dead
  --waitretry=10             # back off between attempts
  --read-timeout=20
  --progress=dot:giga        # one dot line per 1GB — keeps the log readable over days
)

# wget's own --tries=inf covers a stalled *transfer*, but not the two failures actually observed
# on this link: an SSL handshake that never completes, and DNS dying outright when the wifi drops
# ("Name or service not known"). Both make wget exit non-zero, and under `set -e` that killed the
# whole script mid-download. So each fetch gets an outer supervisor loop: sleep through the outage,
# then resume from the byte offset already on disk.
MAX_ATTEMPTS=${MAX_ATTEMPTS:-500}
SLEEP_BETWEEN=${SLEEP_BETWEEN:-60}

get() {  # get <url> ; sizes are the measured content-length, recorded 20 Aug 2026
  local url="$1" name attempt=0 rc
  name=$(basename "$url")
  echo "--- $(date -Is)  fetching $name"
  while (( attempt < MAX_ATTEMPTS )); do
    attempt=$(( attempt + 1 ))
    rc=0
    wget "${WGET_RESILIENT[@]}" -P "$EB" "$url" || rc=$?
    if (( rc == 0 )); then
      echo "--- $(date -Is)  $name complete ($(du -h "$EB/$name" | cut -f1))"
      return 0
    fi
    # rc 8 = server sent an error response. A 404 will never succeed, so don't loop on it forever;
    # a transient 5xx will, hence the small allowance before giving up.
    if (( rc == 8 && attempt > 5 )); then
      echo "--- $(date -Is)  $name: server error (rc=8) x$attempt — giving up, check the URL" >&2
      return 8
    fi
    echo "--- $(date -Is)  $name: wget rc=$rc, attempt $attempt/$MAX_ATTEMPTS — sleeping ${SLEEP_BETWEEN}s"
    sleep "$SLEEP_BETWEEN"
  done
  echo "--- $(date -Is)  $name: exhausted $MAX_ATTEMPTS attempts" >&2
  return 1
}

case "${1:-}" in
  small)
    get $S3/ebnerd_demo.zip                                    # 0.02 GB
    get $S3/ebnerd_small.zip                                   # 0.08 GB
    ;;
  large)
    get $S3/ebnerd_large.zip                                   # 2.97 GB
    get $S3/ebnerd_testset.zip                                 # 1.52 GB
    get $S3/articles_large_only.zip                            # 0.14 GB  bucket ROOT, not artifacts/
    get $S3/artifacts/Ekstra_Bladet_word2vec.zip               # 0.13 GB
    get $S3/artifacts/google_bert_base_multilingual_cased.zip  # 0.34 GB
    ;;
  mind)
    # yjw1029/MIND is gated (`gated: auto`): every resolve/ URL 401s until you have
    # accepted the terms at https://huggingface.co/datasets/yjw1029/MIND and run `hf auth login`.
    # MINDlarge_train/dev are deliberately NOT fetched — A1 needs only MINDlarge_test.zip
    # for the leaderboard, and they would cost several GB of a 53 GB disk budget.
    hf download yjw1029/MIND --repo-type dataset --local-dir "$MIND" \
       --include "MINDsmall_train.zip" "MINDsmall_dev.zip" "MINDlarge_test.zip"
    ;;
  *)
    echo "usage: $0 {small|large|mind}" >&2; exit 2
    ;;
esac

echo "--- $(date -Is)  done: $1"
