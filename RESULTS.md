# RESULTS.md

Every measured number, with the exact command that produced it and the date. A number without a
command is not a result — measure, don't vibe. Headline metrics carry a bootstrap 95% CI, and two
overlapping CIs are **not** a win — do not write "beats" for them.

---

## Environment baseline — 2026-08-20

Machine facts that bound every later measurement.

| Quantity | Value | Command |
|---|---|---|
| Python | 3.12.3 | `python3 --version` |
| RAM total / available | 7 GB / ~2 GB | `free -g` |
| Disk free | 53 GB | `df -h /home/anurag` |
| Network, S3 eu-west-1 | 13,027 B/s | `curl -so /dev/null -r 0-8000000 -w '%{speed_download}' --max-time 20 <ebnerd_large.zip>` |
| Network, Cloudflare | 3,966 B/s | `curl -so /dev/null -r 0-8000000 -w '%{speed_download}' --max-time 20 https://speed.cloudflare.com/__down?bytes=8000000` |
| Network, HuggingFace | 101 B/s | same form, against `huggingface.co/.../README.md` |
| Packet loss / RTT | 50% loss, 305–1225 ms | `ping -c 4 -W 2 1.1.1.1` |
| Interface | `wifi@iiith` | `nmcli -t -f TYPE,STATE,CONNECTION device` |

Three independent endpoints agree, so the constraint is the link rather than any one server.
Implication for the scale analysis (Q6): at 13 KB/s the 2.97 GB `ebnerd_large.zip` needs ~63 h
of uninterrupted transfer. See `SPEC.md` §11.

## Dataset bundle sizes — 2026-08-20

`curl -sI <url> | grep -i content-length`

| Bundle | Bytes | MB |
|---|---:|---:|
| `ebnerd_demo.zip` | 21,499,083 | 20.5 |
| `ebnerd_small.zip` | 84,135,301 | 80.2 |
| `ebnerd_large.zip` | 3,189,066,769 | 3041.3 |
| `ebnerd_testset.zip` | 1,631,004,285 | 1555.4 |
| `articles_large_only.zip` | 149,931,045 | 143.0 |
| `Ekstra_Bladet_word2vec.zip` | 139,510,991 | 133.0 |
| `google_bert_base_multilingual_cased.zip` | 361,239,658 | 344.5 |
| `MINDsmall_train.zip` | 52,953,372 | 50.5 |
| `MINDsmall_dev.zip` | 30,946,172 | 29.5 |
| `MINDlarge_test.zip` | 604,624,665 | 576.6 |

MIND sizes required an `Authorization: Bearer` header — the repo is gated, and unauthenticated
HEAD returns 401 with no `content-length`.

**MIND's test set is 576.6 MB against EB-NeRD's 1555.4 MB**, which is why MIND is the first
submission: it is the smaller download on a constrained link, not a modelling preference.

---

## Submission 1 · MIND popularity baseline — 2026-08-21

```bash
PYTHONPATH=. .venv/bin/python scripts/make_submission_mind.py
```

Fitted on `MINDsmall_train` (156,965 impressions → 7,713 distinct clicked articles; most-clicked
`N55689`, 4,316 clicks). Evaluated on `MINDsmall_dev`, which has labels. Log:
`data/processed/run_mind.log`; metrics: `data/processed/mind_popularity_metrics.json`.

### Offline metrics on MINDsmall_dev (73,152 impressions)

| Metric | Value |
|---|---:|
| AUC | 0.5318 |
| MRR | 0.2382 |
| nDCG@5 | 0.2460 |
| nDCG@10 | 0.3098 |

AUC of 0.532 is barely above chance, which is the correct result for this model rather than a
disappointing one: popularity uses no history, no text, and no personalisation, so it is the floor
that submission 2 has to clear. No bootstrap CI yet — that lands with the P2 harness.

### The coverage problem — why this baseline is weak, measured

Popularity can only rank an article it has seen clicked in training. It has not seen most of them.

| Split | Candidate coverage | Impressions with **zero** known candidates |
|---|---:|---:|
| `MINDsmall_dev` | 34.2% | — |
| `MINDlarge_test` (300K sample) | **6.5%** | **28.6%** |

Two distinct causes compound: MIND's train split is 11 Nov 2019 while the test split is
19–22 Nov 2019, and the test news set is 120,961 articles against small-train's 51,282. News
turns over fast, so most test candidates were never clicked in training.

**Consequence, stated plainly: the dev numbers above overstate what the leaderboard will show.**
Dev has five times the coverage of test. For the 28.6% of test impressions where every candidate
is unknown, all scores tie at 0 and the ranking degenerates to the candidate list's own order —
arbitrary, not predicted.

This is also the argument for submission 2. BM25 and embeddings score *text*, which exists for all
120,961 test articles, so their coverage is 100% by construction. The gap between 6.5% and 100% is
the measured reason to expect an improvement, rather than a hoped-for one.

### Scale instrumentation (Q6 evidence)

| Quantity | Value |
|---|---:|
| Test impressions scored | 2,370,727 |
| Wall time, scoring pass | 44 s |
| Throughput | ~54,000 impressions/s |
| **Peak RSS** | **287 MB** |
| `prediction.txt` | 291,329,312 B (277.8 MB) |
| `mind_prediction.zip` | 28,107,729 B (26.8 MB) |

287 MB peak against ~2 GB free is the payoff from streaming the test split in 200K-row slices
instead of reading all 2.37M rows at once as the reference notebooks do. Offline validation passed:
2,370,727 lines, 2,370,727 distinct impression ids, every rank list a permutation of 1..N.

## Q2 · BM25 lexical retrieval — recall@K

*Pending P3.*

## Q3 · Semantic retrieval — recall@K

*Pending P4.*

## Q3.5 · Lexical vs. semantic, by slice

*Pending P4.*

## Q4 · Ranking metrics with 95% CI

*Pending P2.*

## Q9 · With / without serving-unavailable features

*Pending P4.* The EB-NeRD pair is `next_read_time` / `next_scroll_percentage` — present in
train and validation, absent from test by construction. See `SPEC.md` §9.

## Q6 · Scale instrumentation

*Pending P5.* Index build time, peak RSS, per-query latency, embedding cost.
