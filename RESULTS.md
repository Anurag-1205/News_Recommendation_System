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

| Bundle | Bytes | GB |
|---|---:|---:|
| `ebnerd_demo.zip` | 21,442,553 | 0.02 |
| `ebnerd_small.zip` | 89,668,714 | 0.08 |
| `ebnerd_large.zip` | 3,188,930,048 | 2.97 |
| `ebnerd_testset.zip` | 1,632,102,400 | 1.52 |
| `articles_large_only.zip` | 149,931,045 | 0.14 |
| `Ekstra_Bladet_word2vec.zip` | 139,586,437 | 0.13 |
| `google_bert_base_multilingual_cased.zip` | 365,072,220 | 0.34 |
| MIND bundles | — | gated (HTTP 401) |

---

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
