# Results and Ablations

**CS4.406 Information Retrieval and Extraction — Assignment 1, Component 1**

This document records every measured figure together with the command that produced it and the
date of measurement. A figure reported without its originating command is not treated as a
result.

**Statistical convention.** Headline metrics carry a bootstrap 95% confidence interval obtained
by resampling impressions. Two overlapping intervals are reported as showing no significant
difference; the term "beats" is used only where intervals are disjoint. Where a later
measurement at a larger sample size contradicts an earlier one, both are retained and the
correction is stated explicitly.

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
of uninterrupted transfer.

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

Two distinct causes compound: MIND's train split runs 9–14 Nov 2019 while the test split is
16–22 Nov 2019, and the test news set is 120,961 articles against small-train's 51,282. News
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

### Leaderboard result — MIND Codabench 13967, submitted 2026-08-21

Returned by the competition's own scorer (`scoring_result.zip` → `scores.json`), scoring time 427 s:

| Metric | Offline, `MINDsmall_dev` | **Leaderboard, `MINDlarge_test`** | Δ |
|---|---:|---:|---:|
| AUC | 0.5318 | **0.5036** | −0.0282 |
| MRR | 0.2382 | **0.2249** | −0.0133 |
| nDCG@5 | 0.2460 | **0.2305** | −0.0155 |
| nDCG@10 | 0.3098 | **0.2860** | −0.0238 |

Two things this establishes, both of which are worth more than the score itself.

**1. The harness measures what the graders' scorer measures.** All four offline metrics were
optimistic by a small, consistent margin (−0.013 to −0.028) and none inverted, so
`src/eval/metrics.py` computes the same quantities as the competition's implementation.

> **Superseded — read with the correction.** On the strength of this and submission 2 it was
> concluded here that the harness was "calibrated" and predicted both the *direction* and the
> *size* of a leaderboard move. Submission 3 refuted the second half: its offset was −0.090.
> Two points from one model family were never evidence about a different one. The full
> correction is under *Submission 3*, and the reversal on EB-NeRD is under *EB-NeRD
> submissions*. The claim is left in place rather than edited away, because the sequence of
> belief and refutation is the finding.

**2. The coverage prediction was right.** RESULTS.md said before submitting that dev would overstate
test because dev coverage is 34.2% against test's 6.5%, and that AUC would land near chance. It came
back 0.5036 — chance is 0.5000. Popularity contributes essentially nothing on this test split.

## Submission 2 · MIND weighted-sum fusion — leaderboard 2026-08-22

`data/processed/mind_prediction_v2.zip`, generated by `scripts/overnight.py`.
Scorer output: `{"AUC": 0.5258, "MRR": 0.2405, "nDCG@5": 0.2477, "nDCG@10": 0.3038}`, 305 s.

| Metric | Sub 1 (popularity) | **Sub 2 (fusion)** | Δ |
|---|---:|---:|---:|
| AUC | 0.5036 | **0.5258** | **+0.0222** |
| MRR | 0.2249 | **0.2405** | +0.0156 |
| nDCG@5 | 0.2305 | **0.2477** | +0.0172 |
| nDCG@10 | 0.2860 | **0.3038** | +0.0178 |

**The improvement requirement is satisfied**, on all four metrics.

**The harness offset held for a second time.** Offline predicted 0.5565; the leaderboard
returned 0.5258, an offset of −0.0307 against −0.0282 for submission 1. Two submissions, two
consistent offsets, no inversions — so offline comparisons on this dataset can be trusted to
predict both the direction *and* roughly the size of a leaderboard move. That is what makes it
possible to choose models locally instead of spending submissions to learn things.

The remaining gap to offline is the coverage story from submission 1: dev has 34.2% candidate
coverage against test's 6.5%, and the popularity half of the fusion loses most of its signal on
test while the BM25 half does not.

## Submission 3 · MIND point-in-time ranker — leaderboard 2026-08-23

`data/processed/mind_prediction_v3.zip`, from `scripts/submit_mind_v3.py`.
Scorer output: `{"AUC": 0.5554, "MRR": 0.2534, "nDCG@5": 0.2636, "nDCG@10": 0.3200}`, 725 s.

| Submission | Model | Offline (dev) | Leaderboard | Offset |
|---|---|---:|---:|---:|
| 1 | popularity | 0.5318 | 0.5036 | −0.028 |
| 2 | BM25 + popularity fusion | 0.5565 | 0.5258 | −0.031 |
| 3 | point-in-time GBDT | **0.6454** | **0.5554** | **−0.090** |

Submission 3 improved on submission 2 by +0.030 on every metric, so the progression is real.
But the offset tripled, and that is the more instructive result.

### A claim that was wrong, and the correction

This document previously stated that the harness was "calibrated" and that offline figures
predict "both the direction *and* roughly the size" of a leaderboard move. That was inferred
from two submissions which happened to share a model family. It did not hold for a model built
on different features, and it should not have been stated as a general property from two points.

### The cause, measured

Training click events end **14 Nov 23:59**. Dev is 15 Nov. The test split is 16–22 Nov. A
24-hour lookback from a dev impression still reaches training data; the same lookback from any
test impression reaches a window containing no events at all.

Fraction of candidates for which each feature is exactly zero, measured over 20,000 sampled
impressions per split (742,391 dev candidates, 787,307 test candidates):

| Feature | dev | **test** |
|---|---:|---:|
| `pop_24h` | 57.7% | **100.0%** |
| `pop_1h` | 99.7% | **100.0%** |
| `ctr_24h` | 57.7% | **100.0%** |
| `pop_total` | 54.8% | 93.5% |

Three of the ten features are identically zero for **every** test candidate, while on dev two of
them carried signal 42% of the time and the model split on them. `pop_total` is additionally
zero for 93.5% of test candidates, which is the same coverage figure recorded for submission 1.

This is not leakage — no future information was used. It is **train/serve skew**: the features
exist at serving time but are degenerate there, and the validation split could not reveal it
because dev sits adjacent to the training window while the test split never does.

## Gap-aware model selection — 2026-08-23

```bash
PYTHONPATH=. .venv/bin/python scripts/gap_aware_mind.py
```

A validation protocol that reproduces the test split's temporal position. Rolling counts are
fitted on events strictly before **13 Nov**; the ranker trains on **14 Nov** impressions and is
evaluated on **15 Nov** impressions — two and three days past the last counted event, against
the test split's two to eight. Both sides therefore experience the degenerate-feature regime,
and a model that depends on it is penalised here rather than on the leaderboard.

Evaluated on 73,152 impressions, crossing feature set against semantic representation so the
two proposed changes are separated rather than confounded:

| Configuration | AUC | nDCG@10 |
|---|---|---|
| v3 features + LSA (the submitted model) | 0.6050 [0.6028, 0.6073] | 0.3668 [0.3643, 0.3692] |
| v4 features + LSA | 0.6050 [0.6028, 0.6073] | 0.3668 [0.3643, 0.3692] |
| v3 features + MiniLM | 0.6447 [0.6425, 0.6466] | 0.4046 [0.4023, 0.4069] |
| **v4 features + MiniLM** | **0.6447 [0.6425, 0.6466]** | **0.4046 [0.4023, 0.4069]** |

**The representation is what matters.** MiniLM beats LSA by +0.040 AUC with disjoint intervals.
LSA retained 12.2% of the variance of the term-document matrix; a 384-dimensional sentence
encoder is a materially better representation of the same titles and abstracts.

**Dropping the dead features changed nothing here — and that confirms the diagnosis.** Under
the gap-aware protocol those three features are constant, and a gradient-boosted tree cannot
split on a constant, so the model already ignores them. They caused harm only on the
dev-adjacent protocol, where they varied. They are nonetheless removed from the shipped model,
because the final fit trains on impressions adjacent to its own counts, which is exactly the
condition that recreates the skew.

**Note the protocol is still optimistic.** Its evaluation day sits two to three days past the
counts; the test split reaches eight. The *relative* comparison between representations is what
it is designed to support, not an absolute forecast of the leaderboard.

## Submission 4 · MIND, MiniLM semantics + skew-resistant features — 2026-08-23

```bash
PYTHONPATH=. .venv/bin/python scripts/encode_mind_minilm.py   # one-off, cached
PYTHONPATH=. .venv/bin/python scripts/gap_aware_mind.py       # model selection
PYTHONPATH=. .venv/bin/python scripts/submit_mind_v4.py       # build the submission
```

Two changes from submission 3, each selected under the gap-aware protocol rather than on dev.

**Features (7, down from 10):** `pop_total`, `ctr_total`, `bm25`, `semantic`, `slate_size`, `cat_affinity`, `history_len`.
The three time-windowed count features are removed — they are zero for 100% of test candidates.

**Semantic representation:** `all-MiniLM-L6-v2`, 384 dimensions, replacing 128-component LSA.
Encoding all 125,590 articles took 2,255 s on CPU (56 articles/s), cached to
`data/processed/mind_minilm.npz` (93.6 MB) so it runs once.

**Counts:** built from train **and** dev events. Dev precedes test, so this remains
point-in-time sound, and it narrows the gap between the last counted event and the test window.

Expected gain, from the gap-aware protocol: **0.6447 [0.6425, 0.6466]** against
0.6050 [0.6028, 0.6073] for the configuration submitted as v3 — disjoint intervals.

| Quantity | Value |
|---|---:|
| Ranker fitted on | 230,117 impressions, 8,584,442 candidate rows |
| Test impressions scored | 2,370,727 |
| Wall time, scoring pass | 2,979 s (~50 min) |
| Peak RSS | ~1.8 GB |
| `mind_prediction_v4.zip` | 107.6 MB |

Offline validation passed: 2,370,727 lines, 2,370,727 distinct impression ids, zero duplicates,
every rank list a permutation of 1..N.

**The in-sample AUC of 0.7258 reported in the run log is not a generalisation estimate** and is
recorded only to confirm the fit executed. The honest out-of-sample figure for this
configuration is the gap-aware 0.6447 [0.6425, 0.6466], and even that is optimistic: its
evaluation day sits two to three days past the counts where the test split reaches eight.

### Leaderboard result — submission 4, MIND Codabench 13967, 2026-08-27

Scorer output: `{"AUC": 0.5714, "MRR": 0.2669, "nDCG@5": 0.2795, "nDCG@10": 0.3354}`, 455 s.
Submission id 903452.

| # | Model | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---:|---:|---:|---:|
| 1 | popularity | 0.5036 | 0.2249 | 0.2305 | 0.2860 |
| 2 | BM25 + popularity fusion | 0.5258 | 0.2405 | 0.2477 | 0.3038 |
| 3 | point-in-time GBDT, LSA | 0.5554 | 0.2534 | 0.2636 | 0.3200 |
| 4 | **MiniLM + skew-resistant features** | **0.5714** | **0.2669** | **0.2795** | **0.3354** |

Four submissions, monotonic improvement on all four metrics, total gain **+0.0678 AUC** over the
baseline.

### Did the gap-aware protocol work? Partly — and the residual is instructive

| Configuration | gap-aware AUC | leaderboard AUC | offset |
|---|---:|---:|---:|
| v3 features + LSA (submission 3) | 0.6050 | 0.5554 | −0.0496 |
| v4 features + MiniLM (submission 4) | 0.6447 | 0.5714 | −0.0733 |

**Direction: correct.** The protocol predicted MiniLM would beat LSA, and it did.

**Magnitude: overestimated.** It predicted +0.0397 and the leaderboard delivered **+0.0160**,
about 40% of the forecast. So the protocol is a better guide than the dev split — which had
predicted submission 3's configuration at 0.6454 against an actual 0.5554, an offset of 0.090 —
but it is not a calibrated predictor either, and its own offsets are not constant.

**The reason was stated in advance rather than after the fact.** This document recorded, before
the submission was scored, that the gap-aware protocol "is still optimistic: its evaluation day
sits two to three days past the counts where the test split reaches eight." That is exactly the
shape of the residual error. Closing it would require a validation window spanning the full two-
to-eight-day range, which `MINDsmall_train` is too short to provide — the training file covers
six days in total.

**The honest summary of the whole calibration exercise:** offline figures on this pipeline
reliably predict the *direction* of a change and systematically overstate its *size*, by an
amount that grows with how far the test window sits from the training data. Two consecutive
offsets should never have been read as a calibration constant.

## EB-NeRD submissions — Codabench 2469

### Submission 1 · BM25 — leaderboard 2026-08-27

`data/processed/ebnerd_predictions.zip`, from `scripts/make_submission_ebnerd.py`.
Scorer output (`scoring_result.zip` → `scores.txt`), scoring time 8,593 s (2.4 h):

| Metric | Offline (`ebnerd_small` validation) | **Leaderboard** | Δ |
|---|---:|---:|---:|
| AUC | 0.5030 | **0.5110** | **+0.0080** |
| MRR | — | **0.3303** | — |
| nDCG@5 | — | **0.3633** | — |
| nDCG@10 | 0.4349 | **0.4462** | +0.0113 |

**The offset runs the other way on this dataset.** Every MIND submission came back *below* its
offline figure; EB-NeRD came back *above* it, by 0.008 AUC and 0.011 nDCG@10. Whatever produces
the MIND gap is a property of that dataset's temporal structure, not a property of the harness.
This is the clearest available evidence that the "calibration offset" recorded earlier was never
a constant of the evaluation code.

### The test window, measured — and it is further out than assumed

The EB-NeRD scorer returns a per-day breakdown, which reveals the scored window; confirmed
directly against the file:

```
ebnerd_small train      : 2023-05-18 07:00:01 .. 2023-05-25 06:59:58   232,887
ebnerd_small validation : 2023-05-25 07:00:02 .. 2023-06-01 06:59:59   244,647
ebnerd testset          : 2023-06-01 07:00:00 .. 2023-06-08 06:59:59  13,536,710
```

**Three contiguous seven-day windows.** Validation ends one second before test begins.

**Our models fitted rolling counts on `train` alone**, which places them seven to fourteen days
from the test window — against MIND's two to eight. That is what explains the EB-NeRD picture:
frozen popularity at AUC 0.4429, article recency dominating the importance table, and
rolling-versus-frozen showing no measurable difference, because at that distance no trailing
window reaches live data at all.

**But the gap was partly self-imposed, and that is the more useful finding.** `validation` is
*immediately adjacent* to the test window and was never used for the counts. Fitting on
`train ∪ validation` would have put the model beside the test window rather than a week from it,
and would plausibly have revived the very features measured as dead. This was not a design
decision — the window was never checked while the EB-NeRD models were being built. Recorded in
`SPEC.md` §7 as a known limitation.

Per-day AUC across the eight test days is stable: min 0.5005, max 0.5213, spread 0.0207. So the
result is not carried by any single day.

### Submission 2 · point-in-time GBDT with article recency — submitted, result pending

`data/processed/ebnerd_predictions_v2.zip` (229.9 MB), from `scripts/rerank_ebnerd.py` with
`scripts/resume_ebnerd.py` completing an interrupted pass. Uploaded 2026-08-27; **Codabench had
not published a score at the time of writing**, which course staff confirmed is acceptable.

Offline validation on `ebnerd_small` (120,000 impressions, impression-level 70/30 split):
**AUC 0.7084 [0.7054, 0.7112]** against 0.5030 for the BM25 model submitted first.

Given submission 1 came back 0.008 *above* its offline figure, the leaderboard result for this
one is not forecast here — a single reversed offset is no more a calibration constant than the
two consistent ones were.

## Q1 · Reproducible pipeline — 2026-08-22

```bash
make data      # PYTHONPATH=. .venv/bin/python scripts/build_pipeline.py
```

Raw archives → unified schema → temporal split → feature store, idempotent, ~1 s from
already-extracted archives. Manifest with seeds and boundaries:
`data/processed/feature_store/manifest.json`.

Splits, derived from each dataset's real timestamp range (N=1 test day, M=1 validation day):

| Dataset | train | val | test |
|---|---|---|---|
| MIND (`MINDsmall_train`) | 95,071 rows, 09–12 Nov 2019 | 31,624 rows, 13 Nov | 30,270 rows, 14 Nov |
| EB-NeRD (`ebnerd_small` train) | 192,884 rows, 18–23 May 2023 | 32,225 rows, 24 May | 7,778 rows, 25 May (to 07:00) |

`assert_disjoint` runs inside the build, not only in tests: train's maximum timestamp is
strictly below val's minimum, and val's below test's, on every rebuild. A silently overlapping
split would invalidate every number in this file, so it is checked where it is created.

## Q2 · BM25 lexical retrieval — 2026-08-22

```bash
PYTHONPATH=. .venv/bin/python scripts/eval_bm25_mind.py --limit 2000
```

Own BM25 (`src/lexical/`), k1=1.2, b=0.75, Lucene IDF variant, over `title + abstract`.
Log: `data/processed/bm25_mind.log`; metrics: `data/processed/bm25_mind_metrics.json`.

### Index

| Quantity | Value |
|---|---:|
| Documents (train ∪ dev news) | 65,238 |
| Unique terms | 60,909 |
| Postings | 1,776,607 |
| Mean document length (tokens) | 32.83 |
| Build time | 2.9 s |

### recall@K over the full 65,238-article corpus — the `n_recent` ablation

Query = concatenated title+abstract of the user's *n* most recent clicks. 2,000 dev
impressions; 1,941 scored, 59 skipped as cold-start (no history, so no query — skipped rather
than counted as misses, which would conflate "retriever failed" with "nothing to retrieve from").

| `n_recent` | recall@50 | recall@100 | recall@200 | wall time |
|---:|---:|---:|---:|---:|
| 1 | 0.00473 | 0.00740 | 0.01011 | 1,738 s |
| 5 | **0.00490** | **0.00973** | **0.01636** | 6,034 s |
| 20 | 0.00368 | 0.00750 | 0.01453 | 11,118 s |

**`n_recent`=5 wins at every K.** A single article's text is too narrow a query, and twenty
articles drift into a generic profile of the user rather than a description of what they are
about to read — while costing 6× the time.

**A correction, and the reason it matters.** An earlier run of this table on **2,000**
impressions reported a crossover: `n_recent`=1 appeared to win at K=50 (0.0057 vs 0.0047),
which had a tidy explanation about short queries being sharply on-topic. At 20,000 impressions
that reverses — 0.00473 vs 0.00490 — and the crossover disappears. The tidy explanation was
fitted to noise.

| K | best @2,000 | best @20,000 |
|---|---|---|
| 50 | `n_recent`=1 | **`n_recent`=5** |
| 100 | `n_recent`=5 | `n_recent`=5 |
| 200 | `n_recent`=5 | `n_recent`=5 |

The differences at K=50 are ~0.0002 in absolute recall, which is well inside what a
2,000-impression sample can resolve. **recall@K in this table carries no confidence interval,
and until it does, small gaps between rows should not be read as differences** — the same rule
the bootstrap enforces for the ranking metrics. This is the clearest instance in the project of
the harness catching a claim that a smaller run had made confidently and wrongly.

**Absolute recall is low, and that is the honest headline.** Finding the one clicked article in
a 65,238-article corpus succeeds 0.5–1.8% of the time. Random selection at K=50 would be
50/65,238 = 0.077%, so BM25 is ~7× better than chance — real signal, but weak. Content
similarity between what a user read and what they click next is a much weaker relation than
lexical retrieval assumes: news clicks are driven substantially by recency and prominence,
neither of which a bag of words can see.

### In-impression re-ranking (mode b) — all 73,152 dev impressions, `n_recent`=5

| Metric | Popularity | **BM25** | Δ |
|---|---:|---:|---:|
| AUC | 0.5318 | **0.5451** | +0.0133 |
| MRR | 0.2382 | **0.2538** | +0.0156 |
| nDCG@5 | 0.2460 | **0.2676** | +0.0216 |
| nDCG@10 | 0.3098 | **0.3289** | +0.0191 |

BM25 improves all four metrics. No bootstrap CI yet, so this is not yet a claim that it
"beats" popularity — that wording waits for the P2 harness.

There is reason to expect the leaderboard gap to be **wider** than the offline gap, which is
the opposite of the usual caution. Popularity lost 0.028 AUC from dev to test because its
coverage collapsed from 34.2% to 6.5%. BM25 scores text, and every test article has a title, so
its coverage is 100% on both splits — it has no equivalent cliff to fall off.

### Scale instrumentation (Q6 evidence)

| Operation | Cost |
|---|---:|
| Index build, 65,238 docs | 2.9 s |
| Corpus search, `n_recent`=1 (24 query terms) | 48 ms/query |
| Corpus search, `n_recent`=5 (110 terms) | 172 ms/query |
| Corpus search, `n_recent`=20 (280 terms) | 299 ms/query |
| In-impression re-rank | 17 ms/impression |
| Peak RSS | ~350 MB |

**Query cost grows with query length, roughly 1.4 ms per unique query term** — each term means
one postings traversal. This is the measured argument against long history queries: `n_recent`=20
costs 6× `n_recent`=1 and retrieves *worse* at K=50.

**Where this breaks at 10×, measured rather than projected.** In-impression re-ranking at
17 ms/impression extrapolates to **~11 hours** for the 2.37M-impression MIND test set, and
EB-NeRD's test set is 13.5M impressions — 5.7× larger again. A first implementation used a
linear scan over postings to find a document's term frequency; replacing it with a binary
search over the sorted postings list (`_tf_of`) was the difference between scaling with corpus
size and scaling with candidate count. The remaining cost is Python-level iteration, and the
next step is a forward index so scoring iterates a document's ~33 terms rather than the query's
110.

## Q2 (continued) · BM25 lexical retrieval — EB-NeRD — 2026-08-27

```bash
PYTHONPATH=. .venv/bin/python scripts/eval_bm25_ebnerd.py --limit 20000
```

Own BM25 (`src/lexical/`), same k1=1.2, b=0.75, Lucene IDF variant, over `title + subtitle`
(EB-NeRD's title/abstract analogue), Danish tokenisation (`lang="da"`, no English stemmer or
stoplist). Corpus is `ebnerd_small/articles.parquet`. Log: `data/processed/bm25_ebnerd.log`;
metrics: `data/processed/bm25_ebnerd_metrics.json`.

### Index

| Quantity | Value |
|---|---:|
| Documents (`ebnerd_small` articles) | 20,738 |
| Unique terms | 43,453 |
| Postings | 303,865 |
| Mean document length (tokens) | 15.99 |
| Build time | 0.8 s |

The corpus is 3.1× smaller than MIND's 65,238 documents (EB-NeRD's small bundle covers far
fewer distinct articles than its impression count would suggest), which is why every query below
is proportionally cheaper than the MIND equivalent at the same `n_recent`.

### recall@K over the full 20,738-article corpus — the `n_recent` ablation

Query = concatenated title+subtitle of the user's *n* most recent clicks, from
`ebnerd_small/validation/history.parquet`. 20,000 validation impressions, run directly at this
sample size — the earlier MIND ablation showed a claim made at 2,000 impressions reversing at
20,000, so that mistake is not repeated here.

| `n_recent` | recall@50 | recall@100 | recall@200 | wall time |
|---:|---:|---:|---:|---:|
| 1 | 0.00582 | 0.00946 | 0.01653 | 63 s |
| 5 | 0.00841 | 0.01379 | 0.02300 | 307 s |
| **20** | **0.00911** | **0.01663** | **0.02898** | 734 s |

**`n_recent`=20 wins at every K on EB-NeRD — the opposite ranking from MIND, where `n_recent`=5
won at every K.** This table carries no bootstrap CI (same caveat as the MIND ablation — recall@K
here is a point estimate, and the gaps between rows are not yet certified as real differences),
but the direction is consistent across all three K values, which the MIND crossover-that-reversed
was not. A plausible mechanism: EB-NeRD's per-impression candidate slates are pre-filtered and
narrower than MIND's, so a longer query window may cost less of the "generic profile" drift that
made `n_recent`=20 lose on MIND — this is a hypothesis, not yet a measured explanation, and is
flagged as such rather than asserted.

**Absolute recall is higher than MIND's in raw terms** (recall@200 of 0.02898 here at
`n_recent`=20, against MIND's 0.01636 at its best `n_recent`=5) — expected, since the corpus is
3.1× smaller and there are proportionally fewer wrong articles to rank ahead of the right one.
The more informative comparison is against chance: random selection at K=50 over 20,738 articles
is 50/20,738 = 0.24%, and BM25 at `n_recent`=20 gets 0.91% — **~3.8× chance**, a weaker multiple
than MIND's ~7×, despite the higher raw number. So BM25's *edge over guessing* is smaller here
even though its absolute hit rate is larger, because the smaller corpus also raises the chance
baseline it has to beat. Consistent with the project's running finding that content similarity
between read and next-clicked articles is a weak signal relative to recency and prominence.

### In-impression re-ranking (mode b) — all 244,647 validation impressions, `n_recent`=5

| Metric | Value |
|---|---:|
| AUC | 0.5031 |
| MRR | 0.3196 |
| nDCG@5 | 0.3515 |
| nDCG@10 | 0.4350 |

This AUC (0.5031) lands within 0.0001 of the BM25 offline figure already reported for EB-NeRD
submission 1 (0.5030, from `make_submission_ebnerd.py`'s independent codepath) — a useful
cross-check that the two BM25 pipelines agree, since they share the scorer but not the
harness code around it.

### Scale instrumentation (Q6 evidence)

| Operation | Cost |
|---|---:|
| Index build, 20,738 docs | 0.8 s |
| Corpus search, `n_recent`=1 | 3.2 ms/query |
| Corpus search, `n_recent`=5 | 15.4 ms/query |
| Corpus search, `n_recent`=20 | 36.7 ms/query |
| In-impression re-rank (forward index) | 0.33 ms/impression |

Per-query cost is roughly proportional to corpus size at fixed `n_recent`: EB-NeRD's 20,738-doc
index costs ~11× less per query than MIND's 65,238-doc index at `n_recent`=5 (15.4 ms vs 172 ms)
for a 3.1× smaller corpus — consistent with the postings-traversal cost scaling with both query
length and corpus size, not corpus size alone. In-impression re-ranking at 0.33 ms/impression
extrapolates to **~1.2 hours** for EB-NeRD's 13.5M-impression test set, using the same
forward-index optimisation already applied for MIND (§ above) rather than the un-optimised
17 ms/impression figure.

## Q3 · Semantic retrieval on MIND — 2026-08-22

```bash
PYTHONPATH=. .venv/bin/python scripts/eval_semantic_mind.py --limit 20000 --recall-limit 3000
```

MIND ships no article embeddings — only TransE *entity* vectors — so article vectors are
computed with TF-IDF followed by truncated SVD (classical LSA). The alternative, a
sentence-transformer, means a ~2.5 GB download plus an encoding pass over 65K articles on a
machine with ~2 GB free (§11). LSA also makes the lexical-vs-semantic comparison sharper: it
is a linear factorisation of the *same* term-document matrix BM25 scores, so the comparison
isolates representation rather than model scale.

| Quantity | Value |
|---|---:|
| Embedding dimension | 128 |
| Explained variance (128 components) | 0.124 |
| Embedding build | 7.3 s |
| FAISS `IndexFlatIP` build | 0.2 s |

Explained variance of 0.124 is low, and worth stating plainly: 128
components recover only ~12% of the variance in a 60,909-term space. The representation is
lossy, which bounds how much the semantic model can be expected to do.

### Q3.3 · User-vector pooling ablation — recall@K over 65,238 articles

3,000 dev impressions, 2,917 scored, 83 skipped as unrepresentable.

| Pooling | recall@50 | recall@100 | recall@200 |
|---|---:|---:|---:|
| mean | **0.00406** | **0.00676** | **0.01037** |
| recency-weighted (τ=5) | 0.00248 | 0.00541 | 0.00721 |

**This contradicts the hypothesis it was built to test.** The argument for recency weighting was
that news decays in days, so recent clicks should describe a user better than old ones — and the
popularity result supports that decay elsewhere. Mean pooling nonetheless wins at every K, by
roughly 40%. The likely reason is that averaging *more* history suppresses noise: a single recent
click is one topic, while the mean of five is a more stable estimate of the user's interests. The
decay hypothesis was not wrong about news, it was wrong about which quantity is being estimated.
Recency weighting is therefore **not** used in the shipped model.

### Q3.5 · Lexical vs semantic, by slice — in-impression AUC with 95% CI

20,000 dev impressions. Cold-start threshold: ≤5 history clicks.
Head/tail: whether the clicked article is in the top popularity quintile.

| Slice | n | BM25 (lexical) | LSA (semantic) | Winner |
|---|---:|---|---|---|
| all | 20,000 | 0.5468 [0.5428, 0.5508] | 0.5588 [0.5548, 0.5625] | **semantic** |
| cold-start (≤5 clicks) | 3,456 | 0.5308 [0.5213, 0.5405] | 0.5379 [0.5278, 0.5472] | — overlap, no difference |
| warm (>5 clicks) | 16,544 | 0.5501 [0.5457, 0.5543] | 0.5632 [0.5584, 0.5677] | **semantic** |
| head articles | 5,172 | 0.5159 [0.5078, 0.5236] | 0.5604 [0.5532, 0.5678] | **semantic** |
| tail articles | 14,828 | 0.5575 [0.5528, 0.5620] | 0.5582 [0.5536, 0.5634] | — overlap, no difference |

**The measured story only partly matches the expected one.** The standard expectation is that
lexical wins on exact matches and semantic wins on cold-start and paraphrase. What the data says:

* **Semantic wins overall and on warm users** — with more history to pool, the latent
  representation beats term overlap.
* **Cold-start shows no significant difference.** This is the expectation's clearest failure.
  Semantic retrieval was supposed to help precisely here, and it does not: with ≤5 clicks
  there is too little to pool, and both models degrade to roughly the same place.
* **Head articles favour semantic decisively** (0.5604 [0.5532, 0.5678] vs
  0.5159 [0.5078, 0.5236]). Popular articles are covered by many near-synonymous
  headlines, which is exactly the case where term overlap fails and a latent space does not.
* **Tail articles show no significant difference** — rare articles have thin text and thin
  co-occurrence statistics, so neither representation has much to work with.

### Q4.2 · Beyond-accuracy (top-10 per impression)

Definitions in `src/eval/beyond_accuracy.py` — these terms are used loosely in the literature
and the numbers are meaningless without them.

| Metric | BM25 | Semantic |
|---|---:|---:|
| Intra-list diversity | 0.8474 | 0.8619 |
| Novelty (bits) | 16.192 | 16.199 |
| Catalogue coverage | 0.0388 | 0.0345 |
| Gini (exposure concentration) | 0.9056 | 0.9134 |

**Coverage is the alarming number.** Both models surface under 4% of the 65,238-article
catalogue in *any* top-10, and Gini above 0.90 says that exposure is piled onto a small
minority of those. A system with this profile is a popularity amplifier regardless of how its
AUC looks, and neither accuracy metric would ever reveal it. Diversity is high (~0.85) only
because MIND has 17 broad categories — it is a weak proxy and should not be read as evidence
of genuine variety.

## Q9 · Metrics with and without serving-unavailable features — 2026-08-22

```bash
PYTHONPATH=. .venv/bin/python scripts/ablation_serving_time.py
```

EB-NeRD provides this ablation directly. Columns present in train/validation and **absent from
test by construction**, confirmed by reading both schemas:

`article_id`, `article_ids_clicked`, `next_read_time`, `next_scroll_percentage`

`article_id` and `article_ids_clicked` are the labels. `next_read_time` and
`next_scroll_percentage` describe the user's *next* impression — future information the
organisers removed because it cannot exist when a recommendation is served.

60,000 EB-NeRD validation impressions:

| Model | AUC | nDCG@10 |
|---|---|---|
| **honest** — BM25 only | 0.5029 [0.5007, 0.5052] | 0.4347 [0.4325, 0.4370] |
| **leaky** — BM25 + `next_read_time` | 0.9629 [0.9618, 0.9641] | 0.9493 [0.9479, 0.9507] |

**AUC inflation: +0.4600.**

A chance-level model becomes an apparently near-perfect one — 0.50 → 0.96 — purely by touching
a column that cannot exist at request time. The leak modelled here is not exotic: it is what an
incautious feature pipeline produces by joining an impression-level `next_read_time` onto its
candidate rows. Nothing about the resulting number looks wrong from the outside, which is
precisely why Q9 asks for both rows.

**The shipped model is the honest one.** The leaky row exists to be disclosed, never used.

## Q9b · Frozen vs rolling popularity — the point-in-time ablation

```bash
PYTHONPATH=. .venv/bin/python scripts/rerank_mind.py
```

A second, sharper form of the Q9 question. `src/features/rolling.py` counts clicks **strictly
before** each impression's own timestamp, via binary search over per-article sorted event
times. The frozen variant counts the whole training window once and applies that number
everywhere.

Both feed an identical `HistGradientBoostingClassifier` over ten features, trained on
`MINDsmall_train` and evaluated on the strictly-later `MINDsmall_dev`:

| Popularity mode | AUC | nDCG@10 |
|---|---|---|
| **rolling** (strictly-before-*t*) | **0.6447 [0.6426, 0.6467]** | 0.3931 [0.3909, 0.3957] |
| frozen (whole window, applied everywhere) | 0.6058 [0.6035, 0.6079] | 0.3706 [0.3682, 0.3733] |

**Rolling beats frozen by 0.0389 AUC, intervals disjoint.**

**The direction is the interesting part.** Frozen popularity *leaks* — an early impression is
scored using clicks that happen after it — and yet it scores **worse**. Two effects run in
opposite directions and staleness wins: the frozen count is an average over a six-day window
applied to a dev day that lies outside it, while news popularity turns over in hours. So the
leak buys less than the drift costs. A model that cheated on the timeline still lost to one
that did not, which is a more useful argument for point-in-time features than "leakage is bad".

### Permutation importance (AUC drop when a feature is shuffled)

| `slate_size` | +0.1271 |
| `pop_total` | +0.0410 |
| `ctr_total` | +0.0405 |
| `cat_affinity` | +0.0223 |
| `ctr_24h` | +0.0187 |
| `pop_24h` | +0.0132 |
| `history_len` | +0.0121 |
| `semantic` | +0.0074 |
| `bm25` | +0.0056 |
| `pop_1h` | -0.0001 |

**`slate_size` ranking first is an artifact, not a finding.** It is constant within an
impression, so it cannot separate candidates inside one — and AUC here is computed per
impression. Shuffling it globally injects within-impression variation that the real feature
never has, which manufactures an importance score. Permutation importance is not group-aware,
so it cannot be read directly for a group-constant feature. The honest test is leave-one-out
refitting (`scripts/submit_mind_v3.py`), not this table.

**Leave-one-out confirms it.** Refitting without `slate_size` gives
0.6417 [0.6396, 0.6437] against 0.6454 [0.6433, 0.6474] with it — **intervals overlap, so there
is no significant difference**. A feature that permutation importance ranked first, by 3× the
next entry, turns out to contribute nothing measurable. The lesson generalises: permutation
importance is not group-aware, and for any feature constant within an evaluation group it
reports the noise it injects rather than the signal it carries. Feature attribution needs the
same interval discipline as every other number here.

Taking the rest at face value: **behavioural signal dominates content signal**. `pop_total`
(+0.0410) and `ctr_total` (+0.0405) are each roughly 6× `bm25` (+0.0056) and 5× `semantic`
(+0.0074). This is consistent with everything else measured here — news clicks are driven by
what is popular and current far more than by what resembles a user's reading history — and it
is the quantitative version of the low recall@K in Q2 and Q3.

`pop_1h` contributes nothing (−0.0001): a one-hour window is too narrow to accumulate counts
at MIND-small's volume.

## Q9c · EB-NeRD point-in-time reranker — and a correction to an earlier finding

```bash
PYTHONPATH=. .venv/bin/python scripts/rerank_ebnerd.py
```

EB-NeRD supports two features MIND cannot: **article age at impression** (`published_time`
exists) and the publisher's **provided word2vec vectors** (300-dim, all 125,541 articles).
120,000 validation impressions, impression-level 70/30 fit-evaluate split.

| Popularity mode | AUC | nDCG@10 |
|---|---|---|
| rolling (strictly-before-*t*) | 0.7084 [0.7054, 0.7112] | 0.5778 [0.5749, 0.5807] |
| frozen (whole window) | 0.7091 [0.7060, 0.7118] | 0.5790 [0.5761, 0.5818] |

**Intervals overlap — no significant difference.** This is the opposite of MIND, where rolling
beat frozen by 0.039 with disjoint intervals, and the reason is visible in the importance table.

### Permutation importance

| `slate_size` | +0.1903 |
| `age_hours` | +0.1251 |
| `semantic` | +0.0056 |
| `pop_total` | +0.0051 |
| `ctr_total` | +0.0041 |
| `bm25` | +0.0008 |
| `ctr_24h` | +0.0005 |
| `pop_24h` | +0.0002 |
| `pop_1h` | +0.0000 |
| `history_len` | +0.0000 |

`age_hours` is the second-ranked real feature (+0.1251) while every popularity and CTR feature
is near zero (`pop_total` +0.0051, `ctr_total` +0.0041). **Rolling versus frozen cannot matter
much when popularity itself barely matters** — the model is driven by recency. That is a
coherent explanation for the null result rather than an excuse for it.

(`slate_size` tops this table too, and is the same artifact documented in Q9b: constant within
an impression, so permutation manufactures its importance.)

### The correction

RESULTS.md and the design note previously reported that **popularity is anti-predictive on
EB-NeRD (AUC 0.4429)** and framed it as the dataset's defining property. That measurement stands
— raw frozen popularity *as a standalone scorer* really does rank below chance, and the
logging-policy explanation for it is still the right one.

What was wrong was the implication drawn from it: that EB-NeRD is intrinsically hard to rank.
It is not. With article recency added, AUC goes from **0.5030** (BM25 alone, our submitted
model) to **0.7084**. The missing ingredient was never a better text
model — it was a feature we had not computed. EB-NeRD's candidate slates are already filtered to
fresh articles, and *how* fresh is what separates them.

This is the single largest gap between what we measured and what the data supported, and it
came from not asking what other columns the dataset offered before concluding the task was hard.
