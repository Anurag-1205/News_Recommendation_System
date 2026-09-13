# PLAN.md — A2 · Learning from Click-Logs (team of 2)

Companion to `CLAUDE.md` (how to work), `SPEC.md` (what each component is, how it is verified)
and `CONTEXT.md` (what we have decided so far, and what is in flight). This file is **what to
build, in what order, by whom, by when**. Update the status column as phases land. Do not
silently reorder phases: if the order changes, log the reason in `CONTEXT.md`.

A1 is finished and frozen on `main` at `be15ee6`. A2 lives on branch `a2-click-logs`.

---

## 0. The deadline, stated plainly

| Fact | Value |
|---|---|
| Plan written | **Fri 11 Sep 2026** |
| A2 due | **Sun 20 Sep 2026** (report on Moodle, code on GitHub) |
| Calendar days left | **9** |
| Realistic code freeze | **Sat 19 Sep, 18:00**. The 20th is buffer and must not hold planned work |
| Team | **Anurag Kaushal**: P1, P2 (both done), P5 joint, P6 joint. **Aayush Pandey**: all of P3 (NRMS baseline, improvement, ablation, paired bootstrap), P4, P5 joint, P6 joint. See §2 and `CONTEXT.md` C-005, C-019 |
| Compute | **Kaggle: 2× T4 GPU, fp16, RAM not a constraint.** The laptop (7 GB, no GPU) is for dev, tests and toy-scale runs only (C-004) |

Consequences:

1. **Large-test inference is on the critical path, not a final step.** In A1 the EB-NeRD test pass
   (13.5M impressions) was killed partway on the laptop. In A2 full-scale runs go to Kaggle, but a
   13.5M-impression pass is still hours of work, and Kaggle sessions have a time limit. The first A2
   submission on both leaderboards has to land by **Wed 16 Sep**, even if it is weak.
2. **The NRMS baseline is the biggest unknown.** It needs a GPU (Kaggle T4, fp16), an external
   codebase and a different data loader. Since C-019 it is Aayush's, and it should start as soon as
   Aayush is unblocked: Kaggle background jobs, not a step after everything else.
3. **The critical path is now Aayush's NRMS on Kaggle (P3.1), alongside P4.** P1 and P2 are done
   (C-018), and Q3's comparisons need the NRMS score files. P5 is joint so Anurag can carry the
   submission runs while Aayush is on P3/P4 (C-019).
4. **Grading is never on rank.** It is on correctness, system design, ablation rigour, scale
   analysis and note clarity. If a day is being spent chasing rank, the plan has gone wrong.

---

## 1. What A1 already gives us (do not rebuild)

| A1 asset | Where | A2 use |
|---|---|---|
| Unified schema, readers, temporal split | `src/pipeline/` | unchanged |
| BM25 candidate generator (own index) | `src/lexical/` | stage 1 (Q2.1) |
| Semantic candidates: FAISS flat, user vector | `src/semantic/`, `scripts/encode_mind_minilm.py` | stage 1 (Q2.1) |
| recall@K harness, mode (a) | `scripts/eval_bm25_*.py`, `scripts/eval_semantic_*.py` | stage-1 numbers |
| Point-in-time popularity / CTR | `src/features/rolling.py` | article features (Q1.3) |
| GBDT reranker (sklearn `HistGradientBoostingClassifier`, pointwise) | `scripts/rerank_*.py`, `scripts/submit_mind_v4.py` | starting point for Q2 |
| Metrics, beyond-accuracy, bootstrap CIs, slices | `src/eval/` | Q5 |
| Codabench writer + validator | `src/eval/submission.py` | Q5 |
| Gap-aware validation protocol | `scripts/gap_aware_mind.py`, `SPEC.md` §7 | model selection |
| Leakage tests | `tests/test_no_leakage.py`, `tests/test_rolling.py` | extend for every new feature |

**What is missing, and every item is required by A2:**

- A **paired** bootstrap on the per-impression *difference* between two systems. A1's
  `src/eval/bootstrap.py::compare` only checks whether two independent CIs overlap. Q3.4
  requires a paired CI that excludes zero.
- Features are computed **inside scripts**. They have to move into `src/features/` as
  functions shared by training and serving. A1's train/serve skew (submission 3) came from
  exactly this kind of divergence.
- Recency-weighted history, session features, dwell time, position, freshness for MIND,
  category match.
- The official baseline (NRMS), a serving benchmark (`make bench`) and `make eval`.

A1 best leaderboard results, as the reference point: **MIND AUC 0.5714**, **EB-NeRD AUC 0.5110**.

---

## 2. Team and ownership (revised 11 Sep, `CONTEXT.md` C-019; original split C-005)

| Phase | Owner | Scope |
|---|---|---|
| **P0** Setup | Anurag Kaushal & Aayush Pandey | clean-clone check (`make env && make test` on Aayush's machine); Kaggle verification on both accounts |
| **P1** Behavioural features (Q1) | Anurag Kaushal | **done** (C-006 – C-014) |
| **P2** Two-stage reranker (Q2) | Anurag Kaushal | **locked** (C-018): `src/rerank/config.FINAL`. Open: D1 framing (b), retrieving a top-K from the corpus as Q2.1 asks |
| **P3** Baseline + improvement (Q3): P3.1, P3.4a, P3.2–3.4 | **Aayush Pandey** | NRMS reproduction on Kaggle (2× T4, fp16) on both datasets; the paired bootstrap CI harness; the one principled change and its ablation |
| **P4** Serving & scale (Q4) | Aayush Pandey | index memory, p99 latency, SLA cost model, 10× breakdown |
| **P5** Extended eval + Codabench (Q5) | **Anurag Kaushal & Aayush Pandey** | diversity/novelty/coverage, cold/warm and head/tail slices, full test-set submission runs on Kaggle, screenshots |
| **P6** Design note + ship (Q6–Q9) | Anurag Kaushal & Aayush Pandey | report, checklist, final push |

**The shape of the split (revised).**

- **Anurag** built the behavioural features and the reranker (P1 and P2, both done).
- **Aayush** owns the whole Q3 track: reproducing NRMS, the principled change, its ablation, and
  the paired-bootstrap harness that judges it. He also owns serving (P4).
- **Evaluation and submissions (P5) are shared,** so the full-scale Kaggle runs do not queue
  behind P3 and P4.

**Independent check on Q3 claims.** The original split kept the builder and the judge of a model
apart (`CLAUDE.md` rule 2). With all of P3 on one person that is lost for Q3, so **Anurag reviews
every Q3 "beats" claim** (the paired CI and the commands that produced it) before it goes into
`RESULTS.md`.

**The handoff is still a scores file per system per split:** `impression_id, article_id, score`
as Parquet.

- Producers: Anurag's reranker (`config.FINAL`) and Aayush's NRMS and its ablation rows.
- Consumers: the paired bootstrap (P3), `make eval` and the submission writer (P5).
- Neither person imports the other's code. Pin this format in `SPEC.md` before the first NRMS
  scores are written.

**Order of work after C-019.** The reranker is locked, so its score files can be produced now from
`config.FINAL`. The NRMS score files arrive with P3.1. Harnesses are still built and tested first
against existing outputs, so when a model's scores land, producing numbers is a re-run and not
new work.

| When | Who | Builds | Ready for |
|---|---|---|---|
| 12–13 Sep | Aayush | paired bootstrap + its oracle test (P3.4a) | P3.2 (starts 15 Sep) |
| 12–15 Sep | Aayush | NRMS on Kaggle, EB-NeRD demo → both datasets (P3.1) | P3.2–3.4 |
| 13–14 Sep | Anurag | `make eval`: all metrics, both slices, CIs, on the locked reranker's scores | P5 |
| 14–15 Sep | Anurag | resumable Kaggle test-inference + submission pipeline for `config.FINAL` | first A2 submission, **Wed 16 Sep** |
| 14–17 Sep | Aayush | `make bench`: stage 1 + the locked reranker (memory, p99, cost) | P4 |

---

## 3. Phase overview

| Phase | Covers | Window | Owner | Exit gate | Status |
|---|---|---|---|---|---|
| **P0** Setup | Part 0 | 11 Sep | Anurag & Aayush | branch + docs; `make test` green on Aayush's clean clone; Kaggle CLI + 2× T4 verified on both accounts; team registered on both Codabench comps; NRMS runs on toy data | ☐ |
| **P1** Behavioural features | Q1 | 12–13 Sep | Anurag | features in `src/features/`; leakage test covers each one | ☑ done 11 Sep |
| **P2** Two-stage reranker | Q2 | 13–15 Sep | Anurag | before/after-rerank table, both datasets, with CIs | ☑ locked 11 Sep (C-018); D1 framing (b) open |
| **P3.1** Reproduce NRMS | Q3.1 | 12–15 Sep | **Aayush** | NRMS on both datasets (Kaggle, float32 — C-022); our number vs the published one; score files written | ☑ done 14 Sep (C-025) |
| **P3.4a** Paired bootstrap harness | Q3.4 | 12–14 Sep | **Aayush** | oracle test passes: a constructed Δ is recovered, and a zero-Δ CI covers 0 (replaces or adopts the provisional `src/rerank/common.paired_delta`, C-015) | ☑ done 14 Sep (C-026): adopted + moved; `make paired` |
| **P3.2–3.4** Improve + ablate | Q3.2–3.4 | 15–17 Sep | **Aayush** | paired 95% CI excludes zero, or an honest null; each claim reviewed by Anurag | ☐ |
| **P4** Serving & scale | Q4 | 14–17 Sep | Aayush | `make bench`: memory, p99, cost/1k queries, 10× argument | ☐ |
| **P5** Extended eval + submit | Q5 | 13–18 Sep | **Anurag & Aayush** | `make eval`; 2 slices; first submission 16 Sep; final submissions + screenshots 18 Sep | ☐ |
| **P6** Design note + ship | Q6–Q9 | 18–19 Sep | Anurag & Aayush | ~6-page PDF; README reproduce verified from a clean clone | ☐ |

P4 starts early on A1 outputs (§2). P5 can start at once: the reranker is locked in `config.FINAL`.

---

## Phase 0 — Setup · **Fri 11 Sep · Anurag & Aayush**

1. Branch `a2-click-logs`. `CLAUDE.md`, `PLAN.md`, `CONTEXT.md` and `AI_USAGE.md` are tracked so
   both agents see the same context (`CONTEXT.md` C-002). *(Anurag, done.)*
2. **Clean-clone check (Aayush):** clone, run `make env && make test`, and confirm all tests pass
   on Aayush's machine. A failure here is an environment difference and gets fixed first.
3. **Kaggle verification (both):** on each account, the CLI authenticates, a kernel sees 2× T4,
   and an fp16 matmul runs. Anurag's account is already verified for the CLI (C-004).
4. **Codabench:** check whether the competitions support teams. Register the team, or decide which
   account submits, and log it. Aayush runs the submissions (P5).
5. **Kaggle data (D9):** get `MINDlarge_train/dev/test` and the EB-NeRD large bundle attached as
   Kaggle datasets, and verify them against the official files (row counts, file sizes, a hash of
   each parquet/tsv) before any number is computed from them. A1 fitted MIND on `MINDsmall_train`
   only.
6. **Kaggle workflow:** repo code is uploaded as a private Kaggle dataset and run by kernels pushed
   with the `kaggle` CLI (installed via `requirements.txt`, credentials in `~/.kaggle/kaggle.json`).
   Share the datasets between Anurag's and Aayush's accounts. Each account has its own GPU quota:
   Aayush's goes to NRMS training (P3), and Anurag's to the P5 test-set inference runs (C-019).
7. **NRMS smoke test (Aayush, C-019):** clone `ebnerd-benchmark` into `external/` (gitignored, commit
   hash pinned in `CONTEXT.md`), check which framework it uses, and run NRMS on EB-NeRD demo on a
   Kaggle T4 with fp16. This settles D3 below. It is the riskiest unknown and must surface on day 1.
8. Pin the scores-file format (§2) in `SPEC.md` (both agree it, since it is the handoff contract).

**Exit gate:** both machines run `make test` green; both Kaggle accounts verified; NRMS has run
end to end on demo data on Kaggle.

---

## Phase 1 — Behavioural features (Q1) · **12–13 Sep · Anurag**

Every feature is a function of `(user, candidate, t)` that reads only events **strictly before
t**. Each reviewable unit is one feature family, landed with its test.

| Q1 item | Feature | MIND | EB-NeRD | New? |
|---|---|---|---|---|
| 1.1 history | clicked-title BM25, embedding cosine | ✓ | ✓ | A1 has it |
| 1.1 history | click count, history length | ✓ | ✓ | small |
| 1.1 history | **recency-weighted** user vector / category profile (exp. decay, half-life h) | ✓ | ✓ | **new** |
| 1.1 history | category distribution; candidate-category match | ✓ | ✓ | **new** |
| 1.2 session | within-session clicks so far | proxy (time gap) | `session_id` | **new** |
| 1.2 session | dwell time (history `read_time`) | ✗ absent | ✓ | **new** |
| 1.2 session | position of candidate in the impression list | ✓ ⚠ | ✓ ⚠ | **new** |
| 1.3 article | rolling popularity / CTR | ✓ | ✓ | A1 has it |
| 1.3 article | freshness = t − publish time | proxy: first-seen ts | `published_time` | MIND new |
| 1.4 boundary | leakage assertion for every feature above | | | **extend test** |

**Oracles:** a 20-event toy log whose feature values are computed by hand, covering decay
weights, an event at exactly *t* (excluded), an out-of-order event (rejected), and a session
boundary. Write it before the feature code.

**⚠ Position is a leak risk.** If a dataset lists the clicked items first, or the test file orders
candidates differently from train, position is label information and not display position. Check
the position→label correlation on train, and the ordering convention in the test file, before
using it. Log the result either way.

**Serving-unavailable features (Q9):** EB-NeRD `next_read_time`, `next_scroll_percentage`, and
anything derived from the same impression's clicks. Tag each feature `serving_ok: bool` in its
definition, so the "with vs without" ablation row comes from a flag and not from hand-editing.

---

## Phase 2 — Two-stage reranker (Q2) · **13–15 Sep · Anurag**

1. **Stage 1:** A1's BM25 ∪ ANN retrieve top-K, K ∈ {100, 200}. Report recall@K (mode a, A1 harness).
2. **Stage 2:** GBDT over the P1 features, with the stage-1 scores and ranks as features.
3. Evaluate both framings (D1): the retrieved top-K lists, and the impression candidates, which
   is mode (b) and what both leaderboards score.
4. **Q2.4 table:** AUC, MRR, nDCG@5, nDCG@10 *before* (stage-1 score alone) and *after* the rerank,
   with bootstrap CIs, on both datasets.
5. **Model selection uses the gap-aware protocol** (`SPEC.md` §7), not the adjacent dev split.
   A1 submission 3 lost 0.090 AUC to that mistake.

**Exit gate:** the Q2.4 table in `RESULTS.md`, both datasets, commands included.

---

## Phase 3 — Baseline reproduced, then beaten (Q3)

### P3.1 · Reproduce · **12–15 Sep · Aayush** (Kaggle 2× T4, fp16)

1. NRMS from `ebnerd-benchmark` on EB-NeRD, and the same model on MIND (D3).
2. "Reproduced" means *our* number next to the *published* number, with the gap explained. It
   does not have to match exactly, but the gap must be stated and reasoned about.
3. Output: a scores file (§2) for validation and test, on both datasets.

### P3.2–3.4 · Improve and ablate · **15–17 Sep · Aayush**

Aayush owns the change, the ablation runs and the paired bootstrap CI harness (P3.4a, 12–14 Sep).
Anurag reviews every "beats" claim before it is recorded (C-019).

1. **One** principled change (D4), chosen from what A1 measured. On EB-NeRD, recency dominated
   and popularity barely mattered (A1 Q9c). Recency-weighted pooling won on EB-NeRD and lost on
   MIND. Freshness or category-aware signals are therefore the evidence-backed candidates.
2. **Ablation:** baseline, baseline + change, and each component of the change removed.
3. **Paired bootstrap:** resample *impressions* and compute the metric difference on the same
   resampled set for both systems. Report Δ with a 95% CI. "Beats" is written only if the CI
   excludes zero. **Build and test the paired bootstrap before P3.2 starts** (Aayush, P3.4a). It
   is the oracle.
4. If the CI includes zero, report it as a null result with the CI. That is still full marks for
   rigour, and it is not a failure to hide.

---

## Phase 4 — Serving & scale (Q4) · **14–17 Sep · Aayush**

`make bench` produces, on this machine, with the exact command recorded in `RESULTS.md`:

1. **Memory:** FAISS index bytes, BM25 index bytes, feature-store bytes, peak RSS at serve time.
2. **Latency:** p50/p95/p99 for one user request = stage 1 + feature fetch + GBDT scoring, over
   ≥1,000 sampled requests, with the warm-up excluded and stated.
3. **Cost/QPS:** measured single-core throughput → cores needed for a target QPS at p99 < 100 ms →
   cost per 1,000 queries at a stated $/vCPU-hour. Every assumption is written down.
4. **10× argument:** which resource runs out first (index RAM, feature-store lookups, GBDT CPU),
   argued from the measured numbers in RUM terms (reads, updates, memory).

---

## Phase 5 — Extended eval + submissions (Q5) · **13–18 Sep · Anurag & Aayush**

1. `make eval`: all metrics (AUC, MRR, nDCG@5, nDCG@10, diversity, novelty, coverage) for the full
   two-stage pipeline, with bootstrap CIs.
2. Slices: **cold-start vs warm** (A1: ≤5 history clicks) and **head vs tail** (top popularity
   quintile). Both are required, not optional.
3. **First A2 submission on both leaderboards by Wed 16 Sep.** Final submissions by the 18th.
   Screenshots are taken immediately after each one.
4. Full test inference runs on Kaggle and writes its output in chunks, so a run cut off by the
   session limit resumes instead of restarting. This is the A1 EB-NeRD lesson.

---

## Phase 6 — Design note + ship (Q6–Q9) · **18–19 Sep · Anurag & Aayush**

Design note as a PDF, ~6 pages, 11pt, 1-inch margins: what we built and why; baseline vs improved
with the ablation and CIs; serving and scale findings; where it breaks at 10×.

**Ship checklist:**

- [ ] Code pushed; no `*.zip *.pt *.ckpt data/ __pycache__/` anywhere in history
- [ ] `README.md` one-command reproduce, verified by a fresh clone on Aayush's machine
- [ ] `SPEC.md`, `RESULTS.md`, `AI_USAGE.md` complete; AI-generated vs human-written code marked
- [ ] Leakage test green; the "with vs without serving-unavailable features" row is in the note
- [ ] Both leaderboard screenshots in the note
- [ ] Chat exports and prompts for **both** team members collected (Q7.4)
- [ ] No force-push after the deadline

---

## 4. Descope ladder: cut in this order if behind (decide end of **Wed 16 Sep**)

1. Drop the retrieved-top-K framing in D1 and keep impression-candidate reranking only (state it).
2. Drop the MIND session proxy; session features on EB-NeRD only.
3. Reproduce NRMS on EB-NeRD only if it will not run on MIND in time, and state why.
4. Keep the cost model to one hardware assumption.

**Never cut:** the temporal split, the leakage test, paired bootstrap CIs, both leaderboard
submissions, the serving benchmark, `AI_USAGE.md`, or the design note.

---

## 5. Open decisions (settle in plan mode; record the answer in CONTEXT.md)

| # | Fork | Options | Recommendation |
|---|---|---|---|
| D1 | What "two-stage" means when leaderboards score fixed candidates | (a) rerank the impression's candidates only; (b) rerank the retrieved top-K; (c) both | **(c)**: (b) shows the two-stage system, (a) is leaderboard-comparable |
| D2 | Reranker objective | sklearn HistGBDT pointwise (A1, no new dependency) vs LightGBM `lambdarank` grouped by impression | **LightGBM lambdarank**, with pointwise kept as a comparison row |
| D3 | Baseline to reproduce | NRMS from `ebnerd-benchmark` on both datasets vs a separate MIND baseline | one NRMS implementation for both, so the comparison is like-for-like |
| D4 | The one principled change | freshness weighting / category-aware features / recency-weighted user encoding | decide after P3.1 numbers exist; A1 evidence favours freshness |
| D5 | MIND training data | small vs large train | develop on small locally, final fit on `MINDlarge_train` on Kaggle |
| D6 | MIND freshness (no publish time) | first-seen timestamp in logs before t vs drop the feature | first-seen, point-in-time |
| D7 | EB-NeRD fit window | train only (A1) vs train ∪ validation, which is adjacent to test | train ∪ val for the final fit, selected under the gap-aware protocol |
| D8 | Serving SLA and cost basis | target QPS, $/vCPU-hour source, benchmark hardware | p99 < 100 ms (the brief's example), one cited cloud price; state whether latency was measured on CPU or a T4 |
| D9 | Where Kaggle gets the data | community mirrors already on Kaggle (e.g. `tinkhil/ebnerd-large`, `hieunm21/mindlarge`) vs our own private upload of the official files | mirrors, **only after** they match our official copies on row counts and hashes; fall back to uploading if any check fails |
