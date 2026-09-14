# CONTEXT.md — shared team memory for A2

This file is how the two of us, and our two coding agents, stay in sync without re-deriving each
other's work. **Every agent reads this file at the start of a session, before touching code.**
The rules for writing to it are in `CLAUDE.md` §3.

Three sections:

1. **Current state.** A short snapshot that is overwritten, not appended: who is on what, what is
   blocked, what to pick up next. Keep it to roughly 15 lines.
2. **Decision log.** Append-only. One entry per decision. Never edit or delete an old entry; to
   reverse one, add a new entry that supersedes it.
3. **Inherited from A1.** Facts established in A1 that A2 code must respect.

---

## 1 · Current state

_Last updated: 2026-09-15 by Anurag (agent: Claude Code)_

| | |
|---|---|
| Branch | `a2-click-logs`; origin tracks every commit (the agent commits and pushes on Aayush's instruction; no force-push) |
| Phase | **P1, P2 done/locked (Anurag). P3 done (C-025–C-030); the EB-NeRD "beats" line is REVIEWED AND VERIFIED (C-033), so Aayush is unblocked. P4 done (C-031).** P5, P6 open. Open for Q2: D1 framing (b) — note P4 measured framing (b) end to end, so the retrieval path exists |
| Team | Per C-019/C-027/C-032. Anurag: **P5 alone**, reviews Q3 "beats". Aayush: P6 joint, reviews Q5 claims. Kaggle: Anurag's account for P5 inference; Aayush main `aayushpandey18602` (17.8 h left) and alt `aayushpandey602` (27.6 h left) |
| Anurag Kaushal | **P5 part 1 done (C-035)**: `config.FINAL` scores for both splits, `make eval` (was a stub) with both slices, and the reranker-vs-NRMS pairings — the reranker beats NRMS on both datasets. **Remaining: test-set inference + the two Codabench uploads + screenshots**, plus Q9 (C-034). Earlier today: **hand-over archive installed**: sha256 matches, 27 files under `data/` (gitignored), `read_scores` on the MIND NRMS file returns 2,740,998 rows. **C-019 review of the EB-NeRD "beats" line: reproduction step done, verdict not yet recorded.** `make paired` on Aayush's two score files reproduced his record **bit-identically** (Δ AUC +0.0074 [+0.0066, +0.0081], all four verdicts "beats"); my record: `data/processed/paired/ebnerd_row2_vs_row1_review_anurag.json` (his file untouched). Remaining review checks before sign-off, then `RESULTS.md` Q3.2–3.4 gets the reviewer line. Then **P5 end to end (C-032)**: `config.FINAL` score files, `make eval`, test-set inference, both submissions, screenshots. Also open from P0: Kaggle datasets for the large files (5–6) |
| Aayush Pandey | **P3 and P4 done.** P5 handed to Anurag (C-032); hand-over archive sent. Next: P6 — the Q3 and Q4 sections of the note; the reranker-vs-NRMS `make paired` when Anurag's score files arrive; review of his Q5 claims; ship-checklist items on this machine |
| Compute | 12.2 h + 2.5 h of GPU used this week across Aayush's two accounts; laptop rule: nothing may allocate (1,000 × 245k) at once (C-026) |
| Blocked on | Nothing on Aayush's side: Q3 signed off (C-033) and the reranker scores now exist, so his reranker-vs-NRMS pairing is done too (C-035). Anurag: the Codabench submissions (the last mandatory Q5 item) and Q9 (C-034, still `_Not started._`) |
| Next up | **Anurag:** Kaggle datasets for the large files (P0.5–6) → test-set inference for `config.FINAL` on both test files → upload + screenshots; then Q9. Long runs go to Kaggle, not the laptop (C-004). **Aayush:** P6 Q3/Q4 note sections; the Q5 numbers in `RESULTS.md` are ready to review |

---

## 2 · Decision log

Entry format:

```
### C-NNN · short title
- Date / author: YYYY-MM-DD · name (agent used, if any)
- Decision: what was decided, in one or two sentences
- Why: the evidence or reasoning. Link RESULTS.md sections or commands, not prose claims
- Alternatives rejected: what else was considered, and why not
- Affects: files / phases / interfaces touched
- Status: active | superseded by C-NNN
```

### C-001 · Branch and collaboration workflow
- Date / author: 2026-09-11 · Anurag (Claude Code)
- Decision: all A2 work happens on `a2-click-logs`. `main` stays frozen as the A1 submission.
  Both of us commit to this branch in small commits and `git pull --rebase` before starting a
  session. Short-lived personal branches (`a2-click-logs-<name>-<topic>`) are allowed for risky
  work, merged back within a day.
- Why: A1 was graded from `main`, so it must not change under the graders. With one shared branch,
  each of us sees the other's work within hours rather than at merge time.
- Alternatives rejected: a long-lived branch per person (too much merge risk with a 9-day
  deadline); working directly on `main` (it would overwrite the A1 submission).
- Affects: git workflow
- Status: active

### C-002 · Agent-context docs are now tracked in git
- Date / author: 2026-09-11 · Anurag (Claude Code)
- Decision: `CLAUDE.md`, `PLAN.md`, `CONTEXT.md` and `AI_USAGE.md` are committed on this branch.
  `LEARNING_NOTES.md`, the TA notebooks and `prompts/` stay gitignored.
- Why: this **reverses the A1 choice** to keep them out of the repo. In A1 they were one person's
  scratch files. In A2 the partner's agent has to read the same rules, plan and decision log, and
  git is the only channel both agents see. `AI_USAGE.md` must merge both members' entries for
  Q7.4.
- Alternatives rejected: sharing the files out-of-band (they go stale immediately, and the partner's
  agent would not load them).
- Affects: `.gitignore`
- Status: active. Before the final push, decide whether to keep them in the graded repo (they are
  evidence of process) or strip them.

### C-003 · A1-only files removed from this branch
- Date / author: 2026-09-11 · Anurag (Claude Code)
- Decision: removed `A1.pdf` (replaced by `A2.pdf`), `report/design_note.md` (the A1 note),
  `scripts/overnight.py`, `scripts/resume_ebnerd.py`, `scripts/submit_mind_v3.py`,
  `scripts/make_submission_mind.py` (popularity submission 1), and `scripts/make_deliverables.sh`
  (A1 bundle paths). The `make submit-mind` target went with them. `RESULTS.md` was reset to an A2
  skeleton, and `README.md` was rewritten for A2.
- Why: each was a one-off A1 driver or an A1-only deliverable that A2 does not call. Everything
  still exists at `be15ee6`; see `git show be15ee6:RESULTS.md`, for example.
- Kept on purpose: all of `src/` and `tests/`, `build_pipeline.py`, `fetch_data.sh`,
  `encode_mind_minilm.py`, `eval_bm25_*` / `eval_semantic_*` (stage-1 recall), `rerank_*` and
  `submit_mind_v4.py` (the reranker A2 starts from), `make_submission_ebnerd.py` (the EB-NeRD
  streaming writer), `gap_aware_mind.py` (the model-selection protocol), `ablation_serving_time.py`
  (Q9 again in A2). `SPEC.md` stays as the A1 contract for the retrieval stage, and A2 sections are
  appended to it phase by phase.
- Not touched: gitignored local files (`report/deliverables/`, `report/*.zip`, screenshots,
  `prompts/`). Untracked files are shared across branches, so deleting them here would delete the
  A1 submission record from disk.
- Affects: repo layout, `Makefile`
- Status: active

### C-004 · Compute: Kaggle 2× T4 for heavy work; RAM is no longer a design constraint
- Date / author: 2026-09-11 · Anurag (Claude Code)
- Decision: GPU work and every full-scale run (MINDlarge, EB-NeRD large, test-set inference) run on
  Kaggle, with 2× T4 and fp16 autocast. The laptop is for dev, tests and toy-scale runs. The A1
  rule "memory is the constraint, stream everything" is dropped. Chunked, resumable output stays,
  but its purpose is now surviving Kaggle's session time limit, not saving RAM.
- Why: A1 ran on a 7 GB laptop (~1.4 GB free, no GPU), which forced low-memory designs and killed
  the EB-NeRD test pass. Kaggle removes the RAM limit and provides a GPU, which NRMS (Q3.1) needs
  anyway. fp16 measured faster than the alternatives on these T4s (Anurag's earlier Kaggle work);
  T4 has no bf16 tensor cores.
- Tooling: `kaggle` CLI 2.2.4 added to `requirements.txt` and installed in `.venv`. Auth verified
  as `anuragkaushal183` with a read-only `kaggle kernels list --mine`. Credentials live in
  `~/.kaggle/kaggle.json` (mode 600); `kaggle.json` is also gitignored as a guard.
- Alternatives rejected: Colab (no persistent datasets, and the Kaggle setup already exists);
  keeping the laptop as the full-scale runner (it cannot hold MINDlarge features or train NRMS).
- Open: where Kaggle gets the data (`PLAN.md` D9). Community mirrors exist but are unverified.
- Affects: `CLAUDE.md` §6, `PLAN.md` §0/P0/P5/§4/§5, `requirements.txt`, `.gitignore`
- Status: active

### C-005 · Team roster and division of labour
- Date / author: 2026-09-11 · Anurag Kaushal (Claude Code)
- Decision: the team is **Anurag Kaushal** and **Aayush Pandey**. Ownership, final:

  | Phase | Owner |
  |---|---|
  | P0 Setup (clean-clone check, Kaggle verification) | Anurag & Aayush |
  | P1 Behavioural features (Q1) | Anurag |
  | P2 Two-stage reranker (Q2) | Anurag |
  | P3.1 Reproduce NRMS on Kaggle 2× T4, fp16 (Q3.1) | Anurag |
  | P3.2–3.4 Improvement + ablation (Q3.2–3.4) | Aayush & Anurag; Aayush drives the paired bootstrap CI harness |
  | P4 Serving & scale (Q4) | Aayush |
  | P5 Extended eval + Codabench submissions (Q5) | Aayush |
  | P6 Design note + ship (Q6–Q9) | Anurag & Aayush |

- Why: Anurag owns modelling end to end, through baseline reproduction. Features, the reranker
  and NRMS all produce scores, so one person can explain every model and keep their inputs
  consistent. Aayush takes serving benchmarks, statistical significance testing and production
  scaling: the parts that *judge* the models. Keeping builder and judge separate makes every
  claimed gain an independent check rather than a self-assessment.
- How the halves connect: a scores file per system per split (`impression_id, article_id, score`,
  Parquet). Anurag's models write it and Aayush's harnesses read it (`PLAN.md` §2). The format is
  to be pinned in `SPEC.md` during P0.
- Risk and mitigation: Anurag's 12–15 Sep window holds P1, P2 and P3.1 at once, and Aayush's final
  numbers depend on its output. To keep Aayush unblocked, every Aayush harness (paired bootstrap,
  `make eval`, `make bench`, the Kaggle submission pipeline) is built and tested first against
  A1's existing outputs, then re-run on Anurag's score files. This adds a new phase, **P3.4a**
  (paired bootstrap harness, Aayush, 12–14 Sep), and moves the start of P4 to 14 Sep and P5 to
  13 Sep; their end dates are unchanged. If the NRMS smoke test fails on day 0, the `PLAN.md` §4
  descope rung "NRMS on EB-NeRD only" applies before any other phase slips.
- Alternatives rejected: the earlier proposal in `PLAN.md`, where Anurag took features, reranker
  and eval, and the partner took NRMS and serving. The team chose modelling vs measurement instead.
- Affects: `PLAN.md` §0, §2, §3 and the per-phase headings; `CONTEXT.md` §1. Earlier log entries
  that say "partner" (C-002) are left unedited, as the log is append-only; "partner" there means
  Aayush Pandey.
- Status: active

### C-006 · P1 begins: `recency_weighted_profile` specified and oracle written before any code
- Date / author: 2026-09-11 · Anurag Kaushal (Claude Code)
- Decision: the first P1 feature is a **category profile with exponential time decay**,
  `wᵢ = 2^(−(t − tsᵢ)/h)`, normalised to a distribution. The feature value is the profile's mass
  on the candidate's category (`SPEC.md` §11.1). The contract fixes:
  - **the boundary:** strict `ts < t`, so events at exactly t and after t are excluded;
  - **decay by timestamp, not row position:** results are invariant to row order, so out-of-order
    events are handled rather than rejected;
  - **NaN, not 0, when there is no eligible history;**
  - **`ValueError` on `h ≤ 0` or a null `ts`.**

  Code will live in `src/features/behavioural.py` (`decay_weights`, `recency_weighted_profile`).
  The oracle is a 20-event, 3-user toy log in `tests/test_behavioural_features.py`, with every
  expected value hand-computed.
- Why:
  - *Time decay over position decay:* A1's `recency_pool` decays by list position, so two clicks a
    minute apart and two clicks a week apart get the same weights. That is not recency, and the
    out-of-order test is there to catch it.
  - *NaN over 0:* 0 would conflate "never read this category" with "never read anything", and
    both GBDT options handle missing values natively.
  - *Oracle first:* `CLAUDE.md` rule 2. The implementation is deliberately held back so the tests
    are seen failing first.
- Red state, measured: the new file gives 2 passed (oracle self-checks) and 14 failed, all with
  `ModuleNotFoundError: No module named 'src.features.behavioural'`. The other 212 tests are
  unchanged and pass. Command: `PYTHONPATH=. .venv/bin/pytest tests/test_behavioural_features.py -q`.
- Found while specifying: **MIND history has no click timestamps.** `behaviors.history` is an id
  list; only clicks inside impressions carry `time`. EB-NeRD has per-click times
  (`history.impression_time_fixed`). Recorded as open decision **P1-D1** in `SPEC.md` §11.1
  (recommendation: stamp MIND history clicks with the split's window start, a leak-free upper
  bound). The half-life value is open as **P1-D2** (grid 6 h / 24 h / 72 h / ∞; ∞ is the
  no-decay baseline).
- Alternatives rejected: a decayed *embedding* profile as the first feature (harder to hand-compute,
  so a weaker oracle; it can reuse `decay_weights` later); rejecting out-of-order input like A1's
  `RollingCounts` (unnecessary, since the math never depends on order).
- Affects: `SPEC.md` §11 (new), `tests/test_behavioural_features.py` (new); `src/features/`
  deliberately untouched.
- Status: active

### C-007 · `recency_weighted_profile` implemented: out-of-order handling and NaN default confirmed by mutation check
- Date / author: 2026-09-11 · Anurag Kaushal (Claude Code)
- Decision: the implementation in `src/features/behavioural.py` (`decay_weights`,
  `recency_weighted_profile`, `SERVING_OK`) is accepted. It keeps C-006's contract:
  - out-of-order events are **handled by timestamp**, not rejected;
  - no eligible history returns **NaN**;
  - the time boundary is the strict `ts < t`.

  This entry adds the evidence that the oracle actually enforces that contract. (C-006 already
  recorded the out-of-order and NaN choices; the log is append-only, so this entry confirms them
  rather than re-deciding.)
- Why: a test that has only ever passed proves nothing. The oracle was first run against planted
  bugs.
  - **Combined mutant.** All three bugs at once, via `make test`: 10 failed, 218 passed.
  - **Each bug alone.** Inserted separately into an otherwise-correct implementation:

    | Planted bug | Failing tests | Dedicated test that catches it |
    |---|---|---|
    | `ts <= t` | 8 / 16 | `test_event_at_exactly_t_is_excluded` |
    | decay by list position | 6 / 16 | `test_out_of_order_…_not_row_order` |
    | no `user_id` filter | 7 / 16 | `test_other_users_events_are_excluded` |
    | control (no bug) | 0 / 16 | — |

  Logs: session scratchpad `mutant_all3.log` and `mutants_isolated.log`. The driver script is
  `isolate_mutants.py`, not committed; its three one-line mutations are listed in `SPEC.md` §11.1.
- Result: `make test` gives **236 passed**, 0 failed, exit 0. That is 212 from A1, 16 original
  oracle tests and 8 new fallback tests. The 6 warnings are A1's existing Polars deprecations in
  `tests/test_no_leakage.py`.
- Implementation notes worth knowing:
  - Ages are computed in microseconds (`dt.total_microseconds() / 1e6`), because
    `dt.total_seconds()` truncates to whole seconds, which would silently round sub-second ages
    (verified: 1.5 s → `1`, `Int64`, on polars 1.43.2).
  - Output is sorted by `ts`, then `article_id`, so equal timestamps still give a deterministic
    order.
  - `half_life` has **no default**: every caller must state it (P1-D2).
- Observed in passing: the boundary mutant also fails `test_other_users_events_are_excluded`,
  because that test checks U1's rows against the eligible set, which also excludes U1's own
  at-*t* event. The overlap is harmless, but it means that test's name undersells what it checks.
- Affects: `src/features/behavioural.py` (new), `tests/test_behavioural_features.py` (+8 tests,
  docstring), `SPEC.md` §11.1
- Status: active

### C-008 · P1-D1 decided (MIND timestamp fallback); P1-D2 half-life grid fixed
- Date / author: 2026-09-11 · Anurag Kaushal (Claude Code); decision by Anurag
- Decision, **P1-D1**:
  - Clicks with a null `ts` (MIND history) are stamped with their split's start, but only through
    the explicit keyword `untimed_ts`. Without it, a null `ts` still raises `ValueError`. The
    fallback is an opt-in, never a silent default.
  - Split starts: `MINDsmall_train` 2019-11-09 00:00, `MINDsmall_dev` 2019-11-15 00:00,
    `MINDlarge_test` 2019-11-16 00:00.
- Decision, **P1-D2**:
  - The half-life grid is h ∈ {6 h, 24 h, 72 h, ∞}, with ∞ as the no-decay baseline, selected
    under the gap-aware protocol (`SPEC.md` §7).
  - The value is not chosen yet: it needs the batch path to run the grid on real data.
- Why (measured 2026-09-11 from `behaviors.tsv`; log: scratchpad `mind_history_check.log`):
  - **The stamp is strictly before every impression.** The first impressions are 19 s, 1 s and
    5 s after midnight (train / dev / test).
  - **The stamp is an upper bound on the true click time.** MIND history is a frozen snapshot: it
    varies within a split for **0** of 33,617 / 14,826 / 484,059 repeat users (train / dev / test),
    and is identical in train and dev for **all 5,943** users present in both. It predates 9 Nov.
  - **No leak, even with a wrong stamp.** A stamp at or after `t` is excluded by the strict boundary,
    as tested.
  - **Per-split stamps keep history ages comparable:** 0–6 days in train, 0–7 in test. One global
    9 Nov stamp would be truer, but would age test history to 7–14 days against 0–6 in training,
    which is the A1 submission-3 skew.
- **Consequence that limits the feature (important for D4 and the note):**
  - All MIND history clicks share one stamp, so they share one weight, and normalisation cancels
    it. **On MIND, the history-only recency profile equals the undecayed category distribution for
    every h**, pinned by `test_history_only_profile_does_not_depend_on_half_life`.
  - Decay would need timed in-window clicks, which are unlabelled in the MIND test split, so using
    them in training recreates the skew.
  - **The half-life ablation, and "freshness weighting" as the D4 principled change, are therefore
    only testable on EB-NeRD.** Don't claim a MIND decay effect.
- Alternatives rejected: (b) timed in-window clicks only (discards all history and creates
  train/serve skew); (c) position-based decay for MIND (a second definition, and the weakness
  C-006 was written to avoid); a single global stamp (skew, as above).
- Affects: `src/features/behavioural.py` (`untimed_ts`), `SPEC.md` §11.1 (P1-D1/P1-D2 rewritten
  with the measurements), `tests/test_behavioural_features.py` (`TestMindUntimedFallback`)
- Status: active

### C-009 · Batch scaling: `recency_profile_batch`, held to the row-by-row reference by a parity oracle
- Date / author: 2026-09-11 · Anurag Kaushal (Claude Code)
- Decision:
  - Feature values for Kaggle-scale runs come from `recency_profile_batch(log, requests,
    half_life, *, untimed_ts=None)`, one vectorised Polars pass over a frame of
    `(user_id, t, candidate_category)` requests.
  - The row-by-row `recency_weighted_profile` stays as the readable definition and **is the
    oracle**. The batch path must reproduce it on every request.
  - "Same output" means NaN ↔ NaN and 0.0 ↔ 0.0 exactly, and all other values within 1e-12
    relative. It is not bitwise, because float addition is not associative and the batch path sums
    in a different order.
- Algorithm (`SPEC.md` §11.2):
  - distinct (user, t) anchors, so one profile per impression rather than per candidate;
  - an equi-join to that user's events, then the strict `ts < t` filter per anchor;
  - sums per (anchor, category) and per anchor;
  - a left-join lookup of each candidate's category.

  Row order and passthrough columns are preserved.
- Why this design: it is the same arithmetic as the reference, so it is explainable line for line,
  and it avoids per-row Python.
  - **Cost:** memory is linear in Σ over anchors of history length. The Kaggle driver streams
    `requests` one chunk (Parquet row group) at a time to cap it.
  - **Rejected for now:** per-user prefix sums with an as-of lookup at `t`. It is asymptotically
    cheaper (linear in events + requests, no pair table), but subtler (duplicate timestamps,
    as-of strictness). Adopt it only if the measured cost demands it (`CLAUDE.md` rule 3: find the
    bottleneck, don't guess it).
- Evidence (TDD order kept):
  1. **Red.** The 6 parity tests were written first and failed with
     `AttributeError: … no attribute 'recency_profile_batch'`.
  2. **Green.** After implementing, 6/6 pass. The grid is **360 requests** over both toy logs (128
     NaN, 150 exact-zero and 82 non-zero reference values). It includes scoring times equal to
     event timestamps, the MIND fallback, shuffled requests with a passthrough column, and the
     reference's `ValueError` cases. One test also pins the batch to the hand-computed U1 values.
  3. **Mutation.** Planted in the batch path alone: `ts <= t` fails 4/6; a cross join with no user
     filter fails 3/6. The hand-computed-values test alone misses the second bug, because the
     upstream semi-join narrows a single-user request to that user. The many-user parity grid
     catches it. Logs: scratchpad `batch_mutants.log`; the file was restored and verified
     byte-identical.
- **Not yet measured:** throughput and peak memory on real data. The "millions of rows" claim is a
  design property, with a stated cost model, until the Kaggle driver runs it on EB-NeRD. The
  numbers then go to `RESULTS.md` Q1 with the command. Measured in passing, for the cost model:
  mean candidates per impression is 12.0 on EB-NeRD small validation and 37.2 / 37.5 / 39.3 on
  MIND small train / small dev / large test. That is the per-candidate work the anchor
  deduplication avoids.
- Affects: `src/features/behavioural.py` (+`recency_profile_batch`), `tests/test_behavioural_features.py`
  (`TestBatchParity`, `_requests`, `_assert_same`), `SPEC.md` §11.2
- Status: active

### C-010 · `category_match` defined as a cosine (specified and oracle written; not implemented)
- Date / author: 2026-09-11 · Anurag Kaushal (Claude Code). **The definition is pending
  Anurag's confirmation.**
- Decision: `category_match(user, candidate, t) = cos(P, e_c) = P(c) / ‖P‖₂`.
  - P is the recency-weighted category profile (§11.1), and e_c is the candidate category's
    one-hot vector.
  - Everything else is shared with §11.1: boundary, decay, `untimed_ts`, 0.0 / NaN rules and
    `ValueError` cases.
  - Oracle: the toy log is extended 20 → 27 events (U4, U5), with the original 20 rows untouched.
  - The code is **deliberately not written.**
- Why cosine: the dot product P · e_c equals P(c), which *is* `recency_weighted_profile`, so it
  would give the reranker a duplicate column. Cosine adds exactly one piece of information: profile
  concentration, via ‖P‖₂. It is 1.0 when c is the user's only recent category and 1/√k across k
  equal categories.
- Honest caveat: that is a thin addition. A GBDT given P(c) and a concentration feature could learn
  it. Alternatives, if Anurag prefers:
  - a top-1-category indicator;
  - lift over a point-in-time population prior, P(c) / P_all(c);
  - a subcategory-level match (both datasets carry subcategories).
- Oracle (hand-computed, literals self-checked):
  - **U1:** sports 0.9030165, politics 0.1736579, tech 0.3929429. The literals come from the raw
    masses W, and the oracle recomputes them from the normalised P to show scale invariance.
  - **U4, only science eligible:** 1.0. A leaked at-t click would give 0.7287.
  - **U5, two equal-mass categories at one timestamp:** 1/√2 each. A leaked future click would give
    0.9589.
  - **Also:** exact 0.0 for categories seen only at or after t and for other users' categories;
    NaN for U3; invariance to 5 shuffles; `ValueError` on a null `ts`.
- Red state, measured: `make test` gives **10 failed, 244 passed** (exit 2). All 10 failures are
  `AttributeError: module 'src.features.behavioural' has no attribute 'category_match'`. The
  other 244 pass: 212 from A1, 24 recency, 6 batch parity and 2 new oracle self-checks. Log:
  scratchpad `make_test_category_red.log`.
- Affects: `SPEC.md` §11.3, `tests/test_behavioural_features.py` (`_EXTENSION_ROWS`,
  `toy_log_extended`, `EXPECTED_U1_CATEGORY_MATCH`, `TestCategoryMatch`, 2 oracle self-checks)
- Status: active, **pending Anurag's confirmation of the cosine definition**. Confirmed in C-012.

### C-011 · `requirements.txt` pinned to the full dependency closure
- Date / author: 2026-09-11 · Anurag Kaushal (Claude Code)
- Decision: every package `make env` installs is pinned with `==` to the version on the machine
  where the suite is green. That is 11 direct dependencies (polars 1.43.2, pyarrow 25.0.1, numpy
  2.5.2, scikit-learn 1.9.0, faiss-cpu 1.15.0, rank-bm25 0.2.2, pytest 9.1.1, tqdm 4.70.0,
  matplotlib 3.11.1, huggingface_hub[cli] 1.28.0, kaggle 2.2.4) plus their **53 transitive**
  dependencies, 64 in all, on Python 3.12.3. `requirements-embed.txt` is deliberately left
  unpinned.
- Why:
  - Unpinned, Aayush's `make env` could resolve different versions, so "green here" would not
    mean "green there".
  - Pinning only the direct dependencies still lets scipy, pyarrow's dependencies and the rest
    float. The closure was computed from installed metadata, walking every requirement with its
    markers and extras.
  - `requirements-embed.txt` stays unpinned because the local torch is `2.13.0+cpu`, which PyPI
    cannot serve, and Kaggle's GPU image supplies its own CUDA torch.
- Evidence:
  - A **fresh venv** built only from the pinned file (scratchpad `venv_pinned`) installed cleanly.
    `pip check` found no broken requirements, and `pip freeze` equals all 64 pins exactly.
  - The full suite in that venv: **279 passed** (log: scratchpad `pinned_venv_tests.log`).
  - `make test` in the project venv also refreshed it against the pins: 279 passed.
- To change a version: edit the pin, rebuild a fresh venv, run `make test`, and log it here.
- Affects: `requirements.txt`
- Status: active

### C-012 · C-010 confirmed; `category_match` implemented (row-by-row and batch)
- Date / author: 2026-09-11 · decision by Anurag Kaushal; code by Claude Code
- Decision: Anurag confirmed the cosine definition: "It properly captures user interest
  concentration." It is implemented as `category_match` (reference) and `category_match_batch`.
  - Both batch features now share two private helpers: `_decayed_category_mass` (anchors, strict
    boundary, decayed mass per category) and `_lookup` (join back, NaN for no history, order kept).
  - Refactoring `recency_profile_batch` onto them was guarded by its existing parity tests, which
    still pass unchanged.
- Evidence:
  - The 10 C-010 tests went red → green.
  - `TestCategoryMatchBatchParity` was written first: 6 red (`AttributeError`) → green. It covers
    the same 360-request grid, hand values (U1, U4 = 1.0, U5 = 1/√2), the MIND fallback, and
    order/passthrough.
  - **Planted bugs:**
    - a dot product P(c) instead of the cosine, in the reference: 4/10 fail (U4 correctly passes,
      since cosine = dot = 1 for a single-category user);
    - an L1 sum instead of the L2 norm, in the batch: 4/6 fail.

    Logs: scratchpad `category_match_mutant.log`.
- Cost note: each batch function makes its own pass, so computing both costs two pair-joins. A
  combined pass is an easy later change if the Kaggle measurement shows the join dominates.
- Affects: `src/features/behavioural.py`, `tests/test_behavioural_features.py`, `SPEC.md` §11.3
- Status: active

### C-013 · Phase 1.2 schema integration: slate, session and dwell features, and the unsafe registry, with two corrections to the mapping
- Date / author: 2026-09-11 · Anurag Kaushal (Claude Code), from **Aayush Pandey's schema mapping**
- Decision: implemented in `src/features/behavioural.py` per `SPEC.md` §11.4–§11.7:
  - `slate_features`: `cand_position`, `n_candidates`; MIND and EB-NeRD.
  - `session_features`: `session_pos`, `n_prior_clicks_in_session`, `session_len`; EB-NeRD.
  - `dwell_features`: `hist_read_time_mean`, `hist_scroll_mean`; EB-NeRD, from history strictly
    before t.
  - `current_page_features`: `cur_read_time`, `cur_scroll_percentage`; EB-NeRD, **unsafe**.
  - The feature registry: `SERVING_OK` (all 11 features), `UNSAFE_FEATURES`,
    `ABSENT_FROM_TEST_FILE` and `drop_unsafe()`.
- **Correction 1: `session_len` is serving-unsafe, not safe.** It counts the session's impressions
  after t: no live system knows how long a session will last. A test proves it: appending a future
  impression changes `session_len` from 6 to 7 while no safe feature moves. It is in
  `UNSAFE_FEATURES` with `cur_read_time` and `cur_scroll_percentage`. The safe "length so far" is
  `session_pos` − 1.
- **Correction 2: `n_prior_clicks_in_session` cannot feed the submission model.**
  - It is serving-safe in production, but the EB-NeRD **Codabench test file has no
    `article_ids_clicked`** (nor `article_id`, `next_read_time` or `next_scroll_percentage`).
    Training on it would recreate the A1 submission-3 skew.
  - It is in `ABSENT_FROM_TEST_FILE`, and when `clicked` is absent the column is **omitted, not
    zero-filled**.
  - It is also nearly redundant: every EB-NeRD small-train impression has at least one click (0
    without, mean 1.006), so it is ≈ `session_pos` − 1.
- **Design fact: sessions are keyed by `(user_id, session_id)`.** The test file gives all 200,000
  `is_beyond_accuracy` rows `session_id` 0 (and `impression_id` 0). Keyed on `session_id` alone,
  that is a 200,000-impression "session": a ~4×10¹⁰-pair self-join. Keyed properly, the largest
  test session is 118 impressions, and the whole file needs 47.3M pairs.
- `cand_position`, measured:
  - Click rate is nearly flat across list-position quintiles (EB-NeRD 0.080–0.094, MIND
    0.036–0.045), so the order is **not a label artifact**.
  - EB-NeRD lists are not id-sorted (0.2%, the same in train, validation and test).
  - There is a weak position effect with the same shape in train and dev.
  - **Open:** whether list order equals on-screen order is not established, so the feature is
    documented as list position.
- Evidence:
  - Oracles first: `tests/test_session_features.py`, 19 tests, **19 red (`AttributeError`) →
    19 green**.
  - The central leakage check is "append the future, nothing safe changes", plus the tie at
    exactly t.
  - **Planted bugs:**
    - session `<=`: 4/19 fail;
    - session keyed on `session_id` only: 5/19 fail;
    - dwell `<=`: 4/19 fail.
  - **Data facts:** `make check-data` (new target, `scripts/check_phase1_data.py`), with output
    in `RESULTS.md` Q1. Every number in `SPEC.md` §11.4–§11.7 reproduces from it.
  - One number was corrected on re-derivation: tied-timestamp sessions in test are **195**, not
    196. The first count grouped by `session_id` alone, so the placeholder session counted too.
  - A first attempt at the position measurement, which exploded every EB-NeRD train slate, was
    killed by the OOM killer (exit 137). The script samples 50,000 impressions for per-candidate
    rates.
- Consequence for Q2 (the reranker): the model trained **for the Codabench submission** must use
  `drop_unsafe(...)` and also drop `ABSENT_FROM_TEST_FILE`. The Q9 "with unsafe" row is a separate
  ablation model.
- Affects: `src/features/behavioural.py` (registry and 4 functions), `tests/test_session_features.py`
  (new), `scripts/check_phase1_data.py` (new), `Makefile` (`check-data`), `SPEC.md` §11.4–§11.7,
  `RESULTS.md` Q1
- Status: active. **Aayush to review the two corrections.**

### C-014 · Freshness defined (Q1.3); half-life decided: h = ∞, i.e. no decay (P1-D2 closed)
- Date / author: 2026-09-11 · Anurag Kaushal (Claude Code)
- **Freshness.** `freshness_batch` gives `freshness_hours` = t − the article's first known time,
  strictly before t (`SPEC.md` §11.8).
  - EB-NeRD's first known time is `published_time`; MIND's is the first impression that listed
    the article, with history articles stamped at 2019-11-09.
  - One group-by suffices, because the earliest sighting before t is the overall earliest
    whenever that is before t.
  - A publish time ≥ t gives NaN, since it must be a later rewrite of the metadata. Measured:
    23 of 1.2M EB-NeRD rows.
  - Evidence: `tests/test_freshness.py`, 8 tests, red (`AttributeError` / `KeyError`) → green. A
    `<=` boundary mutant fails 3/8.
- **Half-life: h = ∞ (no decay), for both datasets.**
  - The grid ran on 50,000 EB-NeRD validation impressions, single-feature ranking, with paired
    bootstrap vs ∞: 6 h −0.0063 AUC and 24 h −0.0016 (both CIs exclude 0); 72 h +0.0003,
    CI [−0.0007, +0.0014] (includes 0).
  - 72 h had the highest point estimate, but it is not a win (`CLAUDE.md` §4), so the simpler
    no-decay choice was taken.
  - It was then **confirmed inside the full GBDT:** A2 with h = 72 h is worse than h = ∞ by 0.0035
    AUC, CI [−0.0052, −0.0020].
  - On MIND h is irrelevant anyway (C-008).
  - Command: `scripts/tune_half_life_ebnerd.py`; numbers in `RESULTS.md` Q1.
- **Consequence:** `recency_weighted_profile` is in effect the undecayed category distribution on
  both datasets. "Freshness weighting of history" is therefore **not** a promising D4 candidate.
  The time signal that matters is article freshness, already in the base set.
- Also measured: as a *single-feature ranker* `category_match` is identical to
  `recency_weighted_profile`. ‖P‖ is constant within an impression, so its extra information only
  helps across impressions.
- Affects: `src/features/behavioural.py` (`freshness_batch`, registry), `tests/test_freshness.py`,
  `SPEC.md` §11.8, `scripts/tune_half_life_ebnerd.py`
- Status: active

### C-015 · Phase 2 reranker v1: temporal protocol, a correction to A1, and a negative result for the Phase 1 features
- Date / author: 2026-09-11 · Anurag Kaushal (Claude Code)
- Built: `src/rerank/` (`common.py`, `ebnerd.py`, `mind.py`), `scripts/rerank_ebnerd_a2.py` and
  `scripts/rerank_mind_a2.py` (`SPEC.md` §12).
  - It uses A1's stage-1 generators as scores, A1's pointwise HistGBDT with the same
    hyperparameters, and in-impression reranking (PLAN **D1 framing (a)** only; (b) and (c) are
    still open; **D2 still open**).
  - **`model_features` is the single gate:** it drops `UNSAFE_FEATURES` and
    `ABSENT_FROM_TEST_FILE`, and both scripts assert it.
  - The test-file check runs the full pipeline on 5,000 unlabelled test impressions per dataset:
    no label, no click count, finite predictions.
- **Protocol: fit on the train period, evaluate on the next period** (EB-NeRD train → validation
  week; MINDsmall_train → dev). Impressions are seeded random samples within a split.
- **Correction to A1.** A1's EB-NeRD reranker (AUC 0.7084) fitted and evaluated within the
  validation file, split 70/30 by *row order*. Measured: `behaviors.parquet` is **not
  time-sorted** in train or validation, so that split was not temporal, and its number is not
  comparable. Recorded here rather than silently dropped.
- **Results** (`RESULTS.md` Q2; paired bootstrap on the same impressions):
  - **The two-stage gain is large and significant:** EB-NeRD GBDT − BM25 is +0.133 AUC; MIND
    GBDT − MiniLM is +0.037 AUC.
  - **The Phase 1 features do not help yet:**
    - EB-NeRD A2 − base: AUC +0.0009 [−0.0009, +0.0026] (no difference), while MRR/nDCG are
      −0.002 (significant, small);
    - MIND A2 − base: AUC −0.0021 [−0.0031, −0.0009], with all four metrics significantly worse.
  - The cross-implementation check holds: on MIND, `recency_weighted_profile` = A1 `cat_affinity`
    to 1.1e-16.
- Known limitation: absolute values are optimistic (evaluation sits right after training; A1's
  offline-to-leaderboard offset was −0.03 to −0.09). Differences between rows are what is reliable.
- **Next, proposed and not yet decided:**
  1. a drop-one ablation of the Phase 1 features, to find which one costs the drop;
  2. PLAN D2: LightGBM `lambdarank`, since the pointwise objective may be what wastes the
     within-impression features. That needs a new pinned dependency (C-011 procedure).

  Permutation importance must not be used to judge impression-constant features: `n_candidates`
  ranks first while being unable to reorder candidates.
- **Note for Aayush (stream ownership, `CLAUDE.md` §3):** `src/rerank/common.paired_delta` is a
  *minimal, provisional* paired bootstrap written to measure this run (with 4 tests). Your P3.4a
  harness is the team's version. Please replace it or adopt it after review, and log which.
- Affects: `src/rerank/*` (new), `scripts/rerank_ebnerd_a2.py`, `scripts/rerank_mind_a2.py`,
  `tests/test_rerank_*.py` (new), `SPEC.md` §12, `RESULTS.md` Q2
- Status: active

### C-016 · D2: LightGBM lambdarank adopted as the reranker objective (Anurag's decision), with a contrary MIND result recorded
- Date / author: 2026-09-11 · decision by Anurag Kaushal ("switching to LightGBM lambdarank");
  implementation and measurements by Claude Code
- Decision:
  - The reranker objective becomes **listwise**: `LGBMRanker(objective="lambdarank")`, **one query
    per impression**.
  - Groups come from contiguous `imp_row` runs (`common.group_sizes`, which rejects interleaved
    rows). The key is `imp_row`, not `impression_id`, because the EB-NeRD test file repeats
    impression_id 0.
  - The capacity matches the pointwise model: 300 trees, lr 0.08, 31 leaves.
  - It is deterministic: `deterministic=True`, `force_row_wise=True`, `n_jobs=4`; refits are
    bit-identical (tested).
- Dependency (C-011 procedure):
  - `lightgbm==4.7.0` was added to `requirements.txt`, installed under the existing pins as
    constraints. Its dependencies (numpy, scipy, narwhals) were already pinned, so it is the only
    new package: **65 pins**.
  - A fresh venv from the file gives a clean `pip check`, a freeze equal to the pins, and 310 tests
    passing.
- Evidence (`RESULTS.md` Q2; paired bootstrap, same impressions):
  - **EB-NeRD supports the switch strongly.** Lambdarank − pointwise is **+0.0343 AUC** on base
    features and +0.0217 on A2, and all four metrics improve with CIs excluding 0.
  - **MIND contradicts it.** Lambdarank − pointwise is **−0.0084 AUC** on base features and
    −0.0051 on A2, and all four metrics are worse with CIs excluding 0.
  - Untested hypothesis for MIND: long slates (mean 37, max 299, against EB-NeRD's 11 and 100)
    meet LightGBM's default `lambdarank_truncation_level` of 30.
- **The conditional ablation ran on both datasets** (the trigger is fixed in `common.drop_reasons`):
  - **EB-NeRD:** under lambdarank, A2 − base is −0.0118 AUC. The **dwell family is toxic**:
    removing it gives +0.0144 AUC / +0.0135 MRR / +0.0158 nDCG@5 / +0.0120 nDCG@10. **A2 without
    dwell beats the base on all four metrics** (AUC +0.0026, CI [+0.0014, +0.0038]).
  - Also on EB-NeRD: the category profile helps (removing it costs 0.0091 AUC); session and list
    position are mildly negative on AUC and flat or mixed on MRR/nDCG.
  - **MIND:** A2 − base is AUC +0.0013 but MRR/nDCG −0.002. **First-seen freshness** costs top-rank
    quality (removing it gives MRR +0.0027, nDCG@5 +0.0025, nDCG@10 +0.0021) while helping AUC
    (−0.0015 when removed). Category match and list position help.
- **Status: active for EB-NeRD. Contested for MIND,** where the measured result goes against the
  decision as stated. **Anurag to choose** between:
  - (a) lambdarank everywhere, for one model family, accepting about −0.008 AUC on MIND;
  - (b) a per-dataset objective: lambdarank for EB-NeRD, pointwise for MIND;
  - (c) first test `lambdarank_truncation_level` ≥ the maximum slate on MIND, then decide.

  Recommendation: (c). It is one cheap run and tests the only hypothesis that could reconcile the
  two datasets. **Resolved in C-018:** (c) was tested and failed (C-017), and (b) was adopted.
- **Not yet adoptable: "A2 without dwell".** It was identified on the same validation impressions
  it would be chosen on, which is selection bias. Confirm it on held-out impressions (a different
  seeded sample, or the gap-aware protocol) before changing the shipped feature set.
  - Candidate explanation to test: the dwell means are constant within an impression, so under a
    listwise loss they act only through interactions that may not transfer from one week to the
    next.
- Affects: `requirements.txt`, `src/rerank/common.py` (`fit_lambdarank`, `group_sizes`,
  `drop_reasons`, `leave_one_family_out`, `report`, `conditional_ablation`, `matrix`), both
  reranker scripts, `tests/test_rerank_common.py` (+8), `SPEC.md` §12, `RESULTS.md` Q2

### C-017 · Evidence for the final Phase 2 configuration: the truncation check fails on MIND, and the dwell drop holds on a holdout
- Date / author: 2026-09-11 · Anurag Kaushal (Claude Code). **Findings only. The configuration
  lock-in is Anurag's, pending.**
- **MIND, C-016 option (c): tested and ruled out** (`scripts/check_truncation_mind.py`).
  - Setup: `lambdarank_truncation_level=300`, covering the longest slate of 299. `fit_lambdarank`
    now takes parameter overrides, and a test proves LightGBM consumes the parameter (truncation 1
    and 5 train different models).
  - Truncation 300 − 30 on the base features, all dev: AUC −0.0002, CI [−0.0010, +0.0006], and
    the sign flips between dev halves (−0.0021 / +0.0016).
  - **Lambdarank at truncation 300 still loses to pointwise:** −0.0087 AUC over all dev, −0.0134
    in the earlier half and −0.0040 in the later half, with all CIs excluding 0.
  - What the evidence now supports is **C-016 option (b): a per-dataset objective**, lambdarank
    for EB-NeRD and pointwise for MIND. Option (a), lambdarank everywhere, costs about 0.009 AUC on
    MIND.
- **EB-NeRD dwell: the finding holds out of sample** (`scripts/holdout_dwell_ebnerd.py`).
  - Setup: the same 100,000-impression fit. The ablation's 100,000 validation impressions are the
    *selection* set; the other 144,647 validation-week impressions, disjoint and never scored, are
    the *holdout*.
  - Selection reproduces C-016 exactly.
  - On the holdout: dropping dwell gives +0.0151 AUC over A2, and **A2 without dwell beats the base
    on all four metrics** (AUC +0.0029 [+0.0018, +0.0039], nDCG@10 +0.0014 [+0.0004, +0.0023]).
  - Limit: the holdout shares the week, so it rules out sample-fitting but not drift.
- **Proposed lock-in, awaiting Anurag:**
  - **EB-NeRD:** lambdarank, with features A1 base + category profile + list position + session.
    `hist_read_time_mean` and `hist_scroll_mean` stay computed and registered but out of the
    model.
  - **MIND:** pointwise HistGBDT, with the A1 v4 base features. None of the Phase 1 additions beat
    the base there under either objective (pointwise A2 −0.0021 AUC; lambdarank A2 mixed).
- Also measured: every MIND model is weaker in the later half of the dev day (pointwise AUC
  0.7016 → 0.6479), consistent with A1's staleness finding. This is material for the "breaks at
  10×" and drift discussion.
- Refactor: `scripts/rerank_ebnerd_a2.build` now takes an explicit impression sample
  (`seeded_sample`) so a holdout can be the complement. The draw is unchanged, as the reproduced
  selection numbers show.
- Affects: `src/rerank/common.py` (`fit_lambdarank(**overrides)`), `tests/test_rerank_common.py`
  (+2), `scripts/check_truncation_mind.py` and `scripts/holdout_dwell_ebnerd.py` (new),
  `scripts/rerank_ebnerd_a2.py` (`seeded_sample`, `build`), `RESULTS.md` Q2 follow-ups
- Status: evidence recorded; decided in C-018

### C-018 · Phase 2 locked: EB-NeRD lambdarank without dwell; MIND pointwise with A1 v4 features
- Date / author: 2026-09-11 · decision by Anurag Kaushal ("The metrics are validated. Let's lock
  in Phase 2"); code by Claude Code
- Decision:
  - The shipped reranker configuration is `src/rerank/config.FINAL`:
    - **EB-NeRD:** `lambdarank`, 11 features: the A1 base + `recency_weighted_profile`,
      `category_match`, `cand_position`, `session_pos`. `hist_read_time_mean` and
      `hist_scroll_mean` stay computed and registered but out of the model.
    - **MIND:** `pointwise` HistGBDT, with the A1 v4 base's 7 features.
  - That resolves C-016 with option (b), a per-dataset objective, after option (c) failed in C-017.
- Why:
  - EB-NeRD: lambdarank beats pointwise by +0.034 AUC. Without dwell, A2 beats the base on all
    four metrics, and that holds on 144,647 held-out impressions (AUC +0.0029 [+0.0018, +0.0039]).
  - MIND: pointwise beats lambdarank by 0.0087 AUC in both halves of dev, and no Phase 1 addition
    beats the base.
- **Enforced in code:**
  - Both reranker scripts assert that `FINAL` equals the model they measured (EB-NeRD: A2 minus
    the dwell family; MIND: the base). For MIND they also assert that the final model's predictions
    are bit-identical to the measured base fit.
  - The test-file checks now score with the final models.
  - `tests/test_rerank_config.py` (6 tests) pins the invariants.
  - `common.fit_final` / `predict_scores` dispatch on the objective.
- **Still open against the brief (flagged, not decided):** Q2.1 wants A1's generators to
  *retrieve* a top-K (100–200) from the corpus before re-ranking, which is PLAN D1 framing (b).
  The locked configuration re-ranks each impression's own candidates (framing (a)). Framing (b)
  still has to be built and measured for Q2 to be complete.
- Affects: `src/rerank/config.py` (new), `src/rerank/common.py` (`fit_final`, `predict_scores`),
  `tests/test_rerank_config.py` (new), both reranker scripts, `SPEC.md` §12.1
- Status: active

### C-019 · Ownership change: all of Phase 3 goes to Aayush; Phase 5 becomes joint
- Date / author: 2026-09-11 · decision by Anurag Kaushal; logged by Claude Code. This revises
  C-005's split for the remaining phases.
- Decision:
  - **P3, the whole Q3 track, goes to Aayush Pandey exclusively:**
    - P3.1: the Kaggle NRMS baseline reproduction on EB-NeRD and MIND (2× T4, fp16);
    - P3.4a: the paired-bootstrap CI harness;
    - P3.2–3.4: the one principled change and its ablation.
  - **P5 (extended evaluation + Codabench submissions) becomes joint:** Anurag Kaushal & Aayush
    Pandey.
  - **Unchanged:** P4 (Aayush), P6 (joint). P1 and P2 (Anurag) are done or locked (C-014, C-018).
  - `PLAN.md` §0, §2 and §3, plus the P3/P5 headings and the day-0 NRMS step, are updated to match.
- Why: Anurag's modelling phases are complete, so the NRMS reproduction moves to Aayush, who takes
  over the Kaggle baseline work. Sharing P5 means the full-scale submission runs for the locked
  reranker (`config.FINAL`) do not queue behind Aayush's P3 and P4.
- **Consequences to manage:**
  - **Builder and judge are the same person for Q3.** C-005 separated them. The mitigation written
    into `PLAN.md` §2: **Anurag reviews every Q3 "beats" claim**, meaning the paired CI and its
    command, before it goes into `RESULTS.md`.
  - **Aayush's load is now the critical path.** P3.1, P3.4a and P4 overlap across 12–17 Sep, and
    Aayush is **still blocked**: the Phase 1/2 commits are not on GitHub (C-017 note).
  - **Kaggle GPU quota:** NRMS training on Aayush's account; P5 test-set inference on Anurag's.
  - **The provisional `src/rerank/common.paired_delta`** (C-015) falls inside Aayush's P3.4a, to
    replace or adopt.
  - **Proposed P5 split, not yet agreed:** Anurag takes `make eval` and the resumable Kaggle
    submission pipeline for `config.FINAL`; Aayush's `make bench` stays under P4 (`PLAN.md` §2
    table).
- Anurag's other open item: Q2.1 retrieval framing (b) (C-018).
- Affects: `PLAN.md` (§0 team row and consequences, §2, §3, P3/P5 headings, P0 steps 6–7), `CONTEXT.md` §1
- Status: active

### C-020 · Aayush's Kaggle account verified; GPU budget for P3 is 30 h/week, refreshing 19 Sep
- Date / author: 2026-09-12 · Aayush Pandey (Claude Code)
- Decision: P0 step 3 closed for Aayush's account. `~/.kaggle/kaggle.json` (mode 600) authenticates
  the CLI as `aayushpandey18602`; the kernel `scripts/kaggle/gpu_check/` ran on 2× Tesla T4 with
  fp16 autocast + `GradScaler` and passed. Kaggle's image is torch 2.10.0+cu128, so NRMS runs
  against that torch, not a version we pin.
- Why: numbers in `RESULTS.md` P0. The check is a kernel in the repo, pushed by the CLI with
  `--accelerator NvidiaTeslaT4`, so it can be re-run on either account.
- Planning consequence: `kaggle quota` shows **30 h GPU per week**, refreshing 2026-09-19 00:00.
  P3.1 (NRMS on EB-NeRD and MIND), the change and its ablation all draw from this one budget in the
  week of 12–18 Sep, and the refresh lands one day before the deadline. NRMS runs are therefore
  sized to the demo/small data first, and every full run writes resumable chunks (C-004).
- Alternatives rejected: verifying through the Kaggle UI only (not reproducible, and the CLI is
  what P3 uses).
- Affects: `scripts/kaggle/gpu_check/` (new), `RESULTS.md` P0, `PLAN.md` P0 step 3
- Status: active

### C-021 · `ebnerd-benchmark` pinned at `5164e2c`; it runs on Kaggle through a two-function polars shim, not its own pins
- Date / author: 2026-09-13 · Aayush Pandey (Claude Code)
- Decision: the NRMS baseline is `ebnerd-benchmark` (brief URL `jppol-ai/`, cloned via the `ebanalyse/` redirect) at commit
  `5164e2ce7c92b99cbcb853d5f804cc95f0232b2f` (2026-03-16, "dependencies updates"), cloned into
  `external/ebnerd-benchmark` (gitignored). On Kaggle it runs against the image's own
  TensorFlow 2.20 / Keras 3.13 / polars 1.35 / numpy 2.0 / transformers 5.0, with
  `src/baselines/ebrec_compat.install()` replacing two of its polars helpers.
- Why: the benchmark pins `polars==0.20.8`, `numpy<1.26.1`, `scikit-learn==1.4.0`,
  `tensorflow<2.22`, `torch<2.3`: incompatible with our venv (polars 1.43, numpy 2.5) and with the
  Kaggle image. Installing its pins on Kaggle would downgrade numpy under TF. Smoke run v1 died in
  `map_list_article_id_to_value` (polars 1.35 rejects `replace` with list-valued dicts) and v2 in
  `add_prediction_scores` (`drop` of a column the frame never had; 0.20 ignored it, 1.x raises).
  Both shims keep the original signatures and are held to the benchmark's own docstring examples
  in `tests/test_ebrec_compat.py` (8 tests). Nothing in the model, loss, sampling or metrics is
  touched.
- Framework finding (P0.7 / D3): NRMS here is **TensorFlow/Keras**, adapted from Microsoft
  `recommenders`. So the same implementation can serve MIND only through its own data loader,
  which reads EB-NeRD's `behaviors/history/articles` parquet layout: MIND would need a converter.
  D3 stays open until P3.1 decides between that converter and a separate MIND NRMS.
- Reproduction-gap candidate, recorded for P3.1: `examples/reproducibility_scripts/ebnerd_nrms.py`
  computes the xlm-roberta word embeddings and then builds `NRMSModel(hparams, seed)` **without
  passing them**, so the published recipe trains a random 32,000×300 embedding table indexed by
  xlm-roberta token ids (vocab 250,002). The quick-start passes them. Our runs pass them.
- Alternatives rejected: `pip install polars==0.20.8` inside the kernel (a 0.20-era wheel under
  numpy 2.0 is untested and would make Kaggle's environment diverge from the image both accounts
  share); patching the benchmark checkout (not reproducible from a clean clone).
- Affects: `external/ebnerd-benchmark` (pin), `src/baselines/ebrec_compat.py` (new),
  `tests/test_ebrec_compat.py` (new), `scripts/kaggle/nrms_smoke/` (new)
- Status: active

### C-022 · NRMS trains in float32 on the T4: Keras `mixed_float16` fails in the benchmark's `SelfAttention`
- Date / author: 2026-09-13 · Aayush Pandey (Claude Code)
- Decision: the NRMS baseline runs with Keras's default float32 policy. The "fp16 autocast +
  GradScaler" default in `CLAUDE.md` §6 is a PyTorch recipe and does not carry over unchanged.
- Why: with `tf.keras.mixed_precision.set_global_policy("mixed_float16")` the first training
  batch raises `TypeError` in `SelfAttention.call()` ("`x` and `y` must have the same dtype, got
  tf.float16 != tf.float32"): the benchmark's custom attention layers create float32 weights and
  matmul them against float16 activations. Fixing that means editing the baseline's layers, which
  is out of scope for a *reproduction*. Smoke run v2 (`RESULTS.md` P0): one epoch on EB-NeRD demo,
  45,614 train impressions, npratio 4, batch 32, took **271 s in float32** on one T4 (190 ms/step),
  so float32 is affordable on demo/small; the cost at `ebnerd_large` is measured before deciding.
- Alternatives rejected for now: casting inside `AttLayer2`/`SelfAttention` (an edit to the
  baseline, possible later as an explicitly labelled variant if the quota demands it); running
  on P100 (no tensor cores either way).
- Affects: `scripts/kaggle/nrms_smoke/nrms_smoke.py` (policy probe with fallback), P3.1 budget
- Status: active

### C-023 · Codabench has no team feature: submissions go from Anurag's account, the report names both
- Date / author: 2026-09-13 · Aayush Pandey (checked by hand on Codabench; logged by Claude Code)
- Decision: both competitions (MIND 13967, RecSys 2024 / EB-NeRD 2469) are entered from
  **Anurag's Codabench account**, which has been registered and approved since A1. The design
  note and README name both members and state this. P0 step 4 is closed.
- Why: neither competition page offers a team/participants feature, so a shared entry is not
  possible. Anurag's account is already approved; a new registration for Aayush would add a
  day-plus approval wait against the Wed 16 Sep first-submission target for no gain.
- Consequence: the P5 submission runs (C-019: Anurag's Kaggle quota, joint P5) also upload
  from Anurag's account. Leaderboard screenshots for Q5 show that account name; the note explains
  why. Aayush's Codabench registration is not needed and is not pursued.
- Alternatives rejected: registering Aayush too and alternating submitters (splits the
  submission history across two accounts for no benefit).
- Affects: P5 submission workflow, README team table, design note
- Status: active

### C-024 · D3 decided: a native NRMS per dataset; MIND-small on Kaggle as a private dataset; a dated fallback
- Date / author: 2026-09-13 · decision by Aayush Pandey in plan mode; logged by Claude Code
- Decision:
  - **EB-NeRD:** NRMS from `jppol-ai/ebnerd-benchmark` at `5164e2c` (the brief's starter-code
    URL; `ebanalyse/` 301-redirects to it, same history). Already running (C-021, C-022).
  - **MIND:** NRMS from `recommenders-team/recommenders` (`recommenders/models/newsrec`), the
    brief's "MIND baseline" (A2.pdf Q3.1: "NRMS from the ebnerd-benchmark repo, **or the MIND
    baseline**"). Pinned at `0bb4b3690941ffb668118e31ccaf8a7d19f8212a` (HEAD on 13 Sep).
    Vendored `--no-deps` into the kernel: its `install_requires` pins `numpy<2`,
    `transformers<5`, and its GPU extra `tensorflow<2.16`, none of which the Kaggle image
    (numpy 2.0.2, transformers 5.0, TF 2.20 / Keras 3) satisfies. `newsrec` is TF1-style Keras
    (`tf.compat.v1.keras`, `set_session`), so it runs under `tf-keras` with
    `TF_USE_LEGACY_KERAS=1`.
  - **Data on Kaggle:** EB-NeRD small from the public S3 bucket inside the kernel (as the smoke
    test). MIND small as a **private Kaggle dataset** on Aayush's account, built from the
    official `MINDsmall_train.zip` + `MINDsmall_dev.zip` on disk (80 MB); shared with Anurag.
    `MINDsmall_utils.zip` (GloVe `embedding.npy`, `word_dict.pkl`, `uid2index.pkl`, `nrms.yaml`)
    from `huggingface.co/datasets/Recommenders/MIND` (ungated, HTTP 302 on 13 Sep; the old Azure
    blob `recodatasets.z20.web.core.windows.net` no longer answers), sha256 recorded by the kernel.
  - **Fallback, dated:** if the MIND-native kernel has not trained one epoch by **Mon 14 Sep
    20:00 IST**, MIND runs on the EB-NeRD implementation through an adapter from our unified
    frames, and that reversal gets its own entry.
- Why: PLAN.md D3 recommended one implementation for both. The brief names both options, and
  Aayush chose the native MIND baseline as closer to "the official baseline" for MIND (GloVe,
  the MIND paper's own setup). The agent recorded the counter-case — one codebase, one shim,
  identical hyper-parameters, and the adapter is ~40 lines because `src/rerank/mind.load_behaviors`
  already yields EB-NeRD-shaped lists — and the Keras-3 risk; the dated fallback bounds that risk.
- Protocol and scores-file contract: `SPEC.md` §13.2–§13.3 (fit on the shipped train split,
  score every validation impression on its full slate; `imp_row` key; Float64 scores).
- Affects: `SPEC.md` §13 (new), `src/baselines/nrms_data.py` (new), `scripts/kaggle/nrms_*`,
  `scripts/kaggle/nrms_smoke` (URL), P3.1 schedule
- Status: active

### C-025 · P3.1 done: NRMS reproduced on both datasets; the gap and the asymmetry it reveals
- Date / author: 2026-09-14 · Aayush Pandey (Claude Code)
- Result (`RESULTS.md` Q3.1, ledger `a2-nrms-ebnerd` v4 and `a2-nrms-mind` v5):
  - **EB-NeRD** NRMS, 5 epochs, xlm-roberta-large, all 244,647 small-validation impressions:
    **AUC 0.5600 [0.5588, 0.5612]**, MRR 0.3491, nDCG@5 0.3884, nDCG@10 0.4668. Published
    (Kruse et al. 2024, Table 3, hidden test, trained on large): 0.6103 / 0.3975 / 0.4445 / 0.5124.
  - **MIND** NRMS, 5 epochs, GloVe-300d, all 73,152 MINDsmall_dev impressions:
    **AUC 0.6667 [0.6647, 0.6688]**, MRR 0.3220, nDCG@5 0.3557, nDCG@10 0.4184. Published
    (Wu et al. 2020, Table 3, full-MIND test, half the users for training, mean of 10 runs):
    0.6776 / 0.3305 / 0.3594 / 0.4163 — same implementation (Microsoft Recommenders).
  - Gaps: −0.011 on MIND, −0.050 on EB-NeRD; both explained by training-set size (small vs
    large / half of full) and evaluation set (validation vs hidden test); details in RESULTS.md.
- Decision: these two runs are **the** baseline for Q3.2–3.4. Their score files
  (`data/scores/{ebnerd/validation,mind/MINDsmall_dev}/nrms.parquet`, SPEC §13.3, verified
  row-for-row against `src/rerank` frames with metrics recomputed locally to the same digits) are
  the frozen reference every ablation and the paired bootstrap compare against. Weights are in the
  kernels' outputs (`out/weights/`), so a variant can be scored without retraining the baseline.
- Finding that shapes D4: NRMS trails the locked reranker by **0.008 AUC on MIND** but by
  **0.113 on EB-NeRD**. NRMS has no notion of article age, popularity or session; A1 found
  recency dominates on EB-NeRD. The change with the most evidence-backed headroom is therefore a
  recency/freshness-aware user or candidate encoding inside NRMS (D4 is decided in P3.2's plan,
  not here).
- Cost (quota): EB-NeRD 4.0 h, MIND 2.2 h, pre-flight checks 1.5 h; **7.7 h of 30 used**,
  22.3 h left until the 19 Sep refresh. An NRMS variant costs ≈ 4 h (EB-NeRD) + 2.2 h (MIND).
- Determinism: both pipelines reproduce to every digit (ledger twins). MIND needed Python's
  `random` seeded in our kernel — the package seeds TF and numpy only, and `newsample` draws the
  negatives from the stdlib. Worth a sentence in the note: the official baseline is not
  run-to-run reproducible as shipped.
- Not done / deferred: Anurag's `src/eval/scores.py` validator has not yet been run on the two
  files (it does not exist yet); the check is scripted in the kernels and repeated by hand on the
  laptop (RESULTS.md Q3.1), to be turned into a test when the reader lands.
- Affects: `RESULTS.md` Q3.1, `scripts/kaggle/RUN_LEDGER.md`, `PLAN.md` §3 P3.1 status, D4
- Status: active

### C-026 · P3.4a done: `paired_delta` adopted and moved to `src/eval`; the file-level harness; an OOM fix in `bootstrap_ci`
- Date / author: 2026-09-14 · Aayush Pandey (Claude Code)
- Decision: the provisional `src/rerank/common.paired_delta` (C-015) is **adopted as is** — the
  statistic was right — and **moved to `src/eval/bootstrap.py`** beside the unpaired CI; the old
  name is a re-export, so `src/rerank/common.py` changed by one import line (Anurag's file,
  noted here). The harness around it is `src/eval/paired.py` + `scripts/paired_compare.py` +
  `make paired` (SPEC §14): reader/validator for the §13.3 contract (to be swapped for Anurag's
  `src/eval/scores.py` when it lands), manifest compatibility, labels from the split,
  intersection on `imp_row` with a wrong-split guard, Δ per metric, verdict from the CI only.
- Why / evidence: calibration oracles (coverage ≥ 90 % over 200 trials for Δ = 0 and 0.02;
  paired width < ½ unpaired) and the real-file checks in `RESULTS.md` Q3 "Harness validation":
  rank-preserving rescale → Δ = 0 exactly; +0.5 on clicks → every CI > 0; N(0, 0.01) noise →
  a *real* −0.0011 AUC, significant at 20k impressions. That last one corrected the agent's
  first test, which had assumed noise is a zero-Δ change.
- **Incident and fix.** The second full-file `make paired` was OOM-killed at 01:16 on 14 Sep
  (dmesg: python, 3.4 GB RSS, with ~5 GB available) and took the editor session down. Cause:
  A1's `bootstrap_ci` draws the whole (iterations × n) int64 index array at once — 1.96 GB at
  244,647 impressions, plus the same again for the gather — eight times per comparison. Fix:
  draw in blocks of 100 iterations. Because sequential blocks consume the generator stream in
  the same order, the resampled means are **bit-identical** to the one-shot draw
  (`tests/test_bootstrap.py`), so no CI recorded in `RESULTS.md` moves. Peak RSS for a full
  EB-NeRD comparison: 1.3 GB, 49 s (was killed above 3.4 GB). Rule of thumb for this laptop:
  anything that allocates (1,000 × 245k) at once will die.
- Ready for: the reranker-vs-NRMS comparison the moment `config.FINAL`'s score file exists
  (Anurag's P0.8 round trip); the P3.2–3.4 ablation rows.
- Affects: `src/eval/bootstrap.py`, `src/eval/paired.py` (new), `src/rerank/common.py` (one
  line), `scripts/paired_compare.py` (new), `Makefile` (`paired`), `SPEC.md` §14,
  `tests/test_paired.py` (new), `tests/test_bootstrap.py`
- Status: active

### C-027 · A second Kaggle account (`aayushpandey602`) doubles the P3 GPU budget; ablation variants run in parallel
- Date / author: 2026-09-14 · Aayush Pandey (Claude Code)
- Decision: P3.2–3.4 variant runs are split across two of Aayush's Kaggle accounts —
  `aayushpandey18602` (main, 22.3 h left) and `aayushpandey602` (alt, 30 h) — one dataset per
  account, so the two rows of the ablation (baseline+change; change with a component removed)
  take ≈ 5 h wall each instead of ≈ 12 h sequential. Anurag's account stays reserved for P5
  inference (C-019).
- Verified: alt token at `~/.kaggle/alt/kaggle.json` (mode 600), selected per call with
  `KAGGLE_CONFIG_DIR`; `scripts/kaggle/gpu_check` PASS on the alt account (ledger,
  `a2-gpu-check` v1 (aayushpandey602)); the private dataset `mind-small-official` is shared
  with it (collaborator, read).
- Mechanics: `scripts/kaggle/alt_account.sh push|status|logs|quota` runs a committed kernel
  folder from the alt account by rewriting the `id` in a temp copy of `kernel-metadata.json`;
  the committed metadata keeps the main account. `scripts/kaggle/ledger.py --account alt`
  fetches that account's log and tags the row with the username. Which account ran a row is
  therefore always visible in the ledger; nothing scientific changes — same commit, seeds, data.
- The baseline rows (C-025) are **not** rerun: the ablation is valid as long as variants share
  the baseline's commit of the training recipe, seeds, epochs and data. Do not touch the shared
  recipe; if it must change, the baseline reruns too.
- Affects: `scripts/kaggle/alt_account.sh` (new), `scripts/kaggle/ledger.py`, P3.2–3.4 schedule
- Status: active

### C-028 · D4 decided: the one principled change is a candidate-freshness term in NRMS; prediction pre-registered
- Date / author: 2026-09-14 · decision by Aayush Pandey in plan mode; logged by Claude Code
- Decision: `score = user · news + g(freshness)`, `g` = Dense(8, relu) → Dense(1) on
  `(x, unknown)` where `x` is the standardised `log1p(freshness_hours)` from §11.8's
  `freshness_batch` — the reranker's own feature, same sources, same strict `< t` rule. Nothing
  else in the recipe changes (SPEC §15). Ablation: baseline (reused) / +freshness (one run per
  dataset) / +freshness scored with the term masked (no training).
- Why: C-025's asymmetry (NRMS −0.113 vs the reranker on EB-NeRD, −0.008 on MIND) plus A1's
  finding that `age_hours` dominates EB-NeRD (+0.125 permutation importance) point at exactly
  the signal NRMS lacks. Options named to Aayush: (A) this term; (B) A plus recency-decayed
  history attention (two components, 3 rows, ≈ 18.6 h GPU, attention-layer edits in two
  codebases, split A1 evidence); (C) a category embedding in the news encoder (addresses a
  different gap). Aayush chose A: strongest evidence, one code path in both implementations,
  cheapest (≈ 5 h wall on the two accounts, C-027), easiest to explain in the exam.
- **Prediction, recorded before any run:** EB-NeRD Δ AUC ≥ +0.03 with a paired CI excluding 0;
  MIND Δ AUC within ±0.01, CI may include 0. The outcome is reported against this either way.
- Serving-time honesty: the feature is available at serving time on both datasets; the
  "with/without serving-unavailable features" row is not applicable and is stated as such.
- Affects: `SPEC.md` §15, `src/baselines/nrms_fresh_*` (new), `scripts/kaggle/nrms_*_fresh/`
  (new), P3.2–3.4 schedule (target Tue 16 Sep)
- Status: active

### C-029 · Pre-run diagnostic (logged before any full-run number): freshness has almost no within-slate signal on EB-NeRD
- Date / author: 2026-09-14 02:10 · Aayush Pandey's agent (Claude Code), while the variant demo checks ran
- Finding: ranking each evaluation slate by the freshness feature alone (§15.2 `x`, unknown last),
  with the reranker's own metric code:

  | | fresher-first | older-first | within-slate range of x (median) |
  |---|---|---|---|
  | EB-NeRD small validation, 244,647 impressions | **AUC 0.5010**, MRR 0.3077 | AUC 0.4991 | 2.63 sd |
  | MIND dev, 73,152 impressions | **AUC 0.5185**, MRR 0.2153 | AUC 0.4811 | 3.64 sd |

  Freshness varies a lot *inside* EB-NeRD slates but does not separate the click from the rest.
- Why it matters: C-028's motivation leaned on A1's `age_hours` **+0.125 permutation importance**
  on EB-NeRD. That number is consistent with *pooled* importance — age separates impressions
  (time of day, day of week) rather than candidates within one slate — the pitfall SPEC §12
  names for pointwise models. An additive `g(freshness)` term in NRMS can only capture the
  within-slate marginal signal measured here (≈ 0 on EB-NeRD, small on MIND).
- Consequence: the pre-registered prediction (EB-NeRD Δ AUC ≥ +0.03) is now expected to
  **fail**; MIND's small gain remains possible. The full runs go ahead unchanged — the test is
  pre-registered, the budget is available (C-027), and a null with a measured mechanism is the
  honest Q3 result (PLAN.md P3 step 4). If it is a null, the note should say what *would* have
  a within-slate signal on EB-NeRD (the reranker's category profile: −0.009 AUC when removed,
  RESULTS.md Q2) and that the P3.2 change was chosen on pooled evidence.
- Also observed on the MIND demo check (ledger `a2-nrms-mind-fresh` v4): variant 0.5471 vs
  baseline demo 0.5790, and masking the term at inference *raised* it to 0.5650 — a 1-epoch demo
  signal in the same direction.
- Command: the inline check in `AI_USAGE.md` (this session); to be turned into
  `scripts/check_fresh_signal.py` with the outcome entry.
- Affects: interpretation of P3.2–3.4; nothing in the runs
- Status: active

### C-030 · P3.2–3.4 outcome: freshness term — small significant gain on EB-NeRD (pending Anurag's review), null on MIND
- Date / author: 2026-09-14 06:45 · Aayush Pandey's agent (Claude Code), unattended overnight run
- Result (`RESULTS.md` Q3.2–3.4; ledger `a2-nrms-ebnerd-fresh` v5, `a2-nrms-mind-fresh` v6;
  paired records `data/processed/paired/{ebnerd,mind}_row*.json`):
  - **EB-NeRD:** NRMS + freshness AUC 0.5674 vs NRMS 0.5600; paired Δ **+0.0074 [+0.0066, +0.0081]**,
    MRR +0.0094, nDCG@5 +0.0121, nDCG@10 +0.0092, every CI > 0. The harness's verdict is "beats";
    **recorded as pending Anurag Kaushal's review (C-019)**, not yet as a result.
  - **MIND:** Δ AUC −0.0006 [−0.0014, +0.0002] and every other CI covers 0: **null**. The masked
    row equals the unmasked one (Δ 0.000 ± 0.0002): the model learned to ignore the term.
- Against the pre-registration (C-028): MIND as predicted. EB-NeRD right in sign and significance,
  **wrong in magnitude** (+0.007 vs ≥ +0.03), for the reason C-029 gave before the run: the A1
  evidence was pooled importance; within-slate freshness signal is small. The gap to the locked
  reranker moves from 0.113 to 0.106.
- Two findings for the design note:
  1. **Inference-time masking is a dependence check, not a removal.** On EB-NeRD the encoders
     co-adapted to the term (masked: −0.017, below the baseline). The clean "component removed"
     row is the baseline trained without the term (row 1). Q3.3's isolation is row 2 − row 1.
  2. **Sampled-slate validation overstates changes.** Holdout `val_auc` on wu2019 5-candidate
     slates: +0.076; full slates: +0.007. This is why §13.2 evaluates on full slates.
- Decisions: P3.2–3.4 are complete as specified (one change, ablation, paired CIs on both
  datasets). No second change is attempted: the remaining budget (main 17.8 h, alt 27.6 h) is
  reserved for P4/P5 needs and one contingency rerun. `RESULTS.md` carries the "beats" line
  with the pending-review tag until Anurag confirms.
- Process notes: eight demo-check runs preceded the two full runs (ledger); every failure was
  the agent's (dash has no `pipefail`; wu2019 gives k rows per impression; TF1 sessions cannot
  share weights in one process; `Sequential` inside `TimeDistributed` has no standalone input in
  TF1). Two full runs, two accounts, in parallel, 5 h wall (C-027).
- Affects: `RESULTS.md` Q3.2–3.4, `PLAN.md` §3 P3.2–3.4 status, D4 closed
- Status: active

### C-031 · P4 done: the serving benchmark protocol, its numbers, and what breaks at 10×
- Date / author: 2026-09-14 · Aayush Pandey (Claude Code)
- Protocol decisions (SPEC §16): one request = (user, t) → candidates → `config.FINAL` features →
  score; two framings measured, (a) the impression's own slate (what ships) and (b) BM25 ∪ ANN
  top-K then rerank (Q4.2's wording); **one core** (`taskset -c 0`, LightGBM/OpenMP/BLAS single
  thread) so cost is per vCPU; 1,000 seeded validation impressions after 100 warm-ups; cost with
  ρ = 0.5 and AWS c7i on-demand $0.0425/vCPU-h (URL + date in RESULTS); memory as RSS delta per
  component (Python overhead included — the honest number for this implementation).
- The per-request path (`src/serving/`) is held to the batch path by a **bit-identical parity
  test** on 200 real impressions per dataset. It caught two skews before any number was
  recorded: (1) the user store had merged train ∪ validation history while the batch path reads
  the split's own snapshot; (2) the model was fed float64 where training used float32 — 11/8,360
  MIND rows flipped at tree thresholds. Both fixed; both would have shipped silently otherwise.
  The served models are `config.FINAL` refits that reproduce Q2 exactly (0.6728 / 0.6747),
  cached under `data/processed/models/`.
- Numbers (`RESULTS.md` Q4): (a) p99 **1.7 ms** EB-NeRD / **3.4 ms** MIND, 790 / 550 req/s per
  core, ≈ $0.0001 per 1k queries; (b) K=100 p99 **72 / 86.5 ms**, of which BM25 60 / 74 ms and
  flat ANN 8.4 / 8.9 ms; ≈ $0.0012–0.0014 per 1k. RSS after build 1.16 / 1.38 GB.
- Two bottlenecks found by measuring: the category-profile features recomputed per candidate
  (47 → 1.7 ms p99 once cached per request, definitions unchanged, parity kept); and A1's
  pure-Python BM25 postings scan on 82–128-token queries, which is 80 % of framing (b).
- **10× verdict:** (a) scales on every axis, memory only. (b) fails at ≈ 1.4× more articles on
  BM25 (≈ 600–740 ms p99 at 10×); the fix is a compiled index with top-k pruning, not the model.
  The feature store (`RollingCounts`) stores every event as a Python datetime (57 B/event,
  1.6–3.5 GB at 10×) — a live system keeps windowed aggregates. GBDT scoring is O(K), 1–3 ms,
  and never the constraint.
- Assets: `data/interim/ebnerd/Ekstra_Bladet_word2vec/document_vector.parquet` (sha256
  613f27b5…) and `data/processed/mind_minilm.npz` (eb920313…, 125,590 × 384) produced by the
  CPU kernel `a2-assets-p4` (ledger), so the laptop needs no torch. `MINDlarge_test.zip` added to
  the private dataset for the full corpus.
- Affects: `src/serving/` (new), `scripts/bench.py`, `Makefile` (`bench`), `SPEC.md` §16,
  `RESULTS.md` Q4, `tests/test_serving.py`, `tests/test_cost.py`
- Status: active

### C-032 · P5 goes to Anurag end to end; Aayush's hand-over for it
- Date / author: 2026-09-14 · agreed by Anurag Kaushal and Aayush Pandey; logged by Claude Code (Aayush's agent)
- Decision: **P5 (Q5: `make eval`, the two slices, diversity/novelty/coverage, test-set inference,
  both Codabench submissions and their screenshots) is Anurag's alone.** This revises C-019's
  "joint" split. P6 stays joint. Aayush reviews Anurag's Q5 claims (the mirror of C-019).
- Why: Anurag owns the reranker, the submission pipeline and the Codabench account (C-023); P5
  is built on those. Aayush's phases (P3, P4) finished two days early, so the "joint so it does
  not queue" rationale of C-019 no longer applies. Aayush stands by to take `make eval` if the
  Wed 16 Sep first submission is at risk.
- What Anurag needs from Aayush, and where it is:
  - **On GitHub:** the Q3.4 judge (`make paired`, `src/eval/paired.py`, SPEC §14); the
    scores-file reader/writer implementing the P0.8 contract (`write_scores`/`read_scores`,
    SPEC §13.3) — `src/eval/scores.py` need not be written; the serving code (`src/serving/`)
    if P5 wants per-request numbers; every RESULTS/CONTEXT entry.
  - **Sent by Aayush as a file** (its contents live under gitignored `data/`, Q8):
    `aayush_p3_outputs_2026-09-14.tgz`, 112 MB, **sha256 `4c1b11e9320ca036d14dbdd5e6221de063fc7061d961add28a2e8252286f6ab3`**. It holds the NRMS score
    files for both datasets (`nrms`, `nrms_fresh`, `nrms_fresh_masked`, with manifests, SPEC
    §13.3), the paired-bootstrap records behind RESULTS Q3.2–3.4, the refitted `config.FINAL`
    models, and `data/share/README_handover.md` with the same instructions as here.
    **Install:** from the repo root on `a2-click-logs`, `sha256sum` the file and compare, then
    `tar xzf aayush_p3_outputs_2026-09-14.tgz -C .` — it creates `data/scores/ebnerd/validation/`,
    `data/scores/mind/MINDsmall_dev/`, `data/processed/paired/`, `data/processed/models/`.
    **Verify:** `PYTHONPATH=. .venv/bin/python -c "from src.eval.paired import read_scores; f,m=read_scores('data/scores/mind/MINDsmall_dev/nrms.parquet'); print(f.height, m['system'])"`
    → `2740998 nrms`. **Use:** `read_scores` + `split_labels` for `make eval` rows; write
    `config.FINAL`'s scores with `write_scores` (or `src/baselines/nrms_data.scores_frame` +
    `write_scores`) and run `make paired A=…/nrms.parquet B=…/reranker_final.parquet JSON=…`
    per dataset; labels are never in the files (join from the split by `imp_row`, `cand_position`).
  - **Not produced:** NRMS test-set scores. `config.FINAL` is what is submitted (C-023); an NRMS
    leaderboard row would cost ≈ 2 h GPU per dataset and is not asked for.
- What Aayush still owes P5: the reranker-vs-NRMS paired comparison once Anurag's `config.FINAL`
  score files exist (one `make paired` per dataset; Aayush runs it or Anurag does); review of
  Anurag's "beats" claims; the Q3/Q4 sections of the note (P6).
- Affects: `PLAN.md` §2, §3 (P5 row), P5 heading; `CONTEXT.md` §1
- Status: active (supersedes the P5 ownership line of C-019)

### C-033 · C-019 review: the EB-NeRD freshness "beats" line is verified; Aayush is unblocked
- Date / author: 2026-09-15 · Anurag Kaushal (Claude Code)
- Decision: the Q3.2–3.4 EB-NeRD claim — NRMS + freshness beats NRMS, Δ AUC +0.0074
  [+0.0066, +0.0081] and positive on all four metrics — **passes review and is now a result**.
  `RESULTS.md` Q3.2–3.4 carries the reviewer line. **Aayush is unblocked for the note.**
- What was checked (all passed):
  - **Reproduction.** The command stored in the record, rerun through `make paired`, reproduced it
    **bit-identically** — every value in `system_a`, `system_b` and `delta_b_minus_a`, both
    manifests, same 1,000 resamples and seed 0. My record was written to
    `data/processed/paired/ebnerd_row2_vs_row1_review_anurag.json` so his artifact stayed untouched.
  - **No truncation, full overlap.** 2,928,942 rows over 244,647 impressions in both files, equal to
    the split's own slate-length sum and impression count; identical `(imp_row, cand_position)` key
    sets; per-impression counts equal the slate lengths.
  - **Label mechanics.** No score file carries a label column. `split_labels` was cross-checked
    against an independent path (`src.rerank.ebnerd.candidate_frame`, labels by clicked-id
    membership): identical on all 2,928,942 rows, 245,622 positives each.
  - **Masked control.** `nrms_fresh_masked` − `nrms_fresh` reproduced bit-identically at
    −0.0174 AUC [−0.0180, −0.0169].
- Data fact found while checking, worth keeping: the split lists 246,289 clicked ids but only
  245,622 distinct (impression, article) positives. The difference is exactly **667 duplicate
  clicked ids inside impressions**, deduped identically by both label paths. Every clicked id is
  present in its own slate, so no impression has an unrankable click and none is dropped by the
  bootstrap.
- Affects: `RESULTS.md` Q3.2–3.4, `CONTEXT.md` §1
- Status: active

### C-034 · Q9 (anti-gaming: with and without serving-unavailable features) is Anurag's
- Date / author: 2026-09-15 · decision by Anurag Kaushal; logged by Claude Code
- Decision: **Anurag owns Q9 end to end.** It had no owner: `PLAN.md` folded Q9 into P6 ("Q6–Q9,
  joint") and `RESULTS.md` Q9 still read `_Not started._` on 15 Sep, five days before the deadline,
  for a graded, mandatory requirement.
- Why Anurag: the registry it rests on is his — `SERVING_OK`, `UNSAFE_FEATURES`,
  `ABSENT_FROM_TEST_FILE` and `model_features` (C-013, C-015), and the reranker whose features get
  ablated is `config.FINAL` (C-018).
- Scope: metrics for the locked reranker **with and without** the serving-unavailable features on
  both datasets, paired CIs, plus the A1 precedent (`next_read_time` moved AUC 0.50 → 0.96) as the
  cautionary row. `n_prior_clicks_in_session` (`ABSENT_FROM_TEST_FILE`) is a separate row from the
  `UNSAFE_FEATURES` ones, since it is serving-safe but missing from the Codabench test file.
- Affects: `PLAN.md` §2 and §3 (P6 row), `RESULTS.md` Q9, this log
- Status: active

### C-035 · P5 part 1: the reranker's scores files, `make eval`, and the reranker-vs-NRMS verdict
- Date / author: 2026-09-15 · Anurag Kaushal (Claude Code)
- Built:
  - `scripts/score_final.py` — writes `config.FINAL`'s scores to the §13.3 contract via Aayush's
    `write_scores`. It fits on the same seeded training sample as the measured runs and scores
    **every** impression of the evaluation split, so the files align 1:1 with the NRMS ones
    (EB-NeRD 2,928,942 rows / 244,647 impressions; MIND 2,740,998 / 73,152 — both equal to his).
  - `scripts/eval_a2.py` + `make eval SCORES=… [K=…] [JSON=…]` (SPEC.md §17): all four accuracy
    metrics with bootstrap CIs, the beyond-accuracy trio over each impression's top-k, and the two
    required slices. Labels are joined from the split and the join is asserted lossless.
  - `src/eval/slices.py` + `tests/test_eval_slices.py` (11 tests): the slice definitions, kept at
    A1's thresholds — cold ≤ 5 history clicks, head = top popularity quintile of the **training**
    clicks, never-clicked articles are tail.
- Correctness checks that came free with the design:
  - MIND's "all" row **reproduces `RESULTS.md` Q2 exactly** (0.6747 AUC on the same full dev split).
  - EB-NeRD's "all" row (0.6734 over 244,647) sits on the impression-weighted mean of Q2's two
    disjoint samples (0.6728 on 100k, 0.6738 on the 144,647 holdout → 0.67339).
- **Reranker vs NRMS (the note's headline comparison).** The reranker beats NRMS on both datasets,
  every metric, every CI excluding 0: EB-NeRD **+0.1134 AUC** vs the baseline and **+0.1060** vs
  Aayush's freshness variant; MIND **+0.0080** and **+0.0086**. Records in
  `data/processed/paired/<ds>_reranker_vs_<system>.json`.
  - Stated honestly in `RESULTS.md`: this compares the systems as built here (NRMS at a fixed
    recipe and 5 epochs vs a reranker consuming A1 stage-1 scores plus behavioural features), not
    the architectures in general.
- Slice findings worth carrying into the note: head impressions are far easier on both datasets
  (+0.08 AUC over tail); MIND cold users lose 0.050 AUC against warm, while EB-NeRD has only 887
  cold impressions out of 244,647, so its cold CI is wide and that comparison is weak.
- Affects: `scripts/score_final.py`, `scripts/eval_a2.py`, `src/eval/slices.py`,
  `tests/test_eval_slices.py`, `Makefile` (`eval` target, was a stub), `SPEC.md` §17, `RESULTS.md` Q5
- Status: active. **Remaining in P5: test-set inference, the two Codabench uploads, screenshots.**

---

## 3 · Inherited from A1 (facts, not decisions to revisit)

Sources: `SPEC.md` and `git show be15ee6:RESULTS.md`.

- **Split windows.** MIND train 9–14 Nov 2019, dev 15 Nov, test 16–22 Nov. EB-NeRD train 18–25 May
  2023, val 25 May–1 Jun, test 1–8 Jun; the three windows are contiguous, 7 days each.
- **Submission format.** `impression_id [rank_order]`, where rank *i* belongs to candidate *i* in
  list order. Build it only through `ranks_from_scores`. MIND: `prediction.txt`. EB-NeRD:
  `predictions.txt`. Zip with the file at the archive root.
- **EB-NeRD duplicate ids.** `impression_id = 0` on exactly the 200,000 `is_beyond_accuracy` rows.
  This is legitimate, so pass `allow_duplicate_ids`.
- **Train/serve skew (the most important lesson).** Windowed popularity features (`pop_24h`,
  `pop_1h`, `ctr_24h`) were zero on 100% of MIND test candidates, because test starts 2+ days after
  training ends. The dev split cannot detect this. **Use the gap-aware protocol (`SPEC.md` §7) for
  model selection.**
- **Offline → leaderboard.** Offline metrics predict the *direction* of a change, not its size.
  The offset grows with the distance between the test window and training.
- **MIND coverage.** Popularity knows only 6.5% of `MINDlarge_test` candidates. Text-based
  features have 100% coverage by construction.
- **EB-NeRD specifics.** Danish text, so no English stemmer or stoplist. Publisher word2vec covers
  100% of articles. `next_read_time` / `next_scroll_percentage` are serving-unavailable: adding
  them moves AUC 0.50 → 0.96. Recency dominates there and popularity barely helps.
- **Pooling.** Mean-pooled user vectors won on MIND; recency-weighted won on EB-NeRD. The same
  design choice reverses between the datasets.
- **Best A1 leaderboard.** MIND AUC 0.5714 (MiniLM + skew-resistant GBDT, `submit_mind_v4.py`).
  EB-NeRD AUC 0.5110 (BM25).
- **Scale.** The EB-NeRD 13.5M-impression test pass took more than 3 hours on the laptop and was
  killed once. In A2 it runs on Kaggle (C-004), but it should still be resumable.
