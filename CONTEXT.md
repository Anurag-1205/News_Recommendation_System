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

_Last updated: 2026-09-11 by Anurag (agent: Claude Code)_

| | |
|---|---|
| Branch | `a2-click-logs`, from `main` at `be15ee6` (A1 final) |
| Phase | **P1 Behavioural features, mostly built** (Anurag). Done: recency profile, category match (reference + batch), slate, session, dwell, the unsafe registry. Left: freshness (`published_time` / MIND first-seen), click count, and the half-life choice (P1-D2). P0 exit-gate items owned by Aayush are still open. See `PLAN.md` §3 |
| Team | **Anurag Kaushal**: modelling (P1, P2, P3.1). **Aayush Pandey**: measurement (P3.4a paired bootstrap, P4, P5). Joint: P0, P3.2–3.4, P6. Final (C-005) |
| Anurag Kaushal | Commits `54cf185` (recency) and the P1-block commit that follows it (C-009, C-011–C-013; **279 green**) are local, and **neither is pushed yet**: the agent shell has no GitHub credentials, so Anurag pushes both. Next P1 units: freshness, then running the h grid on EB-NeRD through the batch path to settle P1-D2. Still owed from P0: NRMS smoke test on Kaggle (PLAN P0.7) |
| Aayush Pandey | not started. Now: P0 clean-clone check (`make env && make test`) and Kaggle verification on Aayush's account (PLAN P0.2–P0.3). Next: P3.4a paired bootstrap harness (12–14 Sep) |
| Compute | Kaggle 2× T4 (fp16) for GPU and full-scale runs; laptop for dev/tests (C-004) |
| Blocked on | team decisions D1–D9 in `PLAN.md` §5; the scores-file format (`PLAN.md` §2) must be agreed and pinned in `SPEC.md` |
| Next up | **Aayush:** once Anurag pushes, `git pull` and run `make env && make test`. With the pins (C-011), a fresh venv from `requirements.txt` gave 279 passed on this machine, and `make check-data` re-derives every data fact behind C-013. Please review C-013's two corrections to your schema mapping (`session_len` unsafe; `n_prior_clicks_in_session` absent from the test file). **Both:** the submission model must exclude `UNSAFE_FEATURES` and `ABSENT_FROM_TEST_FILE` (`src/features/behavioural.py`) |

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
