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
| Phase | **P1 Behavioural features started** (Anurag). P0 setup committed (`779cad4`); the P0 exit-gate items owned by Aayush are still open. See `PLAN.md` §3 |
| Team | **Anurag Kaushal**: modelling (P1, P2, P3.1). **Aayush Pandey**: measurement (P3.4a paired bootstrap, P4, P5). Joint: P0, P3.2–3.4, P6. Final (C-005) |
| Anurag Kaushal | done: branch, docs, Kaggle CLI (C-001–C-004). **P1, feature 1 of ~6 done:** `recency_weighted_profile` in `src/features/behavioural.py`, with the MIND fallback. Mutation-checked (C-007), P1-D1 decided (C-008), `make test` green at 236. Next P1 unit: the decayed category profile's siblings (click count, category match, session, dwell, position, freshness; `PLAN.md` P1 table), then the batch path that computes features for millions of impressions and must match the reference on the toy log. Still owed from P0: NRMS smoke test on Kaggle (PLAN P0.7) |
| Aayush Pandey | not started. Now: P0 clean-clone check (`make env && make test`) and Kaggle verification on Aayush's account (PLAN P0.2–P0.3). Next: P3.4a paired bootstrap harness (12–14 Sep) |
| Compute | Kaggle 2× T4 (fp16) for GPU and full-scale runs; laptop for dev/tests (C-004) |
| Blocked on | team decisions D1–D9 in `PLAN.md` §5; the scores-file format (`PLAN.md` §2) must be agreed and pinned in `SPEC.md` |
| Next up | Anurag: next P1 feature; choose h (P1-D2) once the batch path can run the grid on EB-NeRD. Aayush: P0 clean-clone check. `make test` is **green (236 passed)**, so pushing is safe |

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
