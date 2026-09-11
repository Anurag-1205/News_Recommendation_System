# AI Usage Log — A2

**CS4.406 Information Retrieval and Extraction — Assignment 2 (team of 2)**

Required by A2 Q7.4: all prompts, chat history exports, and marking of AI-generated vs
human-written code. Both team members append here. Chat exports are submitted with the report.

**Entry format.** Newest at the bottom.

```
### YYYY-MM-DD · member · tool/agent · short title
- Asked: the key prompt(s), verbatim or closely paraphrased
- Produced: files / functions, each marked AI-generated, human-written, or AI-generated + human-edited
- Verified by: the test, oracle or measurement that checked it
- Failed / corrected: what was wrong and how it was caught (these are required, not optional)
```

---

### 2026-09-11 · Anurag · Claude Code (Opus 5) · A2 branch and team-context setup
- Asked: set up A2 on a new branch for two-person work. Replace the A1 `PLAN.md` with an A2 plan,
  create a `CONTEXT.md` where every decision is logged so the partner's agent can see what was
  chosen, put that policy in `CLAUDE.md`, suggest a branch name, and delete what A2 does not use.
- Produced (AI-generated, to be reviewed by Anurag): branch `a2-click-logs`; new `PLAN.md`,
  `CONTEXT.md`, `AI_USAGE.md`; rewritten `CLAUDE.md`, `README.md`, `RESULTS.md` (skeleton);
  `.gitignore` and `Makefile` edits; seven A1-only files removed (list in `CONTEXT.md` C-003).
- Verified by: `make test` still passes after the deletions; a grep for references to the removed
  files found only the `Makefile` target and `README.md`, and both were updated.
- Failed / corrected: A1's `bootstrap.compare` turned out to be an unpaired CI-overlap check, not
  the paired bootstrap that A2 Q3.4 requires. Recorded as a gap in `PLAN.md` §1.

### 2026-09-11 · Anurag · Claude Code (Opus 5) · Kaggle as the compute target
- Asked: "We have kaggle setup also … kaggle.json … 2*t4 (fp16 was faster on the T4s). No issues
  with ram now … remove if you mention (ram restriction or something)".
- Produced (AI-generated, to be reviewed by Anurag): `kaggle` CLI installed in `.venv` and added to
  `requirements.txt`; RAM-constraint wording removed from `CLAUDE.md` §6 and `PLAN.md`
  (the "if RAM allows" in D5, and the descope rung for MINDlarge_train fitting); Kaggle workflow
  added to P0; new decision D9 (data source on Kaggle); `CONTEXT.md` C-004.
- Verified by: `kaggle kernels list --mine` authenticated (read-only). The credential file was
  checked by path and permissions only; its contents were never printed.
- Failed / corrected: the instruction read as "RAM is no longer an issue", but the laptop measured
  7 GB total with ~1.4 GB available and no GPU. The docs therefore say "RAM is not a constraint on
  Kaggle" and keep full-scale runs off the laptop, instead of removing the limit everywhere.

### 2026-09-11 · Anurag Kaushal · Claude Code (Opus 5) · Final team roster and ownership
- Asked: "Update `PLAN.md` and `CONTEXT.md` to reflect our finalized team roster and ownership
  division." Team: Anurag Kaushal, Aayush Pandey. Allocation: P0 both; P1, P2 and P3.1 (NRMS on
  Kaggle 2× T4, fp16) Anurag; P3.2–3.4 joint, with Aayush driving the paired bootstrap CI harness;
  P4 and P5 Aayush; P6 both. Append C-005 and an AI_USAGE entry.
- Produced (AI-generated, to be reviewed by Anurag):
  - `PLAN.md` §0: a team row and a new "critical path" consequence. §2: rewritten as the final
    ownership table, the builder-vs-judge rationale, and the scores-file handoff. §3: phase table
    with named owners.
  - The per-phase headings, P0 steps and the P6 checklist now use names instead of stream A/B and
    "partner".
  - `CONTEXT.md` §1: current state with names and assignments. §2: new entry C-005.
- One addition beyond the literal request, flagged for review: the new ownership puts P1, P2 and
  P3.1 on Anurag in the same 12–15 Sep window, and Aayush's phases depend on their output. To avoid
  Aayush sitting idle until 16 Sep, the plan now has Aayush build each harness against A1's
  existing outputs first. That added a phase row, P3.4a (paired bootstrap, 12–14 Sep), and moved
  the start of P4 to 14 Sep and P5 to 13 Sep; ownership is unchanged. Reasoning is in C-005.
- Verified by: `git diff --name-only` shows only `CONTEXT.md` and `PLAN.md` changed, so no code,
  test, split, feature or `.gitignore` rule was touched. `make test`'s suite still passes (212
  tests). The §4 invariants (temporal split, no leakage, serving-time honesty, gap-aware selection,
  reproducibility, paired CIs, no large files) are unaffected by a documentation-only change.
- Failed / corrected: the first draft of the `CONTEXT.md` state row used a gendered pronoun for
  Aayush, whose pronouns haven't been stated. It was replaced with the name. Old log entry C-002
  still says "partner" and was deliberately not edited (append-only log); C-005 records that it
  means Aayush Pandey.

### 2026-09-11 · Anurag Kaushal · Claude Code (Opus 5) · P1 start: recency profile spec + oracle (red)
- Asked: "We are officially starting Phase 1 … start with just the first new feature: the
  recency-weighted user profile (exponential decay)." In order: (1) a SPEC.md sub-section with
  inputs `(user, candidate, t)`, outputs, the decay math with half-life *h*, and the strict
  before-*t* boundary; (2) a 20-event toy log as a DataFrame in a new test file; (3) a pytest
  leakage test with hand-computed decay weights for one user at one *t*, covering an event
  strictly before *t*, an event at exactly *t*, and an out-of-order event; (4) **do not implement
  the feature yet**, verify the tests fail, update CONTEXT.md, and log here.
- Produced:
  - `SPEC.md` §11 / §11.1 (AI-generated, to be reviewed by Anurag): contract for
    `decay_weights` and `recency_weighted_profile`, wᵢ = 2^(−Δᵢ/h), the strict `ts < t` boundary,
    NaN for no history, `ValueError` on `h ≤ 0` or a null `ts`, open decisions P1-D1 and P1-D2,
    and the verification table.
  - `tests/test_behavioural_features.py` (AI-generated, to be reviewed): a 20-event, 3-user toy
    log and 16 tests. The expected values were hand-computed as literals; the arithmetic was run
    once in a Python scratch shell before writing them. Besides the three required cases, the
    tests also cover an event after *t*, other users' events, a user with no eligible history,
    and invalid input.
  - `CONTEXT.md`: current state and entry C-006.
  - `src/features/`: **not touched**, as instructed.
- Verified by: the new file gives **2 passed / 14 failed**. The 2 passes are the oracle's
  self-checks: the toy log contains every edge case, and every hand-written weight equals
  2^(−age/h) recomputed from the log's timestamps. All 14 failures are
  `ModuleNotFoundError: No module named 'src.features.behavioural'`, with no other failure type.
  The existing 212 tests still pass.
- Failed / corrected:
  - While writing the spec, a check of the real data showed that **MIND's history has no per-click
    timestamps** (an id list only), whereas EB-NeRD's does. A time-decay feature cannot be
    computed from MIND history as-is. Instead of assuming timestamps that don't exist, this is
    recorded as open decision P1-D1.
  - A1's position-based `recency_pool` would fail the out-of-order test. That is documented as
    the reason the A2 definition is time-based.
- Limitation, stated: a `ModuleNotFoundError` proves the tests are wired to the module, not that
  each assertion can catch its bug. That needs a mutation check (a `<=` boundary, position-based
  decay, a missing user filter) when the implementation is written. Deferred, because the
  near-correct mutants would amount to implementing the feature, which this step forbade.

### 2026-09-11 · Anurag Kaushal · Claude Code (Opus 5) · P1: recency_weighted_profile, mutation check, MIND fallback
- Asked: "We are now implementing the recency_weighted_profile feature in
  src/features/behavioural.py … Test the Traps: … write a deliberately broken implementation first
  … Use <= instead of < … Use list-position for decay … Omit the user ID filter … Run make test and
  confirm that the specific boundary, out-of-order, and cross-user tests fail as expected. Correct
  Implementation: … strictly < t, timestamp-based exponential decay (w = 2^(−(t − ts)/h)), exact
  user filtering, and returning NaN for empty histories. MIND Fallback (P1-D1): … If a click lacks
  a timestamp, default to the known start time of the dataset split … Run make test. Ensure all
  228 tests pass green. Update CONTEXT.md with two new decisions: C-006 … and C-007 … Append this
  prompt and execution summary to AI_USAGE.md."
- Produced (AI-generated, to be reviewed by Anurag):
  - `src/features/behavioural.py`: `decay_weights`, `recency_weighted_profile` (keyword
    `untimed_ts` for the MIND fallback) and `SERVING_OK`.
  - `tests/test_behavioural_features.py`: +8 tests (`TestMindUntimedFallback`, hand-computed:
    2⁻⁵ = 0.03125 for a click stamped 120 h before t; 2/4 and 1/4 for a history-only profile).
  - `SPEC.md` §11.1: `untimed_ts` in the contract, P1-D1 and P1-D2 rewritten with measurements,
    fallback rows, and a mutation-check table.
  - `CONTEXT.md`: current state, C-007, C-008.
  - Scratch only, not committed: the three-bug mutant file, `isolate_mutants.py`, and the logs
    `mutant_all3.log`, `mutants_isolated.log`, `mind_history_check.log`, `make_test_final.log`.
- Verified by:
  1. **Three-bug mutant, via `make test`:** 10 failed, 218 passed. The boundary, out-of-order and
     cross-user tests all failed. `test_event_after_t_is_excluded` correctly still passed, since
     `<=` still excludes the future.
  2. **Each bug alone, in an otherwise-correct implementation:** `<=` gives 8/16 failing,
     including `test_event_at_exactly_t_is_excluded`. Position decay gives 6/16, including the
     out-of-order test. No user filter gives 7/16, including the cross-user test. The control
     gives 0/16.
  3. **Correct implementation, via `make test`:** **236 passed**, 0 failed, exit 0. The 6 warnings
     are A1's existing Polars deprecations in `test_no_leakage.py`.
  4. **MIND facts measured from `behaviors.tsv`:**
     - split starts: 9 / 15 / 16 Nov 2019, 00:00;
     - first impressions 19 s / 1 s / 5 s after midnight;
     - history varies within a split for 0 repeat users in all three splits;
     - history is identical in train and dev for all 5,943 shared users.
- Where the result departs from the prompt, and why:
  - **The test count is 236, not 228.** 228 was the count before the MIND fallback existed. The
    fallback is new logic and needed its own oracle (`CLAUDE.md` rule 2), so 8 tests were added.
    All 236 pass.
  - **Entry numbers are C-007 and C-008, not C-006 and C-007.** C-006 already existed from the
    previous step and already recorded the out-of-order and NaN choices. The log is append-only,
    so C-007 confirms them with the mutation evidence, and C-008 records P1-D1 and P1-D2.
  - **The fallback is an explicit keyword (`untimed_ts`), not a silent default for null
    timestamps.** Filling nulls silently would break the spec's rule that a null `ts` raises,
    which was written to prevent exactly this kind of hidden leak or loss. Without the keyword,
    nulls are still rejected.
- Failed / corrected:
  - The first MIND check script crashed. It passed `…/behaviors.tsv` to `scan_behaviors`, which
    expects the split *directory*. Fixed after reading `split_dir`'s contract; no result came
    from the crashed run.
  - After the combined-mutant run, I told Anurag it failed "11 tests"; the log says 10. Corrected.
  - The claim that `dt.total_seconds()` truncates was written into C-007 before it was checked.
    It was then verified (1.5 s → `1`, `Int64`, on polars 1.43.2) before this entry was written.
  - **Finding that limits the feature:** because MIND history is a frozen, untimed snapshot, the
    recency profile on MIND equals the undecayed category distribution for every half-life. The
    decay feature and the half-life ablation are therefore meaningful on EB-NeRD only. This is
    recorded in C-008 and `SPEC.md` so no MIND decay effect gets claimed.
