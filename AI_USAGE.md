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

### 2026-09-11 · Anurag Kaushal · Claude Code (Opus 5) · Commit/push; P1 batch parity; category_match oracle (red)
- Asked: "First, commit the currently staged files with a clear message detailing the recency
  feature, the TDD leakage checks, and the MIND fallback logic. Push the commit to a2-click-logs
  so Aayush can successfully run his Phase 0 clean-clone check. Next, continue Phase 1 … Batch
  Scaling: Write a batch-processing version of the recency_weighted_profile (using Polars or
  PyArrow) … Batch Oracle: Write a test asserting that this new batch implementation produces the
  exact same numerical outputs as our current row-by-row version when run against the 20-event
  toy log. Category Match Spec & Oracle: Update SPEC.md for the category_match feature … Extend
  the toy log and write the failing leakage/correctness tests … Halt: Do not implement the
  category_match logic … Run make test to confirm the new tests fail, log this decision block in
  CONTEXT.md (C-009 for batch scaling parity), and record this prompt in AI_USAGE.md."
- Produced (AI-generated, to be reviewed by Anurag):
  - Commit `54cf185` (message written by the agent, with no AI trailer, per Anurag's standing
    preference).
  - `recency_profile_batch` in `src/features/behavioural.py`.
  - In `tests/test_behavioural_features.py`: `TestBatchParity` (6 tests), the extended toy log
    (27 events), `TestCategoryMatch` (10 tests, failing by design) and 2 oracle self-checks.
  - `SPEC.md` §11.2 (batch path) and §11.3 (`category_match`).
  - `CONTEXT.md`: current state, C-009, C-010.
- Verified by:
  1. **Clean clone before the push.** A fresh `git clone` of the branch into the scratchpad, with
     no `data/`, ran the suite: 236 passed. This checks that no test depends on untracked files.
  2. **Batch TDD.** The parity tests were written first and failed with `AttributeError` (6/6). The
     implementation then passed 6/6 over 360 requests (128 NaN / 150 exact-zero / 82 values).
  3. **Batch mutation check.** `<=` in the batch fails 4/6; no user filter fails 3/6. The file was
     restored and verified byte-identical with `cmp`.
  4. **Category match red state, via `make test`:** 10 failed, 244 passed. All 10 are
     `AttributeError: … no attribute 'category_match'`.
- Where the result departs from the prompt, and why:
  - **The push did not happen.** `git push` failed with "could not read Username for
    'https://github.com'". The agent shell has no GitHub CLI, credential helper, SSH key or
    VS Code askpass. The commit is local, and Anurag must push it. Nothing was changed in the git
    config to work around this.
  - **Parity runs on the extended 27-event log as well as the 20-event one**, because more
    edge cases give a stronger oracle. It is not bitwise-exact: NaN and 0.0 must match exactly,
    and other values within 1e-12 relative. The batch path sums in a different order and float
    addition is not associative, so demanding bitwise identity would be a false requirement.
  - **A C-010 entry was added** alongside the requested C-009. The cosine definition of
    `category_match` is a design decision, and `CLAUDE.md` §3 requires logging it.
  - **`category_match` is specified as a cosine, not a literal dot-product "similarity".** The dot
    product with a one-hot category equals the existing `recency_weighted_profile` exactly, so it
    would be a duplicate feature. This is flagged for Anurag's confirmation, with three
    alternatives listed.
- Failed / corrected:
  - The first SPEC draft said an EB-NeRD impression lists "~11 candidates", copied from an A1
    script comment. Measured, it is **12.0** (EB-NeRD small validation); MIND's "~37" held (37.2
    / 37.5 / 39.3). Corrected before logging.
  - Mutation finding: the hand-computed-values test alone does not catch a missing user filter
    in the batch path (the semi-join hides it for a single user). Only the many-user parity grid
    does. Recorded in SPEC §11.2, as evidence that both kinds of test are needed.
- Open risk, flagged and not acted on: `requirements.txt` has no version pins. Aayush's `make env`
  may resolve different Polars or numpy versions than ours (polars 1.43.2), so the clean-clone
  pass seen locally is not guaranteed on his machine.

### 2026-09-11 · Anurag Kaushal · Claude Code (Opus 5) · Pins, category_match, Phase 1.2 session features (one commit)
- Asked (condensed from a long prompt; the key instructions are verbatim):
  - "Dependency Pinning: … immediately freeze the package versions in requirements.txt
    (including polars==1.43.2, kaggle==2.2.4, etc.)."
  - "I confirm the cosine similarity (P(c)/‖P‖₂) choice for category_match … Implement the
    row-by-row and batch versions."
  - "Phase 1.2 Session Features (Schema Integration): Aayush has provided the exact schema mapping
    …" This covered: slate `cand_position`, `n_candidates`; session `session_pos`, `session_len`,
    `n_prior_clicks_in_session`; dwell `hist_read_time_mean`, `hist_scroll_mean`; and the UNSAFE
    `cur_read_time`, `cur_scroll_percentage`, to be tagged `serving_ok: False`.
  - "Implement … Ensure all tests strictly enforce the behaviour-window boundary."
  - Log the pinning and the Phase 1.2 integration in CONTEXT.md; log here.
  - "Once make test is entirely green, stage all files and create a single, comprehensive
    commit."
- Produced (AI-generated, to be reviewed by Anurag):
  - `requirements.txt`: 64 exact pins.
  - `src/features/behavioural.py`: `category_match`, `category_match_batch`, the shared helpers
    `_decayed_category_mass` / `_lookup`, `slate_features`, `session_features`, `dwell_features`,
    `current_page_features`, and the registry (`SERVING_OK`, `UNSAFE_FEATURES`,
    `ABSENT_FROM_TEST_FILE`, `drop_unsafe`).
  - `tests/test_behavioural_features.py`: +6 batch parity tests.
  - `tests/test_session_features.py`: new, 19 tests.
  - `scripts/check_phase1_data.py` and `make check-data`: new.
  - `SPEC.md` §11.3–§11.7, `RESULTS.md` Q1, `CONTEXT.md` (state, C-011–C-013).
- Verified by:
  1. **Pins.** A fresh venv from the pinned file gave a clean `pip check`, and `pip freeze` equals
     all 64 pins. The full suite in that venv: 279 passed.
  2. **TDD, every unit red before green.** 10 `category_match` tests, 6 batch parity tests and 19
     Phase 1.2 tests all failed first with `AttributeError`, then passed.
  3. **Planted bugs, each alone:**
     - category_match: dot product 4/10; batch L1 norm 4/6;
     - Phase 1.2: session `<=` 4/19; session keyed on `session_id` only 5/19; dwell `<=` 4/19.

     Each file was restored and verified byte-identical with `cmp`.
  4. **`make test`: 279 passed, exit 0.** The 6 warnings are A1's existing ones in
     `test_no_leakage.py`. A new warning from `explode` in our code was found and fixed
     (`empty_as_null=False`).
  5. **Data facts:** `make check-data` reproduces every number cited in SPEC §11.4–§11.7
     (`RESULTS.md` Q1).
- Where the result departs from the prompt, and why:
  - **`session_len` is tagged UNSAFE, although the mapping listed it as safe.** It counts the
    session's future impressions, and a test shows a future impression changes it. Implementing it
    as "safe" would violate the `CLAUDE.md` §4 no-leakage invariant, and the prompt itself asked
    that all tests enforce the boundary. For Aayush to review (C-013).
  - **`n_prior_clicks_in_session` is implemented, but flagged `ABSENT_FROM_TEST_FILE`.** The
    measured test file has no `article_ids_clicked`. When clicks are absent the column is omitted,
    never zero-filled. The feature is also ≈ `session_pos` − 1, since every train impression has
    at least one click.
  - **Sessions are keyed by `(user_id, session_id)`, not `session_id`.** The test file reuses
    `session_id` 0 across 200,000 beyond-accuracy rows.
  - **`cand_position` is documented as list position, not "display order"**, because nothing we
    verified establishes on-screen order. What was measured: click rates are nearly flat by
    position, lists are not id-sorted, and the ordering is the same in train and test.
  - **Additions beyond the prompt:** `scripts/check_phase1_data.py` + `make check-data`, so the
    data facts are reproducible by Aayush rather than living in agent scratch files (`CLAUDE.md`
    rules 3 and 6), and `RESULTS.md` Q1.
  - **C-010's status line gained a pointer** ("Confirmed in C-012"). It is the same kind of pointer
    the append-only rule allows for "superseded by"; its content is unchanged.
- Failed / corrected:
  - **The first position measurement was killed by the OOM killer** (exit 137). It exploded every
    EB-NeRD train slate on the 7 GB laptop. Rerun on 50,000-impression samples; full-split
    statistics are streamed.
  - **The session restarted mid-block.** The background pin-verification job had finished before
    it: its log showed `FREEZE_MATCHES_PINS` and `PIP_CHECK_EXIT=0`. This was checked rather than
    assumed.
  - **Two measurement bugs, caught before they reached a document as fact:**
    - a session-pair count overflowed `UInt32` (fixed by casting to `Int64`);
    - the test-file tied-timestamp count (196 → **195**) had been grouped by `session_id` alone.
      SPEC was corrected.
  - **A second finding from the "Aayush mapping" review:** the assumption that
    `n_prior_clicks_in_session` was usable for submission would have failed silently at test time.
    This was caught by listing the test file's columns before writing the feature.
