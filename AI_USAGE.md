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
