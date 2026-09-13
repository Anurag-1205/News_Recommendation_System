# CLAUDE.md — Assignment 2 (IRE CS4.406)

Operating rules for any coding agent working in this folder. Derived from Module 1 / L1
"Learn the modern dev stack" (slides 9–10) and the A2 brief (`A2.pdf`). Two people work on this
repo, each with their own agent. **Read this file, then `CONTEXT.md`, then `PLAN.md`, before
touching code.**

---

## 1. What this repo is

**A2 — Learning from Click-Logs on EB-NeRD and MIND.** Team of 2, due **Sun 20 Sep 2026**.
It builds on A1 (lexical + semantic retrieval), which is frozen on `main` at `be15ee6`. All A2 work
happens on branch **`a2-click-logs`**.

Add behavioural signals from click logs to A1's retrieval pipeline: a two-stage retrieve-then-rank
system, an official baseline reproduced and then beaten with a paired-bootstrap-significant
ablation, and a serving/scale analysis.

| # | Deliverable | Done means |
|---|---|---|
| Q1 | Click-history & session features | history (recency-decayed), session, dwell, position, popularity, freshness, category match; **leakage test** |
| Q2 | Two-stage reranker | A1 candidates top-K (K 100–200) → GBDT or small neural ranker; AUC·MRR·nDCG@5·nDCG@10 **before and after** |
| Q3 | Baseline reproduced, then beaten | NRMS (ebnerd-benchmark) on both datasets; one principled change; ablation; **paired bootstrap 95% CI excluding 0** |
| Q4 | Serving & scale | index + feature-store memory, **p99 latency** per request, cost/1k queries at p99 < 100 ms, 10× argument |
| Q5 | Extended eval | all metrics + diversity/novelty/coverage; slices **cold vs warm** and **head vs tail**; CIs; both leaderboards + screenshots |
| Q6 | Design note | PDF, ~6 pages, 11pt, 1-inch margins |
| Q7–Q9 | Deliverables, git policy, anti-gaming | README one-command reproduce; AI usage log for both members; with/without serving-unavailable features |

Grading is on pipeline correctness, system design, ablation rigour, scale analysis and note clarity.
**Never on leaderboard rank.** Optimising rank at the cost of any of those is a wrong trade. Say so
if asked to make it.

The exam probes *our* understanding of this code. Code that neither of us can explain is worth
zero, so favour the explainable implementation over the clever one, and explain as you go.

---

## 2. The agent mind — nine rules from L1

These are course rules, not suggestions. They override your defaults.

**1. The spec is the real work.** Start in **plan mode**. Before code, extend `SPEC.md`: inputs,
outputs, data structures, interfaces, **and how it will be verified**. `SPEC.md` is graded.

**2. Never accept work you can't check.** Every task needs an oracle *before* the implementation:
a hand-computed feature on a toy log, a known metric value, a paired-bootstrap result on a
constructed difference. No oracle → build the oracle first.

**3. Measure, don't vibe.** Each claim needs a number, a baseline and the command that produced
it. Report every choice's impact on **Reads / Updates / Memory**, plus latency where it applies.

**4. Right-size the task.** One reviewable unit at a time ("add the decayed category profile with
its toy-log test", never "do Q1"). If handed something too big, decompose it and confirm the
decomposition first.

**5. Context is scarce.** Keep this file lean. Write large outputs to files and cite the path.
Ask for `/clear` on context clash, poisoning or distraction.

**6. Verification is deterministic, not conversational.** Bake checks into `make data`,
`make test`, `make eval`, `make bench`. If you re-type a verification, turn it into a target.

**7. The human reads every diff and owns every line.** Keep diffs small and self-contained, don't
reformat unrelated code, and flag anything subtle in the summary.

**8. Teach, don't just do.** At a real fork (pointwise vs lambdarank; NRMS vs GBDT for the
improvement; decay half-life), name the options, state the trade-off, recommend one, and let the
human decide. Then log the decision (§3).

**9. Disclose and document.** Append to `AI_USAGE.md` as you work, and **tag every entry with the
team member** whose session it was: tool, key prompts, what worked, **what failed**, and
AI-generated vs hand-written code. A missing entry is a broken build.

---

## 3. Team protocol — `CONTEXT.md` is the shared memory

Two agents cannot see each other's conversations. `CONTEXT.md` is the only channel between them,
so these rules are non-negotiable:

- **Session start:** `git pull --rebase`, then read `CONTEXT.md` (current state + decision log)
  and the relevant `PLAN.md` phase. Do not re-open a logged decision without new evidence.
- **Log every decision in the same change as the code it affects.** "Decision" means any fork that
  was resolved, a threshold or hyperparameter that was fixed, a file deleted or moved, an interface
  or file format changed, a dependency added, or a finding that changes the plan. Use the entry
  format at the top of the decision log, with the next free `C-NNN` number.
- **Append-only.** Never edit or delete another entry. To reverse one, add a new entry and mark the
  old one `superseded by C-NNN`.
- **Session end:** overwrite §1 "Current state" (what you finished, what is in flight, what the
  other person should pick up next) and date it.
- **Ownership:** `PLAN.md` §2 assigns streams. Before editing a file in the other stream, note it
  in `CONTEXT.md` so it is not a surprise at the next pull.
- **Git:** agents never commit or push. Stage the work, summarise it, and stop; the human commits.
  No force-push, ever, on the shared branch.

---

## 4. Non-negotiable invariants

Violating any of these silently invalidates every number in the report. Check them before
claiming a task is done. If a request would break one, stop and say so.

- **Temporal split only.** Never random-split interaction data (`SPEC.md` §3).
- **No future-click leakage.** Every feature at time *t* sees only events strictly before *t*,
  at training *and* serving time. **A test must assert it for every new feature** (Q9).
- **Serving-time honesty.** Report metrics with and without serving-unavailable features. Such a
  feature is an ablation row, not a result.
- **Model selection mimics the test gap.** Use the gap-aware protocol (`SPEC.md` §7), not the
  adjacent dev split. A1 lost 0.090 AUC to this.
- **Reproducible from raw.** One command rebuilds everything; seeds are fixed and recorded.
- **Claimed gains carry a paired bootstrap 95% CI that excludes zero.** Overlapping or
  zero-crossing intervals are not a win, so don't write "beats".
- **No large files in git.** `*.zip *.pt *.ckpt __pycache__/ data/` stay ignored, and so do
  external repos (`external/`) and model weights.
- **Never print or commit `kaggle.json`** or any other credential.

---

## 5. Working loop

```
git pull → read CONTEXT.md → plan mode → SPEC.md entry → oracle/test → smallest implementation
   → make test → make eval / make bench → RESULTS.md → CONTEXT.md entry → AI_USAGE.md entry → stage
```

- Announce which rung you're on.
- Land the toy-scale path first (MIND-small, EB-NeRD demo) locally, then scale up on Kaggle. Never
  debug at 13.5M impressions when 500 rows reproduce the bug.
- When a run takes minutes, save its output to a file and cite the path. **Every Kaggle kernel
  run gets a row in `scripts/kaggle/RUN_LEDGER.md`** via `scripts/kaggle/ledger.py <kernel> <vN>`,
  failures included; the log itself goes to `data/logs/kaggle/`.

## 6. Compute, scale & cost discipline

- **Where things run** (`CONTEXT.md` C-004):
  - **Kaggle: 2× T4, RAM not a constraint.** All GPU work and every full-scale run (MINDlarge,
    EB-NeRD large, 13.5M-impression test inference) runs here, via the `kaggle` CLI
    (`~/.kaggle/kaggle.json`).
  - **The laptop (7 GB, no GPU)** is for dev, `make test` and toy-scale runs. Don't launch
    full-scale jobs on it.
- **GPU defaults on T4:**
  - Use **fp16** autocast + `GradScaler`, not bf16. T4 (Turing) has no bf16 tensor cores, and fp16
    measured faster.
  - With two GPUs, running two experiments side by side (one per GPU) is usually simpler than DDP
    for models of NRMS's size.
- **Kaggle sessions have a time limit and a weekly GPU quota.** Long runs write output in chunks
  and resume, so they don't restart from zero. Check the account's current limits rather than
  assuming them.
- **Sequential beats random by ~100×.** Layout is the design.
- Before optimising, find the bottleneck by service demand (D = V·S) instead of guessing.
- Q4 and the "breaks at 10×" section are graded. Collect index sizes, peak RSS, latency
  percentiles and throughput as you build.

## 7. Repo shape

```
CLAUDE.md  PLAN.md  CONTEXT.md   agent rules · plan · shared decision log
SPEC.md                          interfaces + verification        (graded)
RESULTS.md                       every number + its command
AI_USAGE.md                      per-member AI log                (graded)
README.md  Makefile              one-command reproduce · data test eval bench
src/  pipeline/ lexical/ semantic/ features/ baselines/ eval/   (+ rerank/, serving/ as A2 adds them)
tests/                           includes the no-leakage assertions
scripts/                         experiment drivers
external/                        third-party baseline code (gitignored, commit pinned in CONTEXT.md)
data/                            gitignored
```

## 8. Definition of done (per task)

1. `SPEC.md` describes it and how it is verified.
2. A test or oracle passes, and it would have failed before the change.
3. Numbers are in `RESULTS.md` with the exact command, on both datasets where applicable.
4. The invariants in §4 still hold.
5. `CONTEXT.md` has the decision entry and an updated current state.
6. `AI_USAGE.md` records the prompt, the outcome and any failure caught, tagged by member.
7. A small diff is staged for the human to review and commit.

Report honestly: if a metric got worse, say so with the output. If a step was skipped, say which.
Never report a number you did not run.
