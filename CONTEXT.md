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
| Phase | **P0 Setup**, see `PLAN.md` §3 |
| Team | **Anurag Kaushal**: modelling (P1, P2, P3.1). **Aayush Pandey**: measurement (P3.4a paired bootstrap, P4, P5). Joint: P0, P3.2–3.4, P6. Final (C-005) |
| Anurag Kaushal | done: branch, docs, Kaggle CLI (C-001–C-004). Now: P0 NRMS smoke test on Kaggle (PLAN P0.7). Next: P1 behavioural features (12–13 Sep), with P3.1 NRMS runs in the background |
| Aayush Pandey | not started. Now: P0 clean-clone check (`make env && make test`) and Kaggle verification on Aayush's account (PLAN P0.2–P0.3). Next: P3.4a paired bootstrap harness (12–14 Sep) |
| Compute | Kaggle 2× T4 (fp16) for GPU and full-scale runs; laptop for dev/tests (C-004) |
| Blocked on | team decisions D1–D9 in `PLAN.md` §5; the scores-file format (`PLAN.md` §2) must be agreed and pinned in `SPEC.md` |
| Next up | P0 exit gate: both machines green on `make test`, both Kaggle accounts verified, NRMS runs on EB-NeRD demo |

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
