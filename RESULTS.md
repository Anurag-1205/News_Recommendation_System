# Results and Ablations

**CS4.406 Information Retrieval and Extraction — Assignment 2, Learning from Click-Logs**

This document records every measured figure together with the command that produced it, the date
of measurement and the team member who ran it. A figure reported without its originating command
is not treated as a result.

**Statistical convention.** Headline metrics carry a bootstrap 95% confidence interval obtained by
resampling impressions. A claimed improvement is reported only with a **paired** bootstrap 95% CI
on the per-impression difference; the term "beats" is used only where that interval excludes zero.
Where a later measurement contradicts an earlier one, both are retained and the correction is
stated explicitly.

**A1 results** are preserved at the A1 submission commit: `git show be15ee6:RESULTS.md`. The A1
facts that A2 depends on are summarised in `CONTEXT.md` §3.

---

## Q1 · Behavioural features

**No feature-quality number exists yet.** There is no AUC, no ablation and no feature importance,
because nothing has been trained on these features. That arrives with the Q2 reranker. What
follows are (a) the data facts the feature definitions rest on and (b) the verification of the
implementations.

### Data facts behind the Phase 1 definitions — 2026-09-11, Anurag Kaushal

Command: `make check-data` (runs `scripts/check_phase1_data.py`; laptop, about 5 min;
per-candidate click rates use the first 50,000 impressions of each split, and everything else is
the full split).

| Fact | Value | Used by |
|---|---|---|
| MIND split starts (first impression) | 2019-11-09 00:00:19 · 2019-11-15 00:00:01 · 2019-11-16 00:00:05 (train / dev / test) | C-008 `untimed_ts` |
| MIND history varies within a split | for 0 of 33,617 / 14,826 / 484,059 repeat users | C-008 |
| MIND history identical in small train and dev | 5,943 of 5,943 shared users | C-008 |
| Mean candidates per impression | MIND 37.2 / 37.5 / 39.3; EB-NeRD small validation 12.0 | C-009 cost model |
| EB-NeRD columns absent from the test file | `article_id`, `article_ids_clicked`, `next_read_time`, `next_scroll_percentage` | C-013 |
| EB-NeRD inview lists sorted by id | 0.2% (ascending or descending), in train, validation and test alike | C-013 `cand_position` |
| Click rate by list-position quintile, q0 → q4 | EB-NeRD train 0.0942 · 0.0869 · 0.0910 · 0.0914 · 0.0903; val 0.0870 · 0.0830 · 0.0830 · 0.0824 · 0.0801 | C-013 |
| | MIND train 0.0449 · 0.0394 · 0.0430 · 0.0390 · 0.0362; dev 0.0440 · 0.0387 · 0.0433 · 0.0390 · 0.0373 | C-013 |
| Largest "session" when keyed by `session_id` alone (test) | 200,000 impressions: the `is_beyond_accuracy` rows, all `session_id` 0 and `impression_id` 0 | C-013 session key |
| Keyed by (user_id, session_id), test | 6,766,532 sessions, max 118, p99 9, 47,258,574 self-join pairs, 195 with tied timestamps | C-013 |
| Keyed by (user_id, session_id), small train | 120,587 sessions, max 24, p99 8, 731,211 pairs, 5 with tied timestamps | C-013 |
| EB-NeRD impressions with no click (small train) | 0; mean 1.006 clicks per impression | C-013 (`n_prior_clicks_in_session` ≈ `session_pos` − 1) |
| `scroll_percentage` null in behaviors | 70.3% train, 71.6% test | C-013 |
| History clicks with null read time / scroll (small train) | 0 / 10.5% of 2,426,247 | C-013 dwell means |

### Verification of the implementations

| Check | Result | Command |
|---|---|---|
| Full suite, project venv | **279 passed**, 0 failed | `make test` |
| Full suite, a *fresh* venv built from the pinned `requirements.txt` | **279 passed**; `pip check` clean; `pip freeze` equals all 64 pins | see C-011 |
| Planted bugs, each run alone in an otherwise-correct implementation | every one fails its dedicated test (per-bug tables in `SPEC.md` §11.1–§11.7) | scratch scripts, logged in `AI_USAGE.md` |

## Q2 · Two-stage reranker — before vs after

_Not started._

## Q3 · Baseline reproduced, then beaten

_Not started._

## Q4 · Serving and scale

_Not started._

## Q5 · Extended evaluation and leaderboard submissions

_Not started._

## Q9 · With and without serving-unavailable features

_Not started._
