"""The locked Phase 2 reranker configuration (CONTEXT.md C-018, SPEC.md §12.1).

The single source of truth for what ships: the submission runs (Q5) read this, and both reranker
scripts assert that it matches the models they measured. Changing it is a decision — log it in
CONTEXT.md and re-measure.

Why each dataset ended up where it did (RESULTS.md Q2):

* **EB-NeRD — lambdarank, A1 base + category profile + list position + session (11 features).**
  Lambdarank beat pointwise by +0.034 AUC on the base features. The dwell family hurt the listwise
  model (−0.015 AUC when present); without it, A2 beats the base on all four metrics, and that held
  on 144,647 held-out impressions (AUC +0.0029 [+0.0018, +0.0039]).
* **MIND — pointwise HistGBDT, A1 v4's 7 features.** Lambdarank lost to pointwise (−0.0087 AUC)
  and a truncation level covering the longest slate did not change that, in either half of dev.
  No Phase 1 addition beat the base under either objective on MIND.
"""

FINAL = {
    "ebnerd": {
        "objective": "lambdarank",
        "features": ["bm25", "semantic", "pop_total", "ctr_total", "freshness_hours", "n_candidates",
                     "history_len", "recency_weighted_profile", "category_match", "cand_position",
                     "session_pos"],
    },
    "mind": {
        "objective": "pointwise",
        "features": ["bm25", "semantic", "pop_total", "ctr_total", "n_candidates", "cat_affinity",
                     "history_len"],
    },
}
