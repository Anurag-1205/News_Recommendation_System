"""Oracles for src/baselines/nrms_data (SPEC.md §13.4).

The NRMS kernels feed the benchmark's dataloader from frames built here, and write scores keyed
by `imp_row`. If either the row key or the slate order drifted from what `src/rerank` uses, the
paired bootstrap would silently compare the wrong impressions, so the tests are parity checks
against the reranker's own loaders on the first rows of the real split (skipped when `make data`
has not run), plus exact toy cases for the scores frame and the history truncation.
"""
from pathlib import Path

import polars as pl
import pytest

from src.baselines.nrms_data import ebnerd_behaviors, scores_frame, truncate_history

SMALL = Path("data/interim/ebnerd/ebnerd_small")
needs_data = pytest.mark.skipif(not (SMALL / "validation" / "behaviors.parquet").exists(),
                                reason="run `make fetch-small && make data` first")


# ---- (a), (b): parity with the reranker's loaders on real rows ----------------------------------

@needs_data
def test_keys_and_slates_match_reranker_loader():
    from src.rerank.ebnerd import load_behaviors
    ours = ebnerd_behaviors(SMALL / "validation", history_size=20, limit=300)
    theirs = load_behaviors(SMALL / "validation" / "behaviors.parquet", limit=300)
    assert ours.height == theirs.height == 300
    assert ours["imp_row"].to_list() == theirs["imp_row"].to_list()
    assert ours["impression_id"].to_list() == theirs["impression_id"].to_list()
    assert ours["user_id"].to_list() == theirs["user_id"].to_list()
    assert ours["article_ids_inview"].to_list() == theirs["candidates"].to_list()
    assert ours["article_ids_clicked"].to_list() == theirs["clicked"].to_list()


@needs_data
def test_labels_match_reranker_candidate_frame():
    from src.rerank.ebnerd import candidate_frame, load_articles, load_behaviors
    ours = ebnerd_behaviors(SMALL / "validation", history_size=20, limit=300)
    theirs = candidate_frame(load_behaviors(SMALL / "validation" / "behaviors.parquet", limit=300),
                             load_articles())
    flat = (ours.select("imp_row", "article_ids_inview", "labels")
            .explode("article_ids_inview", "labels")
            .with_columns(cand_position=pl.int_range(1, pl.len() + 1).over("imp_row"))
            .join(theirs.select("imp_row", "cand_position", "label"), on=["imp_row", "cand_position"], how="inner"))
    assert flat.height == theirs.height
    assert (flat["labels"] == flat["label"]).all()


@needs_data
def test_history_is_left_padded_to_history_size():
    ours = ebnerd_behaviors(SMALL / "validation", history_size=20, limit=300)
    lens = ours["article_id_fixed"].list.len().unique().to_list()
    assert lens == [20]


# ---- (c), (d): the scores frame --------------------------------------------------------------------

TOY = pl.DataFrame({
    "imp_row": pl.Series([0, 1, 2], dtype=pl.UInt32),
    "impression_id": pl.Series([100, 0, 0], dtype=pl.UInt32),   # repeated impression_id, as in the test file
    "article_ids_inview": pl.Series([[7, 8, 9], [8], [9, 9]], dtype=pl.List(pl.Int32)),  # duplicated id in row 2
})


def test_scores_frame_exact_rows_dtypes_order():
    out = scores_frame(TOY, [[0.1, 0.3, 0.2], [0.5], [0.9, 0.8]])
    assert out.columns == ["imp_row", "impression_id", "article_id", "cand_position", "score"]
    assert out.dtypes == [pl.UInt32, pl.UInt32, pl.Int32, pl.Int64, pl.Float64]
    assert out["imp_row"].to_list() == [0, 0, 0, 1, 2, 2]
    assert out["cand_position"].to_list() == [1, 2, 3, 1, 1, 2]
    assert out["article_id"].to_list() == [7, 8, 9, 8, 9, 9]
    assert out["score"].to_list() == [0.1, 0.3, 0.2, 0.5, 0.9, 0.8]


def test_scores_frame_keeps_duplicate_article_rows():
    out = scores_frame(TOY, [[0.1, 0.3, 0.2], [0.5], [0.9, 0.8]])
    assert out.filter(pl.col("imp_row") == 2).height == 2


def test_scores_frame_rejects_length_mismatch():
    with pytest.raises(ValueError):
        scores_frame(TOY, [[0.1, 0.3], [0.5], [0.9, 0.8]])
    with pytest.raises(ValueError):
        scores_frame(TOY, [[0.1, 0.3, 0.2], [0.5]])


def test_scores_frame_accepts_numpy_rows():
    import numpy as np
    out = scores_frame(TOY, [np.array([0.1, 0.3, 0.2]), np.array([0.5]), np.array([0.9, 0.8])])
    assert out.height == 6


# ---- truncate_history: the benchmark's docstring example --------------------------------------------

def test_truncate_history_matches_benchmark_docstring():
    df = pl.DataFrame({"id": [1, 2, 3], "history": [["a", "b", "c"], ["d", "e", "f", "g"], ["h", "i"]]})
    assert truncate_history(df, "history", 3)["history"].to_list() == [["a", "b", "c"], ["e", "f", "g"], ["h", "i"]]
    assert truncate_history(df, "history", 3, "-")["history"].to_list() == [["a", "b", "c"], ["e", "f", "g"], ["-", "h", "i"]]


# ---- the column names the benchmark's dataloader reads ---------------------------------------------

def test_columns_match_benchmark_constants():
    """Run v1 of the EB-NeRD kernel died on `article_ids_fixed` vs the benchmark's
    `article_id_fixed`: the parity tests compared values, never names. The names come from
    ebrec's constants when external/ is checked out, else the literals they had at 5164e2c."""
    try:
        import sys
        sys.path.insert(0, "external/ebnerd-benchmark/src")
        from ebrec.utils._constants import (DEFAULT_HISTORY_ARTICLE_ID_COL as HIST, DEFAULT_INVIEW_ARTICLES_COL as INVIEW,
                                            DEFAULT_CLICKED_ARTICLES_COL as CLICKED, DEFAULT_LABELS_COL as LABELS,
                                            DEFAULT_USER_COL as USER, DEFAULT_IMPRESSION_TIMESTAMP_COL as TS)
    except ImportError:
        HIST, INVIEW, CLICKED, LABELS, USER, TS = ("article_id_fixed", "article_ids_inview", "article_ids_clicked",
                                                   "labels", "user_id", "impression_time")
    if (SMALL / "validation" / "behaviors.parquet").exists():
        cols = set(ebnerd_behaviors(SMALL / "validation", history_size=20, limit=5).columns)
        assert {HIST, INVIEW, CLICKED, LABELS, USER, TS, "imp_row", "impression_id"} <= cols
    assert HIST == "article_id_fixed"     # the literal the adapter writes
