"""Oracle for the EB-NeRD candidate frame: label construction and row order (SPEC.md §12).

Clicked ids -> per-candidate 0/1 labels is the trap A1's SPEC §2 names: a wrong mapping yields a
well-formed frame that scores like noise. Hand-built toy impressions, hand-written expectations.
"""

from datetime import datetime

import polars as pl

from src.rerank.ebnerd import candidate_frame

T = datetime(2023, 5, 25, 12, 0, 0)


def toy_behaviors(labelled: bool = True) -> pl.DataFrame:
    df = pl.DataFrame({
        "imp_row": [0, 1, 2],
        "impression_id": [500, 0, 0],                      # duplicate ids, as in the test file
        "user_id": [1, 2, 3], "session_id": [9, 8, 7], "t": [T, T, T],
        "candidates": [[30, 10, 20], [40], [10, 50]],
        "clicked": [[10], [40], [50, 50]],                 # a duplicated click id
    }, schema_overrides={"imp_row": pl.UInt32, "candidates": pl.List(pl.Int32), "clicked": pl.List(pl.Int32)})
    return df if labelled else df.drop("clicked")


ARTICLES = pl.DataFrame({"article_id": [10, 20, 30, 40, 50], "category": [1, 2, 1, 3, 2],
                         "published_time": [T] * 5}, schema_overrides={"article_id": pl.Int32})


def test_labels_positions_and_categories_match_hand_computation():
    out = candidate_frame(toy_behaviors(), ARTICLES)
    assert list(out.select("imp_row", "article_id", "cand_position", "n_candidates", "label",
                           "candidate_category").iter_rows()) == [
        (0, 30, 1, 3, 0, 1), (0, 10, 2, 3, 1, 1), (0, 20, 3, 3, 0, 2),
        (1, 40, 1, 1, 1, 3),
        (2, 10, 1, 2, 0, 1), (2, 50, 2, 2, 1, 2),
    ]


def test_duplicate_impression_ids_stay_distinct_rows():
    out = candidate_frame(toy_behaviors(), ARTICLES)
    assert out.filter(pl.col("impression_id") == 0)["imp_row"].unique().sort().to_list() == [1, 2]


def test_unlabelled_split_has_no_label_column():
    """The test file: no clicks, so no label — never a column of zeros."""
    assert "label" not in candidate_frame(toy_behaviors(labelled=False), ARTICLES).columns
