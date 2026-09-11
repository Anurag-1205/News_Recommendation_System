"""Oracle for the MIND candidate frame and first sightings (SPEC.md §12). Hand-built toy data."""

from datetime import datetime

import polars as pl

from src.rerank.mind import candidate_frame, first_sightings, history_log

T = datetime(2019, 11, 15, 12, 0, 0)
CATS = pl.DataFrame({"article_id": ["N1", "N2", "N3", "N9"], "category": ["news", "sports", "news", "tv"]})


def toy(labelled: bool = True) -> pl.DataFrame:
    df = pl.DataFrame({
        "imp_row": [0, 1], "impression_id": [11, 12], "user_id": ["U1", "U2"],
        "t": [T, datetime(2019, 11, 15, 13, 0, 0)],
        "history_ids": [["N9", "N1"], []],
        "candidates": [["N3", "N1", "N2"], ["N2", "N3"]],
        "labels": [[0, 0, 1], [1, 0]],
    }, schema_overrides={"imp_row": pl.UInt32, "labels": pl.List(pl.Int64), "history_ids": pl.List(pl.Utf8)})
    return df if labelled else df.drop("labels")


def test_labels_attach_by_position_not_by_id():
    """N3 is a non-click in impression 0 but a click in impression 1's list order: by position."""
    out = candidate_frame(toy(), CATS)
    assert list(out.select("imp_row", "article_id", "cand_position", "label", "candidate_category").iter_rows()) == [
        (0, "N3", 1, 0, "news"), (0, "N1", 2, 0, "news"), (0, "N2", 3, 1, "sports"),
        (1, "N2", 1, 1, "sports"), (1, "N3", 2, 0, "news"),
    ]


def test_unlabelled_split_has_no_label_column():
    assert "label" not in candidate_frame(toy(labelled=False), CATS).columns


def test_history_log_has_null_times_and_categories():
    log = history_log(toy(), CATS).sort("user_id", "article_id")
    assert list(log.select("user_id", "article_id", "category").iter_rows()) == [("U1", "N1", "news"), ("U1", "N9", "tv")]
    assert log["ts"].null_count() == log.height


def test_first_sightings_take_the_earliest_impression_and_list_history_untimed():
    s = first_sightings([toy()])
    shown = dict(s.drop_nulls("ts").iter_rows())
    assert shown == {"N1": T, "N2": T, "N3": T}          # all three first appear in impression 0 at T
    assert set(s.filter(pl.col("ts").is_null())["article_id"]) == {"N9", "N1"}
