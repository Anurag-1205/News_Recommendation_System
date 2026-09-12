"""Oracles for src/baselines/ebrec_compat: the worked examples in the benchmark's own docstrings
(ebrec/utils/_articles_behaviors.py, ebrec/utils/_behaviors.py), plus the list-valued mapping
that broke on polars 1.35. No `external/` checkout is needed: the expected outputs are
copied from the docstring, which is what "same semantics as the original" means here."""
import polars as pl

import numpy as np

from src.baselines.ebrec_compat import add_prediction_scores, map_list_article_id_to_value

BEHAVIORS = pl.DataFrame(
    {"user_id": [1, 2, 3, 4, 5],
     "article_ids": [["A1", "A2"], ["A2", "A3"], ["A1", "A4"], ["A4", "A4"], None]}
)
MAPPING = {"A1": "News", "A2": "Sports", "A3": "Entertainment"}


def _lists(df):
    return df.sort("user_id")["article_ids"].to_list()


def test_fill_nulls_matches_docstring():
    out = map_list_article_id_to_value(BEHAVIORS, "article_ids", MAPPING, fill_nulls="Unknown")
    assert _lists(out) == [["News", "Sports"], ["Sports", "Entertainment"],
                           ["News", "Unknown"], ["Unknown", "Unknown"], ["Unknown"]]


def test_drop_nulls_matches_docstring():
    out = map_list_article_id_to_value(BEHAVIORS, "article_ids", MAPPING, drop_nulls=True)
    assert _lists(out) == [["News", "Sports"], ["Sports", "Entertainment"], ["News"], None, None]


def test_keep_nulls_matches_docstring():
    out = map_list_article_id_to_value(BEHAVIORS, "article_ids", MAPPING, drop_nulls=False)
    assert _lists(out) == [["News", "Sports"], ["Sports", "Entertainment"],
                           ["News", None], [None, None], [None]]


def test_other_columns_and_row_order_preserved():
    out = map_list_article_id_to_value(BEHAVIORS, "article_ids", MAPPING, fill_nulls="U")
    assert out.columns == BEHAVIORS.columns
    assert out["user_id"].to_list() == [1, 2, 3, 4, 5]


def test_list_valued_mapping_the_nrms_case():
    """Token-id lists per article: the case polars 1.x `replace` rejects."""
    b = pl.DataFrame({"impression_id": [10, 11], "article_ids_inview": [[7, 8], [8, 9]]})
    mapping = {7: pl.Series([1, 2, 0]), 8: pl.Series([3, 4, 5]), 9: [6, 0, 0]}
    out = map_list_article_id_to_value(b, "article_ids_inview", mapping)
    assert out["article_ids_inview"].to_list() == [[[1, 2, 0], [3, 4, 5]], [[3, 4, 5], [6, 0, 0]]]



SLATES = pl.DataFrame({"id": [1, 2], "article_ids_inview": [[1, 2, 3], [4, 5]]})
FLAT_SCORES = [[0.3], [0.4], [0.5], [0.6], [0.7]]


def test_add_prediction_scores_matches_docstring_list():
    out = add_prediction_scores(SLATES, FLAT_SCORES)
    assert out.columns == ["id", "article_ids_inview", "scores"]
    assert out["scores"].to_list() == [[0.3, 0.4, 0.5], [0.6, 0.7]]


def test_add_prediction_scores_matches_docstring_ndarray():
    out = add_prediction_scores(SLATES, np.array(FLAT_SCORES, dtype=np.float32))
    assert [[round(x, 3) for x in row] for row in out["scores"].to_list()] == [[0.3, 0.4, 0.5], [0.6, 0.7]]


def test_add_prediction_scores_row_alignment():
    """Row 2's scores must come after row 1's regardless of slate length."""
    df = pl.DataFrame({"id": [7, 8, 9], "article_ids_inview": [[1], [2, 3, 4], [5, 6]]})
    out = add_prediction_scores(df, [[10.0], [20.0], [21.0], [22.0], [30.0], [31.0]])
    assert out["scores"].to_list() == [[10.0], [20.0, 21.0, 22.0], [30.0, 31.0]]
