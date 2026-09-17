"""Oracle for the Codabench submission format (SPEC.md §6).

Every case here is hand-computed. The rank-orientation test is the load-bearing one: a
transposed rank list is syntactically perfect and scores like noise, so it cannot be
caught by looking at the output file.
"""

import pytest

from src.eval.submission import (
    format_line,
    ranks_from_scores,
    validate_file,
    validate_line,
    zip_submission,
)


class TestRanksFromScores:
    def test_orientation_is_positional_not_ordinal(self):
        """candidates [A,B,C] with B best, A next, C worst -> [2,1,3].

        The list holds *each candidate's rank*, not *the candidate at each rank*. The
        transposed answer here would be [2,1,3] reversed into [3,1,2] via argsort — a
        mistake that produces a valid-looking file.
        """
        assert ranks_from_scores([0.5, 0.9, 0.1]) == [2, 1, 3]

    def test_descending_scores_give_identity(self):
        assert ranks_from_scores([3.0, 2.0, 1.0]) == [1, 2, 3]

    def test_ascending_scores_give_reverse(self):
        assert ranks_from_scores([1.0, 2.0, 3.0]) == [3, 2, 1]

    def test_ties_break_by_position_deterministically(self):
        """All-equal scores must yield input order, and must do so on every run."""
        assert ranks_from_scores([0.0, 0.0, 0.0]) == [1, 2, 3]
        assert ranks_from_scores([1.0, 5.0, 1.0, 5.0]) == [3, 1, 4, 2]

    def test_single_candidate(self):
        assert ranks_from_scores([0.42]) == [1]

    def test_output_is_always_a_permutation(self):
        scores = [0.3, 0.3, 0.9, 0.1, 0.9, 0.0]
        assert sorted(ranks_from_scores(scores)) == [1, 2, 3, 4, 5, 6]


class TestLineFormat:
    def test_matches_the_documented_example(self):
        """SPEC.md §6 quotes this exact line from the EB-NeRD reference notebook."""
        assert format_line(6451339, [8, 1, 6, 7, 4, 2, 9, 5, 3]) == "6451339 [8,1,6,7,4,2,9,5,3]"

    def test_no_spaces_inside_brackets(self):
        assert " " not in format_line(1, [1, 2, 3]).split("[")[1]

    def test_roundtrips_through_validate(self):
        assert validate_line(format_line(99, [2, 1, 3])) == (99, [2, 1, 3])


class TestValidateLine:
    @pytest.mark.parametrize(
        "bad",
        [
            "1 [1, 2, 3]",   # spaces inside the bracket
            "1 [1,2,3",      # unclosed
            "1 1,2,3",       # no brackets
            "x [1,2]",       # non-numeric id
            "1 []",          # empty rank list
            "",              # blank line
        ],
    )
    def test_rejects_malformed(self, bad):
        with pytest.raises(ValueError, match="malformed"):
            validate_line(bad)

    @pytest.mark.parametrize(
        "bad",
        [
            "1 [1,1,3]",   # duplicate rank
            "1 [1,2,4]",   # gap: 3 missing
            "1 [0,1,2]",   # zero-based
            "1 [2,3,4]",   # off by one
        ],
    )
    def test_rejects_non_permutation(self, bad):
        with pytest.raises(ValueError, match="permutation"):
            validate_line(bad)


class TestValidateFile:
    def _write(self, tmp_path, lines):
        p = tmp_path / "predictions.txt"
        p.write_text("".join(f"{ln}\n" for ln in lines))
        return p

    def test_accepts_a_good_file(self, tmp_path):
        p = self._write(tmp_path, ["1 [1,2]", "2 [2,1,3]"])
        assert validate_file(p) == {"lines": 2, "impressions": 2, "duplicate_rows": 0}

    def test_rejects_duplicate_impression(self, tmp_path):
        p = self._write(tmp_path, ["1 [1,2]", "1 [2,1]"])
        with pytest.raises(ValueError, match="duplicate"):
            validate_file(p)

    def test_detects_missing_impression(self, tmp_path):
        p = self._write(tmp_path, ["1 [1,2]"])
        with pytest.raises(ValueError, match="missing"):
            validate_file(p, expected_ids=[1, 2])

    def test_detects_unexpected_impression(self, tmp_path):
        p = self._write(tmp_path, ["1 [1,2]", "7 [1]"])
        with pytest.raises(ValueError, match="unexpected"):
            validate_file(p, expected_ids=[1])

    def test_detects_wrong_candidate_count(self, tmp_path):
        """The silent killer: right ids, right syntax, wrong number of candidates."""
        p = self._write(tmp_path, ["1 [1,2]"])
        with pytest.raises(ValueError, match="ranks but"):
            validate_file(p, expected_lengths={1: 3})


def test_zip_puts_txt_at_archive_root(tmp_path):
    """Codabench reads the .txt at the root; a nested path silently scores nothing."""
    import zipfile

    txt = tmp_path / "prediction.txt"
    txt.write_text("1 [1]\n")
    z = zip_submission(txt, tmp_path / "s.zip")
    assert zipfile.ZipFile(z).namelist() == ["prediction.txt"]


class TestDuplicateIdPolicy:
    """EB-NeRD zeroes impression_id on its 200,000 beyond-accuracy rows, so duplicates are
    legitimate there and a bug on MIND. The policy is therefore explicit, never inferred."""

    def _write(self, tmp_path, lines):
        p = tmp_path / "predictions.txt"
        p.write_text("".join(f"{ln}\n" for ln in lines))
        return p

    def test_duplicates_rejected_by_default(self, tmp_path):
        p = self._write(tmp_path, ["0 [1,2]", "0 [2,1]", "5 [1]"])
        with pytest.raises(ValueError, match="duplicate"):
            validate_file(p)

    def test_duplicates_allowed_when_opted_in(self, tmp_path):
        p = self._write(tmp_path, ["0 [1,2]", "0 [2,1]", "5 [1]"])
        out = validate_file(p, allow_duplicate_ids=True)
        assert out["lines"] == 3
        assert out["impressions"] == 2          # distinct ids: 0 and 5
        assert out["duplicate_rows"] == 1       # one row beyond the distinct count

    def test_duplicate_row_count_is_reported_even_when_zero(self, tmp_path):
        p = self._write(tmp_path, ["1 [1]", "2 [1]"])
        assert validate_file(p)["duplicate_rows"] == 0

    def test_permutation_check_still_applies_to_duplicated_ids(self, tmp_path):
        """Relaxing the id check must not relax anything else."""
        p = self._write(tmp_path, ["0 [1,2]", "0 [1,1]"])
        with pytest.raises(ValueError, match="permutation"):
            validate_file(p, allow_duplicate_ids=True)


def test_archive_member_name_per_competition(tmp_path):
    """SPEC §6: MIND's scorer opens `prediction.txt`, EB-NeRD's `predictions.txt`. The first MIND
    upload of A2 (17 Sep) failed with FileNotFoundError on exactly this; the driver now picks the
    name by dataset. This pins the rule so a renamed file cannot ship again."""
    import zipfile
    from src.eval.submission import zip_submission
    for dataset, name in (("mind", "prediction.txt"), ("ebnerd", "predictions.txt")):
        txt = tmp_path / dataset / name
        txt.parent.mkdir(); txt.write_text("1 [1]\n")
        z = zip_submission(txt, tmp_path / f"{dataset}.zip")
        assert zipfile.ZipFile(z).namelist() == [name]
