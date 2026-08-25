"""Codabench submission format: writing and validating.

Both competitions use the same line format (SPEC.md §6):

    impression_id [rank_order]

`rank_order` is a permutation of 1..N over the impression's candidate list, in the
*candidate list's own order*. Rank 1 is the most likely click. So for candidates
[A, B, C] and the ranking B > A > C, the line is `<id> [2,1,3]` — position 0 holds A's
rank of 2, not "the first-ranked candidate is B".

Getting that backwards produces a file that validates perfectly and scores like noise,
which is why `validate_line` checks structure and `ranks_from_scores` is the only
sanctioned way to build the list.
"""

from __future__ import annotations

import re
import zipfile
from array import array
from pathlib import Path
from typing import Iterable, Sequence

_LINE_RE = re.compile(r"^(\d+) \[(\d+(?:,\d+)*)\]$")


def ranks_from_scores(scores: Sequence[float]) -> list[int]:
    """Convert per-candidate scores into 1-based ranks, highest score = rank 1.

    Ties break by original position, deterministically: two candidates with equal scores
    keep their input order. Determinism matters because a re-run that reshuffles ties
    would produce a different leaderboard score from the same model.
    """
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    ranks = [0] * len(scores)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks


def format_line(impression_id: int, ranks: Sequence[int]) -> str:
    """Render one submission line. No spaces inside the bracket — the graders' parsers split on ', '."""
    return f"{impression_id} [{','.join(map(str, ranks))}]"


def validate_line(line: str) -> tuple[int, list[int]]:
    """Parse and structurally check one line. Raises ValueError with the offending text.

    Checks the two failures that actually happen: malformed syntax, and a rank list that
    is not a permutation of 1..N (duplicates, gaps, or zero).
    """
    m = _LINE_RE.match(line.rstrip("\n"))
    if not m:
        raise ValueError(f"malformed line: {line[:120]!r}")
    impression_id = int(m.group(1))
    ranks = [int(x) for x in m.group(2).split(",")]
    if sorted(ranks) != list(range(1, len(ranks) + 1)):
        raise ValueError(
            f"impression {impression_id}: ranks are not a permutation of 1..{len(ranks)}: {ranks[:20]}"
        )
    return impression_id, ranks


def validate_file(
    path: Path | str,
    expected_ids: Iterable[int] | None = None,
    expected_lengths: dict[int, int] | None = None,
    allow_duplicate_ids: bool = False,
) -> dict:
    """Validate a whole submission offline, before uploading.

    A rejected upload costs a full regeneration pass, so every check that can run locally
    runs locally: syntax, permutation validity, no duplicate impressions, complete
    coverage of the expected id set, and per-impression candidate counts.

    `allow_duplicate_ids` exists because EB-NeRD needs it and MIND does not. EB-NeRD's test
    set carries 200,000 rows whose `impression_id` is **0** -- exactly and only the rows
    flagged `is_beyond_accuracy`, which are scored for diversity/novelty/coverage rather
    than accuracy. Those lines are legitimately repeated and are matched by file order.
    MIND's ids are genuinely unique, so it keeps the check on: a duplicate there is a bug.
    """
    # Ids go into a typed array, not a Python set. EB-NeRD's test file is 13,536,710 lines;
    # a set of that many boxed ints costs upwards of 600 MB, where 8-byte slots cost ~108 MB.
    # Duplicate detection is deferred to a single sort at the end, which is both cheaper and
    # bounded.
    ids = array("q")
    n_lines = 0
    for n_lines, line in enumerate(Path(path).open(), start=1):
        impression_id, ranks = validate_line(line)
        ids.append(impression_id)
        if expected_lengths is not None:
            want = expected_lengths.get(impression_id)
            if want is not None and want != len(ranks):
                raise ValueError(
                    f"impression {impression_id}: {len(ranks)} ranks but {want} candidates"
                )

    import numpy as np

    arr = np.frombuffer(ids, dtype=np.int64)
    uniq = np.unique(arr)
    n_duplicate_rows = len(arr) - len(uniq)
    if n_duplicate_rows and not allow_duplicate_ids:
        counts = np.bincount(np.searchsorted(uniq, arr))
        dupe = uniq[np.argmax(counts)]
        raise ValueError(f"duplicate impression_id {dupe} ({counts.max()} occurrences)")

    if expected_ids is not None:
        expected = np.unique(np.fromiter(expected_ids, dtype=np.int64))
        missing = np.setdiff1d(expected, uniq, assume_unique=True)
        if len(missing):
            raise ValueError(f"{len(missing)} impressions missing, e.g. {missing[:5].tolist()}")
        extra = np.setdiff1d(uniq, expected, assume_unique=True)
        if len(extra):
            raise ValueError(f"{len(extra)} unexpected impressions, e.g. {extra[:5].tolist()}")
    return {"lines": n_lines, "impressions": int(len(uniq)),
            "duplicate_rows": int(n_duplicate_rows)}


def zip_submission(txt_path: Path | str, zip_path: Path | str) -> Path:
    """Zip the predictions file. Codabench expects the .txt at the archive root, not nested."""
    txt_path, zip_path = Path(txt_path), Path(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(txt_path, arcname=txt_path.name)
    return zip_path
