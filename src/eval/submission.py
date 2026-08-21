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
) -> dict:
    """Validate a whole submission offline, before uploading.

    A rejected upload costs a full regeneration pass, so every check that can run locally
    runs locally: syntax, permutation validity, no duplicate impressions, complete
    coverage of the expected id set, and per-impression candidate counts.
    """
    seen: set[int] = set()
    n_lines = 0
    for n_lines, line in enumerate(Path(path).open(), start=1):
        impression_id, ranks = validate_line(line)
        if impression_id in seen:
            raise ValueError(f"duplicate impression_id {impression_id} at line {n_lines}")
        seen.add(impression_id)
        if expected_lengths is not None:
            want = expected_lengths.get(impression_id)
            if want is not None and want != len(ranks):
                raise ValueError(
                    f"impression {impression_id}: {len(ranks)} ranks but {want} candidates"
                )
    if expected_ids is not None:
        expected = set(expected_ids)
        if missing := expected - seen:
            raise ValueError(f"{len(missing)} impressions missing, e.g. {sorted(missing)[:5]}")
        if extra := seen - expected:
            raise ValueError(f"{len(extra)} unexpected impressions, e.g. {sorted(extra)[:5]}")
    return {"lines": n_lines, "impressions": len(seen)}


def zip_submission(txt_path: Path | str, zip_path: Path | str) -> Path:
    """Zip the predictions file. Codabench expects the .txt at the archive root, not nested."""
    txt_path, zip_path = Path(txt_path), Path(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(txt_path, arcname=txt_path.name)
    return zip_path
