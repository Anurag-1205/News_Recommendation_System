"""The Q9 anti-gaming assertion: no future-click leakage.

A user feature computed at impression time *t* may only see events strictly before *t*
-- a project invariant. This test is the enforcement, and it is deliberately RED until P1 lands
the feature store: it must fail before it passes, so that a green suite is evidence of the
invariant rather than evidence of an absent test.

When P1 implements the feature store, replace the xfail body with the real check:

    for every row in the feature store:
        assert max(source_event_timestamps) < impression_timestamp   # strictly

and add a deliberately-leaked fixture row proving the assertion catches it.
"""

import pytest


@pytest.mark.xfail(
    strict=True,
    reason="P1 not implemented: no feature store to assert against yet",
)
def test_history_events_strictly_precede_impression():
    """Every history event feeding a user feature must predate that user's impression."""
    from src.pipeline import feature_store  # noqa: F401  — does not exist until P1

    raise AssertionError("unreachable once P1 lands")


@pytest.mark.xfail(
    strict=True,
    reason="P1 not implemented: no split to assert against yet",
)
def test_splits_are_disjoint_in_time():
    """train < val < test, by timestamp, with no overlap — never a random split."""
    from src.pipeline import splits  # noqa: F401  — does not exist until P1

    raise AssertionError("unreachable once P1 lands")
