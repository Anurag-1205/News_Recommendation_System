"""Oracles for the paired bootstrap harness (SPEC.md §14, PLAN.md P3.4a).

The statistic (`src.eval.bootstrap.paired_delta`) is judged on synthetic per-impression metric
pairs whose true difference is known: a zero-Δ pair must give a CI covering 0, a constructed Δ
must be recovered and must exclude 0, and over many trials the 95% interval must cover the truth
at (close to) its nominal rate. The file-level harness (`src.eval.paired`) is judged on toy score
files with hand-computed metrics, and on the real NRMS files when they are present.
"""
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.eval.bootstrap import bootstrap_ci, paired_delta
from src.eval.paired import check_compatible, paired_compare, read_scores, write_scores

# ---- U1 · calibration of the statistic ------------------------------------------------------------


def _pair(rng, n, delta, noise=0.1, base_sd=0.2):
    """Two systems on the same n impressions: b = a + delta + independent noise."""
    a = 0.6 + base_sd * rng.standard_normal(n)
    b = a + delta + noise * rng.standard_normal(n)
    return a, b


def test_zero_delta_ci_covers_zero():
    a, b = _pair(np.random.default_rng(1), 2_000, delta=0.0)
    ci = paired_delta(a, b, seed=0)
    assert ci.lo <= 0.0 <= ci.hi


def test_constructed_delta_recovered_and_excludes_zero():
    a, b = _pair(np.random.default_rng(2), 5_000, delta=0.02)
    ci = paired_delta(a, b, seed=0)
    assert ci.lo <= 0.02 <= ci.hi
    assert ci.lo > 0.0, "a 0.02 shift over 5,000 impressions must be significant"


@pytest.mark.parametrize("delta", [0.0, 0.02])
def test_coverage_is_close_to_nominal(delta):
    """Over 200 independent trials the 95% CI must contain the true Δ at least 90% of the time."""
    rng = np.random.default_rng(3)
    hits = 0
    for t in range(200):
        a, b = _pair(rng, 1_000, delta=delta)
        ci = paired_delta(a, b, iterations=400, seed=t)
        hits += ci.lo <= delta <= ci.hi
    assert hits >= 180, f"coverage {hits}/200"


def test_paired_interval_is_much_tighter_than_unpaired():
    """The point of pairing: shared impression-level variance cancels."""
    a, b = _pair(np.random.default_rng(4), 5_000, delta=0.02, noise=0.05, base_sd=0.3)
    paired = paired_delta(a, b, seed=0)
    ua, ub = bootstrap_ci(a, seed=0), bootstrap_ci(b, seed=0)
    unpaired_width = (ub.hi - ub.lo) + (ua.hi - ua.lo)   # width of the naive difference-of-CIs
    assert (paired.hi - paired.lo) < 0.5 * unpaired_width


def test_paired_delta_is_deterministic():
    a, b = _pair(np.random.default_rng(5), 1_000, delta=0.01)
    assert paired_delta(a, b, seed=7) == paired_delta(a, b, seed=7)


# ---- U2 · the file-level harness on toy score files -------------------------------------------------

LABELS = pl.DataFrame({"imp_row": [0, 0, 0, 1, 1, 2, 2], "cand_position": [1, 2, 3, 1, 2, 1, 2],
                       "label": [1, 0, 0, 0, 1, 1, 0]}).with_columns(pl.col("imp_row").cast(pl.UInt32),
                                                                       pl.col("cand_position").cast(pl.Int64),
                                                                       pl.col("label").cast(pl.Int8))


def _toy_scores(tmp, name, scores, **manifest):
    frame = pl.DataFrame({"imp_row": pl.Series([0, 0, 0, 1, 1, 2, 2], dtype=pl.UInt32),
                          "impression_id": pl.Series([10, 10, 10, 11, 11, 12, 12], dtype=pl.UInt32),
                          "article_id": pl.Series([5, 6, 7, 6, 8, 5, 9], dtype=pl.Int32),
                          "cand_position": pl.Series([1, 2, 3, 1, 2, 1, 2], dtype=pl.Int64),
                          "score": pl.Series(scores, dtype=pl.Float64)})
    man = {"dataset": "toy", "split": "toy/validation", "system": name, "framing": "in-impression",
           "n_rows": 7, "n_impressions": 3, **manifest}
    return write_scores(frame, man, tmp / f"{name}.parquet")


def test_harness_recovers_hand_computed_metrics(tmp_path):
    # A ranks the click first in every impression (AUC 1, MRR 1); B ranks it last in imp 0 and 2.
    a = _toy_scores(tmp_path, "a", [0.9, 0.1, 0.2, 0.1, 0.9, 0.9, 0.1])
    b = _toy_scores(tmp_path, "b", [0.1, 0.9, 0.5, 0.1, 0.9, 0.1, 0.9])
    rep = paired_compare(a, b, labels=LABELS, iterations=200, seed=0)
    assert rep.n_common == 3
    assert rep.system_a["auc"].mean == pytest.approx(1.0)
    # B: imp0 auc 0 (click below both negatives), imp1 auc 1, imp2 auc 0 -> mean 1/3
    assert rep.system_b["auc"].mean == pytest.approx(1 / 3)
    assert rep.delta["auc"].mean == pytest.approx(-2 / 3)      # b − a
    assert rep.delta["mrr"].mean == pytest.approx((1 / 3 + 1 + 1 / 2) / 3 - 1.0)


def test_harness_refuses_incompatible_manifests(tmp_path):
    a = _toy_scores(tmp_path, "a", [0.9, 0.1, 0.2, 0.1, 0.9, 0.9, 0.1])
    b = _toy_scores(tmp_path, "b", [0.9, 0.1, 0.2, 0.1, 0.9, 0.9, 0.1], framing="retrieved-topk")
    with pytest.raises(ValueError, match="framing"):
        paired_compare(a, b, labels=LABELS)
    _, ma = read_scores(a); _, mb = read_scores(b)
    with pytest.raises(ValueError):
        check_compatible(ma, {**mb, "framing": "in-impression", "split": "other"})


def test_harness_intersects_on_imp_row_and_reports_counts(tmp_path):
    a = _toy_scores(tmp_path, "a", [0.9, 0.1, 0.2, 0.1, 0.9, 0.9, 0.1])
    fb, mb = read_scores(_toy_scores(tmp_path, "b", [0.9, 0.1, 0.2, 0.1, 0.9, 0.9, 0.1]))
    b = write_scores(fb.filter(pl.col("imp_row") < 2), {**mb, "n_rows": 5, "n_impressions": 2}, tmp_path / "b2.parquet")
    rep = paired_compare(a, b, labels=LABELS, iterations=100)
    assert (rep.n_a, rep.n_b, rep.n_common) == (3, 2, 2)
    assert rep.delta["auc"].mean == 0.0


def test_harness_refuses_small_overlap(tmp_path):
    """A subset is a sample and is allowed (the reranker's 100k vs NRMS's 245k). Two files that
    share few impressions relative to the smaller one are a wrong split and are refused."""
    a = _toy_scores(tmp_path, "a", [0.9, 0.1, 0.2, 0.1, 0.9, 0.9, 0.1])
    fb, mb = read_scores(a)
    shifted = fb.with_columns((pl.col("imp_row") + 2).cast(pl.UInt32))   # imp_rows 2, 3, 4: one in common with 0, 1, 2
    b = write_scores(shifted.sort("imp_row", "cand_position"), mb, tmp_path / "b1.parquet")
    with pytest.raises(ValueError, match="overlap"):
        paired_compare(a, b, labels=LABELS, iterations=100)


def test_read_scores_validates_schema(tmp_path):
    bad = pl.DataFrame({"imp_row": [0], "score": [0.5]})
    bad.write_parquet(tmp_path / "bad.parquet")
    (tmp_path / "bad.json").write_text(json.dumps({"dataset": "toy", "split": "s", "system": "x", "framing": "in-impression"}))
    with pytest.raises(ValueError):
        read_scores(tmp_path / "bad.parquet")


# ---- U3 · the real NRMS files, as a smoke of the harness (not a claim) --------------------------------

NRMS_EBNERD = Path("data/scores/ebnerd/validation/nrms.parquet")
needs_scores = pytest.mark.skipif(not NRMS_EBNERD.exists(), reason="NRMS score file not downloaded")


@needs_scores
def test_real_file_zero_delta_sensitivity_and_oracle_gain(tmp_path):
    """Three constructed comparisons against the real EB-NeRD NRMS file (first 20,000 impressions):
    (1) a monotone rescaling of the scores (rank-preserving) must give Δ exactly 0 on every metric;
    (2) N(0, 0.01) noise added to the scores is a small *real* degradation: every metric's Δ
        is negative, and on AUC it is significant at 20k impressions (CI [−0.0020, −0.0004]);
        MRR's is not ([−0.0013, +0.0002]) — a noisier statistic. The first draft expected a
        zero-Δ here and was wrong;
    (3) +0.5 on every clicked candidate must be a significant gain on every metric."""
    from src.eval.paired import split_labels
    frame, man = read_scores(NRMS_EBNERD)
    sub = frame.filter(pl.col("imp_row") < 20_000)
    labels = split_labels("ebnerd", man["split"]).filter(pl.col("imp_row") < 20_000)
    m = {**man, "n_rows": sub.height, "n_impressions": 20_000}
    base = write_scores(sub, m, tmp_path / "base.parquet")
    rescaled = sub.with_columns((pl.col("score") * 2.0 + 1.0).alias("score"))
    rng = np.random.default_rng(0)
    noisy = sub.with_columns(pl.Series("score", sub["score"].to_numpy() + 0.01 * rng.standard_normal(sub.height)))
    boosted = (sub.join(labels, on=["imp_row", "cand_position"], how="left")
               .with_columns((pl.col("score") + 0.5 * pl.col("label")).alias("score")).drop("label"))
    r_same = paired_compare(base, write_scores(rescaled, {**m, "system": "nrms*2+1"}, tmp_path / "same.parquet"), labels=labels, iterations=300)
    r_noise = paired_compare(base, write_scores(noisy, {**m, "system": "nrms+noise"}, tmp_path / "noisy.parquet"), labels=labels, iterations=300)
    r_boost = paired_compare(base, write_scores(boosted, {**m, "system": "nrms+oracle"}, tmp_path / "boost.parquet"), labels=labels, iterations=300)
    for k in ("auc", "mrr", "ndcg@5", "ndcg@10"):
        assert (r_same.delta[k].mean, r_same.delta[k].lo, r_same.delta[k].hi) == (0.0, 0.0, 0.0), (k, r_same.delta[k])
        assert r_noise.delta[k].mean < 0.0, (k, r_noise.delta[k])          # noise never helps
        assert r_boost.delta[k].lo > 0.0, (k, r_boost.delta[k])
    assert r_noise.delta["auc"].hi < 0.0, r_noise.delta["auc"]           # significant on AUC at 20k; MRR is noisier and is not
    assert r_same.verdict("auc") == "no significant difference"
    assert r_boost.verdict("auc") == "nrms+oracle beats nrms"
