"""Hand-computed oracle for the cost model (SPEC.md §16.2)."""
import pytest

from src.serving.cost import cost_per_1k


def test_hand_computed_example():
    # 20 ms mean service, 60 ms p99, 100 QPS, $0.04/vCPU-h, rho 0.5:
    # cores = ceil(100 × 0.020 / 0.5) = 4; queries/h = 360,000 = 360 thousand;
    # cost/1k = 4 × 0.04 / 360 = $0.000444
    e = cost_per_1k(0.020, 0.060, 100, 0.04)
    assert e.cores == 4 and e.sla_holds is True
    assert e.cost_per_1k_queries == pytest.approx(4 * 0.04 / 360)
    assert e.single_core_qps == pytest.approx(50.0)


def test_sla_fails_when_p99_over_budget_and_cores_scale_with_qps():
    e = cost_per_1k(0.020, 0.150, 1000, 0.04)
    assert e.sla_holds is False and e.cores == 40
    assert cost_per_1k(0.020, 0.150, 1000, 0.04, rho=1.0).cores == 20


def test_rho_validated():
    with pytest.raises(ValueError):
        cost_per_1k(0.02, 0.05, 10, 0.04, rho=0)
