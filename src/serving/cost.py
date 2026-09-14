"""Back-of-envelope serving cost (SPEC.md §16.2, Q4.3).

    cores(QPS)      = ceil(QPS × mean_service_s / rho)
    sla_holds       = p99_s < sla_s            (single-request p99 on one core; queueing kept small by rho)
    cost_per_1k     = cores × price_per_vcpu_hour / (QPS × 3600 / 1000)

rho is the target utilisation per core. At rho = 0.5 an M/M/1 server's mean queueing wait equals
one service time, so the request p99 stays within a small multiple of the measured single-request
p99 — that is the assumption behind "p99 holds". Everything is a parameter; RESULTS.md records
the values used and the price's source.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CostEstimate:
    target_qps: float
    mean_service_s: float
    p99_s: float
    sla_s: float
    rho: float
    price_per_vcpu_hour: float
    cores: int
    sla_holds: bool
    cost_per_1k_queries: float
    single_core_qps: float


def cost_per_1k(mean_service_s: float, p99_s: float, target_qps: float, price_per_vcpu_hour: float,
                *, rho: float = 0.5, sla_s: float = 0.100) -> CostEstimate:
    if not (0 < rho <= 1):
        raise ValueError("rho must be in (0, 1]")
    cores = max(1, math.ceil(target_qps * mean_service_s / rho))
    cost = cores * price_per_vcpu_hour / (target_qps * 3600.0 / 1000.0)
    return CostEstimate(target_qps, mean_service_s, p99_s, sla_s, rho, price_per_vcpu_hour,
                        cores, p99_s < sla_s, cost, 1.0 / mean_service_s)
