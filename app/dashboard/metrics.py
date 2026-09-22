"""
AI Governance Dashboard
--------------------------
In-memory metrics aggregator. In production this reads from
CloudWatch/Prometheus; here it accumulates real numbers from requests
actually processed by this running instance, so `/dashboard` reflects
genuine traffic through the API rather than fake canned numbers.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from app.models import DashboardMetrics, GovernanceReport


@dataclass
class _State:
    requests_total: int = 0
    latency_sum_ms: float = 0.0
    tokens_total: int = 0
    cost_total_usd: float = 0.0
    hallucination_flagged: int = 0
    blocked_requests: int = 0
    pii_incidents: int = 0
    tool_calls: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class MetricsStore:
    def __init__(self) -> None:
        self._state = _State()

    def record_request(self, report: GovernanceReport) -> None:
        with self._state.lock:
            self._state.requests_total += 1
            self._state.latency_sum_ms += report.latency_ms
            self._state.tokens_total += report.tokens_used
            self._state.cost_total_usd += report.estimated_cost_usd
            if report.hallucination.risk_level.value in ("MEDIUM", "HIGH"):
                self._state.hallucination_flagged += 1
            if report.blocked:
                self._state.blocked_requests += 1
            if report.pii_detected:
                self._state.pii_incidents += 1

    def record_tool_call(self) -> None:
        with self._state.lock:
            self._state.tool_calls += 1

    def snapshot(self) -> DashboardMetrics:
        with self._state.lock:
            s = self._state
            avg_latency = round(s.latency_sum_ms / s.requests_total, 2) if s.requests_total else 0.0
            hallucination_rate = (
                round((s.hallucination_flagged / s.requests_total) * 100, 2) if s.requests_total else 0.0
            )
            return DashboardMetrics(
                requests_total=s.requests_total,
                avg_latency_ms=avg_latency,
                tokens_total=s.tokens_total,
                estimated_cost_usd=round(s.cost_total_usd, 4),
                hallucination_rate=hallucination_rate,
                blocked_requests=s.blocked_requests,
                pii_incidents=s.pii_incidents,
                tool_calls=s.tool_calls,
            )


metrics_singleton = MetricsStore()
