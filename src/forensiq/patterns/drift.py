"""Failure-mix drift detection: hand-rolled chi-square + CUSUM.

Why chi-square over a KL-divergence-style measure: the failure mix is small
count data (a handful of taxonomy classes, tens of failures per window). A
chi-square goodness-of-fit test against the baseline mix gives a calibrated
p-value with a small-count pooling guard, so "is this window's mix actually
different" has a false-alarm rate we can set (alpha) instead of a threshold
we guess. KL on sparse counts is unstable and has no null distribution
without resampling — the wrong tool at this sample size.

Two detectors:
  * chi-square on a trailing window vs the baseline mix, alerting when at
    least `min_significant` of the last `lookback` windows are significant
    (default 2-of-3) — a sustained shift, not a blip, and robust to one
    noisy window inside otherwise-drifted traffic;
  * CUSUM per taxonomy class on daily counts — catches slow ramps the
    windowed test is slow to notice.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from forensiq._stats import chi2_sf, mean_std
from forensiq.ingest.schema import Trace
from forensiq.taxonomy.classifier import FailureRecord


class DriftReport(BaseModel):
    baseline_days: list[str] = []
    baseline_mix: dict[str, float] = {}
    daily_counts: dict[str, dict[str, int]] = {}
    alerts: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def quiet(self) -> bool:
        return not self.alerts

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(), indent=indent, sort_keys=True, default=str)


class DriftDetector:
    def __init__(
        self,
        alpha: float = 0.01,
        baseline_days: int = 14,
        window_days: int = 3,
        min_significant: int = 2,
        lookback: int = 3,
        cusum_k: float = 0.5,
        cusum_h: float = 5.0,
    ) -> None:
        self.alpha = alpha
        self.baseline_days = baseline_days
        self.window_days = window_days
        self.min_significant = min_significant
        self.lookback = lookback
        self.cusum_k = cusum_k
        self.cusum_h = cusum_h

    def detect(self, traces: list[Trace], failures: list[FailureRecord]) -> DriftReport:
        day_by_trace = {t.trace_id: t.day for t in traces}
        events: list[tuple[str, str]] = []  # (day, taxonomy_id)
        for f in failures:
            day = day_by_trace.get(f.trace_id)
            if day is not None:
                events.append((day, f.taxonomy_id))
        days = sorted({d for d, _ in events}) or sorted({t.day for t in traces})
        daily: dict[str, Counter[str]] = {d: Counter() for d in days}
        for day, tax_id in events:
            daily[day][tax_id] += 1
        daily_counts = {d: dict(sorted(c.items())) for d, c in daily.items()}

        baseline_window = days[: self.baseline_days]
        eval_days = days[self.baseline_days :]
        baseline_counts = Counter()
        for d in baseline_window:
            baseline_counts.update(daily[d])
        total = sum(baseline_counts.values())
        mix = {k: v / total for k, v in sorted(baseline_counts.items())} if total else {}

        report = DriftReport(
            baseline_days=baseline_window,
            baseline_mix={k: round(v, 4) for k, v in mix.items()},
            daily_counts=daily_counts,
        )

        alerts: list[dict[str, Any]] = []
        if len(baseline_window) < 3 or not mix:
            return report

        # --- chi-square over trailing windows -----------------------------
        # 2-of-3 trailing-window rule: overlapping windows share days, so a
        # single noisy window must not reset a sustained signal, but a blip
        # cannot satisfy 2-of-3 either (null probability ~3e-4 per position).
        recent: list[bool] = []
        for end in range(len(baseline_window), len(days)):
            start = end - self.window_days + 1
            if start < len(baseline_window):
                recent = []
                continue
            window = days[start : end + 1]
            observed = Counter()
            for d in window:
                observed.update(daily[d])
            stat, p, cells = self._chi2_vs_baseline(observed, mix)
            sig = p is not None and p < self.alpha
            recent.append(sig)
            if len(recent) > self.lookback:
                recent.pop(0)
            if sig and sum(recent) >= self.min_significant:
                alerts.append(
                    {
                        "type": "chi_square",
                        "day": window[-1],
                        "window_days": self.window_days,
                        "significant_windows": sum(recent),
                        "lookback": self.lookback,
                        "chi2": round(stat, 4),
                        "df": len(cells) - 1,
                        "p_value": round(p, 6) if p is not None else None,
                        "alpha": self.alpha,
                        "observed": dict(sorted(observed.items())),
                        "expected": {k: round(v * sum(observed.values()), 2) for k, v in mix.items()},
                    }
                )
                recent = []  # one alert per sustained shift

        # --- CUSUM per class ------------------------------------------------
        alerts.extend(self._cusum_alerts(daily, baseline_window, eval_days))

        report.alerts = alerts
        return report

    # -- internals ------------------------------------------------------------

    def _chi2_vs_baseline(
        self, observed: Counter[str], mix: dict[str, float]
    ) -> tuple[float, float | None, list[str]]:
        """Chi-square GOF of observed counts vs baseline mix, with a
        small-count guard: cells with expected < 5 pool into 'other'."""
        n = sum(observed.values())
        classes = sorted(set(mix) | set(observed))
        expected = {c: mix.get(c, 0.0) * n for c in classes}
        # pooling
        pooled_classes = [c for c in classes if expected[c] >= 5]
        small = [c for c in classes if expected[c] < 5]
        if not pooled_classes or len(pooled_classes) < 2:
            pooled_classes = classes  # guard defeated the test; run unpooled
            small = []
        obs: list[float] = [sum(observed.get(c, 0) for c in pooled_classes)]
        exp: list[float] = [sum(expected[c] for c in pooled_classes)]
        obs.append(sum(observed.get(c, 0) for c in small))
        exp.append(sum(expected[c] for c in small))
        stat = 0.0
        for o, e in zip(obs, exp, strict=True):
            if e > 0:
                stat += (o - e) ** 2 / e
        df = len(obs) - 1
        p = chi2_sf(stat, df)
        return stat, p, pooled_classes + (["__other__"] if small else [])

    def _cusum_alerts(
        self,
        daily: dict[str, Counter[str]],
        baseline_window: list[str],
        eval_days: list[str],
    ) -> list[dict[str, Any]]:
        alerts: list[dict[str, Any]] = []
        classes = sorted({c for d in baseline_window for c in daily[d]})
        for cls in classes:
            series = [daily[d].get(cls, 0) for d in baseline_window]
            mu, sigma = mean_std(series)
            if mu <= 0 or sigma <= 1e-9:
                continue
            s = 0.0
            for d in eval_days:
                x = daily[d].get(cls, 0)
                s = max(0.0, s + (x - mu) / sigma - self.cusum_k)
                if s > self.cusum_h:
                    alerts.append(
                        {
                            "type": "cusum",
                            "day": d,
                            "class": cls,
                            "cusum": round(s, 3),
                            "threshold": self.cusum_h,
                            "baseline_daily_mean": round(mu, 3),
                            "observed": x,
                        }
                    )
                    s = 0.0  # reset after firing
        alerts.sort(key=lambda a: (a["day"], a["type"], a.get("class", "")))
        return alerts
