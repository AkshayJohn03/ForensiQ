"""Drift detection: chi-square fires on the planted day-15 degradation,
stays quiet on the stable window at the chosen alpha, CUSUM catches the ramp."""

from __future__ import annotations

import json

import pytest

from forensiq.ingest.synthetic import DRIFT_DAY
from forensiq.patterns.drift import DriftDetector


@pytest.fixture(scope="module")
def classified(corpus, failures):
    traces, _ = corpus
    return traces, failures


class TestPlantedDrift:
    def test_fires_after_drift_day(self, classified):
        traces, failures = classified
        report = DriftDetector().detect(traces, failures)
        chi_alerts = [a for a in report.alerts if a["type"] == "chi_square"]
        assert chi_alerts, "expected at least one chi-square alert"
        assert all(a["day"] >= f"2026-08-{DRIFT_DAY:02d}" for a in chi_alerts)

    def test_quiet_on_stable_window(self, classified):
        """Days 1-14 only: the detector must stay quiet at alpha=0.01 —
        no false alarm on the planted-stable prefix."""
        traces, failures = classified
        stable_traces = [t for t in traces if t.ts.day < DRIFT_DAY]
        stable_failures = [f for f in failures if f.trace_id in {t.trace_id for t in stable_traces}]
        report = DriftDetector().detect(stable_traces, stable_failures)
        assert report.alerts == []
        assert report.quiet

    def test_cusum_catches_slow_ramp(self, classified):
        traces, failures = classified
        report = DriftDetector().detect(traces, failures)
        cusum = [a for a in report.alerts if a["type"] == "cusum"]
        assert cusum
        assert all(a.get("class") for a in cusum)

    def test_alert_payload_shape(self, classified):
        traces, failures = classified
        report = DriftDetector().detect(traces, failures)
        payload = json.loads(report.to_json())
        assert "alerts" in payload and "baseline_mix" in payload
        alert = report.alerts[0]
        assert alert["type"] in ("chi_square", "cusum")
        if alert["type"] == "chi_square":
            assert alert["p_value"] < alert["alpha"]
            assert "observed" in alert and "expected" in alert

    def test_baseline_mix_sums_to_one(self, classified):
        traces, failures = classified
        report = DriftDetector().detect(traces, failures)
        # 4-dp display rounding per class -> allow tiny accumulation error
        assert sum(report.baseline_mix.values()) == pytest.approx(1.0, abs=1e-3)


class TestChiSquareGuard:
    def test_small_counts_pool_into_other(self):
        det = DriftDetector()
        from collections import Counter

        observed = Counter({"F-A": 12, "F-B": 8, "F-C": 1, "F-D": 0})
        # expected: F-A 10.5, F-B 8.4 (>= 5, stay), F-C 1.05 / F-D 1.05 pool
        stat, p, cells = det._chi2_vs_baseline(observed, {"F-A": 0.5, "F-B": 0.4, "F-C": 0.05, "F-D": 0.05})
        assert p is not None
        assert "__other__" in cells
        assert len(cells) == 3  # F-A, F-B, other -> df 2

    def test_identical_mix_never_alerts(self):
        det = DriftDetector()
        from collections import Counter

        mix = {"F-A": 0.5, "F-B": 0.3, "F-C": 0.2}
        observed = Counter({"F-A": 50, "F-B": 30, "F-C": 20})
        stat, p, _ = det._chi2_vs_baseline(observed, mix)
        assert p > 0.99
