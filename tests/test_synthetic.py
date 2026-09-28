"""Synthetic corpus: scale, determinism, planted rates, drift shift."""

from __future__ import annotations

from collections import Counter

import pytest

from forensiq.ingest.synthetic import (
    DRIFT_DAY,
    ROOT_CAUSE_STAGE,
    SyntheticTraceGenerator,
)

EXPECTED_RATES = {  # normal (pre-drift) planted rates
    "F-RET-001": 0.12,
    "F-RET-002": 0.08,
    "F-PROMPT-001": 0.06,
    "F-GEN-001": 0.05,
    "F-GEN-002": 0.04,
    "F-GEN-003": 0.03,
    "F-INFRA-001": 0.02,
    "F-INFRA-002": 0.02,
}


class TestCorpus:
    def test_scale(self, corpus):
        traces, planted = corpus
        assert len(traces) >= 300
        assert len(planted) > 0.3 * len(traces)

    def test_deterministic_same_seed(self):
        t1, p1 = SyntheticTraceGenerator(seed=7).generate()
        t2, p2 = SyntheticTraceGenerator(seed=7).generate()
        assert t1 == t2
        assert p1 == p2

    def test_different_seed_different_corpus(self):
        t1, _ = SyntheticTraceGenerator(seed=1).generate()
        t2, _ = SyntheticTraceGenerator(seed=2).generate()
        assert t1 != t2

    def test_planted_rates_match_plan(self, corpus):
        traces, planted = corpus
        stable = [p for p in planted if p.day < DRIFT_DAY]
        n_stable_traces = len([t for t in traces if t.ts.day < DRIFT_DAY])
        counts = Counter(p.taxonomy_id for p in stable)
        for tax_id, rate in EXPECTED_RATES.items():
            actual = counts[tax_id] / n_stable_traces
            # 3-sigma tolerance for a binomial(n, rate) — statistical honesty
            sigma = (rate * (1 - rate) / n_stable_traces) ** 0.5
            tol = max(0.025, 3.0 * sigma)
            assert actual == pytest.approx(rate, abs=tol), f"{tax_id}: {actual} vs {rate}"

    def test_unique_trace_ids_and_stage_spans(self, corpus):
        traces, _ = corpus
        ids = [t.trace_id for t in traces]
        assert len(ids) == len(set(ids))
        for t in traces[:100]:
            assert t.stage_span("embed") is not None
            assert t.stage_span("retrieve") is not None
            assert t.stage_span("generate") is not None

    def test_planted_root_cause_stage_mapping(self, corpus):
        _, planted = corpus
        for p in planted:
            assert ROOT_CAUSE_STAGE[p.taxonomy_id] == p.stage

    def test_drift_shifts_failure_mix(self, corpus):
        """After DRIFT_DAY the retrieval-failure RATE must jump (rates, not
        shares — shares are noise-sensitive when the total failure count
        moves between windows)."""
        traces, planted = corpus
        n_stable = len([t for t in traces if t.ts.day < DRIFT_DAY])
        n_drifted = len(traces) - n_stable
        stable = Counter(p.taxonomy_id for p in planted if p.day < DRIFT_DAY)
        drifted = Counter(p.taxonomy_id for p in planted if p.day >= DRIFT_DAY)
        assert drifted["F-RET-002"] / n_drifted > stable["F-RET-002"] / n_stable + 0.10
        assert drifted["F-RET-001"] / n_drifted > stable["F-RET-001"] / n_stable + 0.01

    def test_drift_degrades_healthy_scores(self, corpus):
        traces, _ = corpus
        stable_scores, drifted_scores = [], []
        for t in traces:
            if len(t.stage_span("retrieve").attrs.get("hit_scores", [])) == 0:
                continue
            scores = t.stage_span("retrieve").attrs["hit_scores"]
            (drifted_scores if t.ts.day >= DRIFT_DAY else stable_scores).extend(scores)
        assert sum(drifted_scores) / len(drifted_scores) < sum(stable_scores) / len(stable_scores) - 0.03

    def test_write_jsonl_round_trips(self, corpus, tmp_path):
        traces, _ = corpus
        path = SyntheticTraceGenerator(seed=42).write_jsonl(tmp_path / "c.jsonl", traces[:10])
        assert path.exists()
        assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 10
