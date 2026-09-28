"""Blame ranker: top-1 root-cause stage vs planted truth, evidence chains,
cohort aggregation."""

from __future__ import annotations

from collections import Counter

import pytest

from forensiq.attribution.blame import BlameRanker
from forensiq.ingest.synthetic import ROOT_CAUSE_STAGE


class TestTopOneAccuracy:
    def test_true_root_cause_stage_first_in_at_least_80pct(self, corpus, failures):
        """The spec target: >= 80% of targeted fixtures rank the truly
        injected root-cause stage first."""
        traces, planted = corpus
        ranker = BlameRanker.from_traces(traces, failures)
        by_id = {t.trace_id: t for t in traces}
        hits = 0
        for p in planted:
            trace = by_id[p.trace_id]
            failure = next(f for f in failures if f.trace_id == p.trace_id)
            if ranker.top_stage(trace, failure) == ROOT_CAUSE_STAGE[p.taxonomy_id]:
                hits += 1
        accuracy = hits / len(planted)
        assert accuracy >= 0.80, f"top-1 accuracy {accuracy:.3f}"


class TestUnitBehavior:
    def test_empty_retrieval_blames_retrieve(self, corpus, failures, ranker):
        traces, _ = corpus
        failure = next(f for f in failures if f.taxonomy_id == "F-RET-001")
        trace = next(t for t in traces if t.trace_id == failure.trace_id)
        ranked = ranker.rank(trace, failure)
        assert ranked[0].stage == "retrieve"
        assert ranked[0].evidence, "evidence chain must be present"
        assert any(ev.stage == "retrieve" for ev in ranked[0].evidence)

    def test_timeout_blames_generate(self, corpus, failures, ranker):
        traces, _ = corpus
        failure = next(f for f in failures if f.taxonomy_id == "F-INFRA-001")
        trace = next(t for t in traces if t.trace_id == failure.trace_id)
        assert ranker.top_stage(trace, failure) == "generate"

    def test_cost_spike_blames_generate(self, corpus, failures, ranker):
        traces, _ = corpus
        failure = next(f for f in failures if f.taxonomy_id == "F-INFRA-002")
        trace = next(t for t in traces if t.trace_id == failure.trace_id)
        assert ranker.top_stage(trace, failure) == "generate"

    def test_ranking_is_deterministic(self, corpus, failures, ranker):
        traces, _ = corpus
        trace = next(t for t in traces if t.trace_id == failures[0].trace_id)
        first = [c.stage for c in ranker.rank(trace, failures[0])]
        second = [c.stage for c in ranker.rank(trace, failures[0])]
        assert first == second

    def test_baseline_excludes_failed_traces(self, corpus, failures):
        traces, _ = corpus
        clean = BlameRanker.from_traces(traces, failures)
        dirty = BlameRanker.from_traces(traces, None)
        assert clean.baseline.n_traces < dirty.baseline.n_traces


class TestCohortBlame:
    def test_distribution_per_family(self, corpus, failures, ranker):
        traces, _ = corpus
        dist = ranker.cohort_blame(traces, failures)
        assert set(dist) >= {"F-RET-001", "F-RET-002", "F-INFRA-001", "F-INFRA-002"}
        assert dist["F-RET-001"]["retrieve"] >= 0.8
        assert dist["F-RET-002"]["retrieve"] >= 0.8
        assert dist["F-INFRA-001"]["generate"] >= 0.8

    def test_shares_sum_to_one(self, corpus, failures, ranker):
        traces, _ = corpus
        dist = ranker.cohort_blame(traces, failures)
        for _tax_id, shares in dist.items():
            assert sum(shares.values()) == pytest.approx(1.0)

    def test_every_family_present(self, corpus, failures, ranker):
        traces, planted = corpus
        dist = ranker.cohort_blame(traces, failures)
        fam_sizes = Counter(f.taxonomy_id for f in failures)
        assert set(dist) == set(fam_sizes)
        for shares in dist.values():
            assert sum(shares.values()) == pytest.approx(1.0)
