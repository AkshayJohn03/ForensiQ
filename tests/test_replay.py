"""Counterfactual replay: the simulated adapter's effect model — including
its deliberately honest refusals (top_k cannot fix an empty index, and it
dilutes mean scores on weak context)."""

from __future__ import annotations

import asyncio

import pytest

from forensiq.attribution.replay import (
    SimulatedAdapter,
    StageMutation,
    WhatIfReplay,
)
from forensiq.taxonomy.classifier import FailureClassifier


@pytest.fixture(scope="module")
def replayer():
    return WhatIfReplay(adapter=SimulatedAdapter(), classifier=FailureClassifier())


def _first(corpus, failures, tax_id):
    traces, _ = corpus
    failure = next(f for f in failures if f.taxonomy_id == tax_id)
    trace = next(t for t in traces if t.trace_id == failure.trace_id)
    return trace, failure


def _replay(replayer, trace, stage: str, param: str, value, failure):
    mutation = StageMutation(stage=stage, param=param, value=value)
    return asyncio.run(replayer.replay(trace, [mutation], failure))


class TestSimulatedAdapter:
    def test_query_rewrite_resolves_low_recall(self, corpus, failures, replayer):
        trace, failure = _first(corpus, failures, "F-RET-002")
        result = _replay(replayer, trace, "retrieve", "query_rewrite", True, failure)
        attempt = result.attempts[0]
        assert attempt.resolved is True
        assert result.original_taxonomy_id == "F-RET-002"
        assert attempt.predicted_lift["mean_hit_score"] > 0.1

    def test_top_k_honestly_does_not_fix_low_recall(self, corpus, failures, replayer):
        """Marginal documents land below the current mean — the effect model
        must say so instead of selling top_k as a fix for score quality."""
        trace, failure = _first(corpus, failures, "F-RET-002")
        result = _replay(replayer, trace, "retrieve", "top_k", 10, failure)
        attempt = result.attempts[0]
        assert attempt.resolved is False
        assert attempt.predicted_lift["mean_hit_score"] < 0.0  # dilution

    def test_top_k_cannot_fix_empty_index(self, corpus, failures, replayer):
        trace, failure = _first(corpus, failures, "F-RET-001")
        result = _replay(replayer, trace, "retrieve", "top_k", 10, failure)
        attempt = result.attempts[0]
        assert attempt.resolved is False
        # and the empty result set is untouched — no documents conjured
        mutated = asyncio.run(SimulatedAdapter().rerun(trace, StageMutation(stage="retrieve", param="top_k", value=10)))
        assert mutated.stage_span("retrieve").attrs["hit_scores"] == []

    def test_context_limit_resolves_truncation(self, corpus, failures, replayer):
        trace, failure = _first(corpus, failures, "F-PROMPT-001")
        result = _replay(replayer, trace, "generate", "context_limit", 16384, failure)
        assert result.attempts[0].resolved is True

    def test_unmodeled_mutation_is_flagged_noop(self, corpus, failures, replayer):
        trace, failure = _first(corpus, failures, "F-RET-001")
        result = _replay(replayer, trace, "generate", "model", "gpt-5", failure)
        attempt = result.attempts[0]
        assert attempt.resolved is False
        assert attempt.note is not None and "un-modeled" in attempt.note

    def test_simulated_adapter_has_no_actual_outcome(self):
        assert SimulatedAdapter().actual_outcome(None, None) is None


class TestReplayResult:
    def test_best_picks_resolving_lift(self, corpus, failures, replayer):
        trace, failure = _first(corpus, failures, "F-RET-002")
        mutations = [
            StageMutation(stage="retrieve", param="top_k", value=10),
            StageMutation(stage="retrieve", param="query_rewrite", value=True),
        ]
        result = asyncio.run(replayer.replay(trace, mutations, failure))
        best = result.best
        assert best is not None
        assert "query_rewrite" in best.mutation

    def test_protocol_satisfied_by_simulated_adapter(self):
        from forensiq.attribution.replay import PipelineAdapter

        assert isinstance(SimulatedAdapter(), PipelineAdapter)

    def test_real_adapter_records_actual(self, corpus, failures):
        """An adapter that provides actual_outcome gets it recorded — the
        predicted-vs-actual hook for calibration."""

        class RealAdapter:
            def __init__(self):
                self.calls = 0

            async def rerun(self, trace, mutation):
                mutated = trace.model_copy(deep=True)
                ret = mutated.stage_span("retrieve")
                ret.attrs["hit_scores"] = [0.9] * int(mutation.value or 5)
                ret.attrs["hit_count"] = int(mutation.value or 5)
                return mutated

            def actual_outcome(self, trace, mutation):
                return True

        trace, failure = _first(corpus, failures, "F-RET-002")
        replayer = WhatIfReplay(adapter=RealAdapter(), classifier=FailureClassifier())
        result = asyncio.run(replayer.replay(trace, [StageMutation(stage="retrieve", param="top_k", value=8)], failure))
        assert result.attempts[0].actual is True
        assert result.attempts[0].resolved is True
