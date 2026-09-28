"""Counterfactual replay ("what if we had run it with X?").

A :class:`PipelineAdapter` re-runs one stage of a trace with mutated params
(top_k 5->10, a different reranker, a bigger context window). ForensiQ ships
a :class:`SimulatedAdapter` whose effect model is a documented ESTIMATE —
it mutates span attrs by a deterministic rule and lets the classifier decide
whether the failure would have resolved. Honest limits (stated in README):
a simulation cannot know what the real retrieval layer would have returned,
so predicted lift is a hypothesis to verify with the real adapter, not a
measurement.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from forensiq.attribution.health import trace_health
from forensiq.ingest.schema import Trace
from forensiq.taxonomy.classifier import FailureClassifier


class StageMutation(BaseModel):
    """One parameter mutation applied to one stage during replay."""

    stage: str
    param: str
    value: Any
    label: str | None = None

    def describe(self) -> str:
        return self.label or f"{self.stage}.{self.param} -> {self.value}"


@runtime_checkable
class PipelineAdapter(Protocol):
    """Caller-supplied counterfactual executor.

    Implement `rerun` against your real pipeline (re-run retrieval with a new
    top_k, swap the reranker, expand context) and optionally `actual_outcome`
    to report whether the failure really resolved. ForensiQ never requires
    one — the SimulatedAdapter keeps everything offline.
    """

    async def rerun(self, trace: Trace, mutation: StageMutation) -> Trace: ...

    def actual_outcome(self, trace: Trace, mutation: StageMutation) -> bool | None:
        """True/False when the adapter knows the real outcome; None otherwise."""
        ...


class SimulatedAdapter:
    """Offline effect model. Deterministic, documented, honest about limits.

    Effects:
      * retrieve.top_k (increase): adds (new_k - old_k) hits scored slightly
        BELOW the current mean — marginal documents dilute precision. An
        empty result set stays empty: top_k cannot conjure documents that
        the index does not have.
      * retrieve.query_rewrite: lifts hit scores by +0.18 (bounded 0.97) —
        models "a better query finds better documents".
      * rerank.reranker: rerank_delta += 0.08 (a stronger cross-encoder).
      * generate.context_limit: caps prompt_chars and clears the length
        finish_reason (the truncation would not happen).
      * generate.model: explicitly un-modeled — the trace is returned
        unchanged and the attempt is flagged.
    """

    SCORE_LIFT = 0.18

    async def rerun(self, trace: Trace, mutation: StageMutation) -> Trace:
        mutated = trace.model_copy(deep=True)
        if mutation.stage == "retrieve" and mutation.param == "top_k":
            self._mutate_top_k(mutated, int(mutation.value))
        elif mutation.stage == "retrieve" and mutation.param == "query_rewrite":
            self._mutate_query_rewrite(mutated)
        elif mutation.stage == "rerank" and mutation.param == "reranker":
            for span in mutated.stage_spans("rerank"):
                delta = float(span.attr("rerank_delta") or 0.0)
                span.attrs["rerank_delta"] = round(delta + 0.08, 4)
                span.attrs["reranker"] = str(mutation.value)
        elif mutation.stage == "generate" and mutation.param == "context_limit":
            limit = int(mutation.value)
            for span in mutated.stage_spans("generate"):
                span.attrs["prompt_chars"] = min(int(span.attr("prompt_chars") or 0), int(limit * 0.6))
                if span.attr("finish_reason") == "length":
                    span.attrs["finish_reason"] = "stop"
        else:
            # Un-modeled mutation: recorded as a no-op so results stay honest.
            for span in mutated.spans:
                span.attrs.setdefault("_replay_note", f"un-modeled mutation: {mutation.describe()}")
        return mutated

    def actual_outcome(self, trace: Trace, mutation: StageMutation) -> bool | None:
        return None  # simulation only — there is no actual outcome

    # -- effect model internals ---------------------------------------------

    def _mutate_top_k(self, trace: Trace, new_k: int) -> None:
        for span in trace.stage_spans("retrieve"):
            old_k = int(span.attr("top_k") or 0)
            scores = list(span.attr("hit_scores") or [])
            span.attrs["top_k"] = new_k
            if not scores or new_k <= old_k:
                continue  # empty index stays empty; shrinking k is not a fix
            mean = sum(scores) / len(scores)
            # Marginal hits land below the current mean — honest dilution.
            extra = [round(max(0.05, mean - 0.05 - i * 0.02), 3) for i in range(new_k - old_k)]
            merged = sorted(scores + extra, reverse=True)
            span.attrs["hit_scores"] = merged
            span.attrs["hit_count"] = len(merged)

    def _mutate_query_rewrite(self, trace: Trace) -> None:
        for span in trace.stage_spans("retrieve"):
            scores = list(span.attr("hit_scores") or [])
            if not scores:
                continue
            lifted = [round(min(0.97, s + self.SCORE_LIFT), 3) for s in scores]
            span.attrs["hit_scores"] = lifted
            span.attrs["hit_count"] = len(lifted)
            span.attrs["query"] = f"{span.attr('query', '')} [rewritten]"


class ReplayAttempt(BaseModel):
    mutation: str
    resolved: bool
    predicted_lift: dict[str, float]
    actual: bool | None = None
    note: str | None = None


class ReplayResult(BaseModel):
    trace_id: str
    original_taxonomy_id: str | None = None
    attempts: list[ReplayAttempt] = Field(default_factory=list)

    @property
    def best(self) -> ReplayAttempt | None:
        """The resolving attempt with the largest predicted lift, if any."""
        resolving = [a for a in self.attempts if a.resolved]
        if not resolving:
            return None
        return max(resolving, key=lambda a: sum(a.predicted_lift.values()))


class WhatIfReplay:
    def __init__(self, adapter: PipelineAdapter, classifier: FailureClassifier | None = None) -> None:
        self.adapter = adapter
        self.classifier = classifier or FailureClassifier()

    async def replay(
        self, trace: Trace, mutations: list[StageMutation], failure: Any = None
    ) -> ReplayResult:
        """Replay one failed trace under each mutation.

        resolved  — would the classifier still flag this trace?
        predicted_lift — health-metric deltas (mean_hit_score, groundedness,
                         rerank_delta) between original and replayed trace.
        actual    — filled only when the adapter knows the real outcome.
        """
        before_metrics = trace_health(trace)
        before_failure = self.classifier.classify(trace)
        original_id = before_failure.taxonomy_id if before_failure else None
        result = ReplayResult(trace_id=trace.trace_id, original_taxonomy_id=original_id)
        reference_id = failure.taxonomy_id if failure is not None else (
            before_failure.taxonomy_id if before_failure is not None else None
        )
        for mutation in mutations:
            replayed = await self.adapter.rerun(trace, mutation)
            after_failure = self.classifier.classify(replayed)
            resolved = after_failure is None or after_failure.taxonomy_id != reference_id
            after_metrics = trace_health(replayed)
            lift = {
                key: round(after_metrics.get(key, 0.0) - before_metrics.get(key, 0.0), 4)
                for key in before_metrics
            }
            note = None
            actual = None
            actual_fn = getattr(self.adapter, "actual_outcome", None)
            if callable(actual_fn):
                actual = actual_fn(trace, mutation)
            mutated_span = replayed.stage_span(mutation.stage)
            if mutated_span is not None and "_replay_note" in mutated_span.attrs:
                note = str(mutated_span.attrs["_replay_note"])
            result.attempts.append(
                ReplayAttempt(
                    mutation=mutation.describe(),
                    resolved=bool(resolved),
                    predicted_lift=lift,
                    actual=actual,
                    note=note,
                )
            )
        return result


DEFAULT_MUTATIONS: list[StageMutation] = [
    StageMutation(stage="retrieve", param="top_k", value=10),
    StageMutation(stage="retrieve", param="query_rewrite", value=True),
    StageMutation(stage="rerank", param="reranker", value="cross-encoder-ms-marco"),
    StageMutation(stage="generate", param="context_limit", value=16384),
]
