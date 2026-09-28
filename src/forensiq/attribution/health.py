"""Per-stage health scoring for RAG traces.

Metric names align with RAG_showcase's eval module (backend/app/eval/metrics.py:
Context Precision/Recall, Faithfulness, Answer Relevance) so reports read as a
continuation of that pipeline's own numbers. The groundedness heuristic here is
a deliberate improvement on its lexical faithfulness: numbers and capitalized
entities are treated as ANCHORED tokens — a fabricated number or entity
penalizes double — and a negation guard penalizes answers that negate what the
context affirms. Still a lexical heuristic, not an NLI judge; the limitation
is documented, not discovered by a reviewer.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from forensiq._stats import mean_std, percentile
from forensiq.ingest.schema import Trace

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "is", "are", "was", "were", "be", "been", "it", "its", "this", "that",
    "as", "at", "from", "has", "have", "had", "per", "within", "under",
}
_NEGATIONS = {"not", "no", "never", "without", "none", "cannot", "doesn't", "don't", "isn't"}
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)?")
_NUMBER_RE = re.compile(r"\d")


def _content_tokens(text: str) -> list[tuple[str, bool]]:
    """(token, is_anchored) pairs. Anchored = number-bearing token.

    Number-bearing tokens are kept REGARDLESS of length — dropping '8'/'24'
    would make a fabricated figure invisible to the penalty, which defeats
    the anchored-token improvement over plain lexical overlap.
    """
    out: list[tuple[str, bool]] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in _STOPWORDS:
            continue
        anchored = bool(_NUMBER_RE.search(tok))
        if len(tok) <= 2 and not anchored:
            continue
        out.append((tok, anchored))
    return out


def _entities(text: str) -> set[str]:
    """Capitalized multi-letter words in the ORIGINAL text (proper nouns)."""
    return {
        w.lower()
        for w in re.findall(r"\b[A-Z][a-zA-Z0-9-]{2,}\b", text)
        if w.lower() not in _STOPWORDS
    }


def groundedness(output: str, context: str) -> float:
    """Claim grounding heuristic in [0, 1], sentence-averaged.

    Improvements over plain token overlap:
      * anchored tokens (numbers, entities) count double when missing —
        fabricated figures/names are the costliest hallucination class;
      * negation guard: output negations absent from the context penalize
        (answer denying what the source affirms).

    Empty/abstaining output scores 0.0 — callers decide whether an abstention
    is correct behavior (RAG_showcase's faithfulness treats abstention as 1.0;
    here abstention surfaces as low groundedness AND an F-GEN-003 refusal
    classification, which keeps the two signals separable).
    """
    if not output.strip():
        return 0.0
    if not context.strip():
        return 0.0
    ctx_tokens = {t for t, _ in _content_tokens(context)}
    ctx_entities = _entities(context)
    ctx_all = ctx_tokens | ctx_entities
    ctx_negations = {t for t in _TOKEN_RE.findall(context.lower()) if t in _NEGATIONS}

    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", output) if s.strip()]
    if not sentences:
        sentences = [output]
    scores: list[float] = []
    for sent in sentences:
        toks = _content_tokens(sent)
        if not toks:
            continue
        sent_entities = _entities(sent)
        missing = 0
        for tok, anchored in toks:
            if tok in ctx_all:
                continue
            missing += 2 if (anchored or tok in sent_entities) else 1
        base = 1.0 - min(1.0, missing / max(1, len(toks) + sum(1 for t in sent_entities if t not in ctx_all)))
        out_negations = {t for t in _TOKEN_RE.findall(sent.lower()) if t in _NEGATIONS}
        unshared_neg = out_negations - ctx_negations
        base -= 0.25 * len(unshared_neg)
        scores.append(max(0.0, min(1.0, base)))
    return sum(scores) / len(scores) if scores else 0.0


class StageMetrics(BaseModel):
    """Health card for one pipeline stage across a trace cohort."""

    stage: str
    metrics: dict[str, float]
    status: str = "ok"  # ok | warn | crit
    flags: list[str] = []


class HealthReport(BaseModel):
    stages: list[StageMetrics]
    trace_count: int

    def get(self, stage: str) -> StageMetrics | None:
        return next((s for s in self.stages if s.stage == stage), None)


def _status(value: float, warn_below: float, crit_below: float) -> str:
    if value < crit_below:
        return "crit"
    if value < warn_below:
        return "warn"
    return "ok"


class StageHealth:
    """Cohort-level, stage-conditional health scoring.

    Why stage-conditional: a single "trace health" number hides where the
    pipeline is sick. Retrieval sickness (scores/hits) and generation sickness
    (grounding/format) have different owners, different playbooks and
    different SLOs — so each stage gets its own card with its own thresholds.
    """

    def __init__(
        self,
        low_score_threshold: float = 0.6,
        groundedness_warn: float = 0.6,
        groundedness_crit: float = 0.35,
    ) -> None:
        self.low_score_threshold = low_score_threshold
        self.groundedness_warn = groundedness_warn
        self.groundedness_crit = groundedness_crit

    def score(self, traces: list[Trace]) -> HealthReport:
        cards: list[StageMetrics] = []
        for stage in ("embed", "retrieve", "rerank", "graph", "generate", "guard", "tool"):
            stage_traces = [t for t in traces if t.stage_span(stage) is not None]
            if not stage_traces:
                continue
            card = self._score_stage(stage, stage_traces)
            if card is not None:
                cards.append(card)
        return HealthReport(stages=cards, trace_count=len(traces))

    # -- per-stage cards ----------------------------------------------------

    def _score_stage(self, stage: str, stage_traces: list[Trace]) -> StageMetrics | None:
        metrics: dict[str, float] = {}
        flags: list[str] = []
        status = "ok"

        latencies = [s.duration_ms for t in stage_traces for s in t.stage_spans(stage)]
        metrics["p50_ms"] = round(percentile(latencies, 50), 2)
        metrics["p95_ms"] = round(percentile(latencies, 95), 2)
        errors = [s for t in stage_traces for s in t.stage_spans(stage) if s.status != "ok"]
        metrics["error_rate"] = round(len(errors) / max(1, len(latencies)), 4)

        if stage == "retrieve":
            scores: list[float] = []
            hits: list[float] = []
            ks: list[float] = []
            for t in stage_traces:
                span = t.stage_span("retrieve")
                assert span is not None
                arr = span.attr("hit_scores") or []
                if arr:
                    scores.append(_mean_of(arr))
                top_k = float(span.attr("top_k") or 0)
                hit_count = float(span.attr("hit_count") or len(arr))
                hits.append(hit_count)
                if top_k:
                    ks.append(min(1.0, hit_count / top_k))
            hit_rate = sum(1 for h in hits if h > 0) / max(1, len(hits))
            mean_score = _mean_of(scores) if scores else 0.0
            metrics["hit_rate"] = round(hit_rate, 4)
            metrics["mean_hit_score"] = round(mean_score, 4)
            metrics["k_coverage"] = round(_mean_of(ks) if ks else 0.0, 4)
            if hit_rate < 0.9:
                status = max_status(status, _status(hit_rate, 0.9, 0.7))
                flags.append(f"hit_rate {hit_rate:.2f} — some queries return nothing")
            if mean_score < self.low_score_threshold:
                status = max_status(status, _status(mean_score, 0.6, 0.45))
                flags.append(
                    f"mean_hit_score {mean_score:.2f} below {self.low_score_threshold} — weak context"
                )
        elif stage == "rerank":
            deltas = [
                float(s.attr("rerank_delta") or 0.0)
                for t in stage_traces
                for s in t.stage_spans("rerank")
            ]
            metrics["mean_rerank_delta"] = round(_mean_of(deltas), 4)
            if metrics["mean_rerank_delta"] < 0.01:
                status = "warn"
                flags.append("reranker adds ~nothing — candidate fusion may be the bottleneck")
        elif stage == "generate":
            grounds: list[float] = []
            for t in stage_traces:
                span = t.stage_span("generate")
                assert span is not None
                output = str(span.attr("output", ""))
                context = str(span.attr("context", ""))
                if output.strip():
                    grounds.append(groundedness(output, context))
            if grounds:
                metrics["groundedness_mean"] = round(_mean_of(grounds), 4)
                metrics["groundedness_p25"] = round(percentile(grounds, 25), 4)
                if metrics["groundedness_mean"] < self.groundedness_warn:
                    status = max_status(
                        status, _status(metrics["groundedness_mean"], self.groundedness_warn, self.groundedness_crit)
                    )
                    flags.append(
                        f"groundedness {metrics['groundedness_mean']:.2f} — answers drift from context"
                    )
        elif stage == "embed":
            toks = [
                float(s.attr("token_count") or 0)
                for t in stage_traces
                for s in t.stage_spans(stage)
            ]
            m, sd = mean_std(toks)
            metrics["token_count_mean"] = round(m, 1)
            metrics["token_count_std"] = round(sd, 1)

        return StageMetrics(stage=stage, metrics=metrics, status=status, flags=flags)


def _mean_of(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def max_status(a: str, b: str) -> str:
    order = {"ok": 0, "warn": 1, "crit": 2}
    return a if order[a] >= order[b] else b


def trace_health(trace: Trace) -> dict[str, Any]:
    """Single-trace health snapshot (used by replay to measure lift)."""
    out: dict[str, Any] = {}
    ret = trace.stage_span("retrieve")
    if ret is not None:
        scores = ret.attr("hit_scores") or []
        out["mean_hit_score"] = round(_mean_of(scores), 4)
    gen = trace.stage_span("generate")
    if gen is not None:
        out["groundedness"] = round(groundedness(str(gen.attr("output", "")), str(gen.attr("context", ""))), 4)
    rr = trace.stage_span("rerank")
    if rr is not None:
        out["rerank_delta"] = float(rr.attr("rerank_delta") or 0.0)
    return out
