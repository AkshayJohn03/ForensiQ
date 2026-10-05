"""Causal-chain extraction for the single-page incident postmortem.

The chain is the security-incident reading of a trace (SIEM for AI):

    TRIGGER      the user prompt that started the request
    POISONING    which retrieved context/chunks entered the model's prompt
    DIVERGENCE   what the model produced under that influence
    VALIDATOR    which guardrail should have caught it — and why it didn't
    BLAST RADIUS which downstream steps consumed the bad output

Everything is derived from the recorded trace + failure records; nothing is
inferred at report time, so the chain is quotable in a postmortem and two
runs over the same trace are byte-identical.
"""

from __future__ import annotations

from typing import Any

from forensiq.ingest.schema import Span, Trace
from forensiq.taxonomy.classifier import FailureRecord

# The guardrail that SHOULD have stopped each failure mode, and the generic
# reason it fails when the trace shows no guard activity (the "why it didn't").
EXPECTED_VALIDATORS: dict[str, str] = {
    "F-RET-001": (
        "no-hits fallback guard (an empty retrieval set should trigger reformulation or escalation before generation)"
    ),
    "F-RET-002": "retrieval score SLO (mean hit score below threshold should block or re-query before generation)",
    "F-RET-003": "drift monitor on retrieval scores (degrading hit scores should trip the weekly drift alarm)",
    "F-PROMPT-001": "context-budget guard (prompt near the context ceiling should trim before send)",
    "F-GEN-001": "schema/JSON validator on the model output (unparseable output should trigger a repair retry)",
    "F-GEN-002": "repetition detector on decoded text (n-gram loops should trip a decoding guard)",
    "F-GEN-003": (
        "answer-relevance check (a refusal despite usable context should be caught before it reaches the user)"
    ),
    "F-GEN-004": "groundedness/citation validator (fabricated claims should fail the grounding gate)",
    "F-TOOL-001": "tool error guard (tool exceptions should be surfaced to the model for replanning, not propagated)",
    "F-INFRA-001": "stage timeout watchdog (a hung stage should be aborted with a fallback)",
    "F-INFRA-002": "cost guardrail (per-trace budget should cap runaway prompt growth)",
    "F-INFRA-003": "the guard itself (it fired as designed — review whether the block was correct)",
}

SNIPPET = 160  # evidence snippet length in the causal chain


def clip(text: Any, limit: int = SNIPPET) -> str:
    text = str(text).strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def quote(value: Any) -> str:
    return f"`{clip(value, 80)}`"


def find_trigger(trace: Trace) -> tuple[Span | None, str]:
    """The user prompt that started it: first retrieve query, else the
    generate-side prompt, else the first span's input-ish attrs."""
    for span in trace.spans:
        query = span.attr("query")
        if query:
            return span, str(query)
    for span in trace.spans:
        for key in ("input", "prompt", "input.value", "question"):
            if span.attr(key):
                return span, str(span.attr(key))
    return (trace.spans[0] if trace.spans else None), ""


def find_poisoning(trace: Trace) -> dict[str, Any]:
    """What entered the prompt from retrieval: chunk counts, the weakest
    chunk (the prime suspect), and the context snippet that reached the model."""
    retrieve = trace.stage_span("retrieve")
    generate = trace.stage_span("generate")
    out: dict[str, Any] = {"span": retrieve, "hit_count": None, "top_k": None, "weakest": None, "context": None}
    if retrieve is not None:
        out["hit_count"] = retrieve.attr("hit_count")
        out["top_k"] = retrieve.attr("top_k")
        scores = retrieve.attr("hit_scores") or []
        if scores:
            out["weakest"] = min(float(s) for s in scores)
            out["mean"] = sum(float(s) for s in scores) / len(scores)
    if generate is not None:
        context = generate.attr("context")
        if context:
            out["context"] = clip(context)
    return out


def find_divergence(trace: Trace) -> dict[str, Any]:
    """What the model produced."""
    generate = trace.stage_span("generate")
    out: dict[str, Any] = {"span": generate, "output": None, "finish_reason": None}
    if generate is not None:
        out["output"] = clip(generate.attr("output"))
        out["finish_reason"] = generate.attr("finish_reason")
    return out


def find_validator_gap(trace: Trace, failure: FailureRecord) -> dict[str, Any]:
    """Which guardrail should have caught it, and why it didn't.

    Reads the guard stage of the trace: absent guard span -> the pipeline has
    no validator checkpoint at all; present-but-passing guard -> a coverage
    gap in the rules; guard blocked -> the validator worked (only meaningful
    for F-INFRA-003).
    """
    guard_spans = trace.stage_spans("guard")
    expected = EXPECTED_VALIDATORS.get(failure.taxonomy_id, "a guardrail between retrieval and generation")
    if not guard_spans:
        why = "no guard stage span in the trace — the pipeline has no validator checkpoint on this path"
    else:
        blocked = [bool(g.attr("blocked", False)) for g in guard_spans]
        if any(blocked):
            why = "a guard DID block on this trace — if the failure still shipped, the block landed after the damage"
        else:
            rules = ", ".join(sorted({str(g.attr("rules") or g.name) for g in guard_spans}))
            why = (
                f"guard ran and passed (blocked=false; rules: {rules or 'unknown'}) — "
                f"rule coverage gap for {failure.taxonomy_id}"
            )
    return {"expected": expected, "why": why, "guards": guard_spans}


def find_blast_radius(trace: Trace, failure: FailureRecord) -> dict[str, Any]:
    """Downstream steps after the first failing-stage span (the stages that
    consumed the poisoned context / bad output), plus their cost share."""
    evidence_span_id = failure.evidence[0].span_id if failure.evidence else None
    failing_index = None
    if evidence_span_id:
        failing_index = next((i for i, s in enumerate(trace.spans) if s.span_id == evidence_span_id), None)
    if failing_index is None:
        failing_index = next((i for i, s in enumerate(trace.spans) if s.stage == failure.stage), None)
    downstream = trace.spans[failing_index + 1 :] if failing_index is not None else []
    stages = sorted({s.stage for s in downstream})
    downstream_cost = sum(s.cost for s in downstream)
    return {
        "count": len(downstream),
        "stages": stages,
        "cost": downstream_cost,
        "user_facing": any(s.stage == "generate" for s in downstream) or failure.stage == "generate",
    }
