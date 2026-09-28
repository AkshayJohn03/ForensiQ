"""Two-pass failure classification.

Pass 1 — deterministic rules per stage (fast, explainable, offline):
every rule quotes span refs + attr values as evidence and carries a
confidence. Ordered by precedence; the first match is the primary failure.

Pass 2 — optional LLM classification for ambiguous leftovers (spans that
errored or were blocked without a deterministic signature). Behind the
:class:`forensiq.llm.LLMClient` Protocol; offline the EchoMockClient provides
a deterministic heuristic verdict, and any provider error degrades to the
same fallback. Failure forensics has to work during the outage that caused
the failures it is analyzing.

Taxonomy ids live in docs/TAXONOMY.md (F-RET-*, F-PROMPT-*, F-GEN-*,
F-TOOL-*, F-INFRA-*); the catalog is mirrored in forensiq.report.rca.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from pydantic import BaseModel

from forensiq._stats import percentile
from forensiq.ingest.schema import EvidenceItem, Span, Trace
from forensiq.llm import LLMClient, parse_llm_verdict

# Refusal phrasings mirror RAG_showcase's abstention detector (metrics.py).
REFUSAL_PHRASES: tuple[str, ...] = (
    "i cannot answer",
    "i can't answer",
    "i do not have",
    "i don't have",
    "no information",
    "unable to answer",
    "i don't know",
    "not found in the provided",
    "does not contain",
    "do not have the relevant information",
)

NEGATIONS = {"not", "no", "never", "n't", "without", "none"}


class FailureRecord(BaseModel):
    """One classified failure with its evidence chain."""

    taxonomy_id: str
    stage: str
    trace_id: str
    confidence: float
    evidence: list[EvidenceItem]
    source: str = "rule"  # rule | llm | fallback


class RuleContext(BaseModel):
    """Thresholds handed to pass-1 rules."""

    low_score_threshold: float = 0.5
    context_limit: int = 8192
    truncation_fraction: float = 0.9
    repetition_min_repeat: int = 5
    cost_threshold: float | None = None


Rule = Callable[[Trace, RuleContext], FailureRecord | None]


def _span_evidence(span: Span, key: str, note: str | None = None) -> EvidenceItem:
    return EvidenceItem(
        span_id=span.span_id,
        stage=span.stage,
        key=key,
        value=str(span.attr(key)),
        note=note,
    )


def _mean(scores: list[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0


def _rule_timeout(trace: Trace, ctx: RuleContext) -> FailureRecord | None:
    for span in trace.spans:
        if span.status == "timeout":
            return FailureRecord(
                taxonomy_id="F-INFRA-001",
                stage=span.stage,
                trace_id=trace.trace_id,
                confidence=0.95,
                evidence=[
                    _span_evidence(span, "status", "span timed out"),
                    EvidenceItem(
                        span_id=span.span_id,
                        stage=span.stage,
                        key="duration_ms",
                        value=f"{span.duration_ms:.0f}",
                        note="duration before abort",
                    ),
                ],
            )
    return None


def _rule_tool_error(trace: Trace, ctx: RuleContext) -> FailureRecord | None:
    for span in trace.spans:
        if span.stage == "tool" and span.status == "error":
            return FailureRecord(
                taxonomy_id="F-TOOL-001",
                stage="tool",
                trace_id=trace.trace_id,
                confidence=0.90,
                evidence=[
                    _span_evidence(span, "status"),
                    _span_evidence(span, "error", "tool raised"),
                    _span_evidence(span, "tool"),
                ],
            )
    return None


def _rule_empty_retrieval(trace: Trace, ctx: RuleContext) -> FailureRecord | None:
    span = trace.stage_span("retrieve")
    if span is None:
        return None
    hit_count = span.attr("hit_count")
    scores = span.attr("hit_scores") or []
    if (hit_count == 0 or (hit_count is None and not scores)) and not scores:
        return FailureRecord(
            taxonomy_id="F-RET-001",
            stage="retrieve",
            trace_id=trace.trace_id,
            confidence=0.95,
            evidence=[
                _span_evidence(span, "hit_count", "empty result set — no hits at any k"),
                _span_evidence(span, "query"),
                _span_evidence(span, "top_k"),
            ],
        )
    return None


def _rule_low_recall(trace: Trace, ctx: RuleContext) -> FailureRecord | None:
    span = trace.stage_span("retrieve")
    if span is None:
        return None
    scores = span.attr("hit_scores") or []
    if not scores:
        return None
    mean_score = span.attr("mean_hit_score")
    mean_score = float(mean_score) if mean_score is not None else _mean(scores)
    if mean_score < ctx.low_score_threshold:
        return FailureRecord(
            taxonomy_id="F-RET-002",
            stage="retrieve",
            trace_id=trace.trace_id,
            confidence=0.85,
            evidence=[
                EvidenceItem(
                    span_id=span.span_id,
                    stage=span.stage,
                    key="hit_scores",
                    value=str([round(s, 3) for s in scores]),
                    note=f"weak context: mean {mean_score:.3f} below threshold {ctx.low_score_threshold}",
                ),
                _span_evidence(span, "query"),
            ],
        )
    return None


def _rule_prompt_truncation(trace: Trace, ctx: RuleContext) -> FailureRecord | None:
    span = trace.stage_span("generate")
    if span is None:
        return None
    finish = str(span.attr("finish_reason", ""))
    prompt_chars = span.attr("prompt_chars")
    near_limit = prompt_chars is not None and prompt_chars >= ctx.truncation_fraction * ctx.context_limit
    if finish == "length" and near_limit:
        return FailureRecord(
            taxonomy_id="F-PROMPT-001",
            stage="generate",
            trace_id=trace.trace_id,
            confidence=0.90,
            evidence=[
                _span_evidence(span, "finish_reason", "hit the context ceiling"),
                _span_evidence(
                    span,
                    "prompt_chars",
                    f">= {ctx.truncation_fraction:.0%} of context_limit {ctx.context_limit}",
                ),
            ],
        )
    return None


def _rule_format_break(trace: Trace, ctx: RuleContext) -> FailureRecord | None:
    span = trace.stage_span("generate")
    if span is None:
        return None
    output = str(span.attr("output", ""))
    declared_json = span.attr("output_format") == "json"
    looks_json = output.lstrip()[:1] in ("{", "[")
    if not (declared_json or looks_json) or not output.strip():
        return None
    try:
        json.loads(output)
    except json.JSONDecodeError as exc:
        return FailureRecord(
            taxonomy_id="F-GEN-001",
            stage="generate",
            trace_id=trace.trace_id,
            confidence=0.90,
            evidence=[
                _span_evidence(span, "output_format", "structured output expected"),
                EvidenceItem(
                    span_id=span.span_id,
                    stage=span.stage,
                    key="output",
                    value=output[:120],
                    note=f"JSON parse failed: {exc.msg} (pos {exc.pos})",
                ),
            ],
        )
    return None


def _rule_repetition_loop(trace: Trace, ctx: RuleContext) -> FailureRecord | None:
    span = trace.stage_span("generate")
    if span is None:
        return None
    output = str(span.attr("output", ""))
    if len(output) < 80:
        return None
    tokens = re.findall(r"\w+", output.lower())
    best_ngram, best_count = "", 0
    for n in (3, 5, 8):
        counts: dict[tuple[str, ...], int] = {}
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i : i + n])
            counts[gram] = counts.get(gram, 0) + 1
        for gram, count in counts.items():
            if count > best_count:
                best_ngram, best_count = " ".join(gram), count
    if best_count >= ctx.repetition_min_repeat:
        evidence = [
            EvidenceItem(
                span_id=span.span_id,
                stage=span.stage,
                    key="output",
                    value=f'"{best_ngram}" x{best_count}',
                    note=f"repetition loop: {ctx.repetition_min_repeat}+ repeats of the same n-gram",
            )
        ]
        if span.tokens_out:
            evidence.append(_span_evidence(span, "tokens_out"))
        return FailureRecord(
            taxonomy_id="F-GEN-002",
            stage="generate",
            trace_id=trace.trace_id,
            confidence=0.90,
            evidence=evidence,
        )
    return None


def _rule_refusal(trace: Trace, ctx: RuleContext) -> FailureRecord | None:
    span = trace.stage_span("generate")
    if span is None:
        return None
    output = str(span.attr("output", "")).lower()
    matched = next((p for p in REFUSAL_PHRASES if p in output), None)
    if matched is None:
        return None
    # Only a failure when retrieval actually delivered usable context;
    # an empty-context refusal is a retrieval problem (F-RET-001 wins above).
    ret = trace.stage_span("retrieve")
    scores = (ret.attr("hit_scores") or []) if ret else []
    if scores and _mean(scores) >= ctx.low_score_threshold:
        return FailureRecord(
            taxonomy_id="F-GEN-003",
            stage="generate",
            trace_id=trace.trace_id,
            confidence=0.80,
            evidence=[
                EvidenceItem(
                    span_id=span.span_id,
                    stage=span.stage,
                    key="output",
                    value=str(span.attr("output"))[:120],
                    note=f"refusal phrase: '{matched}'",
                ),
                EvidenceItem(
                    span_id=ret.span_id if ret else span.span_id,
                    stage="retrieve",
                    key="hit_scores",
                    value=str([round(s, 3) for s in scores]),
                    note="context was present, model declined anyway",
                ),
            ],
        )
    return None


def _rule_cost_spike(trace: Trace, ctx: RuleContext) -> FailureRecord | None:
    if ctx.cost_threshold is None or trace.total_cost <= ctx.cost_threshold:
        return None
    gen = trace.stage_span("generate")
    evidence = [
        EvidenceItem(
            span_id=gen.span_id if gen else trace.trace_id,
            stage=gen.stage if gen else "trace",
            key="total_cost",
            value=f"{trace.total_cost:.4f}",
            note=f"> threshold {ctx.cost_threshold:.4f}",
        )
    ]
    if gen:
        evidence.append(_span_evidence(gen, "tokens_in", "prompt exploded"))
    return FailureRecord(
        taxonomy_id="F-INFRA-002",
        stage=gen.stage if gen else "generate",
        trace_id=trace.trace_id,
        confidence=0.85,
        evidence=evidence,
    )


# Precedence order: hard signals first, upstream causes before downstream
# symptoms, cost last (it is a cohort-relative anomaly, not a signature).
RULES: tuple[Rule, ...] = (
    _rule_timeout,
    _rule_tool_error,
    _rule_empty_retrieval,
    _rule_low_recall,
    _rule_prompt_truncation,
    _rule_format_break,
    _rule_repetition_loop,
    _rule_refusal,
    _rule_cost_spike,
)

# Signals that mark a trace as "ambiguous" for pass 2 (no rule fired but
# something is off): a non-tool error span, a guard block, an empty output.
AMBIGUITY_SIGNALS: tuple[str, ...] = ("span_error", "guard_blocked", "empty_output")

# Canonical stage per taxonomy id (mirrors docs/TAXONOMY.md).
TAXONOMY_STAGE: dict[str, str] = {
    "F-RET-001": "retrieve",
    "F-RET-002": "retrieve",
    "F-RET-003": "retrieve",
    "F-PROMPT-001": "generate",
    "F-GEN-001": "generate",
    "F-GEN-002": "generate",
    "F-GEN-003": "generate",
    "F-GEN-004": "generate",
    "F-TOOL-001": "tool",
    "F-INFRA-001": "generate",
    "F-INFRA-002": "generate",
    "F-INFRA-003": "guard",
}


class FailureClassifier:
    """Rules-first classifier with an optional LLM second pass."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        low_score_threshold: float = 0.5,
        context_limit: int = 8192,
        repetition_min_repeat: int = 5,
        cost_spike_factor: float = 3.0,
        refusal_phrases: tuple[str, ...] | None = None,
    ) -> None:
        self.llm = llm
        self.low_score_threshold = low_score_threshold
        self.context_limit = context_limit
        self.repetition_min_repeat = repetition_min_repeat
        self.cost_spike_factor = cost_spike_factor
        self.refusal_phrases = refusal_phrases or REFUSAL_PHRASES
        self.last_cost_threshold: float | None = None

    # -- pass 1 ---------------------------------------------------------

    def context(self, cost_threshold: float | None = None) -> RuleContext:
        return RuleContext(
            low_score_threshold=self.low_score_threshold,
            context_limit=self.context_limit,
            repetition_min_repeat=self.repetition_min_repeat,
            cost_threshold=cost_threshold,
        )

    def classify(self, trace: Trace, cost_threshold: float | None = None) -> FailureRecord | None:
        """Pass 1 only: first matching rule wins (precedence-ordered)."""
        ctx = self.context(cost_threshold)
        for rule in RULES:
            record = rule(trace, ctx)
            if record is not None:
                return record
        return None

    def classify_cohort(self, traces: list[Trace]) -> list[FailureRecord]:
        """Classify a cohort; the cost-spike threshold is derived from the
        cohort p95 x cost_spike_factor so the rule is deployment-relative."""
        costs = [t.total_cost for t in traces if t.total_cost > 0]
        threshold = None
        if len(costs) >= 20:
            threshold = max(1e-6, percentile(costs, 95.0) * self.cost_spike_factor)
        self.last_cost_threshold = threshold
        records: list[FailureRecord] = []
        for trace in traces:
            record = self.classify(trace, cost_threshold=threshold)
            if record is not None:
                records.append(record)
        return records

    # -- pass 2 ---------------------------------------------------------

    def ambiguity_signals(self, trace: Trace) -> list[str]:
        signals: list[str] = []
        for span in trace.spans:
            if span.status == "error" and span.stage != "tool":
                signals.append("span_error")
            if span.stage == "guard" and bool(span.attr("blocked", False)):
                signals.append("guard_blocked")
            if span.stage == "generate" and not str(span.attr("output", "")).strip() and span.status == "ok":
                signals.append("empty_output")
        return sorted(set(signals))

    def _ambiguity_prompt(self, trace: Trace, signals: list[str]) -> str:
        candidates = "\n".join(
            f"  - {tid}: {' '.join(kws)}"
            for tid, kws in [
                ("F-RET-001", "empty retrieval hit_count=0"),
                ("F-RET-002", "low hit scores weak context"),
                ("F-PROMPT-001", "prompt truncation finish_reason=length"),
                ("F-GEN-001", "structured output json parse format"),
                ("F-GEN-002", "repetition loop n-gram"),
                ("F-GEN-003", "refusal cannot answer"),
                ("F-TOOL-001", "tool execution error"),
                ("F-INFRA-001", "timeout"),
                ("F-INFRA-002", "cost spike"),
                ("F-INFRA-003", "guard policy block abort"),
            ]
        )
        evidence = "; ".join(
            f"{s.span_id}:{s.key}={s.value}" for s in self._trace_evidence_summary(trace)
        )
        return (
            "You are a failure forensics classifier for AI pipelines. "
            "No deterministic rule matched this trace, but these ambiguity "
            f"signals fired: {signals}.\n"
            f"Trace spans:\n{evidence}\n"
            "Pick exactly one taxonomy id from:\n"
            f"{candidates}\n"
            "Answer in exactly this format:\n"
            "TAXONOMY: <id>\nCONFIDENCE: <0..1>\nREASON: <one line>"
        )

    def _trace_evidence_summary(self, trace: Trace) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        for span in trace.spans:
            items.append(
                EvidenceItem(
                    span_id=span.span_id,
                    stage=span.stage,
                    key="status",
                    value=span.status,
                )
            )
            for key in ("hit_count", "mean_hit_score", "finish_reason", "output", "blocked"):
                if key in span.attrs:
                    items.append(
                        EvidenceItem(
                            span_id=span.span_id,
                            stage=span.stage,
                            key=key,
                            value=str(span.attrs[key])[:100],
                        )
                    )
        return items

    def _fallback_verdict(self, trace: Trace, signals: list[str]) -> FailureRecord | None:
        """Deterministic heuristic used when no LLM is configured, the LLM is
        unreachable, or its answer is unparseable."""
        if "guard_blocked" in signals:
            tax_id, stage = "F-INFRA-003", "guard"
        elif "span_error" in signals:
            span = next(s for s in trace.spans if s.status == "error")
            tax_id, stage = ("F-TOOL-001", "tool") if span.stage == "tool" else ("F-INFRA-003", span.stage)
        elif "empty_output" in signals:
            tax_id, stage = "F-GEN-001", "generate"
        else:
            return None
        return FailureRecord(
            taxonomy_id=tax_id,
            stage=stage,
            trace_id=trace.trace_id,
            confidence=0.45,
            evidence=self._trace_evidence_summary(trace)[:4],
            source="fallback",
        )

    async def classify_ambiguous(self, trace: Trace) -> FailureRecord | None:
        """Pass 2: classify a trace whose pass-1 verdict was None but which
        carries ambiguity signals. Uses the configured LLM; degrades to the
        deterministic fallback on any provider/parse error."""
        signals = self.ambiguity_signals(trace)
        if not signals:
            return None
        if self.llm is None:
            return self._fallback_verdict(trace, signals)
        try:
            response = await self.llm.complete(self._ambiguity_prompt(trace, signals))
            tax_id, confidence = parse_llm_verdict(response)
        except Exception:
            # Defensive by design: the LLM pass is optional telemetry; any
            # provider failure must degrade to the offline fallback.
            tax_id, confidence = None, None
        if tax_id is None:
            record = self._fallback_verdict(trace, signals)
            return record
        stage = TAXONOMY_STAGE.get(tax_id, "generate")
        return FailureRecord(
            taxonomy_id=tax_id,
            stage=stage,
            trace_id=trace.trace_id,
            confidence=confidence if confidence is not None else 0.5,
            evidence=self._trace_evidence_summary(trace)[:4],
            source="llm",
        )
