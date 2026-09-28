"""Root-cause blame ranking.

For each failed trace, every stage gets an anomaly score:
  score(stage) = max directional z-score of its signals vs the pipeline
                 baseline  +  taxonomy evidence weight (if the classifier
                 pinned a failure to that stage).
The z-scores use a healthy baseline (failed traces excluded when ground
truth/failures are supplied), so "how abnormal is this stage vs when the
pipeline works" is the yardstick. Output: ranked candidates with evidence
chains, plus cohort-level blame distribution per taxonomy class.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from pydantic import BaseModel

from forensiq._stats import mean_std, zscore
from forensiq.ingest.schema import EvidenceItem, Stage, Trace
from forensiq.taxonomy.classifier import FailureRecord


# Signal specs per stage: (key, source, direction).
#   source "span" -> Span field (duration_ms, cost, tokens_in/out)
#   source "attr" -> Span.attrs key
#   direction: "high" = bigger is worse, "low" = smaller is worse,
#              "both" = any deviation is suspicious.
class SignalSpec(NamedTuple):
    key: str
    source: str
    direction: str

SIGNAL_SPECS: dict[Stage, tuple[SignalSpec, ...]] = {
    "embed": (
        SignalSpec("duration_ms", "span", "high"),
        SignalSpec("token_count", "attr", "high"),
    ),
    "retrieve": (
        SignalSpec("hit_count", "attr", "low"),
        SignalSpec("mean_hit_score", "attr", "low"),
        SignalSpec("duration_ms", "span", "high"),
    ),
    "rerank": (
        SignalSpec("rerank_delta", "attr", "low"),
        SignalSpec("duration_ms", "span", "high"),
    ),
    "graph": (SignalSpec("duration_ms", "span", "high"),),
    "generate": (
        SignalSpec("duration_ms", "span", "high"),
        SignalSpec("prompt_chars", "attr", "high"),
        SignalSpec("tokens_in", "span", "high"),
        SignalSpec("tokens_out", "span", "both"),
        SignalSpec("cost", "span", "high"),
    ),
    "guard": (SignalSpec("duration_ms", "span", "high"),),
    "tool": (SignalSpec("duration_ms", "span", "high"),),
    "agent": (SignalSpec("duration_ms", "span", "high"),),
}

EVIDENCE_WEIGHT = 1.5


def _signal_value(span: Any, spec: SignalSpec) -> float | None:
    if spec.source == "span":
        value = getattr(span, spec.key, None)
    else:
        value = span.attr(spec.key)
        if spec.key == "mean_hit_score" and value is None:
            scores = span.attr("hit_scores") or []
            value = sum(scores) / len(scores) if scores else None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class Baseline(BaseModel):
    """Per-stage signal baselines from healthy traces."""

    values: dict[str, dict[str, tuple[float, float]]] = {}  # stage -> signal -> (mean, std)
    n_traces: int = 0


class BlameCandidate(BaseModel):
    stage: str
    score: float
    z_score: float
    evidence: list[EvidenceItem]


class BlameRanker:
    def __init__(self, baseline: Baseline, evidence_weight: float = EVIDENCE_WEIGHT) -> None:
        self.baseline = baseline
        self.evidence_weight = evidence_weight

    @classmethod
    def from_traces(
        cls, traces: list[Trace], failures: list[FailureRecord] | None = None
    ) -> BlameRanker:
        """Build stage baselines; exclude failed traces when known so the
        yardstick is 'healthy pipeline behavior'."""
        failed_ids = {f.trace_id for f in failures} if failures else set()
        healthy = [t for t in traces if t.trace_id not in failed_ids]
        raw: dict[str, dict[str, list[float]]] = {}
        for trace in healthy:
            for stage, specs in SIGNAL_SPECS.items():
                for span in trace.stage_spans(stage):
                    for spec in specs:
                        value = _signal_value(span, spec)
                        if value is None:
                            continue
                        raw.setdefault(stage, {}).setdefault(spec.key, []).append(value)
        values: dict[str, dict[str, tuple[float, float]]] = {}
        for stage, signals in raw.items():
            values[stage] = {key: mean_std(vals) for key, vals in signals.items()}
        return cls(Baseline(values=values, n_traces=len(healthy)))

    def rank(self, trace: Trace, failure: FailureRecord | None = None) -> list[BlameCandidate]:
        """Rank stages by anomaly contribution for one trace."""
        candidates: list[BlameCandidate] = []
        for stage, specs in SIGNAL_SPECS.items():
            spans = trace.stage_spans(stage)
            if not spans:
                continue
            best_z = 0.0
            best_spec: SignalSpec | None = None
            best_span: Any = None
            for span in spans:
                for spec in specs:
                    value = _signal_value(span, spec)
                    if value is None:
                        continue
                    mean, std = self.baseline.values.get(stage, {}).get(spec.key, (0.0, 0.0))
                    if mean == 0.0 and std == 0.0:
                        continue  # no baseline for this signal
                    z = zscore(value, mean, std)
                    directional = z if spec.direction in ("high", "both") else -z
                    if spec.direction == "both":
                        directional = abs(z)
                    if directional > best_z:
                        best_z, best_spec, best_span = directional, spec, span
            if best_spec is None or best_span is None:
                continue
            score = best_z
            if failure is not None and failure.stage == stage:
                score += self.evidence_weight
            mean_base, std_base = self.baseline.values[stage][best_spec.key]
            value = _signal_value(best_span, best_spec) or 0.0
            evidence = [
                EvidenceItem(
                    span_id=best_span.span_id,
                    stage=stage,
                    key=best_spec.key,
                    value=f"{value:.4g}",
                    note=(
                        f"baseline {mean_base:.4g} ± {std_base:.4g} "
                        f"(z={zscore(value, mean_base, std_base):+.1f})"
                    ),
                )
            ]
            if failure is not None and failure.stage == stage:
                evidence.extend(failure.evidence[:2])
            candidates.append(
                BlameCandidate(stage=stage, score=round(score, 4), z_score=round(best_z, 4), evidence=evidence)
            )
        # Deterministic order: score desc, then canonical stage order.
        stage_order = {s: i for i, s in enumerate(
            ("embed", "retrieve", "rerank", "graph", "generate", "guard", "tool", "agent")
        )}
        candidates.sort(key=lambda c: (-c.score, stage_order.get(c.stage, 99)))
        return candidates

    def top_stage(self, trace: Trace, failure: FailureRecord | None = None) -> str | None:
        candidates = self.rank(trace, failure)
        return candidates[0].stage if candidates else None

    def cohort_blame(
        self, traces: list[Trace], failures: list[FailureRecord]
    ) -> dict[str, dict[str, float]]:
        """Aggregate blame distribution: taxonomy_id -> {stage: share of top-1}.

        Answers "which stage is to blame for each failure family" over a
        cohort — the number that decides which team gets paged.
        """
        by_id: dict[str, dict[str, int]] = {}
        failure_by_trace = {f.trace_id: f for f in failures}
        for trace in traces:
            failure = failure_by_trace.get(trace.trace_id)
            if failure is None:
                continue
            top = self.top_stage(trace, failure)
            if top is None:
                top = failure.stage
            by_id.setdefault(failure.taxonomy_id, {})
            by_id[failure.taxonomy_id][top] = by_id[failure.taxonomy_id].get(top, 0) + 1
        out: dict[str, dict[str, float]] = {}
        for tax_id, counts in sorted(by_id.items()):
            total = sum(counts.values())
            out[tax_id] = {stage: n / total for stage, n in sorted(counts.items(), key=lambda kv: -kv[1])}
        return out
