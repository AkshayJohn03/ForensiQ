"""Session replay — deterministic step-through of one recorded trace.

The flight-recorder reading of a trace: print each span in execution order
with its stage, duration, status and the *state delta* it introduced (which
attrs it added or changed vs. everything before it). ``--at Tn`` aggregates
the cumulative state up to and including step Tn — "what did the pipeline
know when step 4 ran?" — which is the question an incident responder asks
when deciding where a bad decision entered.

Deterministic by construction: spans render in stored (ingest) order, keys
render sorted, no wall-clock values — same trace, byte-identical replay.
This is the *descriptive* replay; the *counterfactual* replay (what-if with
mutated parameters) lives in :mod:`forensiq.attribution.replay`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from forensiq.ingest.schema import Trace

__all__ = ["ReplayStep", "render_replay", "replay_steps", "replay_to_json", "state_at"]


@dataclass
class ReplayStep:
    """One step (Tn, 1-based) of the session replay."""

    index: int  # 1-based step number (T1, T2, ...)
    span_id: str
    name: str
    stage: str
    duration_ms: float
    status: str
    delta: dict[str, Any] = field(default_factory=dict)  # attrs added/changed by this span
    cumulative_tokens_in: int = 0
    cumulative_tokens_out: int = 0
    cumulative_cost: float = 0.0

    def line(self) -> str:
        """One deterministic timeline line."""
        delta = ", ".join(f"{k}={self.delta[k]}" for k in sorted(self.delta))
        if len(delta) > 72:
            shown = delta[:69] + "..."
            count = len(self.delta)
            line = f"T{self.index:<3} {self.stage:<9} {self.status:<7} {self.duration_ms:>8.1f}ms  {shown}"
            return f"{line}  (+{count} attrs)"
        suffix = f"  {delta}" if delta else ""
        return f"T{self.index:<3} {self.stage:<9} {self.status:<7} {self.duration_ms:>8.1f}ms{suffix}"


def replay_steps(trace: Trace) -> list[ReplayStep]:
    """Build the step-through: spans in order, each with its state delta."""
    steps: list[ReplayStep] = []
    seen: dict[str, Any] = {}
    tokens_in = tokens_out = 0
    cost = 0.0
    for i, span in enumerate(trace.spans, start=1):
        delta = {k: v for k, v in span.attrs.items() if k not in seen or seen[k] != v}
        tokens_in += span.tokens_in
        tokens_out += span.tokens_out
        cost += span.cost
        steps.append(
            ReplayStep(
                index=i,
                span_id=span.span_id,
                name=span.name,
                stage=span.stage,
                duration_ms=span.duration_ms,
                status=span.status,
                delta=delta,
                cumulative_tokens_in=tokens_in,
                cumulative_tokens_out=tokens_out,
                cumulative_cost=cost,
            )
        )
        seen.update(span.attrs)
    return steps


def state_at(trace: Trace, at: int) -> dict[str, Any]:
    """Aggregated pipeline state up to and including step ``at`` (1-based).

    Merges the attrs of spans[0..at-1] (later spans win on conflicts) plus
    cumulative counters. ``at`` is clamped to [0, len(spans)].
    """
    spans = trace.spans
    at = max(0, min(at, len(spans)))
    state: dict[str, Any] = {}
    for span in spans[:at]:
        state.update(span.attrs)
    state["_meta"] = {
        "step": at,
        "spans_seen": at,
        "stages_seen": sorted({s.stage for s in spans[:at]}),
        "tokens_in": sum(s.tokens_in for s in spans[:at]),
        "tokens_out": sum(s.tokens_out for s in spans[:at]),
        "cost": round(sum(s.cost for s in spans[:at]), 6),
    }
    return state


def _state_block(state: dict[str, Any]) -> list[str]:
    lines = [f"  {k} = {state[k]}" for k in sorted(state) if k != "_meta"]
    meta = state.get("_meta", {})
    if meta:
        lines.append(
            "  counters: spans_seen={spans_seen} tokens_in={tokens_in} tokens_out={tokens_out} cost={cost:.4f}".format(
                **meta
            )
        )
    return lines


def render_replay(trace: Trace, at: int | None = None) -> str:
    """Render the step-through timeline (plus cumulative state when ``at``)."""
    steps = replay_steps(trace)
    lines = [
        f"Session replay: {trace.trace_id} ({trace.pipeline_name}, {trace.ts.isoformat()}, {len(steps)} spans)",
    ]
    for step in steps:
        lines.append("  " + step.line())
    if steps:
        last = steps[-1]
        totals = (
            f"  totals: tokens_in={last.cumulative_tokens_in} tokens_out={last.cumulative_tokens_out} "
            f"cost={last.cumulative_cost:.4f}"
        )
        lines.append(totals)
    else:
        lines.append("  (trace has no spans)")
    if at is not None:
        clamped = max(0, min(at, len(steps)))
        lines.append("")
        lines.append(f"state at T{clamped} (cumulative through step {clamped}):")
        lines.extend(_state_block(state_at(trace, clamped)))
    return "\n".join(lines)


def replay_to_json(trace: Trace, at: int | None = None) -> str:
    """JSON form of the replay (steps + optional cumulative state)."""
    steps = replay_steps(trace)
    payload: dict[str, Any] = {
        "trace_id": trace.trace_id,
        "pipeline_name": trace.pipeline_name,
        "ts": trace.ts.isoformat(),
        "steps": [
            {
                "step": s.index,
                "span_id": s.span_id,
                "name": s.name,
                "stage": s.stage,
                "duration_ms": s.duration_ms,
                "status": s.status,
                "delta": s.delta,
                "cumulative": {
                    "tokens_in": s.cumulative_tokens_in,
                    "tokens_out": s.cumulative_tokens_out,
                    "cost": round(s.cumulative_cost, 6),
                },
            }
            for s in steps
        ],
    }
    if at is not None:
        clamped = max(0, min(at, len(steps)))
        payload["state_at"] = state_at(trace, clamped)
    return json.dumps(payload, indent=2, default=str, sort_keys=True)
