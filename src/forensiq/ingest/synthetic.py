"""Seeded synthetic RAG corpus with PLANTED failures — offline demo + fixtures.

Generates 28 days of traces for the fictional ``acme-support-rag`` pipeline.
Every failure is planted with known ground truth so the classifier, blame
ranker and drift detector can be scored honestly. The planted failure rates:

======================  =======  ==========
failure                 normal   post-drift
======================  =======  ==========
F-RET-001 empty retr.    12%       18%
F-RET-002 low recall      8%       28%
F-PROMPT-001 truncation   6%        6%
F-GEN-001 format break    5%        5%
F-GEN-002 repetition      4%        4%
F-GEN-003 refusal         3%        3%
F-INFRA-001 timeout       2%        2%
F-INFRA-002 cost spike    2%        2%
======================  =======  ==========

A slow drift degrades retrieval quality starting on day 15 (DRIFT_DAY): the
failure mix shifts toward retrieval failures and healthy hit scores sag —
exactly what the chi-square/CUSUM drift detector is built to catch while the
stable window (days 1-14) must stay quiet.

Content reuses RAG_showcase's fictional enterprise corpus (Nordwerk
Precision GmbH, XK-7, Helios 4.2, SAF-114...) so demo reports read like the
real sibling project.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from forensiq.ingest.schema import Span, Trace

PIPELINE_NAME = "acme-support-rag"
DRIFT_DAY = 15  # first degraded day (days 1..14 stable)
CONTEXT_LIMIT = 8192

ROOT_CAUSE_STAGE: dict[str, str] = {
    "F-RET-001": "retrieve",
    "F-RET-002": "retrieve",
    "F-PROMPT-001": "generate",
    "F-GEN-001": "generate",
    "F-GEN-002": "generate",
    "F-GEN-003": "generate",
    "F-INFRA-001": "generate",
    "F-INFRA-002": "generate",
}

# Ordered failure plan: (taxonomy_id, p_normal, p_drift)
FAILURE_PLAN: tuple[tuple[str, float, float], ...] = (
    ("F-RET-001", 0.12, 0.18),
    ("F-RET-002", 0.08, 0.28),
    ("F-PROMPT-001", 0.06, 0.06),
    ("F-GEN-001", 0.05, 0.05),
    ("F-GEN-002", 0.04, 0.04),
    ("F-GEN-003", 0.03, 0.03),
    ("F-INFRA-001", 0.02, 0.02),
    ("F-INFRA-002", 0.02, 0.02),
)


class PlantedFailure(BaseModel):
    """Ground-truth label for one synthetic trace."""

    trace_id: str
    taxonomy_id: str
    stage: str
    day: int


# --- RAG_showcase-flavored content -------------------------------------------------

QUERIES = (
    "Which supplier provides the component used in Product X?",
    "What fixed the rev-B jitter defect?",
    "What is the rate limit for endpoint API-009?",
    "How does the SAF-114 runbook connect to the X-207 postmortem?",
    "What late penalty does the supplier SLA define?",
    "Which policy governs security key rotation?",
)

CONTEXTS = (
    "Nordwerk Precision GmbH supplies component XK-7 used by Product X. "
    "The SLA applies a 5 percent late penalty per week and 8 hours maximum response.",
    "Helios firmware 4.2 fixed the rev-B jitter defect. The runbook links SAF-114 "
    "to RMA-77 within the X-207 incident postmortem.",
    "The api reference documents endpoint API-009 with a 100 per minute rate limit. "
    "Security policy SEC-301 requires a 90 day key rotation.",
)

OUTPUTS = (
    "Component XK-7 is supplied by Nordwerk Precision GmbH. The SLA applies a "
    "5 percent late penalty per week and 8 hours maximum response.",
    "Helios firmware 4.2 fixed the rev-B jitter defect. The runbook links SAF-114 "
    "to RMA-77 in the X-207 postmortem.",
    "Endpoint API-009 has a 100 per minute rate limit. Security policy SEC-301 "
    "requires a 90 day key rotation.",
)

FILLER_CONTEXTS = (
    "The batch ingest pipeline buffers kafka events nightly and rotates audit "
    "tags per retention policy.",
    "Quarterly reviews track vendor onboarding progress across regional offices "
    "and procurement tiers.",
)

HALLUCINATION_OUTPUT = (
    "The component is supplied by Vertex Dynamics under a 12 month contract "
    "with 24 hour support."
)

REFUSAL_OUTPUT = (
    "I cannot answer this question because I do not have the relevant "
    "information in the provided context."
)

REPEAT_SENTENCE = "The supplier is Nordwerk Precision GmbH. "

EMBED_MODEL = "bge-small-en-v1.5"
GEN_MODEL = "qwen3:1.7b"
RERANKER = "ms-marco-MiniLM"


class SyntheticTraceGenerator:
    """Deterministic (seeded) generator: same seed -> identical corpus."""

    def __init__(
        self,
        seed: int = 42,
        days: int = 28,
        traces_per_day: int = 20,
        pipeline_name: str = PIPELINE_NAME,
    ) -> None:
        if days < 3:
            raise ValueError("need at least 3 days of traces")
        self.seed = seed
        self.days = days
        self.traces_per_day = traces_per_day
        self.pipeline_name = pipeline_name
        # Two independent streams: the failure-mix choice must honor the
        # planted-rate contract exactly (it is what the classifier/drift tests
        # are scored against), while cosmetic build randomness (queries,
        # durations, templates) may consume draws freely. Sharing one stream
        # couples the mix to build-consumption patterns via the fixed MT19937
        # stream positions — with some seeds the mix drifts by multiple sigma.
        self._choice_rng = random.Random(seed)
        self.rng = random.Random(seed * 100003 + 7)
        self._base_ts = datetime(2026, 8, 1, 6, 0, 0, tzinfo=UTC)

    # -- public API ---------------------------------------------------------

    def generate(self) -> tuple[list[Trace], list[PlantedFailure]]:
        """Return (traces, planted ground truth)."""
        traces: list[Trace] = []
        failures: list[PlantedFailure] = []
        n = self.traces_per_day
        for day in range(1, self.days + 1):
            for idx in range(n):
                ts = self._base_ts + timedelta(days=day - 1, minutes=17 * idx, seconds=day * 3)
                trace_id = f"tr-{day:02d}-{idx:03d}"
                tax_id = self._choose_failure(day)
                trace = self._build_trace(trace_id, ts, tax_id)
                traces.append(trace)
                if tax_id is not None:
                    failures.append(
                        PlantedFailure(
                            trace_id=trace_id,
                            taxonomy_id=tax_id,
                            stage=ROOT_CAUSE_STAGE[tax_id],
                            day=day,
                        )
                    )
        return traces, failures

    def write_jsonl(self, path: str | Path, traces: list[Trace] | None = None) -> Path:
        """Export traces only (no ground truth) as JSONL for CLI/demo flows."""
        if traces is None:
            traces, _ = self.generate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for t in traces:
                fh.write(t.model_dump_json() + "\n")
        return path

    # -- internals ----------------------------------------------------------

    def _choose_failure(self, day: int) -> str | None:
        drift = day >= DRIFT_DAY
        r = self._choice_rng.random()
        acc = 0.0
        for tax_id, p_normal, p_drift in FAILURE_PLAN:
            acc += p_drift if drift else p_normal
            if r < acc:
                return tax_id
        return None

    def _build_trace(self, trace_id: str, ts: datetime, tax_id: str | None) -> Trace:
        rng = self.rng
        spans: list[Span] = []
        query = rng.choice(QUERIES)

        # embed
        spans.append(
            Span(
                span_id=f"{trace_id}-embed",
                name="embed",
                stage="embed",
                duration_ms=round(rng.uniform(6.0, 18.0), 2),
                attrs={"model": EMBED_MODEL, "token_count": rng.randint(20, 60)},
            )
        )

        # retrieve
        hit_scores: list[float] = []
        top_k = 5
        if tax_id == "F-RET-001":
            hit_count = 0
        elif tax_id == "F-RET-002":
            hit_count = top_k
            hit_scores = sorted((round(rng.uniform(0.30, 0.46), 3) for _ in range(top_k)), reverse=True)
        else:
            hit_count = top_k
            lo, hi = 0.68, 0.94
            if ts.day >= DRIFT_DAY:
                lo, hi = lo - 0.06, hi - 0.06  # slow degradation of healthy scores
            hit_scores = sorted((round(rng.uniform(lo, hi), 3) for _ in range(top_k)), reverse=True)
        retrieve_span = Span(
            span_id=f"{trace_id}-retrieve",
            parent_id=f"{trace_id}-embed",
            name="hybrid_retrieve",
            stage="retrieve",
            duration_ms=round(rng.uniform(25.0, 110.0), 2),
            attrs={
                "query": query,
                "top_k": top_k,
                "hit_count": hit_count,
                "hit_scores": hit_scores,
                "index": "04_vectors",
            },
        )
        spans.append(retrieve_span)

        # rerank (skipped when retrieval came back empty)
        mean_score = sum(hit_scores) / len(hit_scores) if hit_scores else 0.0
        if hit_count > 0 and rng.random() < 0.85:
            delta = rng.uniform(0.01, 0.05) if tax_id == "F-RET-002" else rng.uniform(0.04, 0.22)
            spans.append(
                Span(
                    span_id=f"{trace_id}-rerank",
                    parent_id=retrieve_span.span_id,
                    name="rerank",
                    stage="rerank",
                    duration_ms=round(rng.uniform(12.0, 45.0), 2),
                    attrs={
                        "reranker": RERANKER,
                        "rerank_top1": round(min(0.99, mean_score + delta), 3),
                        "rerank_delta": round(delta, 3),
                    },
                )
            )

        # graph hop (occasionally, mirroring the ReAct/graph path)
        if rng.random() < 0.15:
            spans.append(
                Span(
                    span_id=f"{trace_id}-graph",
                    parent_id=retrieve_span.span_id,
                    name="graph_merge",
                    stage="graph",
                    duration_ms=round(rng.uniform(30.0, 90.0), 2),
                    attrs={"entities": rng.randint(2, 6), "hops": rng.randint(1, 2)},
                )
            )

        # generate
        gen_span = self._generate_span(trace_id, retrieve_span, tax_id)
        spans.append(gen_span)

        # guard
        if rng.random() < 0.5:
            spans.append(
                Span(
                    span_id=f"{trace_id}-guard",
                    parent_id=gen_span.span_id,
                    name="guardrails",
                    stage="guard",
                    duration_ms=round(rng.uniform(2.0, 8.0), 2),
                    attrs={"pii_redactions": rng.randint(0, 2), "blocked": False},
                )
            )

        # tool call
        if rng.random() < 0.25:
            spans.append(
                Span(
                    span_id=f"{trace_id}-tool",
                    parent_id=gen_span.span_id,
                    name="crm_lookup",
                    stage="tool",
                    duration_ms=round(rng.uniform(40.0, 300.0), 2),
                    attrs={"tool": "crm_lookup"},
                )
            )

        total_cost = sum(s.cost for s in spans)
        return Trace(
            trace_id=trace_id,
            ts=ts,
            pipeline_name=self.pipeline_name,
            spans=spans,
            total_cost=round(total_cost, 6),
            meta={"env": "demo", "user": f"u{self.rng.randint(1, 40)}"},
        )

    def _generate_span(self, trace_id: str, retrieve_span: Span, tax_id: str | None) -> Span:
        rng = self.rng
        idx = rng.randrange(len(CONTEXTS))
        output = OUTPUTS[idx]
        context = CONTEXTS[idx]
        prompt_chars = rng.randint(2800, 5200)
        tokens_in = prompt_chars // 4 + rng.randint(-40, 40)
        tokens_out = rng.randint(150, 380)
        duration = rng.uniform(500.0, 2400.0)
        finish_reason = "stop"
        cost = rng.uniform(0.0015, 0.0045)
        status: str = "ok"
        attrs: dict[str, Any] = {
            "model": GEN_MODEL,
            "prompt_chars": prompt_chars,
            "finish_reason": finish_reason,
            "context": context,
        }

        if tax_id == "F-RET-001":
            # Nothing retrieved -> the model improvises.
            attrs["context"] = ""
            output = HALLUCINATION_OUTPUT
            attrs["context_scores"] = []
        elif tax_id == "F-RET-002":
            # Weak context: unrelated filler sneaks into the prompt.
            attrs["context"] = rng.choice(FILLER_CONTEXTS)
        elif tax_id == "F-PROMPT-001":
            attrs["prompt_chars"] = rng.randint(CONTEXT_LIMIT - 40, CONTEXT_LIMIT - 1)
            attrs["finish_reason"] = "length"
            tokens_in = rng.randint(1900, 2100)
            output = OUTPUTS[idx][: int(len(OUTPUTS[idx]) * 0.6)]
            tokens_out = rng.randint(60, 110)
        elif tax_id == "F-GEN-001":
            attrs["output_format"] = "json"
            output = '{"answer": "Component XK-7 is supplied by Nordwerk'
            tokens_out = rng.randint(30, 60)
            attrs["finish_reason"] = "length"
        elif tax_id == "F-GEN-002":
            output = REPEAT_SENTENCE * 14
            tokens_out = rng.randint(500, 900)
        elif tax_id == "F-GEN-003":
            output = REFUSAL_OUTPUT
            tokens_out = rng.randint(25, 40)
        elif tax_id == "F-INFRA-001":
            status = "timeout"
            duration = rng.uniform(29000.0, 42000.0)
            output = ""
            tokens_out = 0
        elif tax_id == "F-INFRA-002":
            tokens_in = rng.randint(90000, 150000)
            attrs["prompt_chars"] = tokens_in * 4
            duration = rng.uniform(4000.0, 9000.0)
            cost = rng.uniform(0.6, 1.4)

        attrs["output"] = output
        return Span(
            span_id=f"{trace_id}-generate",
            parent_id=retrieve_span.span_id,
            name="generate",
            stage="generate",
            duration_ms=round(duration, 2),
            status=status,  # type: ignore[arg-type]
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=round(cost, 6),
            attrs=attrs,
        )
