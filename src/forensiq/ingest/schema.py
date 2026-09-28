"""Normalized trace/span model — the lingua franca of every ForensiQ stage.

The span-stage vocabulary is aligned with the author's RAG_showcase pipeline,
which emits per-stage spans ``embed -> hybrid -> rerank -> graph -> generate``
(plus guardrails). ForensiQ normalizes ``hybrid``/``retrieval``/``search`` to
the canonical ``retrieve`` stage so RAG_showcase exports ingest unchanged, and
reuses its eval metric names (faithfulness, context precision/recall, answer
relevance) in :mod:`forensiq.attribution.health`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Stage = Literal["embed", "retrieve", "rerank", "graph", "generate", "guard", "tool", "agent"]
SpanStatus = Literal["ok", "error", "timeout"]

CANONICAL_STAGES: tuple[Stage, ...] = (
    "embed",
    "retrieve",
    "rerank",
    "graph",
    "generate",
    "guard",
    "tool",
    "agent",
)

# RAG_showcase emits "hybrid" for the fusion retrieval stage; other common
# spellings are normalized here so exports ingest out of the box.
STAGE_ALIASES: dict[str, Stage] = {
    "embed": "embed",
    "embedding": "embed",
    "embeddings": "embed",
    "retrieve": "retrieve",
    "retrieval": "retrieve",
    "hybrid": "retrieve",
    "hybrid_retrieve": "retrieve",
    "search": "retrieve",
    "vector_search": "retrieve",
    "rerank": "rerank",
    "reranker": "rerank",
    "cross_encoder": "rerank",
    "graph": "graph",
    "graphrag": "graph",
    "kg": "graph",
    "generate": "generate",
    "generation": "generate",
    "llm": "generate",
    "answer": "generate",
    "guard": "guard",
    "guardrail": "guard",
    "guardrails": "guard",
    "tool": "tool",
    "tool_call": "tool",
    "agent": "agent",
    "react": "agent",
}

# Pipeline order used for rendering blame graphs and timelines.
STAGE_ORDER: dict[Stage, int] = {s: i for i, s in enumerate(CANONICAL_STAGES)}


def normalize_stage(raw: str) -> Stage:
    """Map a raw stage/span name onto the canonical vocabulary.

    Unknown names degrade to ``tool`` (the generic operational span) and keep
    their original name in ``Span.name`` — ingestion never hard-fails on a new
    stage spelling, it just loses stage-specific analytics until the alias is
    registered here.
    """
    key = raw.strip().lower().replace("-", "_").replace(" ", "_")
    stage = STAGE_ALIASES.get(key)
    if stage is not None:
        return stage
    for alias, mapped in STAGE_ALIASES.items():
        if alias in key:
            return mapped
    return "tool"


class EvidenceItem(BaseModel):
    """One quoted piece of evidence: which span, which attr, what value."""

    model_config = ConfigDict(extra="ignore")

    span_id: str
    stage: str
    key: str
    value: str
    note: str | None = None


class Span(BaseModel):
    """One pipeline stage execution inside a trace.

    ``attrs`` carries the stage evidence ForensiQ reasons over:
      * embed:    model, token_count
      * retrieve: query, top_k, hit_count, hit_scores (list[float])
      * rerank:   reranker, rerank_top1, rerank_delta
      * graph:    entities, hops
      * generate: model, prompt_chars, finish_reason, output, context, output_format
      * guard:    pii_redactions, blocked
      * tool:     tool, error
    """

    model_config = ConfigDict(extra="ignore")

    span_id: str
    parent_id: str | None = None
    name: str
    stage: Stage
    duration_ms: float = 0.0
    status: SpanStatus = "ok"
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    attrs: dict[str, Any] = Field(default_factory=dict)

    def attr(self, key: str, default: Any = None) -> Any:
        return self.attrs.get(key, default)


class Trace(BaseModel):
    """A normalized pipeline trace."""

    model_config = ConfigDict(extra="ignore")

    trace_id: str
    ts: datetime
    pipeline_name: str = "unknown"
    spans: list[Span] = Field(default_factory=list)
    total_cost: float = 0.0
    meta: dict[str, Any] = Field(default_factory=dict)

    def stage_span(self, stage: Stage) -> Span | None:
        """First span of the given stage (pipelines usually run one per trace)."""
        for span in self.spans:
            if span.stage == stage:
                return span
        return None

    def stage_spans(self, stage: Stage) -> list[Span]:
        return [s for s in self.spans if s.stage == stage]

    @property
    def day(self) -> str:
        """ISO date bucket used by drift detection."""
        return self.ts.date().isoformat()

    def recompute_cost(self) -> float:
        """Sum of span costs (used when exporters omit trace-level cost)."""
        self.total_cost = sum(s.cost for s in self.spans)
        return self.total_cost
