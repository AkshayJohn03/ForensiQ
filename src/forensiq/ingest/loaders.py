"""Trace loaders: JSONL, JSON, Langfuse REST, and the span-dict adapter.

Accepted shapes, in order of specificity (see record_to_trace):

1. Full trace dicts: ``{"trace_id", "ts", "pipeline_name", "spans": [...],
   "total_cost", "meta"}`` — the native ForensiQ export.
2. RAG_showcase-style eval records: ``{"query", "retrieved": [...],
   "answer", "metrics": {...}}`` — converted into a two-span trace
   (retrieve + generate) with metric values preserved in ``meta``.
3. Bare span dicts (the hook format emitted by portfolio systems such as
   AegisGate / SwarmResearch): ``{"span_id", "parent_id", "name", "stage",
   "duration_ms", "status", "attrs"}`` — each becomes a single-span trace
   unless a ``trace_id`` groups them.

Stage names are normalized through :func:`forensiq.ingest.schema.normalize_stage`,
so RAG_showcase's ``hybrid`` stage maps to the canonical ``retrieve``.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from forensiq.ingest.schema import Span, Trace, normalize_stage


class IngestError(ValueError):
    """Raised when a source cannot be parsed at all (not per-record glitches)."""


def _parse_ts(raw: Any) -> datetime:
    """Accept ISO strings ('Z' ok), epoch seconds or epoch milliseconds."""
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=UTC)
    if isinstance(raw, (int, float)):
        # Heuristic: > 1e11 is ms-epoch.
        secs = raw / 1000.0 if raw > 1e11 else float(raw)
        return datetime.fromtimestamp(secs, tz=UTC)
    if isinstance(raw, str):
        text = raw.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError as exc:
            raise IngestError(f"unparseable timestamp {raw!r}") from exc
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    raise IngestError(f"unsupported timestamp {raw!r}")


def spans_from_dicts(dicts: Iterable[dict[str, Any]]) -> list[Span]:
    """Adapter for the span dicts emitted by portfolio systems
    (AegisGate / SwarmResearch hooks): ``{"span_id", "parent_id", "name",
    "stage", "duration_ms", "status", "attrs"}`` plus optional
    ``tokens_in``/``tokens_out``/``cost``.

    Missing span_ids are synthesized; stage aliases are normalized; unknown
    stages degrade to ``tool`` instead of failing ingestion.
    """
    spans: list[Span] = []
    for i, raw in enumerate(dicts):
        if not isinstance(raw, dict):
            raise IngestError(f"span #{i} is not a dict: {raw!r}")
        name = str(raw.get("name", raw.get("stage", "span")))
        stage = normalize_stage(str(raw.get("stage", name)))
        span_id = str(raw.get("span_id", raw.get("id") or f"{name}-{i}"))
        status = str(raw.get("status", "ok")).lower()
        if status not in ("ok", "error", "timeout"):
            status = "error" if status else "ok"
        spans.append(
            Span(
                span_id=span_id,
                parent_id=raw.get("parent_id") or raw.get("parentObservationId"),
                name=name,
                stage=stage,
                duration_ms=float(raw.get("duration_ms", raw.get("durationMs", 0.0)) or 0.0),
                status=status,  # type: ignore[arg-type]
                tokens_in=int(raw.get("tokens_in", raw.get("tokensIn", 0)) or 0),
                tokens_out=int(raw.get("tokens_out", raw.get("tokensOut", 0)) or 0),
                cost=float(raw.get("cost", 0.0) or 0.0),
                attrs=dict(raw.get("attrs", {}) or {}),
            )
        )
    return spans


def _eval_record_to_trace(rec: dict[str, Any], fallback_id: str) -> Trace:
    """Convert a RAG_showcase-style eval record into a retrieve+generate trace.

    Accepted keys (all optional except one of query/answer):
      query, retrieved (list of scores or dicts with "score"/"chunk_id"),
      hits (alias), top_k, answer/output, context, metrics
      (faithfulness / context_precision / context_recall / answer_relevance / ...).
    """
    query = str(rec.get("query", rec.get("question", "")))
    retrieved = rec.get("retrieved", rec.get("hits", []))
    scores: list[float] = []
    for item in retrieved or []:
        if isinstance(item, dict):
            if "score" in item:
                scores.append(float(item["score"]))
        else:
            scores.append(float(item))
    top_k = int(rec.get("top_k", len(scores)) or len(scores))
    answer = str(rec.get("answer", rec.get("output", "")))
    metrics = dict(rec.get("metrics", {}) or {})
    ts = _parse_ts(rec.get("ts", rec.get("timestamp", 0)) or 0)
    return Trace(
        trace_id=str(rec.get("trace_id", rec.get("id") or fallback_id)),
        ts=ts,
        pipeline_name=str(rec.get("pipeline_name", "rag_showcase")),
        spans=[
            Span(
                span_id=f"{rec.get('trace_id', fallback_id)}-retrieve",
                name="hybrid_retrieve",
                stage="retrieve",
                duration_ms=float(rec.get("retrieval_ms", 0.0) or 0.0),
                attrs={
                    "query": query,
                    "top_k": top_k,
                    "hit_count": len(scores),
                    "hit_scores": scores,
                },
            ),
            Span(
                span_id=f"{rec.get('trace_id', fallback_id)}-generate",
                parent_id=f"{rec.get('trace_id', fallback_id)}-retrieve",
                name="generate",
                stage="generate",
                duration_ms=float(rec.get("generation_ms", 0.0) or 0.0),
                attrs={
                    "output": answer,
                    "context": str(rec.get("context", "")),
                    "prompt_chars": int(rec.get("prompt_chars", len(answer) * 4) or 0),
                    **metrics,
                },
            ),
        ],
        meta={"source": "eval_record", "metrics": metrics},
    )


def record_to_trace(rec: dict[str, Any], fallback_id: str) -> Trace:
    """Convert one generic record into a Trace (see module docstring)."""
    if "spans" in rec:
        spans = spans_from_dicts(rec["spans"])
        ts = _parse_ts(rec.get("ts", rec.get("timestamp", 0)) or 0)
        trace = Trace(
            trace_id=str(rec.get("trace_id", rec.get("id") or fallback_id)),
            ts=ts,
            pipeline_name=str(rec.get("pipeline_name", rec.get("name", "unknown"))),
            spans=spans,
            total_cost=float(rec.get("total_cost", 0.0) or 0.0),
            meta=dict(rec.get("meta", {}) or {}),
        )
        if trace.total_cost == 0.0:
            trace.recompute_cost()
        return trace
    if "query" in rec or "question" in rec:
        return _eval_record_to_trace(rec, fallback_id)
    # Bare span dict → single-span trace.
    span = spans_from_dicts([rec])[0]
    return Trace(
        trace_id=str(rec.get("trace_id", fallback_id)),
        ts=_parse_ts(rec.get("ts", 0) or 0),
        pipeline_name=str(rec.get("pipeline_name", "unknown")),
        spans=[span],
        total_cost=float(span.cost),
    )


def records_to_traces(records: Iterable[dict[str, Any]]) -> list[Trace]:
    return [record_to_trace(r, f"trace-{i}") for i, r in enumerate(records)]


class JSONLTraceLoader:
    """One JSON object per line; per-line failures degrade to warnings."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def load(self, source: str | Path) -> list[Trace]:
        text = Path(source).read_text(encoding="utf-8")
        return self.load_text(text)

    def load_text(self, text: str) -> list[Trace]:
        traces: list[Trace] = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                traces.append(record_to_trace(rec, f"trace-l{lineno}"))
            except (IngestError, ValidationError, KeyError, TypeError, ValueError) as exc:
                # Defensive on purpose: one bad line must not kill the batch.
                self.warnings.append(f"line {lineno}: {exc}")
        if not traces and self.warnings:
            raise IngestError(f"no valid traces; {len(self.warnings)} parse failures")
        return traces


class JSONTraceLoader:
    """A JSON file holding a single trace, a list of traces, or mixed records."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def load(self, source: str | Path) -> list[Trace]:
        text = Path(source).read_text(encoding="utf-8")
        return self.load_text(text)

    def load_text(self, text: str) -> list[Trace]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise IngestError(f"invalid JSON document: {exc}") from exc
        records: list[Any] = data if isinstance(data, list) else [data]
        traces: list[Trace] = []
        for i, rec in enumerate(records):
            if not isinstance(rec, dict):
                self.warnings.append(f"record #{i} is not an object")
                continue
            try:
                traces.append(record_to_trace(rec, f"trace-{i}"))
            except (IngestError, ValidationError, KeyError, TypeError, ValueError) as exc:
                self.warnings.append(f"record #{i}: {exc}")
        if not traces:
            raise IngestError(f"no valid traces; {len(self.warnings)} parse failures")
        return traces


class LangfuseTraceLoader:
    """Pull traces + observations from Langfuse's public REST API via httpx.

    Env-guarded: the loader refuses to construct without credentials so a
    missing key can never turn into a surprise network call or a silent
    empty result. Tests exercise :meth:`from_pages` against recorded fixture
    JSON — the network path is a thin, untested-by-design wrapper.
    """

    def __init__(
        self,
        base_url: str | None = None,
        public_key: str | None = None,
        secret_key: str | None = None,
        timeout: float = 15.0,
    ) -> None:
        base_url = base_url or os.environ.get("LANGFUSE_HOST", "")
        public_key = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY", "")
        secret_key = secret_key or os.environ.get("LANGFUSE_SECRET_KEY", "")
        if not (base_url and public_key and secret_key):
            raise IngestError(
                "Langfuse ingestion is env-guarded: set LANGFUSE_HOST, "
                "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY (see .env.example)."
            )
        self.base_url = base_url.rstrip("/")
        self.auth = (public_key, secret_key)
        self.timeout = timeout

    @classmethod
    def from_pages(
        cls, traces_page: dict[str, Any] | list[Any], observations_page: dict[str, Any] | list[Any]
    ) -> list[Trace]:
        """Map recorded Langfuse API pages onto ForensiQ traces.

        traces_page:      GET /api/public/traces      → {"data": [...]}
        observations_page: GET /api/public/observations → {"data": [...]}
        """
        traces_raw = traces_page.get("data", []) if isinstance(traces_page, dict) else traces_page
        obs_raw = (
            observations_page.get("data", []) if isinstance(observations_page, dict) else observations_page
        )
        by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for obs in obs_raw:
            tid = obs.get("traceId")
            if tid:
                by_trace[str(tid)].append(obs)

        out: list[Trace] = []
        for tr in traces_raw:
            tid = str(tr.get("id", ""))
            spans = spans_from_dicts([_observation_to_span_dict(o) for o in by_trace.get(tid, [])])
            total_cost = sum(s.cost for s in spans)
            ts_raw = tr.get("timestamp")
            ts = _parse_ts(ts_raw) if ts_raw else datetime.now(UTC)
            out.append(
                Trace(
                    trace_id=tid or f"langfuse-{len(out)}",
                    ts=ts,
                    pipeline_name=str(tr.get("name", "langfuse")),
                    spans=spans,
                    total_cost=total_cost,
                    meta={
                        "source": "langfuse",
                        "release": tr.get("release"),
                        "tags": tr.get("tags") or [],
                    },
                )
            )
        return out

    def fetch(self, limit: int = 100) -> list[Trace]:
        """Live REST pull. Only reachable when credentials were provided."""
        traces_page = self._get("/api/public/traces", limit)
        observations_page = self._get("/api/public/observations", limit * 10)
        return self.from_pages(traces_page, observations_page)

    def _get(self, path: str, limit: int) -> dict[str, Any]:
        # Network path: intentionally minimal; guarded by __init__ credential check.
        resp = httpx.get(
            f"{self.base_url}{path}",
            params={"limit": limit},
            auth=self.auth,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return dict(resp.json())


def _observation_to_span_dict(obs: dict[str, Any]) -> dict[str, Any]:
    """Map one Langfuse observation onto the portfolio span-dict hook shape."""
    level = str(obs.get("level", "DEFAULT")).upper()
    name = str(obs.get("name", obs.get("type", "span")))
    status = "ok"
    if "timeout" in name.lower():
        status = "timeout"
    elif level in ("ERROR",):
        status = "error"
    usage = obs.get("usage") or {}
    start = obs.get("startTime")
    end = obs.get("endTime")
    duration_ms = 0.0
    if start and end:
        try:
            t0 = _parse_ts(start)
            t1 = _parse_ts(end)
            duration_ms = max(0.0, (t1 - t0).total_seconds() * 1000.0)
        except IngestError:
            duration_ms = 0.0
    cost = obs.get("calculatedTotalCost", obs.get("cost"))
    metadata = obs.get("metadata") or {}
    return {
        "span_id": obs.get("id"),
        "parent_id": obs.get("parentObservationId"),
        "name": name,
        "stage": name,
        "duration_ms": duration_ms,
        "status": status,
        "tokens_in": usage.get("input", 0) or 0,
        "tokens_out": usage.get("output", 0) or 0,
        "cost": float(cost) if cost is not None else 0.0,
        "attrs": dict(metadata) if isinstance(metadata, dict) else {},
    }
