"""OpenTelemetry OTLP/JSON ingestion — accept any OTel-instrumented app.

Parses the OTLP/JSON ``ExportTraceServiceRequest`` shape produced by the OTel
HTTP exporter (the body POSTed to ``/v1/traces`` on any collector):

    resourceSpans[] -> resource.attributes (KV list)
                    -> scopeSpans[] -> scope + spans[]
    span: traceId / spanId / parentSpanId (hex), name,
          startTimeUnixNano / endTimeUnixNano / durationNanos (strings of int64),
          attributes (KV list with typed oneof values), status{code,message}

Everything is normalized onto the ForensiQ ``Trace``/``Span`` model:

* ids        — hex is canonicalized (lowercase, ``0x`` stripped); protobuf-JSON
               base64-encoded ids are decoded; empty/all-zero parent -> root.
* timestamps — unix nanos are converted to tz-aware datetimes without float
               precision loss (divmod on nanoseconds); ``durationNanos`` is
               honored when ``endTimeUnixNano`` is absent.
* stages     — span name + attributes are mapped onto the canonical vocabulary
               (embed/retrieve/rerank/graph/generate/guard/tool/agent) via
               OpenInference ``span.kind``, ``gen_ai.*`` hints and name rules;
               anything unknown degrades to ``tool`` (never a parse failure).
* attrs      — both attribute shapes (KV lists and plain maps) are accepted;
               well-known ``gen_ai.*`` / OpenInference keys are overlaid with
               the canonical ForensiQ keys (model, tokens_in/out, output,
               finish_reason, top_k, hit_count, hit_scores, query, context).

Tolerant by contract: a malformed span/scope/resource is skipped and counted
in an :class:`OTLPSkipReport`; parsing never raises on partial data. Only a
document that is not JSON at all raises :class:`IngestError`.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from forensiq.ingest.loaders import IngestError
from forensiq.ingest.schema import Span, Stage, Trace, normalize_stage

__all__ = [
    "MAX_SKIP_REASONS",
    "OTLPParseResult",
    "OTLPSkipReport",
    "infer_stage",
    "parse_otlp",
    "parse_otlp_json",
]

# Skip reasons are capped so a poisoned 1M-span export cannot balloon memory.
MAX_SKIP_REASONS = 200

_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")  # case-insensitive: uppercase hex is valid base64 too


# -- skip report -------------------------------------------------------------


class OTLPSkipReport(BaseModel):
    """What was dropped and why (bounded list, counts always exact)."""

    skipped_spans: int = 0
    skipped_scopes: int = 0
    skipped_resources: int = 0
    reasons: list[str] = Field(default_factory=list)

    def note(self, reason: str) -> None:
        if len(self.reasons) < MAX_SKIP_REASONS:
            self.reasons.append(reason)

    @property
    def total_skipped(self) -> int:
        return self.skipped_spans + self.skipped_scopes + self.skipped_resources


class OTLPParseResult(BaseModel):
    """Parsed traces plus the skip report for everything dropped."""

    traces: list[Trace] = Field(default_factory=list)
    skip: OTLPSkipReport = Field(default_factory=OTLPSkipReport)


# -- stage mapping -----------------------------------------------------------

# OpenInference span.kind (Traceloop/Arize/Langfuse OTel exporters) -> stage.
OPENINFERENCE_KINDS: dict[str, Stage] = {
    "LLM": "generate",
    "RETRIEVER": "retrieve",
    "EMBEDDING": "embed",
    "RERANKER": "rerank",
    "TOOL": "tool",
    "AGENT": "agent",
    "GUARDRAIL": "guard",
    "CHAIN": "agent",
}

# Substring rules over the lower-cased span name, first hit wins. Vector DB
# client names (qdrant/chroma/...) land on retrieve; LLM client calls land on
# generate; anything unrecognized falls through to the alias table and then
# degrades to `tool`.
NAME_STAGE_PATTERNS: tuple[tuple[tuple[str, ...], Stage], ...] = (
    (("embed", "embedding"), "embed"),
    (
        ("retriev", "search", "vector", "qdrant", "chroma", "milvus", "pgvector", "pinecone", "weaviate", "hybrid"),
        "retrieve",
    ),
    (("rerank",), "rerank"),
    (("graph", "knowledge", "kg_"), "graph"),
    (("generate", "generation", "llm", "completion", "chat", "openai", "anthropic", "gpt"), "generate"),
    (("guard", "moderation", "safety"), "guard"),
    (("agent", "planner", "react"), "agent"),
    (("tool", "function", "execute"), "tool"),
)

# gen_ai.* keys whose presence marks an LLM call even when the span name is
# opaque (e.g. "invoke_model").
GENAI_GENERATE_KEYS: frozenset[str] = frozenset(
    {
        "gen_ai.prompt",
        "gen_ai.completion",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "gen_ai.response.finish_reasons",
    }
)


def infer_stage(name: str, attrs: dict[str, Any]) -> Stage:
    """Map an OTel span name + attributes onto the canonical stage vocabulary.

    Order: explicit ForensiQ hint > OpenInference span.kind > gen_ai.*/retrieval
    attribute hints > name patterns > exact alias table > `tool`.
    """
    for key in ("forensiq.stage", "stage"):
        if key in attrs:
            return normalize_stage(str(attrs[key]))
    for key in ("openinference.span.kind", "openinference_span_kind"):
        kind = attrs.get(key)
        if kind is not None:
            mapped = OPENINFERENCE_KINDS.get(str(kind).upper())
            if mapped is not None:
                return mapped
    keys = {str(k).lower() for k in attrs}
    if keys & GENAI_GENERATE_KEYS:
        return "generate"
    retrieval_keys = {"retrieval.documents", "retrieval_documents", "match.count", "db.system"}
    if keys & retrieval_keys:
        return "retrieve"
    low = name.strip().lower()
    for needles, stage in NAME_STAGE_PATTERNS:
        if any(needle in low for needle in needles):
            return stage
    return normalize_stage(name)  # exact alias table; falls back to "tool"


# -- attribute flattening ----------------------------------------------------

_TOKENS_IN_KEYS: tuple[str, ...] = (
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.prompt_tokens",
    "llm.token_count.prompt",
    "usage.input_tokens",
    "prompt_tokens",
)
_TOKENS_OUT_KEYS: tuple[str, ...] = (
    "gen_ai.usage.output_tokens",
    "gen_ai.usage.completion_tokens",
    "llm.token_count.completion",
    "usage.output_tokens",
    "completion_tokens",
)
_COST_KEYS: tuple[str, ...] = ("gen_ai.usage.cost", "llm.cost.total", "openinference.span.cost", "cost")
_MODEL_KEYS: tuple[str, ...] = (
    "gen_ai.request.model",
    "gen_ai.response.model",
    "llm.model_name",
    "model_name",
    "model",
)
_FINISH_KEYS: tuple[str, ...] = (
    "gen_ai.response.finish_reasons",
    "finish_reason",
    "response.finish_reason",
)
_OUTPUT_KEYS: tuple[str, ...] = (
    "gen_ai.completion",
    "output.value",
    "output_value",
    "llm.output_messages",
    "output",
    "completion",
)
_PROMPT_KEYS: tuple[str, ...] = ("gen_ai.prompt", "input.value", "input_value", "prompt", "llm.input_messages")
_QUERY_KEYS: tuple[str, ...] = ("retrieval.query", "search.query", "query", "input.value")
_TOP_K_KEYS: tuple[str, ...] = ("retrieval.top_k", "top_k", "topk")


def _first_key(attrs: dict[str, Any], keys: tuple[str, ...], prefixes: tuple[str, ...] = ()) -> Any:
    for key in keys:
        if key in attrs:
            return attrs[key]
    for prefix in prefixes:
        for key, value in attrs.items():
            if key.startswith(prefix):
                return value
    return None


def _flatten_value(value: Any) -> Any:
    """One OTLP KeyValue ``value`` oneof -> a plain python value."""
    if not isinstance(value, dict):
        return value
    if "stringValue" in value:
        return value["stringValue"]
    if "boolValue" in value:
        return value["boolValue"]
    if "intValue" in value:
        raw = value["intValue"]
        try:
            return int(raw)
        except (TypeError, ValueError):
            return raw
    if "doubleValue" in value:
        return value["doubleValue"]
    if "bytesValue" in value:
        try:
            return base64.b64decode(value["bytesValue"], validate=False).hex()
        except (binascii.Error, ValueError):
            return value["bytesValue"]
    if "arrayValue" in value:
        return [_flatten_value(v) for v in (value["arrayValue"] or {}).get("values", [])]
    if "kvlistValue" in value:
        return _flatten_attrs((value["kvlistValue"] or {}).get("values", []))
    return value


def _flatten_attrs(raw: Any) -> dict[str, Any]:
    """Accept both OTLP attribute shapes: KV lists and plain maps."""
    if isinstance(raw, dict):
        return {str(k): _flatten_value(v) for k, v in raw.items()}
    if isinstance(raw, list):
        out: dict[str, Any] = {}
        for item in raw:
            if isinstance(item, dict) and "key" in item:
                out[str(item["key"])] = _flatten_value(item.get("value"))
        return out
    return {}


def _messages_text(raw: Any) -> str:
    """gen_ai.prompt/completion can be a string or a list of message dicts."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for msg in raw:
            if isinstance(msg, dict):
                content = msg.get("content", msg.get("text", ""))
                parts.append(str(content))
            else:
                parts.append(str(msg))
        return "\n".join(parts)
    if raw is not None:
        return str(raw)
    return ""


def _canonical_attrs(attrs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Overlay well-known OTel semantic-convention keys with the canonical
    ForensiQ attr vocabulary. Returns (span_fields, canon_attrs)."""
    span_fields: dict[str, Any] = {}
    canon: dict[str, Any] = {}

    tokens_in = _first_key(attrs, _TOKENS_IN_KEYS)
    if tokens_in is not None:
        span_fields["tokens_in"] = _as_int(tokens_in)
    tokens_out = _first_key(attrs, _TOKENS_OUT_KEYS)
    if tokens_out is not None:
        span_fields["tokens_out"] = _as_int(tokens_out)
    cost = _first_key(attrs, _COST_KEYS)
    if cost is not None:
        try:
            span_fields["cost"] = float(cost)
        except (TypeError, ValueError):
            pass

    model = _first_key(attrs, _MODEL_KEYS)
    if model is not None:
        canon["model"] = str(model)

    finish = _first_key(attrs, _FINISH_KEYS)
    if isinstance(finish, list) and finish:
        finish = finish[0]
    if finish is not None:
        canon["finish_reason"] = str(finish)

    output = _first_key(attrs, _OUTPUT_KEYS)
    if output is not None:
        text = _messages_text(output)
        if text:
            canon["output"] = text
    prompt = _first_key(attrs, _PROMPT_KEYS)
    if prompt is not None:
        prompt_text = _messages_text(prompt)
        if prompt_text:
            canon["prompt_chars"] = len(prompt_text)

    query = _first_key(attrs, _QUERY_KEYS, prefixes=("retrieval.query",))
    if query is not None and not isinstance(query, (dict, list)):
        canon["query"] = str(query)
    top_k = _first_key(attrs, _TOP_K_KEYS)
    if top_k is not None:
        top_k = _as_int(top_k)
        if top_k is not None:
            canon["top_k"] = top_k

    docs = attrs.get("retrieval.documents", attrs.get("retrieval_documents"))
    if isinstance(docs, list) and docs:
        scores: list[float] = []
        contents: list[str] = []
        for doc in docs:
            if isinstance(doc, dict):
                score = doc.get("document.score", doc.get("score"))
                if isinstance(score, (int, float)):
                    scores.append(float(score))
                content = doc.get("document.content", doc.get("content"))
                if content:
                    contents.append(str(content))
            elif isinstance(doc, (int, float)):
                scores.append(float(doc))
        canon["hit_count"] = len(docs)
        if scores:
            canon["hit_scores"] = scores
        if contents:
            canon["context"] = "\n".join(contents)

    match_count = attrs.get("match.count")
    if match_count is not None and "hit_count" not in canon:
        count = _as_int(match_count)
        if count is not None:
            canon["hit_count"] = count

    return span_fields, canon


def _as_int(raw: Any) -> int | None:
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return None


# -- ids and timestamps ------------------------------------------------------

_INVALID = object()  # sentinel: id present on the wire but not parseable


def _normalize_hex_id(raw: Any) -> str | object | None:
    """Canonicalize a trace/span id: hex lowercase, ``0x`` stripped;
    protobuf-JSON base64-encoded byte ids are decoded to hex.
    None = absent (missing parent / root), _INVALID = present but bad."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.lower().startswith("0x"):
        text = text[2:]
    if _HEX_RE.fullmatch(text) and len(text) % 2 == 0:
        return text.lower()
    try:  # protobuf-JSON encodes bytes fields as base64
        return base64.b64decode(text, validate=True).hex()
    except (binascii.Error, ValueError):
        return _INVALID


def _is_root_parent(parent_hex: str | None) -> bool:
    """All-zero or empty parent ids mean 'no parent' (root span)."""
    return parent_hex is None or parent_hex.strip("0") == ""


def _ns_to_datetime(ns: int) -> datetime:
    """Unix nanoseconds -> tz-aware datetime with microsecond fidelity
    (divmod first: ns/1e9 as a float would lose the sub-microsecond digits)."""
    secs, rem = divmod(ns, 1_000_000_000)
    base = datetime.fromtimestamp(secs, tz=UTC)
    return base + timedelta(microseconds=rem // 1000)


def _read_nanos(raw: Any) -> int | None:
    """OTLP 64-bit ints arrive JSON-encoded as strings; accept both."""
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    return _as_int(raw)


# -- span / trace assembly ---------------------------------------------------


def _span_status(raw: Any, name: str) -> tuple[str, dict[str, Any]]:
    """OTel status -> (ForensiQ status, extra attrs). OTel has no timeout
    status; spans named '*timeout*' keep the richer ForensiQ status."""
    status, extra = "ok", {}
    if isinstance(raw, dict):
        code = raw.get("code")
        message = raw.get("message")
        if isinstance(code, str):
            code_text = code.upper().replace("STATUS_CODE_", "")
        else:
            code_text = {0: "UNSET", 1: "OK", 2: "ERROR"}.get(code if isinstance(code, int) else -1, "UNSET")
        if code_text == "ERROR":
            status = "error"
            if message:
                extra["error"] = str(message)
    if "timeout" in name.lower():
        status = "timeout"
    return status, extra


def _parse_span(raw: Any, report: OTLPSkipReport, where: str) -> tuple[str, int | None, Span] | None:
    """One OTLP span -> (trace_id, start_ns, Span) or None (skipped, reported)."""
    if not isinstance(raw, dict):
        report.skipped_spans += 1
        report.note(f"{where}: span entry is not an object")
        return None
    where = f"{where}/span {raw.get('name', '?')!r}"

    trace_id = _normalize_hex_id(raw.get("traceId"))
    if trace_id is None or trace_id is _INVALID:
        report.skipped_spans += 1
        report.note(f"{where}: missing/unparseable traceId")
        return None
    span_hex = _normalize_hex_id(raw.get("spanId"))
    if span_hex is None or span_hex is _INVALID:
        report.skipped_spans += 1
        report.note(f"{where}: missing/unparseable spanId")
        return None
    parent_hex = _normalize_hex_id(raw.get("parentSpanId"))
    if parent_hex is _INVALID:
        # Not fatal: the span becomes a root, but the wire glitch is reported.
        report.note(f"{where}: unparseable parentSpanId treated as root")
        parent_hex = None
    root = parent_hex is None or _is_root_parent(parent_hex)

    name = str(raw.get("name") or "span")
    attrs = _flatten_attrs(raw.get("attributes"))
    status, status_extra = _span_status(raw.get("status"), name)
    attrs.update(status_extra)

    start_ns = _read_nanos(raw.get("startTimeUnixNano"))
    end_ns = _read_nanos(raw.get("endTimeUnixNano"))
    duration_ms = 0.0
    if start_ns is not None and end_ns is not None and end_ns >= start_ns:
        duration_ms = (end_ns - start_ns) / 1_000_000.0
    else:
        duration_nanos = _read_nanos(raw.get("durationNanos"))
        if duration_nanos is not None:
            duration_ms = duration_nanos / 1_000_000.0

    span_fields, canon = _canonical_attrs(attrs)
    merged_attrs = {**attrs, **canon}
    return (
        str(trace_id),
        start_ns,
        Span(
            span_id=str(span_hex),
            parent_id=None if root else str(parent_hex),
            name=name,
            stage=infer_stage(name, attrs),
            duration_ms=round(duration_ms, 3),
            status=status,  # type: ignore[arg-type]
            tokens_in=int(span_fields.get("tokens_in", 0) or 0),
            tokens_out=int(span_fields.get("tokens_out", 0) or 0),
            cost=float(span_fields.get("cost", 0.0) or 0.0),
            attrs=merged_attrs,
        ),
    )


def _resource_attrs(resource: Any) -> dict[str, Any]:
    if not isinstance(resource, dict):
        return {}
    return _flatten_attrs(resource.get("attributes"))


def _build_trace(
    trace_id: str,
    entries: list[tuple[int | None, Span]],
    pipeline_name: str,
    meta: dict[str, Any],
) -> Trace:
    """Group entries into a Trace; spans are ordered chronologically when all
    starts are known (stable for ties -> document order preserved). ForensiQ
    spans carry no start time, so this is the one place ordering is fixed up
    for session replay."""
    indexed = [(ns, span, i) for i, (ns, span) in enumerate(entries)]
    if all(ns is not None for ns, _, _ in indexed):
        ordered = [span for _, span, _ in sorted(indexed, key=lambda t: (t[0], t[2]))]
    else:
        ordered = [span for _, span in entries]

    starts = [ns for ns, _ in entries if ns is not None]
    ts = _ns_to_datetime(min(starts)) if starts else datetime.fromtimestamp(0, tz=UTC)
    trace = Trace(trace_id=trace_id, ts=ts, pipeline_name=pipeline_name, spans=ordered, meta=dict(meta))
    trace.recompute_cost()
    return trace


def parse_otlp(payload: str | bytes | dict[str, Any] | list[Any]) -> OTLPParseResult:
    """Parse one OTLP/JSON trace export into ForensiQ traces.

    Tolerant by contract: malformed spans/scopes/resources are skipped and
    counted in the :class:`OTLPSkipReport`; parsing never raises on partial
    data. Only a document that is not JSON at all raises :class:`IngestError`.
    Accepts a single ``ExportTraceServiceRequest`` or a list of them.
    """
    if isinstance(payload, (str, bytes)):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise IngestError(f"OTLP payload is not valid JSON: {exc}") from exc
    requests: list[Any] = payload if isinstance(payload, list) else [payload]

    report = OTLPSkipReport()
    grouped: dict[str, list[tuple[int | None, Span]]] = {}
    metas: dict[str, dict[str, Any]] = {}

    for i, request in enumerate(requests):
        if not isinstance(request, dict) or not isinstance(request.get("resourceSpans", []), list):
            report.skipped_resources += 1
            report.note(f"request #{i}: no resourceSpans list")
            continue
        for r_index, entry in enumerate(request["resourceSpans"]):
            where = f"resourceSpans[{r_index}]"
            if not isinstance(entry, dict):
                report.skipped_resources += 1
                report.note(f"{where}: not an object")
                continue
            resource = _resource_attrs(entry.get("resource"))
            service = str(resource.get("service.name", "unknown"))
            scopes = entry.get("scopeSpans") or []
            if not isinstance(scopes, list):
                report.skipped_resources += 1
                report.note(f"{where}: scopeSpans is not a list")
                continue
            for s_index, scope_entry in enumerate(scopes):
                scope_where = f"{where}/scopeSpans[{s_index}]"
                if not isinstance(scope_entry, dict):
                    report.skipped_scopes += 1
                    report.note(f"{scope_where}: not an object")
                    continue
                spans = scope_entry.get("spans") or []
                if not isinstance(spans, list):
                    report.skipped_scopes += 1
                    report.note(f"{scope_where}: spans is not a list")
                    continue
                for sp_index, raw_span in enumerate(spans):
                    parsed = _parse_span(raw_span, report, f"{scope_where}/spans[{sp_index}]")
                    if parsed is None:
                        continue  # counter + reason already recorded
                    trace_id, start_ns, span = parsed
                    grouped.setdefault(trace_id, []).append((start_ns, span))
                    meta = metas.setdefault(trace_id, {"source": "otlp", "service.name": service, "scopes": []})
                    scope = scope_entry.get("scope")
                    if isinstance(scope, dict) and scope.get("name"):
                        meta["scopes"] = sorted(set(meta["scopes"]) | {str(scope["name"])})
                    extra_resource = {k: resource[k] for k in sorted(resource) if k != "service.name"}
                    if extra_resource:
                        meta.setdefault("resource", {}).update(extra_resource)

    built = [
        _build_trace(
            trace_id,
            entries,
            str(metas[trace_id].get("service.name", "unknown")),
            metas[trace_id],
        )
        for trace_id, entries in sorted(grouped.items())
    ]
    built.sort(key=lambda t: (t.ts, t.trace_id))
    return OTLPParseResult(traces=built, skip=report)


def parse_otlp_json(payload: str | bytes | dict[str, Any] | list[Any]) -> list[Trace]:
    """Convenience wrapper: parse and return just the traces."""
    return parse_otlp(payload).traces
