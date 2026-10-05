"""DuckDB-backed :class:`~forensiq.ingest.store.TraceStore` (optional extra).

DuckDB is the analytical engine for large corpora: an embedded, pip-installable
columnar database — no server, no daemon, works offline, one file on disk. It
sits behind the same :class:`TraceStore` Protocol as the pure-python
:class:`~forensiq.ingest.store.JSONLTraceStore`, so the forensics layer never
notices which engine is underneath.

``duckdb`` is imported lazily: the base install never needs it. Use
``pip install 'forensiq[ingest]'`` to enable. Stores are keyed on trace id:
re-ingesting a trace replaces its stored copy, so idempotent re-exports never
duplicate flight records.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from forensiq.ingest.loaders import IngestError, _parse_ts
from forensiq.ingest.schema import Span, Trace
from forensiq.taxonomy.classifier import FailureRecord

__all__ = ["DuckDBTraceStore"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id      VARCHAR NOT NULL,
    ts            TIMESTAMPTZ NOT NULL,
    pipeline_name VARCHAR,
    total_cost    DOUBLE,
    meta          JSON
);
CREATE TABLE IF NOT EXISTS spans (
    seq        BIGINT,          -- insertion order (session replay order)
    trace_id   VARCHAR NOT NULL,
    span_id    VARCHAR NOT NULL,
    parent_id  VARCHAR,
    name       VARCHAR,
    stage      VARCHAR,
    duration_ms DOUBLE,
    status     VARCHAR,
    tokens_in  INTEGER,
    tokens_out INTEGER,
    cost       DOUBLE,
    attrs      JSON
);
CREATE TABLE IF NOT EXISTS failures (
    trace_id    VARCHAR NOT NULL,
    taxonomy_id VARCHAR NOT NULL,
    stage       VARCHAR,
    confidence  DOUBLE,
    day         DATE
);
"""

_SPAN_COLUMNS = (
    "seq, trace_id, span_id, parent_id, name, stage, duration_ms, status, tokens_in, tokens_out, cost, attrs"
)


class DuckDBTraceStore:
    """Embedded DuckDB store implementing the TraceStore Protocol."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        try:
            import duckdb  # lazy: only the [ingest] extra ships this
        except ImportError as exc:
            raise IngestError(
                "DuckDBTraceStore needs the optional duckdb dependency: "
                "pip install 'forensiq[ingest]' (or use JSONLTraceStore)."
            ) from exc
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(self.path)
        self.con.execute(_SCHEMA)
        self._next_seq = 0

    # -- writes -------------------------------------------------------------

    def insert_batch(
        self, traces: list[Trace], failures: list[FailureRecord] | None = None
    ) -> int:
        """Append a batch of traces (+ optional failure records). Returns count.

        Keyed on trace id: re-ingesting a trace replaces its stored copy, so
        idempotent re-exports never duplicate flight records."""
        trace_ids = [t.trace_id for t in traces]
        if trace_ids:
            placeholders = ", ".join("?" for _ in trace_ids)
            for table in ("traces", "spans", "failures"):
                self.con.execute(f"DELETE FROM {table} WHERE trace_id IN ({placeholders})", trace_ids)
        trace_rows, span_rows = [], []
        for trace in traces:
            trace_rows.append(
                (
                    trace.trace_id,
                    trace.ts,
                    trace.pipeline_name,
                    trace.total_cost,
                    json.dumps(trace.meta, default=str),
                )
            )
            for span in trace.spans:
                span_rows.append(
                    (
                        self._next_seq,
                        trace.trace_id,
                        span.span_id,
                        span.parent_id,
                        span.name,
                        span.stage,
                        span.duration_ms,
                        span.status,
                        span.tokens_in,
                        span.tokens_out,
                        span.cost,
                        json.dumps(span.attrs, default=str),
                    )
                )
                self._next_seq += 1
        if trace_rows:
            self.con.executemany(
                "INSERT INTO traces (trace_id, ts, pipeline_name, total_cost, meta) VALUES (?, ?, ?, ?, ?)",
                trace_rows,
            )
        if span_rows:
            self.con.executemany(
                f"INSERT INTO spans ({_SPAN_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                span_rows,
            )
        if failures:
            self.con.executemany(
                "INSERT INTO failures (trace_id, taxonomy_id, stage, confidence, day) VALUES (?, ?, ?, ?, ?)",
                [
                    (f.trace_id, f.taxonomy_id, f.stage, f.confidence, self._day_of(traces, f.trace_id))
                    for f in failures
                ],
            )
        return len(traces)

    @staticmethod
    def _day_of(traces: list[Trace], trace_id: str) -> str | None:
        for trace in traces:
            if trace.trace_id == trace_id:
                return trace.day
        return None

    # -- reads ----------------------------------------------------------------

    def _spans_for(self, trace_id: str) -> list[Span]:
        rows = self.con.execute(
            f"SELECT {_SPAN_COLUMNS} FROM spans WHERE trace_id = ? ORDER BY seq",
            [trace_id],
        ).fetchall()
        return [
            Span(
                span_id=str(r[2]),
                parent_id=r[3],
                name=str(r[4] or "span"),
                stage=str(r[5] or "tool"),  # type: ignore[arg-type]
                duration_ms=float(r[6] or 0.0),
                status=str(r[7] or "ok"),  # type: ignore[arg-type]
                tokens_in=int(r[8] or 0),
                tokens_out=int(r[9] or 0),
                cost=float(r[10] or 0.0),
                attrs=json.loads(r[11]) if r[11] else {},
            )
            for r in rows
        ]

    def _trace_meta(self, trace_id: str) -> tuple[datetime, str, float, dict[str, Any]] | None:
        row = self.con.execute(
            "SELECT ts, pipeline_name, total_cost, meta FROM traces WHERE trace_id = ? ORDER BY rowid LIMIT 1",
            [trace_id],
        ).fetchone()
        if row is None:
            return None
        meta = json.loads(row[3]) if row[3] else {}
        return row[0], str(row[1] or "unknown"), float(row[2] or 0.0), meta

    def get_trace(self, trace_id: str) -> Trace | None:
        head = self._trace_meta(trace_id)
        if head is None:
            return None
        ts, pipeline_name, total_cost, meta = head
        if ts.tzinfo is None:  # defensive: TIMESTAMPTZ should always be aware
            ts = ts.replace(tzinfo=UTC)
        return Trace(
            pipeline_name=pipeline_name,
            spans=self._spans_for(trace_id),
            total_cost=total_cost,
            meta=meta,
        )

    def traces_by_time(self, start: Any, end: Any) -> list[Trace]:
        """Traces with ts in [start, end] inclusive, ordered by (ts, trace_id)."""
        s, e = _parse_ts(start), _parse_ts(end)
        rows = self.con.execute(
            "SELECT trace_id FROM traces WHERE ts >= ? AND ts <= ? ORDER BY ts, trace_id",
            [s, e],
        ).fetchall()
        out = []
        for (trace_id,) in rows:
            trace = self.get_trace(str(trace_id))
            if trace is not None:
                out.append(trace)
        return out

    def spans_by_trace(self, trace_id: str) -> list[Span]:
        return self._spans_for(trace_id)

    def failure_mix_per_day(self) -> dict[str, dict[str, int]]:
        """{(ISO day): {taxonomy_id: count}}; ties broken deterministically."""
        rows = self.con.execute(
            """
            SELECT CAST(day AS VARCHAR), taxonomy_id, COUNT(*) AS n
            FROM failures
            WHERE day IS NOT NULL
            GROUP BY day, taxonomy_id
            ORDER BY day, taxonomy_id
            """
        ).fetchall()
        mix: dict[str, dict[str, int]] = {}
        for day, tax_id, n in rows:
            mix.setdefault(str(day), {})[str(tax_id)] = int(n)
        return mix

    def count_traces(self) -> int:
        """Distinct traces currently stored."""
        return int(self.con.execute("SELECT COUNT(DISTINCT trace_id) FROM traces").fetchone()[0])

    def close(self) -> None:
        self.con.close()
