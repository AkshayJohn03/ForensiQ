"""Trace storage seam: a minimal Protocol with two implementations.

* :class:`JSONLTraceStore` — pure-python, file-backed (JSONL append). The
  default path: zero dependencies, human-inspectable files, works everywhere
  ForensiQ works.
* :class:`~forensiq.ingest.duckdb_store.DuckDBTraceStore` — embedded DuckDB
  (optional extra ``pip install 'forensiQ[ingest]'``) for high-throughput
  analytical queries over large corpora. Lazy import: the base install never
  needs it.

Both implement the same queries the forensics layer needs: ``traces_by_time``,
``spans_by_trace``, ``failure_mix_per_day`` and ``get_trace``. Stores are
keyed on trace id: re-ingesting a trace replaces its stored copy, so
idempotent re-exports never duplicate flight records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from forensiq.ingest.loaders import IngestError, _parse_ts
from forensiq.ingest.schema import Span, Trace
from forensiq.taxonomy.classifier import FailureRecord

__all__ = ["JSONLTraceStore", "TraceStore"]


@runtime_checkable
class TraceStore(Protocol):
    """Write/read seam for normalized traces (and their failure records)."""

    def insert_batch(
        self, traces: list[Trace], failures: list[FailureRecord] | None = None
    ) -> int: ...

    def traces_by_time(self, start: Any, end: Any) -> list[Trace]: ...

    def spans_by_trace(self, trace_id: str) -> list[Span]: ...

    def get_trace(self, trace_id: str) -> Trace | None: ...

    def failure_mix_per_day(self) -> dict[str, dict[str, int]]: ...


class JSONLTraceStore:
    """Pure-python fallback: traces.jsonl + failures.jsonl in one directory.

    Append-only writes, full-scan reads — correctness over speed, which is the
    right trade for a forensics tool whose analytics only need hundreds of
    failures. On read, the first copy of a trace_id wins (later re-ingests are
    ignored). Use :class:`~forensiq.ingest.duckdb_store.DuckDBTraceStore` when
    the corpus outgrows this.
    """

    def __init__(self, path: str | Path) -> None:
        self.dir = Path(path)
        self.traces_file = self.dir / "traces.jsonl"
        self.failures_file = self.dir / "failures.jsonl"

    # -- writes ----------------------------------------------------------

    def insert_batch(
        self, traces: list[Trace], failures: list[FailureRecord] | None = None
    ) -> int:
        """Append traces (and optional failure records); returns count written."""
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.traces_file.open("a", encoding="utf-8") as fh:
            for trace in traces:
                fh.write(json.dumps(trace.model_dump(mode="json"), default=str) + "\n")
        if failures:
            with self.failures_file.open("a", encoding="utf-8") as fh:
                for failure in failures:
                    fh.write(json.dumps(failure.model_dump(mode="json"), default=str) + "\n")
        return len(traces)

    # -- reads (full scans) ------------------------------------------------

    def _read_traces(self) -> list[Trace]:
        if not self.traces_file.exists():
            return []
        traces: list[Trace] = []
        for lineno, line in enumerate(self.traces_file.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                traces.append(Trace.model_validate(json.loads(line)))
            except (ValueError, TypeError) as exc:
                # A torn last line (crash mid-write) must not kill queries.
                raise IngestError(f"{self.traces_file}: line {lineno}: {exc}") from exc
        return traces

    def _read_failures(self) -> list[FailureRecord]:
        if not self.failures_file.exists():
            return []
        records: list[FailureRecord] = []
        for lineno, line in enumerate(self.failures_file.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(FailureRecord.model_validate(json.loads(line)))
            except (ValueError, TypeError) as exc:
                raise IngestError(f"{self.failures_file}: line {lineno}: {exc}") from exc
        return records

    # -- queries ------------------------------------------------------------

    def _dedupe(self, traces: list[Trace]) -> dict[str, Trace]:
        out: dict[str, Trace] = {}
        for trace in traces:
            out.setdefault(trace.trace_id, trace)
        return out

    def traces_by_time(self, start: Any, end: Any) -> list[Trace]:
        """Traces with start ts in [start, end] (inclusive), ordered by ts."""
        s, e = _parse_ts(start), _parse_ts(end)
        matched = [t for t in self._dedupe(self._read_traces()).values() if s <= t.ts <= e]
        return sorted(matched, key=lambda t: (t.ts, t.trace_id))

    def spans_by_trace(self, trace_id: str) -> list[Span]:
        trace = self.get_trace(trace_id)
        return list(trace.spans) if trace else []

    def get_trace(self, trace_id: str) -> Trace | None:
        return self._dedupe(self._read_traces()).get(trace_id)

    def failure_mix_per_day(self) -> dict[str, dict[str, int]]:
        """{(ISO day): {taxonomy_id: count}}, ordered; unknown trace ids skipped."""
        day_by_trace = {t.trace_id: t.day for t in self._dedupe(self._read_traces()).values()}
        mix: dict[str, dict[str, int]] = {}
        for failure in self._read_failures():
            day = day_by_trace.get(failure.trace_id)
            if day is None:
                continue
            mix.setdefault(day, {})
            mix[day][failure.taxonomy_id] = mix[day].get(failure.taxonomy_id, 0) + 1
        return {day: dict(sorted(mix[day].items())) for day in sorted(mix)}
