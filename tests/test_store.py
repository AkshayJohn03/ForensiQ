"""Trace stores: JSONLTraceStore (always available) round-trips; DuckDBTraceStore
skips cleanly when the optional extra is missing."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from forensiq.ingest.loaders import IngestError
from forensiq.ingest.schema import Span, Trace
from forensiq.ingest.store import JSONLTraceStore
from forensiq.taxonomy.classifier import FailureRecord


def make_trace(trace_id: str, minutes: int = 0) -> Trace:
    return Trace(
        trace_id=trace_id,
        ts=datetime(2026, 8, 14, 9, minutes, tzinfo=UTC),
        pipeline_name="rag_showcase",
        spans=[
            Span(
                span_id=f"{trace_id}-r",
                name="hybrid",
                stage="retrieve",
                duration_ms=44.0,
                attrs={"query": f"q-{trace_id}", "top_k": 5, "hit_count": 5, "hit_scores": [0.8, 0.6]},
            ),
            Span(
                span_id=f"{trace_id}-g",
                parent_id=f"{trace_id}-r",
                name="generate",
                stage="generate",
                duration_ms=120.0,
                tokens_in=100,
                tokens_out=20,
                cost=0.001,
                attrs={"model": "m", "output": "answer"},
            ),
        ],
        total_cost=0.001,
    )


def make_failure(trace_id: str, tax_id: str = "F-RET-002") -> FailureRecord:
    return FailureRecord(
        taxonomy_id=tax_id,
        stage="retrieve",
        trace_id=trace_id,
        confidence=0.85,
        evidence=[],
    )


class TestJSONLTraceStore:
    def test_insert_and_get_trace_roundtrip(self, tmp_path):
        store = JSONLTraceStore(tmp_path)
        assert store.insert_batch([make_trace("t1")]) == 1
        loaded = store.get_trace("t1")
        assert loaded is not None
        assert loaded.trace_id == "t1"
        assert loaded.pipeline_name == "rag_showcase"
        assert loaded.ts == datetime(2026, 8, 14, 9, 0, tzinfo=UTC)
        assert [s.span_id for s in loaded.spans] == ["t1-r", "t1-g"]
        assert loaded.spans[0].attrs["hit_scores"] == [0.8, 0.6]

    def test_missing_trace_returns_none(self, tmp_path):
        assert JSONLTraceStore(tmp_path).get_trace("nope") is None

    def test_spans_by_trace_in_order(self, tmp_path):
        store = JSONLTraceStore(tmp_path)
        store.insert_batch([make_trace("t1"), make_trace("t2", minutes=5)])
        assert [s.stage for s in store.spans_by_trace("t1")] == ["retrieve", "generate"]
        assert store.spans_by_trace("ghost") == []

    def test_traces_by_time_window(self, tmp_path):
        store = JSONLTraceStore(tmp_path)
        store.insert_batch([make_trace("a", 0), make_trace("b", 10), make_trace("c", 30)])
        start = datetime(2026, 8, 14, 9, 5, tzinfo=UTC)
        end = start + timedelta(minutes=20)
        assert [t.trace_id for t in store.traces_by_time(start, end)] == ["b"]

    def test_failure_mix_per_day(self, tmp_path):
        store = JSONLTraceStore(tmp_path)
        store.insert_batch(
            [make_trace("a", 0), make_trace("b", 0), make_trace("c", 30)],
            failures=[make_failure("a"), make_failure("b"), make_failure("c", "F-GEN-001")],
        )
        mix = store.failure_mix_per_day()
        assert mix == {"2026-08-14": {"F-GEN-001": 1, "F-RET-002": 2}}

    def test_failure_for_unknown_trace_is_skipped(self, tmp_path):
        store = JSONLTraceStore(tmp_path)
        store.insert_batch([], failures=[make_failure("ghost")])
        assert store.failure_mix_per_day() == {}

    def test_reinsert_replaces_first_copy_wins(self, tmp_path):
        store = JSONLTraceStore(tmp_path)
        store.insert_batch([make_trace("t1")])
        mutated = make_trace("t1")
        mutated.pipeline_name = "changed"
        store.insert_batch([mutated])
        assert store.get_trace("t1").pipeline_name == "rag_showcase"  # first copy wins

    def test_torn_last_line_raises_ingest_error(self, tmp_path):
        store = JSONLTraceStore(tmp_path)
        store.insert_batch([make_trace("t1")])
        with store.traces_file.open("a", encoding="utf-8") as fh:
            fh.write('{"trace_id": "t9", "ts": "2026')  # simulated crash mid-write
        with pytest.raises(IngestError):
            store.get_trace("t1")

    def test_implements_protocol(self, tmp_path):
        from forensiq.ingest.store import TraceStore

        assert isinstance(JSONLTraceStore(tmp_path), TraceStore)


class TestDuckDBTraceStore:
    def test_roundtrip_when_available(self, tmp_path):
        pytest.importorskip("duckdb")
        from forensiq.ingest.duckdb_store import DuckDBTraceStore

        store = DuckDBTraceStore(tmp_path / "traces.duckdb")
        assert store.insert_batch([make_trace("t1"), make_trace("t2", minutes=5)]) == 2
        loaded = store.get_trace("t1")
        assert loaded is not None
        assert loaded.ts == datetime(2026, 8, 14, 9, 0, tzinfo=UTC)
        assert [s.span_id for s in loaded.spans] == ["t1-r", "t1-g"]
        assert loaded.spans[0].attrs["hit_scores"] == [0.8, 0.6]
        assert [s.span_id for s in store.spans_by_trace("t1")] == ["t1-r", "t1-g"]

    def test_traces_by_time_and_mix_when_available(self, tmp_path):
        pytest.importorskip("duckdb")
        from forensiq.ingest.duckdb_store import DuckDBTraceStore

        store = DuckDBTraceStore(":memory:")
        store.insert_batch(
            [make_trace("a", 0), make_trace("b", 10), make_trace("c", 30)],
            failures=[make_failure("a"), make_failure("b"), make_failure("c", "F-GEN-001")],
        )
        start = datetime(2026, 8, 14, 9, 5, tzinfo=UTC)
        assert [t.trace_id for t in store.traces_by_time(start, start + timedelta(minutes=20))] == ["b"]
        assert store.failure_mix_per_day() == {"2026-08-14": {"F-GEN-001": 1, "F-RET-002": 2}}
        assert store.count_traces() == 3

    def test_reinsert_replaces(self, tmp_path):
        pytest.importorskip("duckdb")
        from forensiq.ingest.duckdb_store import DuckDBTraceStore

        store = DuckDBTraceStore(":memory:")
        store.insert_batch([make_trace("t1")])
        store.insert_batch([make_trace("t1")])  # idempotent re-export
        assert store.count_traces() == 1
        assert len(store.spans_by_trace("t1")) == 2

    def test_proper_column_types(self, tmp_path):
        pytest.importorskip("duckdb")
        from forensiq.ingest.duckdb_store import DuckDBTraceStore

        store = DuckDBTraceStore(":memory:")
        store.insert_batch([make_trace("t1")], failures=[make_failure("t1")])
        types = {
            row[0]: row[1]
            for row in store.con.execute(
                "SELECT table_name, column_name FROM information_schema.columns "
                "WHERE table_schema = 'main'"
            ).fetchall()
        }
        _ = types  # existence checks per table below
        spans_types = {
            r[0]: r[1]
            for r in store.con.execute(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'spans'"
            ).fetchall()
        }
        assert spans_types["duration_ms"] == "DOUBLE"
        assert spans_types["tokens_in"] == "INTEGER"
        assert spans_types["attrs"] == "JSON"
        traces_types = {
            r[0]: r[1]
            for r in store.con.execute(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'traces'"
            ).fetchall()
        }
        assert traces_types["ts"] == "TIMESTAMP WITH TIME ZONE"

    def test_missing_extra_raises_actionable_error(self):
        try:
            import duckdb  # noqa: F401
        except ImportError:
            from forensiq.ingest.duckdb_store import DuckDBTraceStore

            with pytest.raises(IngestError, match="forensiq\\[ingest\\]"):
                DuckDBTraceStore(":memory:")
            return
        pytest.skip("duckdb installed; import-error path not exercisable")
