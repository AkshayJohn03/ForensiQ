"""Loaders: portfolio span-dict round-trip, RAG_showcase exports, Langfuse
fixture mapping, and env-guard behavior. All offline."""

from __future__ import annotations

import json

import pytest

from forensiq.ingest.loaders import (
    IngestError,
    JSONLTraceLoader,
    JSONTraceLoader,
    LangfuseTraceLoader,
    record_to_trace,
    spans_from_dicts,
)
from forensiq.ingest.schema import Trace, normalize_stage


class TestStageNormalization:
    def test_rag_showcase_aliases(self):
        # RAG_showcase emits "hybrid" for fused retrieval — must map to retrieve.
        assert normalize_stage("hybrid") == "retrieve"
        assert normalize_stage("Hybrid-Retrieve") == "retrieve"
        assert normalize_stage("retrieval") == "retrieve"
        assert normalize_stage("generation") == "generate"
        assert normalize_stage("guardrail") == "guard"

    def test_unknown_degrades_to_tool(self):
        assert normalize_stage("mystery_step") == "tool"

    def test_canonical_passthrough(self):
        for stage in ("embed", "retrieve", "rerank", "graph", "generate", "tool", "agent"):
            assert normalize_stage(stage) == stage


class TestSpansFromDicts:
    def test_portfolio_hook_round_trip(self):
        """AegisGate/SwarmResearch hook shape: span_id/parent_id/name/stage/
        duration_ms/status/attrs (+ optional token/cost fields)."""
        dicts = [
            {
                "span_id": "s1",
                "parent_id": None,
                "name": "embed",
                "stage": "embed",
                "duration_ms": 12.5,
                "status": "ok",
                "attrs": {"model": "bge-small"},
                "tokens_in": 40,
                "tokens_out": 0,
                "cost": 0.0001,
            },
            {
                "span_id": "s2",
                "parent_id": "s1",
                "name": "hybrid",
                "stage": "hybrid",
                "duration_ms": 40,
                "status": "ok",
                "attrs": {"top_k": 5, "hit_scores": [0.9, 0.4]},
            },
        ]
        spans = spans_from_dicts(dicts)
        assert [s.stage for s in spans] == ["embed", "retrieve"]  # alias normalized
        assert spans[0].tokens_in == 40
        assert spans[1].parent_id == "s1"
        assert spans[1].attrs["top_k"] == 5
        # round-trip: dump -> load -> equal
        restored = spans_from_dicts([json.loads(s.model_dump_json()) for s in spans])
        assert restored == spans

    def test_missing_ids_synthesized_and_status_normalized(self):
        spans = spans_from_dicts(
            [
                {"name": "llm-call", "stage": "llm", "durationMs": 5, "status": "TIMEOUT"},
                {"stage": "tool"},
            ]
        )
        assert spans[0].stage == "generate"
        assert spans[0].status == "timeout"
        assert spans[0].duration_ms == 5
        assert spans[1].span_id  # synthesized
        assert spans[1].status == "ok"

    def test_non_dict_raises(self):
        with pytest.raises(IngestError):
            spans_from_dicts(["nope"])


class TestRecordToTrace:
    def test_full_trace_dict(self):
        rec = {
            "trace_id": "t1",
            "ts": "2026-08-14T09:12:00Z",
            "pipeline_name": "rag_showcase",
            "spans": [
                {"span_id": "s1", "name": "embed", "stage": "embed", "duration_ms": 10, "cost": 0.001},
                {"span_id": "s2", "name": "hybrid", "stage": "hybrid", "duration_ms": 20, "cost": 0.002},
            ],
        }
        trace = record_to_trace(rec, "fb")
        assert isinstance(trace, Trace)
        assert trace.trace_id == "t1"
        assert trace.ts.year == 2026 and trace.ts.month == 8
        assert trace.total_cost == 0.003  # recomputed
        assert trace.stage_span("retrieve") is not None

    def test_epoch_ms_and_epoch_s(self):
        rec = {"trace_id": "t", "ts": 1755000000, "spans": []}
        assert record_to_trace(rec, "fb").ts.year == 2025
        rec["ts"] = 1755000000000
        assert record_to_trace(rec, "fb").ts == record_to_trace(
            {"trace_id": "t", "ts": 1755000000, "spans": []}, "fb"
        ).ts

    def test_eval_style_record(self):
        """RAG_showcase eval records: query + retrieved scores + metrics."""
        rec = {
            "query": "Which supplier provides XK-7?",
            "retrieved": [{"chunk_id": "a#1", "score": 0.91}, {"chunk_id": "a#2", "score": 0.42}],
            "answer": "Nordwerk Precision GmbH supplies XK-7.",
            "metrics": {"faithfulness": 0.91, "context_recall": 0.94},
        }
        trace = record_to_trace(rec, "fb")
        ret = trace.stage_span("retrieve")
        gen = trace.stage_span("generate")
        assert ret is not None and gen is not None
        assert ret.attrs["hit_scores"] == [0.91, 0.42]
        assert ret.attrs["query"].startswith("Which supplier")
        assert gen.attrs["output"].startswith("Nordwerk")
        assert trace.meta["metrics"]["faithfulness"] == 0.91

    def test_bare_span_becomes_single_span_trace(self):
        trace = record_to_trace(
            {"span_id": "x", "name": "rerank", "stage": "rerank", "duration_ms": 3}, "fb"
        )
        assert len(trace.spans) == 1
        assert trace.spans[0].stage == "rerank"


class TestJSONLLoader:
    def test_round_trip_of_synthetic_export(self, corpus):
        import io

        traces, _ = corpus
        buf = io.StringIO()
        for t in traces[:50]:
            buf.write(t.model_dump_json() + "\n")
        loaded = JSONLTraceLoader().load_text(buf.getvalue())
        assert loaded == traces[:50]

    def test_bad_lines_degrade_to_warnings(self):
        text = "\n".join(
            ['{"trace_id": "ok", "ts": 0, "spans": []}', "not json", '{"trace_id": "ok2", "ts": 0, "spans": []}', ""]
        )
        loader = JSONLTraceLoader()
        traces = loader.load_text(text)
        assert len(traces) == 2
        assert len(loader.warnings) == 1
        assert "line 2" in loader.warnings[0]

    def test_all_bad_raises(self):
        with pytest.raises(IngestError):
            JSONLTraceLoader().load_text("garbage\ngarbage2")


class TestJSONLoader:
    def test_list_and_single(self):
        single = json.dumps({"trace_id": "a", "ts": 0, "spans": []})
        assert JSONTraceLoader().load_text(single)[0].trace_id == "a"
        many = json.dumps(
            [{"trace_id": "a", "ts": 0, "spans": []}, {"trace_id": "b", "ts": 0, "spans": []}]
        )
        assert len(JSONTraceLoader().load_text(many)) == 2

    def test_invalid_json_raises(self):
        with pytest.raises(IngestError):
            JSONTraceLoader().load_text("{oops")


LANGFUSE_TRACES_PAGE = {
    "data": [
        {"id": "lf-1", "timestamp": "2026-08-14T09:12:00Z", "name": "rag_showcase", "release": "1.2.0"},
        {"id": "lf-2", "timestamp": "2026-08-14T09:13:00Z", "name": "rag_showcase"},
    ]
}
LANGFUSE_OBS_PAGE = {
    "data": [
        {
            "id": "o1",
            "traceId": "lf-1",
            "parentObservationId": None,
            "name": "embed",
            "type": "SPAN",
            "level": "DEFAULT",
            "startTime": "2026-08-14T09:12:00Z",
            "endTime": "2026-08-14T09:12:00.012Z",
            "usage": {"input": 30, "output": 0},
        },
        {
            "id": "o2",
            "traceId": "lf-1",
            "parentObservationId": "o1",
            "name": "hybrid",
            "type": "SPAN",
            "level": "DEFAULT",
            "startTime": "2026-08-14T09:12:00.012Z",
            "endTime": "2026-08-14T09:12:00.056Z",
            "usage": {},
        },
        {
            "id": "o3",
            "traceId": "lf-1",
            "parentObservationId": "o2",
            "name": "generate",
            "type": "GENERATION",
            "level": "ERROR",
            "startTime": "2026-08-14T09:12:00.06Z",
            "endTime": "2026-08-14T09:12:01.60Z",
            "usage": {"input": 1204, "output": 210},
            "calculatedTotalCost": 0.0021,
            "metadata": {"model": "qwen3:1.7b", "finish_reason": "stop"},
        },
    ]
}


class TestLangfuseLoader:
    def test_env_guard_blocks_construction(self, monkeypatch):
        for var in ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(IngestError, match="env-guarded"):
            LangfuseTraceLoader()

    def test_from_pages_maps_recorded_fixture(self):
        traces = LangfuseTraceLoader.from_pages(LANGFUSE_TRACES_PAGE, LANGFUSE_OBS_PAGE)
        assert [t.trace_id for t in traces] == ["lf-1", "lf-2"]
        t1 = traces[0]
        assert t1.pipeline_name == "rag_showcase"
        assert t1.meta["release"] == "1.2.0"
        # hybrid observation normalized to the retrieve stage
        assert [s.stage for s in t1.spans] == ["embed", "retrieve", "generate"]
        gen = t1.stage_span("generate")
        assert gen.status == "error"  # level ERROR
        assert gen.tokens_in == 1204 and gen.tokens_out == 210
        assert gen.cost == pytest.approx(0.0021)
        assert gen.duration_ms == pytest.approx(1540.0, rel=0.01)
        assert gen.attrs["finish_reason"] == "stop"
        assert t1.total_cost == pytest.approx(0.0021)

    def test_timeout_level_detection(self):
        obs = dict(LANGFUSE_OBS_PAGE["data"][2])
        obs["name"] = "generate-timeout"
        obs["level"] = "DEFAULT"
        traces = LangfuseTraceLoader.from_pages(LANGFUSE_TRACES_PAGE, {"data": [obs]})
        assert traces[0].spans[0].status == "timeout"
