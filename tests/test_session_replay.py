"""Session replay: step timeline, state deltas, --at cumulative state, JSON
form, CLI wiring (forensiq replay <trace_id> --at N), determinism. Offline."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from forensiq.cli import main
from forensiq.ingest.schema import Span, Trace
from forensiq.replay import render_replay, replay_steps, replay_to_json, state_at


@pytest.fixture(scope="module")
def trace():
    return Trace(
        trace_id="q-1042",
        ts=datetime(2026, 8, 14, 9, 12, tzinfo=UTC),
        pipeline_name="rag_showcase",
        spans=[
            Span(
                span_id="s1",
                name="embed",
                stage="embed",
                duration_ms=12,
                attrs={"model": "MiniLM-L6", "token_count": 42},
            ),
            Span(
                span_id="s2",
                parent_id="s1",
                name="hybrid",
                stage="retrieve",
                duration_ms=44,
                attrs={"model": "MiniLM-L6", "query": "SAF-114?", "top_k": 5, "hit_count": 5, "hit_scores": [0.8, 0.6]},
            ),
            Span(
                span_id="s3",
                parent_id="s2",
                name="generate",
                stage="generate",
                duration_ms=1800,
                tokens_in=1204,
                tokens_out=210,
                cost=0.002,
                attrs={"query": "SAF-114?", "output": "answer", "finish_reason": "stop"},
            ),
        ],
    )


class TestReplaySteps:
    def test_steps_in_span_order_with_indices(self, trace):
        steps = replay_steps(trace)
        assert [s.index for s in steps] == [1, 2, 3]
        assert [s.stage for s in steps] == ["embed", "retrieve", "generate"]
        assert [s.status for s in steps] == ["ok", "ok", "ok"]

    def test_delta_is_only_new_or_changed_attrs(self, trace):
        steps = replay_steps(trace)
        assert set(steps[0].delta) == {"model", "token_count"}  # first span: full attrs
        assert set(steps[1].delta) == {"query", "top_k", "hit_count", "hit_scores"}  # model unchanged
        assert set(steps[2].delta) == {"output", "finish_reason"}  # query repeated

    def test_cumulative_counters(self, trace):
        steps = replay_steps(trace)
        assert steps[2].cumulative_tokens_in == 1204
        assert steps[2].cumulative_cost == pytest.approx(0.002)
        assert steps[0].cumulative_tokens_in == 0

    def test_changed_attr_counts_as_delta(self):
        t = Trace(
            trace_id="t",
            ts=datetime(2026, 1, 1, tzinfo=UTC),
            spans=[
                Span(span_id="a", name="s", stage="tool", attrs={"k": 1}),
                Span(span_id="b", name="s", stage="tool", attrs={"k": 2}),
            ],
        )
        steps = replay_steps(t)
        assert steps[1].delta == {"k": 2}


class TestStateAt:
    def test_aggregates_attrs_through_step(self, trace):
        state = state_at(trace, 2)
        assert state["model"] == "MiniLM-L6"
        assert state["query"] == "SAF-114?"
        assert state["top_k"] == 5
        assert "output" not in state  # T3 attrs not yet seen
        assert state["_meta"]["spans_seen"] == 2
        assert state["_meta"]["stages_seen"] == ["embed", "retrieve"]

    def test_clamped_bounds(self, trace):
        assert state_at(trace, 0)["_meta"]["spans_seen"] == 0
        assert state_at(trace, 99)["_meta"]["spans_seen"] == 3

    def test_later_spans_win_conflicts(self, trace):
        assert state_at(trace, 3)["query"] == "SAF-114?"  # repeated value, no crash


class TestRendering:
    def test_timeline_lines(self, trace):
        out = render_replay(trace)
        assert out.startswith("Session replay: q-1042 (rag_showcase, 2026-08-14T09:12:00+00:00, 3 spans)")
        assert "T1   embed" in out
        assert "T3   generate  ok        1800.0ms" in out
        assert "totals: tokens_in=1204" in out

    def test_at_block_is_cumulative_state(self, trace):
        out = render_replay(trace, at=2)
        assert "state at T2 (cumulative through step 2):" in out
        assert "top_k = 5" in out
        assert "counters: spans_seen=2" in out

    def test_deterministic(self, trace):
        assert render_replay(trace, at=2) == render_replay(trace, at=2)
        assert replay_to_json(trace, at=2) == replay_to_json(trace, at=2)

    def test_json_form(self, trace):
        payload = json.loads(replay_to_json(trace, at=1))
        assert payload["trace_id"] == "q-1042"
        assert [s["step"] for s in payload["steps"]] == [1, 2, 3]
        assert payload["state_at"]["_meta"]["spans_seen"] == 1

    def test_empty_trace_renders(self):
        empty = Trace(trace_id="e", ts=datetime(2026, 1, 1, tzinfo=UTC), spans=[])
        out = render_replay(empty)
        assert "(trace has no spans)" in out


@pytest.fixture()
def jsonl_file(tmp_path, trace):
    path = tmp_path / "traces.jsonl"
    path.write_text(trace.model_dump_json() + "\n", encoding="utf-8")
    return path


class TestCLI:
    def test_session_replay_text_with_at(self, jsonl_file, capsys):
        code = main(["replay", "q-1042", "--file", str(jsonl_file), "--at", "2"])
        assert code == 0
        out = capsys.readouterr().out
        assert "Session replay: q-1042" in out
        assert "state at T2" in out

    def test_session_replay_json(self, jsonl_file, capsys):
        code = main(["replay", "q-1042", "--file", str(jsonl_file), "--json"])
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["trace_id"] == "q-1042"

    def test_session_replay_from_jsonl_store(self, jsonl_file, capsys, tmp_path):
        from forensiq.ingest.store import JSONLTraceStore

        store_dir = tmp_path / "store"
        traces = [
            Trace.model_validate(json.loads(line))
            for line in jsonl_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        JSONLTraceStore(store_dir).insert_batch(traces)
        code = main(["replay", "q-1042", "--db", str(store_dir), "--at", "1"])
        assert code == 0
        assert "state at T1" in capsys.readouterr().out

    def test_unknown_trace_id_errors(self, jsonl_file):
        assert main(["replay", "ghost", "--file", str(jsonl_file)]) == 2

    def test_counterfactual_mode_still_works(self, tmp_path, capsys):
        """Regression guard: the pre-existing no-trace-id replay flow."""
        demo = tmp_path / "out"
        assert main(["demo", "--out", str(demo), "--seed", "7"]) == 0
        code = main(["replay", "--file", str(demo / "traces.jsonl"), "--taxonomy", "F-RET-002"])
        assert code == 0
        assert "replayed" in capsys.readouterr().out
