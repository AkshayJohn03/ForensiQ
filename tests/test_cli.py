"""CLI: demo end-to-end into a temp dir, ingest/analyze/report/watch flows,
and error paths. Offline; called in-process via cli.main()."""

from __future__ import annotations

import json

import pytest

from forensiq.cli import main


@pytest.fixture()
def demo_out(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    code = main(["demo", "--out", str(out), "--seed", "7"])
    assert code == 0
    return out


class TestDemo:
    def test_writes_artifacts(self, demo_out):
        for name in ("traces.jsonl", "rca.md", "failures.json", "planted_ground_truth.json", "drift_alerts.json"):
            assert (demo_out / name).exists(), name

    def test_report_has_content(self, demo_out):
        report = (demo_out / "rca.md").read_text(encoding="utf-8")
        assert "graph TD" in report and "timeline" in report
        assert "F-RET-" in report

    def test_failures_json_parses(self, demo_out):
        failures = json.loads((demo_out / "failures.json").read_text(encoding="utf-8"))
        assert failures
        assert {"taxonomy_id", "stage", "trace_id", "confidence", "evidence"} <= set(failures[0])

    def test_drift_alerts_json(self, demo_out):
        payload = json.loads((demo_out / "drift_alerts.json").read_text(encoding="utf-8"))
        assert isinstance(payload["alerts"], list)


class TestIngestAndAnalyze:
    def test_ingest_jsonl(self, demo_out, capsys):
        code = main(["ingest", "--file", str(demo_out / "traces.jsonl")])
        assert code == 0
        out = capsys.readouterr().out
        assert "ingested" in out and "traces" in out

    def test_analyze_prints_failures_and_health(self, demo_out, capsys):
        code = main(["analyze", "--file", str(demo_out / "traces.jsonl")])
        assert code == 0
        out = capsys.readouterr().out
        assert "F-RET-002" in out
        assert "[WARN" in out or "[CRIT" in out or "[OK" in out

    def test_report_writes_file(self, demo_out, tmp_path):
        target = tmp_path / "rca-custom.md"
        code = main(["report", "--file", str(demo_out / "traces.jsonl"), "--out", str(target)])
        assert code == 0
        assert "graph TD" in target.read_text(encoding="utf-8")

    def test_watch_prints_json_alerts(self, demo_out, capsys):
        code = main(["watch", "--file", str(demo_out / "traces.jsonl"), "--baseline-days", "14"])
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert "alerts" in payload

    def test_replay_command(self, demo_out, capsys):
        code = main(["replay", "--file", str(demo_out / "traces.jsonl"), "--taxonomy", "F-RET-002"])
        assert code == 0
        assert "replayed" in capsys.readouterr().out


class TestErrorPaths:
    def test_missing_file(self, tmp_path, capsys):
        assert main(["ingest", "--file", str(tmp_path / "nope.jsonl")]) != 0

    def test_langfuse_without_env_is_rejected(self, monkeypatch, tmp_path):
        for var in ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
            monkeypatch.delenv(var, raising=False)
        assert main(["ingest", "--langfuse"]) == 2

    def test_no_source_given(self, tmp_path):
        assert main(["ingest"]) == 2
