"""RCA report: deterministic rendering, mermaid blocks, playbook mapping,
postmortem + ticket exports."""

from __future__ import annotations

import pytest

from forensiq.attribution.blame import BlameRanker
from forensiq.attribution.health import StageHealth
from forensiq.ingest.synthetic import SyntheticTraceGenerator
from forensiq.patterns.cluster import FailureClusterer
from forensiq.patterns.drift import DriftDetector
from forensiq.report.rca import RCAReportGenerator, ReportInputs
from forensiq.taxonomy.classifier import FailureClassifier


@pytest.fixture(scope="module")
def analysis():
    traces, planted = SyntheticTraceGenerator(seed=42).generate()
    failures = FailureClassifier().classify_cohort(traces)
    health = StageHealth().score(traces)
    ranker = BlameRanker.from_traces(traces, failures)
    blame = ranker.cohort_blame(traces, failures)
    clusters = FailureClusterer().fit(traces, failures)
    drift = DriftDetector().detect(traces, failures)
    inputs = ReportInputs(
        failures=failures, health=health, blame_distribution=blame, clusters=clusters, drift=drift
    )
    return traces, planted, failures, inputs


class TestReportRendering:
    def test_contains_all_key_sections(self, analysis):
        traces, _, _, inputs = analysis
        report = RCAReportGenerator().generate(traces, inputs)
        for section in (
            "# ForensiQ RCA Report",
            "## Executive summary",
            "## Failure mix",
            "## Top root causes (blame-ranked)",
            "## Blame graph",
            "## Failure timeline",
            "## Stage health",
            "## Failure families (clusters)",
            "## Drift",
            "## Mapped fixes (playbook)",
        ):
            assert section in report, section

    def test_renders_mermaid_blocks(self, analysis):
        traces, _, _, inputs = analysis
        report = RCAReportGenerator().generate(traces, inputs)
        assert "```mermaid" in report
        assert "graph TD" in report
        assert "timeline" in report
        # blame graph nodes carry shares
        assert "retrieve:" in report

    def test_deterministic_output(self, analysis):
        traces, _, _, inputs = analysis
        r1 = RCAReportGenerator().generate(traces, inputs)
        r2 = RCAReportGenerator().generate(traces, inputs)
        assert r1 == r2

    def test_failure_mix_and_share(self, analysis):
        traces, _, failures, inputs = analysis
        report = RCAReportGenerator().generate(traces, inputs)
        assert f"{len(failures)} failures across {len(traces)} traces" in report
        assert "F-RET-002" in report and "F-RET-001" in report

    def test_evidence_quotes_present(self, analysis):
        traces, _, _, inputs = analysis
        report = RCAReportGenerator().generate(traces, inputs)
        assert "Representative evidence" in report
        assert "- `" in report  # span-ref quoting

    def test_playbook_mapping(self, analysis):
        traces, _, _, inputs = analysis
        report = RCAReportGenerator().generate(traces, inputs)
        assert "Check index freshness" in report  # F-RET-002 playbook line
        assert "no playbook entry" not in report  # every observed id is mapped

    def test_yaml_override_changes_fixes(self, analysis, tmp_path):
        traces, _, _, inputs = analysis
        yaml_path = tmp_path / "playbooks.yaml"
        yaml_path.write_text("playbooks:\n  F-RET-001: 'Custom runbook step.'\n", encoding="utf-8")
        report = RCAReportGenerator.with_yaml(yaml_path).generate(traces, inputs)
        assert "Custom runbook step." in report

    def test_drift_alerts_appear_in_report(self, analysis):
        traces, _, _, inputs = analysis
        report = RCAReportGenerator().generate(traces, inputs)
        assert "Drift alert" in report


class TestExports:
    def test_postmortem_template(self, analysis):
        traces, _, failures, _ = analysis
        gen = RCAReportGenerator()
        trace = next(t for t in traces if t.trace_id == failures[0].trace_id)
        text = gen.postmortem(trace, failures[0])
        for header in ("# Postmortem template", "## Evidence", "## Root cause", "## Action items"):
            assert header in text
        assert failures[0].taxonomy_id in text
        assert trace.trace_id in text

    def test_ticket_text(self, analysis):
        traces, _, failures, _ = analysis
        gen = RCAReportGenerator()
        trace = next(t for t in traces if t.trace_id == failures[0].trace_id)
        ticket = gen.ticket(failures[0], trace)
        assert ticket.startswith(f"[{failures[0].taxonomy_id}]")
        assert "trace_id:" in ticket
        assert "evidence:" in ticket
        assert "suggested_fix:" in ticket
