"""Incident postmortem: causal-chain sections (trigger/poisoning/validator/
blast radius), mermaid flow, ticket export, determinism. Planted-failure
fixture, fully offline."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from forensiq.ingest.schema import Span, Trace
from forensiq.report.rca import RCAReportGenerator
from forensiq.taxonomy.classifier import FailureClassifier


@pytest.fixture(scope="module")
def planted():
    """A trace with planted weak retrieval + ungrounded output (no guard)."""
    trace = Trace(
        trace_id="q-1042",
        ts=datetime(2026, 8, 14, 9, 12, tzinfo=UTC),
        pipeline_name="rag_showcase",
        spans=[
            Span(
                span_id="s1",
                name="hybrid",
                stage="retrieve",
                duration_ms=44,
                attrs={
                    "query": "What is the SAF-114 maintenance interval?",
                    "top_k": 5,
                    "hit_count": 5,
                    "hit_scores": [0.31, 0.28, 0.22, 0.19, 0.11],
                },
            ),
            Span(
                span_id="s2",
                parent_id="s1",
                name="rerank",
                stage="rerank",
                duration_ms=30,
                attrs={"rerank_delta": 0.02},
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
                attrs={
                    "model": "qwen3:1.7b",
                    "prompt_chars": 4800,
                    "finish_reason": "stop",
                    "output": "The SAF-114 requires maintenance every 3 days per the XK-7 handbook.",
                    "context": (
                        "SAF-114 is a hydraulic fluid for the XK-7 press line. "
                        "Helios 4.2 controllers require SAF-114 grade A."
                    ),
                },
            ),
        ],
        total_cost=0.002,
    )
    failure = FailureClassifier().classify(trace)
    assert failure is not None and failure.taxonomy_id == "F-RET-002"
    return trace, failure


@pytest.fixture(scope="module")
def markdown(planted):
    trace, failure = planted
    return RCAReportGenerator().generate_postmortem(trace, [failure])


class TestCausalChainSections:
    def test_has_causal_chain_section(self, markdown):
        assert "## CAUSAL CHAIN" in markdown

    def test_trigger_section_quotes_user_prompt(self, markdown):
        assert "### 1. Trigger (user prompt)" in markdown
        assert "SAF-114 maintenance interval" in markdown
        assert "`s1` (retrieve)" in markdown

    def test_poisoning_section_identifies_weak_chunks(self, markdown):
        assert "### 2. Poisoning" in markdown
        assert "Chunks delivered: 5" in markdown
        assert "Weakest chunk score: **0.110**" in markdown
        assert "prime suspect" in markdown

    def test_divergence_section_quotes_output(self, markdown):
        assert "### 3. Divergence" in markdown
        assert "finish_reason: `stop`" in markdown
        assert "The SAF-114 requires maintenance every 3 days" in markdown

    def test_validator_section_names_missing_guard(self, markdown):
        assert "### 4. Validator failure" in markdown
        assert "retrieval score SLO" in markdown  # the guard that should exist
        assert "no guard stage span in the trace" in markdown  # why it didn't fire

    def test_blast_radius_section(self, markdown):
        assert "### 5. Blast radius" in markdown
        assert "generate, rerank" in markdown  # downstream stages
        assert "User-facing: YES" in markdown


class TestPostmortemRendering:
    def test_mermaid_flow_present(self, markdown):
        assert "```mermaid" in markdown
        assert "T[\"TRIGGER:" in markdown
        assert "P[\"POISONING:" in markdown
        assert "V[\"VALIDATOR GAP:" in markdown
        assert "B[\"BLAST RADIUS:" in markdown
        assert "    T --> P --> D --> V --> B" in markdown

    def test_classification_and_playbook_present(self, markdown, planted):
        _, failure = planted
        assert failure.taxonomy_id in markdown
        assert "## Resolution" in markdown
        assert "Check index freshness" in markdown  # F-RET-002 playbook
        assert "## Action items" in markdown

    def test_deterministic_output(self, planted):
        trace, failure = planted
        gen = RCAReportGenerator()
        assert gen.generate_postmortem(trace, [failure]) == gen.generate_postmortem(trace, [failure])


class TestValidatorGapVariants:
    def test_guard_present_but_passing_is_coverage_gap(self, planted):
        trace, failure = planted
        trace.spans.append(
            Span(span_id="s4", parent_id="s3", name="guard", stage="guard", attrs={"blocked": False, "rules": "pii"})
        )
        markdown = RCAReportGenerator().generate_postmortem(trace, [failure])
        assert "guard ran and passed (blocked=false; rules: pii)" in markdown
        assert "rule coverage gap" in markdown

    def test_guard_blocked_notes_late_block(self, planted):
        trace, failure = planted
        trace.spans.append(Span(span_id="s5", name="guard", stage="guard", attrs={"blocked": True}))
        markdown = RCAReportGenerator().generate_postmortem(trace, [failure])
        assert "a guard DID block on this trace" in markdown

    def test_unknown_taxonomy_gets_generic_validator(self, planted):
        from forensiq.taxonomy.classifier import FailureRecord

        trace, _ = planted
        bogus = FailureRecord(
            taxonomy_id="F-XXXX-999", stage="generate", trace_id=trace.trace_id, confidence=0.5, evidence=[]
        )
        markdown = RCAReportGenerator().generate_postmortem(trace, [bogus])
        assert "a guardrail between retrieval and generation" in markdown


class TestTicketExport:
    def test_plain_text_ticket_with_causal_chain(self, planted):
        trace, failure = planted
        ticket = RCAReportGenerator().postmortem_ticket(trace, [failure])
        assert ticket.startswith("[F-RET-002]")
        assert "trace_id: q-1042" in ticket
        assert "CAUSAL CHAIN:" in ticket
        assert "TRIGGER   :" in ticket
        assert "POISONING :" in ticket
        assert "DIVERGENCE:" in ticket
        assert "VALIDATOR :" in ticket
        assert "BLAST     :" in ticket
        assert "suggested_fix:" in ticket
        assert "```" not in ticket  # plain text, no markdown fences

    def test_ticket_deterministic(self, planted):
        trace, failure = planted
        gen = RCAReportGenerator()
        assert gen.postmortem_ticket(trace, [failure]) == gen.postmortem_ticket(trace, [failure])

    def test_empty_records_rejected(self, planted):
        trace, _ = planted
        with pytest.raises(ValueError):
            RCAReportGenerator().generate_postmortem(trace, [])
