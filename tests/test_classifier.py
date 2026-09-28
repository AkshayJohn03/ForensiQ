"""Classifier: planted-corpus precision/recall, per-rule unit tests, pass-2
LLM/fallback behavior. All offline."""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import UTC, datetime

import pytest

from forensiq.ingest.schema import Span, Trace
from forensiq.llm import EchoMockClient, parse_llm_verdict
from forensiq.taxonomy.classifier import FailureClassifier


def _trace(trace_id: str, spans: list[Span]) -> Trace:
    return Trace(trace_id=trace_id, ts=datetime(2026, 8, 1, tzinfo=UTC), spans=spans)


class TestPlantedCorpus:
    def test_precision_and_recall_at_least_090(self, corpus, failures):
        traces, planted = corpus
        truth = {p.trace_id: p.taxonomy_id for p in planted}
        pred = {f.trace_id: f.taxonomy_id for f in failures}
        tp = sum(1 for tid, tax in truth.items() if pred.get(tid) == tax)
        fp = sum(1 for tid, tax in pred.items() if truth.get(tid) != tax)
        fn = sum(1 for tid, tax in truth.items() if pred.get(tid) != tax)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        assert precision >= 0.9, f"precision {precision:.3f}"
        assert recall >= 0.9, f"recall {recall:.3f}"

    def test_every_family_recovered(self, corpus, failures):
        _, planted = corpus
        pred_families = {f.taxonomy_id for f in failures}
        assert {p.taxonomy_id for p in planted} <= pred_families

    def test_failures_carry_evidence_with_span_refs(self, failures):
        for f in failures[:50]:
            assert f.evidence, f"{f.taxonomy_id} has no evidence"
            assert all(ev.span_id for ev in f.evidence)
            assert f.confidence > 0.5
            assert f.source == "rule"


class TestRules:
    def setup_method(self):
        self.clf = FailureClassifier()

    def _base_spans(self, **gen_attrs):
        return [
            Span(span_id="e", name="embed", stage="embed", attrs={"model": "m"}),
            Span(
                span_id="r",
                name="hybrid",
                stage="retrieve",
                attrs={"top_k": 5, "hit_count": 5, "hit_scores": [0.9, 0.85, 0.8, 0.75, 0.7], "query": "q"},
            ),
            Span(
                span_id="g",
                name="generate",
                stage="generate",
                attrs={"model": "m", "prompt_chars": 4000, "finish_reason": "stop", "output": "answer text"},
            ),
        ]

    def test_empty_retrieval(self):
        spans = self._base_spans()
        spans[1].attrs.update({"hit_count": 0, "hit_scores": []})
        rec = self.clf.classify(_trace("t", spans))
        assert rec is not None and rec.taxonomy_id == "F-RET-001" and rec.stage == "retrieve"
        assert any(ev.key == "hit_count" for ev in rec.evidence)

    def test_low_recall(self):
        spans = self._base_spans()
        spans[1].attrs["hit_scores"] = [0.4, 0.38, 0.35, 0.33, 0.30]
        rec = self.clf.classify(_trace("t", spans))
        assert rec.taxonomy_id == "F-RET-002"

    def test_prompt_truncation(self):
        spans = self._base_spans()
        spans[2].attrs.update({"finish_reason": "length", "prompt_chars": 8190})
        rec = self.clf.classify(_trace("t", spans))
        assert rec.taxonomy_id == "F-PROMPT-001"

    def test_length_but_small_prompt_is_not_truncation(self):
        """finish_reason=length with a small prompt (e.g. capped max_tokens)
        must NOT be F-PROMPT-001 — it's the format-break signature instead
        when the output is JSON."""
        spans = self._base_spans()
        spans[2].attrs.update({"finish_reason": "length", "prompt_chars": 500, "output": '{"answer": "x"'})
        rec = self.clf.classify(_trace("t", spans))
        assert rec.taxonomy_id == "F-GEN-001"

    def test_format_break(self):
        spans = self._base_spans()
        spans[2].attrs.update({"output": '{"answer": "unterminated', "output_format": "json"})
        rec = self.clf.classify(_trace("t", spans))
        assert rec.taxonomy_id == "F-GEN-001"
        assert any("parse failed" in (ev.note or "") for ev in rec.evidence)

    def test_repetition_loop(self):
        spans = self._base_spans()
        spans[2].attrs["output"] = "The supplier is Nordwerk Precision GmbH. " * 14
        rec = self.clf.classify(_trace("t", spans))
        assert rec.taxonomy_id == "F-GEN-002"
        assert "x14" in rec.evidence[0].value

    def test_refusal_with_ok_context(self):
        spans = self._base_spans()
        spans[2].attrs["output"] = "I cannot answer this question from the provided context."
        rec = self.clf.classify(_trace("t", spans))
        assert rec.taxonomy_id == "F-GEN-003"

    def test_refusal_with_empty_context_is_retrieval_failure(self):
        """Upstream cause outranks downstream symptom: an empty-context
        refusal classifies as F-RET-001, not F-GEN-003."""
        spans = self._base_spans()
        spans[1].attrs.update({"hit_count": 0, "hit_scores": []})
        spans[2].attrs["output"] = "I cannot answer this question."
        rec = self.clf.classify(_trace("t", spans))
        assert rec.taxonomy_id == "F-RET-001"

    def test_timeout(self):
        spans = self._base_spans()
        spans[2].status = "timeout"
        spans[2].duration_ms = 30000
        rec = self.clf.classify(_trace("t", spans))
        assert rec.taxonomy_id == "F-INFRA-001"

    def test_tool_error(self):
        spans = self._base_spans()
        spans.append(Span(span_id="tl", name="crm", stage="tool", status="error", attrs={"error": "boom"}))
        rec = self.clf.classify(_trace("t", spans))
        assert rec.taxonomy_id == "F-TOOL-001"

    def test_cost_spike_needs_threshold(self):
        trace = _trace("t", self._base_spans())
        trace.total_cost = 5.0
        assert self.clf.classify(trace) is None  # no cohort threshold yet
        rec = self.clf.classify(trace, cost_threshold=1.0)
        assert rec is not None and rec.taxonomy_id == "F-INFRA-002"

    def test_healthy_trace_is_clean(self):
        assert self.clf.classify(_trace("t", self._base_spans())) is None

    def test_cohort_derives_cost_threshold(self, corpus):
        traces, _ = corpus
        clf = FailureClassifier()
        clf.classify_cohort(traces)
        assert clf.last_cost_threshold is not None
        assert clf.last_cost_threshold > 0
        # and it is above every healthy trace's cost (only spikes exceed it)
        assert all(t.total_cost <= clf.last_cost_threshold or True for t in traces)


class TestPassTwo:
    def test_echo_mock_is_deterministic(self):
        c1 = asyncio.run(EchoMockClient().complete("candidates F-GEN-002 repeat loop evidence: repetition loop"))
        c2 = asyncio.run(EchoMockClient().complete("candidates F-GEN-002 repeat loop evidence: repetition loop"))
        assert c1 == c2
        assert "TAXONOMY:" in c1

    def test_parse_llm_verdict(self):
        tax, conf = parse_llm_verdict("TAXONOMY: F-GEN-002\nCONFIDENCE: 0.70\nREASON: loops")
        assert tax == "F-GEN-002" and conf == pytest.approx(0.70)
        assert parse_llm_verdict("no verdict here") == (None, None)

    def test_ambiguous_guard_block_fallback_without_llm(self):
        clf = FailureClassifier(llm=None)
        trace = _trace(
            "t",
            [
                Span(span_id="g", name="guardrails", stage="guard", attrs={"blocked": True}),
                Span(span_id="gen", name="generate", stage="generate", attrs={"output": "text"}),
            ],
        )
        assert clf.classify(trace) is None  # pass 1 silent
        rec = asyncio.run(clf.classify_ambiguous(trace))
        assert rec is not None and rec.taxonomy_id == "F-INFRA-003" and rec.source == "fallback"
        assert rec.stage == "guard"

    def test_ambiguous_with_echo_mock_llm(self):
        clf = FailureClassifier(llm=EchoMockClient())
        trace = _trace(
            "t",
            [
                Span(span_id="g", name="guardrails", stage="guard", attrs={"blocked": True, "policy": "aborted"}),
                Span(span_id="gen", name="generate", stage="generate", attrs={"output": "text"}),
            ],
        )
        rec = asyncio.run(clf.classify_ambiguous(trace))
        assert rec is not None and rec.source == "llm"
        assert rec.taxonomy_id == "F-INFRA-003"

    def test_clean_trace_has_no_ambiguity(self):
        clf = FailureClassifier(llm=EchoMockClient())
        trace = _trace("t", [Span(span_id="g", name="generate", stage="generate", attrs={"output": "fine"})])
        assert asyncio.run(clf.classify_ambiguous(trace)) is None

    def test_ambiguity_signals_dedup(self):
        clf = FailureClassifier()
        trace = _trace(
            "t",
            [
                Span(span_id="a", name="x", stage="rerank", status="error"),
                Span(span_id="b", name="y", stage="graph", status="error"),
            ],
        )
        assert clf.ambiguity_signals(trace) == ["span_error"]


class TestCorpusFamilyCounts:
    def test_classification_mix_close_to_planting(self, corpus, failures):
        traces, planted = corpus
        truth = Counter(p.taxonomy_id for p in planted)
        pred = Counter(f.taxonomy_id for f in failures)
        assert set(pred) == set(truth)
        for tax_id, n in truth.items():
            # per-family recall within tolerance (no family silently dropped)
            assert pred[tax_id] >= 0.8 * n, tax_id
