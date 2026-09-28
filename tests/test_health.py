"""Stage health: groundedness heuristic (anchored tokens, negation guard),
stage-conditional cards, latency percentiles."""

from __future__ import annotations

from datetime import UTC

from forensiq.attribution.health import StageHealth, groundedness, trace_health

GROUND_CTX = (
    "Nordwerk Precision GmbH supplies component XK-7. The SLA allows 8 hours maximum response."
)


class TestGroundedness:
    def test_fully_grounded_scores_high(self):
        out = "Nordwerk Precision GmbH supplies XK-7. The SLA allows 8 hours maximum response."
        assert groundedness(out, GROUND_CTX) >= 0.8

    def test_fabricated_number_penalizes_double(self):
        grounded = "Nordwerk Precision GmbH supplies XK-7 within 8 hours."
        fabricated = "Vertex Dynamics supplies XK-7 within 3 hours."
        assert groundedness(grounded, GROUND_CTX) > groundedness(fabricated, GROUND_CTX) + 0.4

    def test_negation_guard(self):
        """Answer denying what the context affirms must score below the
        affirming version (mirrors RAG_showcase's polarity guard)."""
        ctx = "The supplier delivers within 8 hours. Component XK-7 is available."
        affirming = "The supplier delivers within 8 hours and the component is available."
        denying = "The supplier does not deliver within 8 hours and the component is not available."
        assert groundedness(denying, ctx) < groundedness(affirming, ctx)

    def test_empty_output_or_context(self):
        assert groundedness("", GROUND_CTX) == 0.0
        assert groundedness("some answer", "") == 0.0

    def test_improvement_over_plain_overlap_anchored_tokens(self):
        """The whole point vs RAG_showcase's lexical faithfulness: a sentence
        with high plain overlap but ONE fabricated figure must sink."""
        ctx = "Nordwerk Precision GmbH supplies component XK-7. The SLA allows 8 hours."
        plain_overlap_wrong_number = "Nordwerk Precision GmbH supplies component XK-7 within 24 hours."
        # plain overlap would be generous (7/8 content tokens shared); the
        # anchored penalty for the fabricated '24' drags the score to <= 0.8.
        assert groundedness(plain_overlap_wrong_number, ctx) < 0.8
        # and the fully accurate paraphrase stays high
        assert groundedness("Nordwerk Precision GmbH supplies component XK-7 within 8 hours.", ctx) > 0.85


class TestStageCards:
    def test_cards_for_all_present_stages(self, corpus):
        traces, _ = corpus
        report = StageHealth().score(traces)
        stages = {c.stage for c in report.stages}
        assert {"embed", "retrieve", "generate"} <= stages
        assert report.trace_count == len(traces)

    def test_retrieval_card_metrics(self, corpus):
        traces, _ = corpus
        card = StageHealth().score(traces).get("retrieve")
        assert card is not None
        assert 0.0 < card.metrics["hit_rate"] <= 1.0
        assert 0.0 < card.metrics["mean_hit_score"] < 1.0
        assert card.metrics["p95_ms"] >= card.metrics["p50_ms"]

    def test_generation_groundedness_drops_on_weak_context(self, corpus):
        """Low-recall traces put filler context under a normal answer, so the
        cohort's groundedness must sit clearly below a healthy corpus's."""
        traces, _ = corpus
        card = StageHealth().score(traces).get("generate")
        assert card.metrics["groundedness_mean"] < 0.75  # ~45% of traces are degraded
        assert card.status in ("warn", "crit")

    def test_healthy_corpus_scores_ok(self):
        from datetime import datetime

        from forensiq.ingest.schema import Span, Trace

        def mk(i: int) -> Trace:
            ctx = "Nordwerk Precision GmbH supplies component XK-7 within 8 hours."
            return Trace(
                trace_id=f"t{i}",
                ts=datetime(2026, 8, 1, tzinfo=UTC),
                spans=[
                    Span(span_id=f"e{i}", name="embed", stage="embed", duration_ms=10),
                    Span(
                        span_id=f"r{i}",
                        name="retrieve",
                        stage="retrieve",
                        duration_ms=40,
                        attrs={"top_k": 5, "hit_count": 5, "hit_scores": [0.9, 0.88, 0.86, 0.84, 0.82]},
                    ),
                    Span(
                        span_id=f"g{i}",
                        name="generate",
                        stage="generate",
                        duration_ms=900,
                        attrs={"output": "Nordwerk supplies XK-7 within 8 hours.", "context": ctx},
                    ),
                ],
            )

        report = StageHealth().score([mk(i) for i in range(10)])
        assert report.get("retrieve").status == "ok"
        assert report.get("retrieve").metrics["mean_hit_score"] > 0.8
        assert report.get("generate").status == "ok"
        assert report.get("generate").metrics["groundedness_mean"] > 0.8

    def test_trace_health_snapshot(self, corpus):
        traces, _ = corpus
        snap = trace_health(traces[0])
        assert "mean_hit_score" in snap
        assert "groundedness" in snap
