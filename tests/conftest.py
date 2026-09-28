"""Shared fixtures: the seeded synthetic corpus + its classified failures.

Everything is offline and deterministic (seed 42); heavy artifacts are built
once per session and shared across test modules.
"""

from __future__ import annotations

import pytest

from forensiq.attribution.blame import BlameRanker
from forensiq.ingest.synthetic import SyntheticTraceGenerator
from forensiq.taxonomy.classifier import FailureClassifier


@pytest.fixture(scope="session")
def corpus():
    """(traces, planted ground truth) for the default seed."""
    return SyntheticTraceGenerator(seed=42).generate()


@pytest.fixture(scope="session")
def failures(corpus):
    """Pass-1 classification of the corpus (primary failure per trace)."""
    traces, _ = corpus
    return FailureClassifier().classify_cohort(traces)


@pytest.fixture(scope="session")
def ranker(corpus, failures):
    traces, planted = corpus
    return BlameRanker.from_traces(traces, failures)


@pytest.fixture(scope="session")
def planted_by_id(corpus):
    _, planted = corpus
    return {p.trace_id: p for p in planted}
