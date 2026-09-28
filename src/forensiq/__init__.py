"""ForensiQ — failure forensics for AI pipelines.

Ingest traces (JSONL / JSON / Langfuse REST), classify failures against a
staged taxonomy, attribute root causes per pipeline stage, mine failure
families and drift, and render RCA reports. Runs fully offline; LLM and
Langfuse access are optional and env-guarded.
"""

from __future__ import annotations

from forensiq.ingest.loaders import (
    IngestError,
    JSONLTraceLoader,
    JSONTraceLoader,
    LangfuseTraceLoader,
    spans_from_dicts,
)
from forensiq.ingest.schema import Span, Trace
from forensiq.taxonomy.classifier import FailureClassifier, FailureRecord

__version__ = "0.1.0"

__all__ = [
    "FailureClassifier",
    "FailureRecord",
    "IngestError",
    "JSONLTraceLoader",
    "JSONTraceLoader",
    "LangfuseTraceLoader",
    "Span",
    "Trace",
    "__version__",
    "spans_from_dicts",
]
