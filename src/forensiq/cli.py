"""ForensiQ CLI.

    forensiq demo                              # synthetic corpus + full RCA into ./forensiq-output/
    forensiq ingest --file traces.jsonl        # ingest + quick stats
    forensiq ingest --langfuse                 # REST pull (needs LANGFUSE_* env)
    forensiq analyze --file traces.jsonl       # classify + stage health to stdout
    forensiq report --file traces.jsonl --out rca.md
    forensiq watch --file traces.jsonl --baseline-days 14   # drift alerts (JSON)
    forensiq replay <trace_id> --file traces.jsonl --at 4   # session step-through
    forensiq replay --file traces.jsonl --taxonomy F-RET-002  # counterfactual replay
    forensiq serve                             # OTLP receiver (needs [serve] extra)

Everything runs offline; --langfuse, the LLM second pass and the OTLP receiver
extras are env-guarded (see .env.example).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

from forensiq import __version__
from forensiq.attribution.blame import BlameRanker
from forensiq.attribution.health import StageHealth
from forensiq.attribution.replay import DEFAULT_MUTATIONS, SimulatedAdapter, WhatIfReplay
from forensiq.config import load_settings
from forensiq.ingest.loaders import (
    IngestError,
    JSONLTraceLoader,
    JSONTraceLoader,
    LangfuseTraceLoader,
)
from forensiq.ingest.synthetic import SyntheticTraceGenerator
from forensiq.llm import EchoMockClient, OpenAICompatClient
from forensiq.patterns.cluster import FailureClusterer
from forensiq.patterns.drift import DriftDetector
from forensiq.replay import render_replay, replay_to_json
from forensiq.report.rca import RCAReportGenerator, ReportInputs
from forensiq.taxonomy.classifier import FailureClassifier


def _load_traces(file: str | None, langfuse: bool) -> tuple[list, list[str]]:
    if langfuse:
        loader = LangfuseTraceLoader()  # raises IngestError without env keys
        return loader.fetch(), []
    if not file:
        raise IngestError("provide --file PATH or --langfuse")
    path = Path(file)
    if not path.exists():
        raise IngestError(f"file not found: {path}")
    loader: JSONLTraceLoader | JSONTraceLoader = (
        JSONLTraceLoader() if path.suffix.lower() in (".jsonl", ".ndjson") else JSONTraceLoader()
    )
    return loader.load(path), loader.warnings


def _load_trace_from_store(db: str, trace_id: str):
    """Session-replay lookup from a store file (.jsonl → pure-python store,
    .duckdb/.ddb → DuckDBTraceStore; duckdb missing → actionable error)."""
    path = Path(db)
    if path.suffix.lower() in (".duckdb", ".ddb"):
        from forensiq.ingest.duckdb_store import DuckDBTraceStore

        store = DuckDBTraceStore(path)
    else:
        from forensiq.ingest.store import JSONLTraceStore

        store = JSONLTraceStore(path)
    trace = store.get_trace(trace_id)
    if trace is None:
        raise IngestError(f"trace {trace_id!r} not found in store {path}")
    return trace


def _analyze(traces, settings) -> ReportInputs:
    llm = None
    if settings.llm_enabled and settings.llm_api_key:
        llm = OpenAICompatClient(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )
    else:
        llm = EchoMockClient()  # offline fallback for the ambiguous pass
    classifier = FailureClassifier(
        llm=llm,
        low_score_threshold=settings.low_score_threshold,
        context_limit=settings.context_limit,
        repetition_min_repeat=settings.repetition_min_repeat,
        cost_spike_factor=settings.cost_spike_factor,
    )
    failures = classifier.classify_cohort(traces)
    health = StageHealth().score(traces)
    ranker = BlameRanker.from_traces(traces, failures)
    blame_distribution = ranker.cohort_blame(traces, failures)
    clusters = FailureClusterer().fit(traces, failures)
    drift = DriftDetector(
        alpha=settings.alpha,
        baseline_days=settings.baseline_days,
        window_days=settings.window_days,
    ).detect(traces, failures)
    return ReportInputs(
        failures=failures,
        health=health,
        blame_distribution=blame_distribution,
        clusters=clusters,
        drift=drift,
    )


def cmd_demo(args: argparse.Namespace) -> int:
    settings = load_settings()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gen = SyntheticTraceGenerator(seed=args.seed)
    traces, planted = gen.generate()
    traces_file = gen.write_jsonl(out / "traces.jsonl", traces)
    inputs = _analyze(traces, settings)
    report = RCAReportGenerator.with_yaml("docs/playbooks.yaml").generate(
        traces, inputs, title="ForensiQ demo — acme-support-rag"
    )
    (out / "rca.md").write_text(report, encoding="utf-8")
    (out / "failures.json").write_text(
        json.dumps([f.model_dump() for f in inputs.failures], indent=2, default=str),
        encoding="utf-8",
    )
    (out / "planted_ground_truth.json").write_text(
        json.dumps([p.model_dump() for p in planted], indent=2), encoding="utf-8"
    )
    if inputs.drift is not None:
        (out / "drift_alerts.json").write_text(inputs.drift.to_json(), encoding="utf-8")
    counts = Counter(f.taxonomy_id for f in inputs.failures)
    print(f"synthetic corpus : {len(traces)} traces -> {traces_file}")
    print(f"classified       : {len(inputs.failures)} failures ({dict(sorted(counts.items()))})")
    print(f"drift alerts     : {len(inputs.drift.alerts) if inputs.drift else 0}")
    print(f"report           : {out / 'rca.md'}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    traces, warnings = _load_traces(args.file, args.langfuse)
    stages = Counter(s.stage for t in traces for s in t.spans)
    total_cost = sum(t.total_cost for t in traces)
    print(f"ingested {len(traces)} traces")
    print(f"stage spans: {dict(sorted(stages.items()))}")
    print(f"total cost : {total_cost:.4f}")
    if warnings:
        print(f"warnings   : {len(warnings)} (first: {warnings[0]})")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    settings = load_settings()
    traces, warnings = _load_traces(args.file, args.langfuse)
    inputs = _analyze(traces, settings)
    counts = Counter(f.taxonomy_id for f in inputs.failures)
    print(f"traces: {len(traces)}  failures: {len(inputs.failures)}")
    for tax_id, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {tax_id:<13} x{n}")
    for card in inputs.health.stages:
        flags = (" -- " + "; ".join(card.flags)) if card.flags else ""
        print(f"  [{card.status.upper():<4}] {card.stage:<9} {card.metrics}{flags}")
    if warnings:
        print(f"ingest warnings: {len(warnings)}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    settings = load_settings()
    traces, _ = _load_traces(args.file, args.langfuse)
    inputs = _analyze(traces, settings)
    report = RCAReportGenerator.with_yaml("docs/playbooks.yaml").generate(traces, inputs)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"wrote {args.out} ({len(inputs.failures)} failures over {len(traces)} traces)")
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    settings = load_settings()
    traces, _ = _load_traces(args.file, args.langfuse)
    classifier = FailureClassifier(
        low_score_threshold=settings.low_score_threshold,
        cost_spike_factor=settings.cost_spike_factor,
    )
    failures = classifier.classify_cohort(traces)
    detector = DriftDetector(
        alpha=settings.alpha,
        baseline_days=args.baseline_days,
        window_days=settings.window_days,
    )
    report = detector.detect(traces, failures)
    print(report.to_json())
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    if args.trace_id:
        return cmd_session_replay(args)
    return cmd_counterfactual_replay(args)


def cmd_session_replay(args: argparse.Namespace) -> int:
    """Step-through session replay of one recorded trace."""
    if args.db:
        trace = _load_trace_from_store(args.db, args.trace_id)
    else:
        traces, _ = _load_traces(args.file, args.langfuse)
        trace = next((t for t in traces if t.trace_id == args.trace_id), None)
        if trace is None:
            raise IngestError(f"trace {args.trace_id!r} not found in the given source")
    if args.json:
        print(replay_to_json(trace, at=args.at))
    else:
        print(render_replay(trace, at=args.at))
    return 0


def cmd_counterfactual_replay(args: argparse.Namespace) -> int:
    settings = load_settings()
    traces, _ = _load_traces(args.file, args.langfuse)
    classifier = FailureClassifier(
        low_score_threshold=settings.low_score_threshold,
        cost_spike_factor=settings.cost_spike_factor,
    )
    failures = classifier.classify_cohort(traces)
    if not failures:
        print("no failures to replay")
        return 0
    sample = next(f for f in failures if f.taxonomy_id == args.taxonomy) if args.taxonomy else failures[0]
    trace = next(t for t in traces if t.trace_id == sample.trace_id)
    replayer = WhatIfReplay(adapter=SimulatedAdapter(), classifier=classifier)
    result = asyncio.run(replayer.replay(trace, DEFAULT_MUTATIONS, failure=sample))
    print(f"trace {result.trace_id} ({result.original_taxonomy_id}) replayed:")
    for attempt in result.attempts:
        lift = ", ".join(f"{k}{v:+.2f}" for k, v in attempt.predicted_lift.items())
        print(f"  {attempt.mutation:<42} resolved={attempt.resolved} lift[{lift}]")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from forensiq.serve import main_serve

    return main_serve(host=args.host, port=args.port, buffer_dir=args.buffer_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forensiq", description="Failure forensics for AI pipelines."
    )
    parser.add_argument("--version", action="version", version=f"forensiq {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="synthetic corpus + full RCA into ./forensiq-output/")
    demo.add_argument("--out", default="forensiq-output")
    demo.add_argument("--seed", type=int, default=42)
    demo.set_defaults(func=cmd_demo)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--file", help="trace file (.jsonl or .json)")
        p.add_argument("--langfuse", action="store_true", help="pull traces from Langfuse REST (env-guarded)")

    ingest = sub.add_parser("ingest", help="load traces and print quick stats")
    add_common(ingest)
    ingest.set_defaults(func=cmd_ingest)

    analyze = sub.add_parser("analyze", help="classify failures + stage health")
    add_common(analyze)
    analyze.set_defaults(func=cmd_analyze)

    report = sub.add_parser("report", help="generate an RCA markdown report")
    add_common(report)
    report.add_argument("--out", default="rca.md")
    report.set_defaults(func=cmd_report)

    watch = sub.add_parser("watch", help="drift watch mode (JSON alerts)")
    add_common(watch)
    watch.add_argument("--baseline-days", type=int, default=14)
    watch.set_defaults(func=cmd_watch)

    replay = sub.add_parser(
        "replay",
        help="session step-through of one trace (with TRACE_ID) or counterfactual replay of one failure (without)",
    )
    replay.add_argument("trace_id", nargs="?", default=None, help="trace id to step through (session replay)")
    add_common(replay)
    replay.add_argument("--db", help="store file to look up the trace (.jsonl store or .duckdb/.ddb)")
    replay.add_argument("--at", type=int, default=None, help="session replay: show cumulative state at step Tn")
    replay.add_argument("--json", action="store_true", help="session replay: emit JSON instead of text")
    replay.add_argument("--taxonomy", help="counterfactual: replay the first failure of this taxonomy id")
    replay.set_defaults(func=cmd_replay)

    serve = sub.add_parser("serve", help="run the OTLP trace receiver (needs 'forensiq[serve]')")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=4318)
    serve.add_argument("--buffer-dir", default="forensiq-buffer")
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
