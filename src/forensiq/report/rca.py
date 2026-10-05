"""RCA report generation: markdown with mermaid, postmortem + ticket exports.

Deterministic by construction: every table/graph is sorted, the report window
comes from trace timestamps (never wall clock), and representative evidence
picks the first failure in input order — same inputs, byte-identical report.
The taxonomy catalog and playbook mapping mirror docs/TAXONOMY.md and
docs/playbooks.yaml; a YAML file can override the built-ins.
"""

from __future__ import annotations

from collections import Counter
from datetime import date as _date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from forensiq.attribution.blame import BlameCandidate
from forensiq.attribution.health import HealthReport
from forensiq.ingest.schema import Stage, Trace
from forensiq.patterns.cluster import ClusterCard
from forensiq.patterns.drift import DriftReport
from forensiq.report import causal_chain
from forensiq.report.causal_chain import clip, quote
from forensiq.taxonomy.classifier import FailureRecord

TAXONOMY: dict[str, dict[str, str]] = {
    "F-RET-001": {
        "title": "Empty retrieval set",
        "family": "F-RET",
        "severity": "high",
        "stage": "retrieve",
        "description": "Vector/BM25 search returned zero hits for a legitimate query.",
    },
    "F-RET-002": {
        "title": "Low-recall context",
        "family": "F-RET",
        "severity": "high",
        "stage": "retrieve",
        "description": "Hits returned but mean relevance score is below threshold — weak context for generation.",
    },
    "F-RET-003": {
        "title": "Retrieval score collapse (drift)",
        "family": "F-RET",
        "severity": "medium",
        "stage": "retrieve",
        "description": (
            "Gradual degradation of hit scores across days — index freshness or embedding-model "
            "drift. Detected by the drift monitor, not per-trace."
        ),
    },
    "F-PROMPT-001": {
        "title": "Prompt truncation",
        "family": "F-PROMPT",
        "severity": "high",
        "stage": "generate",
        "description": (
            "finish_reason=length with prompt_chars at the context ceiling — instructions/context "
            "silently cut."
        ),
    },
    "F-GEN-001": {
        "title": "Structured output format break",
        "family": "F-GEN",
        "severity": "medium",
        "stage": "generate",
        "description": "JSON/structured output expected but unparseable.",
    },
    "F-GEN-002": {
        "title": "Repetition loop",
        "family": "F-GEN",
        "severity": "medium",
        "stage": "generate",
        "description": "n-gram loop in the generated text — decoding parameters or a degenerate prompt state.",
    },
    "F-GEN-003": {
        "title": "Ungrounded refusal",
        "family": "F-GEN",
        "severity": "medium",
        "stage": "generate",
        "description": (
            "Model declined although usable context was retrieved — over-strict alignment or "
            "prompt confusion."
        ),
    },
    "F-GEN-004": {
        "title": "Ungrounded claim",
        "family": "F-GEN",
        "severity": "high",
        "stage": "generate",
        "description": "Groundedness heuristic far below cohort baseline — health-flagged, not rule-classified.",
    },
    "F-TOOL-001": {
        "title": "Tool execution error",
        "family": "F-TOOL",
        "severity": "high",
        "stage": "tool",
        "description": "A tool call raised inside the pipeline.",
    },
    "F-INFRA-001": {
        "title": "Stage timeout",
        "family": "F-INFRA",
        "severity": "high",
        "stage": "generate",
        "description": "A span aborted on timeout (usually the LLM call).",
    },
    "F-INFRA-002": {
        "title": "Cost spike",
        "family": "F-INFRA",
        "severity": "medium",
        "stage": "generate",
        "description": "Trace cost far above the cohort p95 — runaway prompt growth or retry storm.",
    },
    "F-INFRA-003": {
        "title": "Guard abort",
        "family": "F-INFRA",
        "severity": "low",
        "stage": "guard",
        "description": "A guardrail blocked/aborted the pipeline. Pass-2 (LLM/fallback) classified.",
    },
}

PLAYBOOKS: dict[str, str] = {
    "F-RET-001": (
        "Check index freshness and embed-model consistency; verify the query reached the right "
        "collection/namespace; add a no-hits fallback query."
    ),
    "F-RET-002": (
        "Check index freshness / embedding model drift; re-embed stale chunks; raise hybrid BM25 "
        "weight for keyword-heavy queries; raise top_k only after score review."
    ),
    "F-RET-003": (
        "Correlate score decay with deploy/embedding-model changes; schedule re-embedding; add "
        "canary queries with score SLOs."
    ),
    "F-PROMPT-001": (
        "Trim retrieved context (fewer/smaller chunks, parent-child summarization) or raise the "
        "context window; alert on prompt_chars near the ceiling."
    ),
    "F-GEN-001": (
        "Enforce structured output (JSON schema / function calling), lower temperature for "
        "extraction, add a parse-repair retry with one re-ask."
    ),
    "F-GEN-002": (
        "Lower temperature / add repetition penalty, check for duplicated context chunks feeding "
        "the loop, cap max_tokens."
    ),
    "F-GEN-003": (
        "Review the system prompt's refusal instructions; check the guard prompt for over-broad "
        "safety wording; verify citations render correctly."
    ),
    "F-GEN-004": (
        "Tighten the grounding prompt, require citations per claim, add an NLI/LLM-judge spot "
        "check on the worst offenders."
    ),
    "F-TOOL-001": (
        "Check tool dependency health and retries; add timeout + idempotency on the tool call; "
        "surface the error to the model for replanning."
    ),
    "F-INFRA-001": (
        "Raise the stage timeout or stream tokens; check provider latency SLOs; add a smaller "
        "fallback model for degradation."
    ),
    "F-INFRA-002": (
        "Cap prompt growth (context budget), check for retry loops re-sending the prompt, add "
        "per-trace cost guardrails."
    ),
    "F-INFRA-003": (
        "Review guard rules hit rate vs true positives; if blocks are wrong, tune the guard; "
        "otherwise route the user to a safe fallback."
    ),
}

STAGE_ORDER: tuple[Stage, ...] = ("embed", "retrieve", "rerank", "graph", "generate", "guard", "tool", "agent")


class ReportInputs(BaseModel):
    failures: list[FailureRecord]
    health: HealthReport
    blame_distribution: dict[str, dict[str, float]] = {}
    clusters: list[ClusterCard] = []
    drift: DriftReport | None = None


class RCAReportGenerator:
    def __init__(
        self,
        playbooks: dict[str, str] | None = None,
        taxonomy: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self.playbooks = {**PLAYBOOKS, **(playbooks or {})}
        self.taxonomy = {**TAXONOMY, **(taxonomy or {})}

    @classmethod
    def with_yaml(cls, path: str | Path) -> RCAReportGenerator:
        """Override playbooks/taxonomy from a YAML file (docs/playbooks.yaml)."""
        data: dict[str, Any] = {}
        p = Path(path)
        if p.exists():
            loaded = yaml.safe_load(p.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        return cls(playbooks=data.get("playbooks"), taxonomy=data.get("taxonomy"))

    # -- main report ---------------------------------------------------------

    def generate(self, traces: list[Trace], inputs: ReportInputs, title: str = "ForensiQ RCA Report") -> str:
        failures = sorted(inputs.failures, key=lambda f: f.trace_id)
        traces_sorted = sorted(traces, key=lambda t: t.trace_id)
        counts = Counter(f.taxonomy_id for f in failures)
        n_traces = len(traces_sorted)
        lines: list[str] = []
        add = lines.append

        add(f"# {title}")
        add("")
        if traces_sorted:
            add(
                f"*Pipeline(s): {', '.join(sorted({t.pipeline_name for t in traces_sorted}))} · "
                f"Trace window: {min(t.ts for t in traces_sorted).isoformat()} → "
                f"{max(t.ts for t in traces_sorted).isoformat()} · {n_traces} traces*"
            )
        add("")

        # Executive summary
        add("## Executive summary")
        add("")
        rate = (len(failures) / n_traces) if n_traces else 0.0
        top_class, top_n = (counts.most_common(1)[0] if counts else ("—", 0))
        top_class_share = (top_n / len(failures)) if failures else 0.0
        blamed = self._dominant_blame_stage(inputs.blame_distribution, top_class)
        add(f"- **{len(failures)} failures across {n_traces} traces** (failure rate {rate:.1%}).")
        add(
            f"- Dominant family: **{top_class} — {self._title(top_class)}** "
            f"({top_n} traces, {top_class_share:.0%} of failures)"
            + (f", primarily blamed on the **{blamed}** stage." if blamed else ".")
        )
        crit = [s.stage for s in inputs.health.stages if s.status == "crit"]
        warn = [s.stage for s in inputs.health.stages if s.status == "warn"]
        if crit or warn:
            add(
                f"- Stage health: {'CRIT ' + ', '.join(crit) if crit else ''}"
                f"{'; warn: ' + ', '.join(warn) if warn else ''}".rstrip("; ")
                + (" — see stage cards." if (crit or warn) else "")
            )
        else:
            add("- Stage health: all monitored stages within thresholds.")
        if inputs.drift and inputs.drift.alerts:
            first = inputs.drift.alerts[0]
            add(
                f"- **Drift alert**: {first['type']} fired around day {first['day']} — "
                "failure mix shifted vs the baseline window."
            )
        else:
            add("- Drift: failure mix stable vs baseline window.")
        add("")

        # Failure mix
        add("## Failure mix")
        add("")
        add("| taxonomy id | failure | stage | count | share |")
        add("|---|---|---|---:|---:|")
        for tax_id, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            meta = self.taxonomy.get(tax_id, {})
            add(
                f"| {tax_id} | {meta.get('title', tax_id)} | {meta.get('stage', '?')} "
                f"| {n} | {n / len(failures):.1%} |"
            )
        add("")

        # Top root causes with blame + evidence
        add("## Top root causes (blame-ranked)")
        add("")
        for tax_id, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]:
            meta = self.taxonomy.get(tax_id, {})
            add(f"### {tax_id} — {meta.get('title', tax_id)} ({n} traces)")
            add("")
            add(f"*{meta.get('description', '')}*")
            add("")
            dist = inputs.blame_distribution.get(tax_id, {})
            if dist:
                add(
                    "Blame: "
                    + ", ".join(f"{stage} {share:.0%}" for stage, share in dist.items())
                )
                add("")
            sample = next(f for f in failures if f.taxonomy_id == tax_id)
            add(f"Representative evidence (`{sample.trace_id}`, confidence {sample.confidence:.2f}):")
            add("")
            for ev in sample.evidence[:3]:
                note = f" — {ev.note}" if ev.note else ""
                add(f"- `{ev.span_id}` `{ev.key}={ev.value}`{note}")
            add("")

        # Mermaid blame graph
        add("## Blame graph")
        add("")
        add("```mermaid")
        add("graph TD")
        stage_share = self._aggregate_blame_by_stage(inputs.blame_distribution, failures)
        stages_present = [s for s in STAGE_ORDER if s in stage_share and stage_share[s] > 0]
        if not stages_present:
            stages_present = ["retrieve", "generate"]
        for i, stage in enumerate(stages_present):
            share = stage_share[stage]
            add(f'    S{i}["{stage}: {share:.0%} of top-1 blame"]')
        for i in range(len(stages_present) - 1):
            add(f"    S{i} --> S{i + 1}")
        add("```")
        add("")

        # Mermaid timeline
        add("## Failure timeline")
        add("")
        add("```mermaid")
        add("timeline")
        add("    title Failures per day")
        by_day = self._failures_by_day(traces_sorted, failures)
        if by_day:
            for section, day_entries in self._day_sections(by_day):
                add(f"    {section}")
                for day_label, day_counter in day_entries:
                    events = " : ".join(f"{tid} x{n}" for tid, n in sorted(day_counter.items()))
                    add(f"        {day_label} : {events}")
        else:
            add("    section Days")
            add("        No failures observed")
        add("```")
        add("")

        # Stage health cards
        add("## Stage health")
        add("")
        for card in inputs.health.stages:
            flags = f" — {'; '.join(card.flags)}" if card.flags else ""
            key_metrics = ", ".join(f"{k}={v}" for k, v in sorted(card.metrics.items()))
            add(f"- **{card.stage}** [{card.status.upper()}]: {key_metrics}{flags}")
        add("")

        # Cluster cards
        if inputs.clusters:
            add("## Failure families (clusters)")
            add("")
            for card in inputs.clusters:
                terms = ", ".join(card.top_terms) or "—"
                add(
                    f"- **#{card.cluster_id}** size {card.size} ({card.share:.0%}) — terms: {terms} · "
                    f"dominant: {card.dominant_taxonomy} (purity {card.taxonomy_purity:.0%}) · "
                    f"e.g. `{card.representative_trace_id}`"
                )
            add("")

        # Drift section
        if inputs.drift is not None:
            add("## Drift")
            add("")
            if inputs.drift.alerts:
                add("| type | day | detail |")
                add("|---|---|---|")
                for alert in inputs.drift.alerts:
                    if alert["type"] == "chi_square":
                        detail = f"chi2={alert.get('chi2')} p={alert.get('p_value')}"
                    else:
                        detail = (
                            f"{alert.get('class')} cusum={alert.get('cusum')} "
                            f"(baseline mean {alert.get('baseline_daily_mean')})"
                        )
                    add(f"| {alert['type']} | {alert['day']} | {detail} |")
                add("")
            else:
                add("No drift alerts within the analyzed window.")
                add("")

        # Mapped fixes
        add("## Mapped fixes (playbook)")
        add("")
        for tax_id in sorted(counts):
            fix = self.playbooks.get(tax_id, "no playbook entry — add one in docs/playbooks.yaml")
            add(f"- **{tax_id} {self._title(tax_id)}** → {fix}")
        add("")

        add("## Exports")
        add("")
        add(
            "Postmortem template: `RCAReportGenerator.postmortem(trace, failure, candidates)` · "
            "Incident postmortem with causal chain: `RCAReportGenerator.generate_postmortem(trace, failure_records)` · "
            "Plain ticket text: `RCAReportGenerator.ticket(failure, trace)` / "
            "`RCAReportGenerator.postmortem_ticket(trace, failure_records)`. "
            "Full per-trace evidence is attached to every FailureRecord."
        )
        add("")
        return "\n".join(lines)

    # -- exports ---------------------------------------------------------------

    def generate_postmortem(self, trace: Trace, failure_records: list[FailureRecord]) -> str:
        """Single-page incident postmortem with an explicit CAUSAL CHAIN.

        The chain reads like a security-incident timeline (SIEM for AI):
        trigger (user prompt) -> poisoning (context that entered the prompt)
        -> divergence (what the model produced) -> validator failure (which
        guardrail should have caught it and why it didn't) -> blast radius
        (downstream steps affected). Deterministic: same trace + records,
        byte-identical markdown.
        """
        records = list(failure_records)
        primary = records[0] if records else None
        if primary is None:
            raise ValueError("generate_postmortem needs at least one FailureRecord")
        meta = self.taxonomy.get(primary.taxonomy_id, {})

        trigger_span, trigger_text = causal_chain.find_trigger(trace)
        poison = causal_chain.find_poisoning(trace)
        divergence = causal_chain.find_divergence(trace)
        validator = causal_chain.find_validator_gap(trace, primary)
        blast = causal_chain.find_blast_radius(trace, primary)

        lines: list[str] = []
        add = lines.append
        add(f"# Incident postmortem — {trace.trace_id}")
        add("")
        add(
            f"**Failure**: {primary.taxonomy_id} — {meta.get('title', primary.taxonomy_id)}"
            + (f" (+{len(records) - 1} correlated)" if len(records) > 1 else "")
        )
        add(f"**Pipeline**: {trace.pipeline_name} · **Timestamp**: {trace.ts.isoformat()}")
        add(
            f"**Confidence**: {primary.confidence:.2f} ({primary.source}) · "
            f"**Blamed stage**: {primary.stage}"
        )
        add("")

        add("## CAUSAL CHAIN")
        add("")
        add("### 1. Trigger (user prompt)")
        add("")
        trigger_ref = f"`{trigger_span.span_id}` ({trigger_span.stage})" if trigger_span else "—"
        add(f"- Span: {trigger_ref}")
        add(f"- Prompt: {quote(trigger_text) if trigger_text else '(not recorded in trace)'}")
        add("")
        add("### 2. Poisoning (context that entered the prompt)")
        add("")
        poison_span = poison.get("span")
        add(f"- Retrieval span: `{poison_span.span_id}`" if poison_span else "- Retrieval span: —")
        add(f"- Chunks delivered: {poison.get('hit_count', 'n/a')} (top_k={poison.get('top_k', 'n/a')})")
        if poison.get("weakest") is not None:
            add(
                f"- Weakest chunk score: **{poison['weakest']:.3f}**"
                + (f" (mean {poison['mean']:.3f})" if poison.get("mean") is not None else "")
                + " — prime suspect for the degraded answer"
            )
        if poison.get("context"):
            add(f"- Context as sent to the model: \"{poison['context']}\"")
        add("")
        add("### 3. Divergence (what the model produced)")
        add("")
        divergence_span = divergence.get("span")
        add(f"- Generation span: `{divergence_span.span_id}`" if divergence_span else "- Generation span: —")
        if divergence.get("finish_reason"):
            add(f"- finish_reason: `{divergence['finish_reason']}`")
        add(f"- Output: \"{divergence.get('output') or '(empty)'}\"")
        add("")
        add("### 4. Validator failure (guardrail gap)")
        add("")
        add(f"- Should have caught it: {validator['expected']}")
        add(f"- Why it didn't: {validator['why']}")
        add("")
        add("### 5. Blast radius (downstream impact)")
        add("")
        if blast["count"]:
            stages = ", ".join(blast["stages"]) or "none"
            add(f"- {blast['count']} downstream span(s) consumed the output: {stages}")
            add(f"- Downstream cost: {blast['cost']:.4f}")
        else:
            add("- No downstream pipeline spans — the divergence reached the consumer directly.")
        add(f"- User-facing: {'YES — the bad output left the pipeline' if blast['user_facing'] else 'no (contained)'}")
        add("")

        add("```mermaid")
        add("graph TD")
        trigger_label = clip(trigger_text, 60) if trigger_text else "user prompt"
        poison_label = (
            f'{poison.get("hit_count", "?")} chunks, weakest {poison["weakest"]:.3f}'
            if poison.get("weakest") is not None
            else "context unknown"
        )
        divergence_label = divergence.get("output") or "empty output"
        facing = " — user-facing" if blast["user_facing"] else ""
        for label, text in (
            ("TRIGGER", trigger_label),
            ("POISONING", poison_label),
            ("DIVERGENCE", divergence_label),
            ("VALIDATOR GAP", f'{meta.get("stage", primary.stage)} not guarded'),
            ("BLAST RADIUS", f'{blast["count"]} downstream span(s){facing}'),
        ):
            safe = str(text).replace('"', "'")  # mermaid labels cannot contain raw double quotes
            add(f'    {label[0]}["{label}: {safe}"]')
        add("    T --> P --> D --> V --> B")
        add("```")
        add("")

        add("## Evidence")
        add("")
        for record in records:
            add(f"- **{record.taxonomy_id}** ({record.source}, confidence {record.confidence:.2f})")
            for ev in record.evidence[:4]:
                note = f" — {ev.note}" if ev.note else ""
                add(f"  - `{ev.span_id}` `{ev.key}={ev.value}`{note}")
        add("")
        add("## Resolution")
        add("")
        add(self.playbooks.get(primary.taxonomy_id, "<mapped fix>"))
        add("")
        add("## Action items")
        add("")
        add("- [ ] Close the validator gap (see section 4) or document why the guard is absent.")
        add("- [ ] Add an alert for this taxonomy id so the next occurrence pages instead of waiting for a human.")
        add("- [ ] Verify the fix with `forensiq replay` counterfactual mode before shipping.")
        add("")
        return "\n".join(lines)

    def postmortem_ticket(self, trace: Trace, failure_records: list[FailureRecord]) -> str:
        """Plain-text ticket export of the causal chain (no markdown)."""
        records = list(failure_records)
        primary = records[0] if records else None
        if primary is None:
            raise ValueError("postmortem_ticket needs at least one FailureRecord")
        meta = self.taxonomy.get(primary.taxonomy_id, {})
        trigger_span, trigger_text = causal_chain.find_trigger(trace)
        poison = causal_chain.find_poisoning(trace)
        divergence = causal_chain.find_divergence(trace)
        validator = causal_chain.find_validator_gap(trace, primary)
        blast = causal_chain.find_blast_radius(trace, primary)

        lines = [
            f"[{primary.taxonomy_id}] {meta.get('title', primary.taxonomy_id)} — incident ticket",
            f"trace_id: {trace.trace_id}",
            f"ts: {trace.ts.isoformat()}  pipeline: {trace.pipeline_name}",
            "CAUSAL CHAIN:",
            f"  TRIGGER   : {trigger_span.stage if trigger_span else '?'} span "
            f"{trigger_span.span_id if trigger_span else '-'} prompt={clip(trigger_text, 80) or '(not recorded)'}",
            f"  POISONING : {poison.get('hit_count', 'n/a')} chunks entered the prompt"
            + (f", weakest score {poison['weakest']:.3f}" if poison.get("weakest") is not None else ""),
            f"  DIVERGENCE: \"{clip(divergence.get('output') or '(empty)', 80)}\""
            + (f" (finish_reason={divergence['finish_reason']})" if divergence.get("finish_reason") else ""),
            f"  VALIDATOR : {validator['expected']}",
            f"              why it didn't: {validator['why']}",
            f"  BLAST     : {blast['count']} downstream span(s) [{', '.join(blast['stages']) or 'none'}]"
            + (" — USER-FACING" if blast["user_facing"] else ""),
            "evidence:",
        ]
        for ev in primary.evidence:
            note = f" ({ev.note})" if ev.note else ""
            lines.append(f"  - {ev.span_id}.{ev.key} = {clip(ev.value, 100)}{note}")
        lines.append(f"suggested_fix: {self.playbooks.get(primary.taxonomy_id, 'triage manually')}")
        return "\n".join(lines)

    def postmortem(
        self, trace: Trace, failure: FailureRecord, candidates: list[BlameCandidate] | None = None
    ) -> str:
        meta = self.taxonomy.get(failure.taxonomy_id, {})
        blamed = candidates[0].stage if candidates else failure.stage
        lines = [
            "# Postmortem template",
            "",
            f"**Failure**: {failure.taxonomy_id} — {meta.get('title', failure.taxonomy_id)}",
            f"**Trace**: `{trace.trace_id}` at {trace.ts.isoformat()} ({trace.pipeline_name})",
            f"**Blamed stage**: {blamed} (classifier pinned: {failure.stage}, "
            f"confidence {failure.confidence:.2f}, source: {failure.source})",
            "",
            "## Summary",
            "<one paragraph: what the user experienced>",
            "",
            "## Evidence",
        ]
        for ev in failure.evidence:
            note = f" — {ev.note}" if ev.note else ""
            lines.append(f"- `{ev.span_id}` `{ev.key}={ev.value}`{note}")
        lines += [
            "",
            "## Root cause",
            "<what actually broke, per the blame ranking above>",
            "",
            "## Resolution",
            self.playbooks.get(failure.taxonomy_id, "<mapped fix>"),
            "",
            "## Action items",
            "- [ ] <preventive change>",
            "- [ ] <detection gap>",
        ]
        return "\n".join(lines)

    def ticket(self, failure: FailureRecord, trace: Trace | None = None) -> str:
        meta = self.taxonomy.get(failure.taxonomy_id, {})
        lines = [
            f"[{failure.taxonomy_id}] {meta.get('title', failure.taxonomy_id)} ({failure.stage})",
            f"trace_id: {failure.trace_id}",
        ]
        if trace is not None:
            lines.append(f"ts: {trace.ts.isoformat()} pipeline: {trace.pipeline_name}")
        lines.append(f"confidence: {failure.confidence:.2f} source: {failure.source}")
        lines.append("evidence:")
        for ev in failure.evidence:
            note = f" ({ev.note})" if ev.note else ""
            lines.append(f"  - {ev.span_id}.{ev.key} = {ev.value}{note}")
        lines.append(f"suggested_fix: {self.playbooks.get(failure.taxonomy_id, 'triage manually')}")
        return "\n".join(lines)

    # -- helpers ---------------------------------------------------------------

    def _title(self, tax_id: str) -> str:
        return self.taxonomy.get(tax_id, {}).get("title", tax_id)

    def _dominant_blame_stage(self, blame_distribution: dict[str, dict[str, float]], tax_id: str) -> str | None:
        dist = blame_distribution.get(tax_id)
        if not dist:
            return None
        return sorted(dist.items(), key=lambda kv: -kv[1])[0][0]

    def _aggregate_blame_by_stage(
        self, blame_distribution: dict[str, dict[str, float]], failures: list[FailureRecord]
    ) -> dict[str, float]:
        """Stage share weighted by class frequency (for the mermaid graph)."""
        counts = Counter(f.taxonomy_id for f in failures)
        total = sum(counts.values()) or 1
        share: dict[str, float] = {}
        for tax_id, n in counts.items():
            for stage, s in blame_distribution.get(tax_id, {}).items():
                share[stage] = share.get(stage, 0.0) + s * (n / total)
        return share

    def _failures_by_day(
        self, traces: list[Trace], failures: list[FailureRecord]
    ) -> dict[str, Counter[str]]:
        if not traces:
            return {}
        day_by_trace = {t.trace_id: t.day for t in traces}
        by_day: dict[str, Counter[str]] = {}
        for f in failures:
            day = day_by_trace.get(f.trace_id)
            if day is None:
                continue
            by_day.setdefault(day, Counter())[f.taxonomy_id] += 1
        return by_day

    def _day_sections(
        self, by_day: dict[str, Counter[str]]
    ) -> list[tuple[str, list[tuple[str, Counter[str]]]]]:
        """Group ISO days into 'Days a-b' mermaid timeline sections, with
        1-based day indices relative to the first day seen (deterministic)."""
        days = sorted(by_day)
        if not days:
            return []
        base = days[0]
        sections: dict[tuple[int, int], list[tuple[str, Counter[str]]]] = {}
        for day in days:
            day_index = _day_to_index(day, base)
            week = (day_index - 1) // 7
            start, end = week * 7 + 1, week * 7 + 7
            sections.setdefault((start, end), []).append((f"Day {day_index}", by_day[day]))
        return [(f"section Days {s}-{e}", entries) for (s, e), entries in sorted(sections.items())]


def _day_to_index(day: str, base: str) -> int:
    return (_date.fromisoformat(day) - _date.fromisoformat(base)).days + 1
