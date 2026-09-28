# Security Policy — ForensiQ

ForensiQ ingests traces that may contain sensitive prompts, completions, and
costs. Threat model and defenses:

| Threat | Defense |
|---|---|
| PII in traces | Ingestion is file/pull-based under your control; the README documents sampling and retention guidance; the classifier consumes attrs, never echoes full prompts into reports (evidence quotes are attr-scoped) |
| Trace-file attacks (zip/xml bombs via exported bundles) | Loaders parse JSON/JSONL only; no archive extraction; hardened parsing at part boundaries |
| Tampered evidence | Reports embed trace/span ids so every claim is re-checkable against the raw store; analysis is deterministic (seeded) and re-runnable |
| Supply chain | Minimal deps; Langfuse path is optional and env-guarded; CI runs pip-audit |

Report issues marked `security`.