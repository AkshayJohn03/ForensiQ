# ForensiQ Failure Taxonomy

Every failure ForensiQ emits carries one of these ids. Families map to
pipeline stages (`embed → retrieve → rerank → graph → generate`, plus
`guard / tool / agent`), matching the span vocabulary of
[RAG_showcase](https://github.com/AkshayJohn03/RAG_showcase) (`hybrid`
retrieval normalizes to `retrieve`). Detection column: `rule` = deterministic
pass-1 rule, `llm/fallback` = pass-2 classification for ambiguous leftovers,
`drift` = cohort-level detector in `forensiq.patterns.drift`.

| id | title | stage | severity | detection | signature |
|---|---|---|---|---|---|
| F-RET-001 | Empty retrieval set | retrieve | high | rule | `hit_count == 0` (or empty `hit_scores`) on the retrieve span |
| F-RET-002 | Low-recall context | retrieve | high | rule | mean of `hit_scores` < `low_score_threshold` (default 0.5) |
| F-RET-003 | Retrieval score collapse (drift) | retrieve | medium | drift | hit-score decay / mix shift across days (chi-square + CUSUM) |
| F-PROMPT-001 | Prompt truncation | generate | high | rule | `finish_reason == "length"` AND `prompt_chars ≥ 0.9 × context_limit` |
| F-GEN-001 | Structured output format break | generate | medium | rule | output declared/looks JSON but `json.loads` fails |
| F-GEN-002 | Repetition loop | generate | medium | rule | any 3/5/8-gram repeated ≥ `repetition_min_repeat` (default 5) |
| F-GEN-003 | Ungrounded refusal | generate | medium | rule | refusal phrase present AND retrieval context was usable |
| F-GEN-004 | Ungrounded claim | generate | high | health | groundedness heuristic far below cohort baseline (`attribution.health`) |
| F-TOOL-001 | Tool execution error | tool | high | rule | tool span `status == "error"` |
| F-INFRA-001 | Stage timeout | any | high | rule | span `status == "timeout"` |
| F-INFRA-002 | Cost spike | generate | medium | rule (cohort) | `total_cost > p95 × cost_spike_factor` (default ×3) |
| F-INFRA-003 | Guard abort | guard | low | llm/fallback | guard span `blocked == true` or non-tool `status == "error"` without a rule signature |

## Notes on precedence

Pass-1 rules are evaluated in this order and the first match wins:

```
F-INFRA-001 → F-TOOL-001 → F-RET-001 → F-RET-002 → F-PROMPT-001
→ F-GEN-001 → F-GEN-002 → F-GEN-003 → F-INFRA-002
```

Rationale: hard infrastructure signals first; upstream causes before
downstream symptoms (an empty-context refusal is a retrieval problem, so
F-RET-001 outranks the refusal signature F-GEN-003); cohort-relative cost
anomalies last. A trace yields one primary failure — cohorts still expose the
full mix through aggregates.

## Extending

Add ids in this file, the catalog in `src/forensiq/report/rca.py`, a playbook
line in `docs/playbooks.yaml`, and (for rule detection) a rule in
`taxonomy/classifier.py`. Unknown stages from foreign exporters degrade to
`tool` spans and keep ingestion alive — register aliases in
`ingest/schema.py::STAGE_ALIASES`.
