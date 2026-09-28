# ForensiQ Fix Playbooks

Mapped from taxonomy ids to first-response actions. The report generator
maps every observed failure family to the playbook line below (overridable
via `docs/playbooks.yaml`, see `RCAReportGenerator.with_yaml`).

| id | failure | first response |
|---|---|---|
| F-RET-001 | Empty retrieval set | Check index freshness and embed-model consistency; verify the query reached the right collection/namespace; add a no-hits fallback query. |
| F-RET-002 | Low-recall context | Check index freshness / embedding model drift; re-embed stale chunks; raise hybrid BM25 weight for keyword-heavy queries; raise top_k only after score review. |
| F-RET-003 | Retrieval score collapse | Correlate score decay with deploy/embedding-model changes; schedule re-embedding; add canary queries with score SLOs. |
| F-PROMPT-001 | Prompt truncation | Trim retrieved context (fewer/smaller chunks, parent-child summarization) or raise the context window; alert on prompt_chars near the ceiling. |
| F-GEN-001 | Structured output format break | Enforce structured output (JSON schema / function calling), lower temperature for extraction, add a parse-repair retry with one re-ask. |
| F-GEN-002 | Repetition loop | Lower temperature / add repetition penalty, check for duplicated context chunks feeding the loop, cap max_tokens. |
| F-GEN-003 | Ungrounded refusal | Review the system prompt's refusal instructions; check the guard prompt for over-broad safety wording; verify citations render correctly. |
| F-GEN-004 | Ungrounded claim | Tighten the grounding prompt, require citations per claim, add an NLI/LLM-judge spot check on the worst offenders. |
| F-TOOL-001 | Tool execution error | Check tool dependency health and retries; add timeout + idempotency on the tool call; surface the error to the model for replanning. |
| F-INFRA-001 | Stage timeout | Raise the stage timeout or stream tokens; check provider latency SLOs; add a smaller fallback model for degradation. |
| F-INFRA-002 | Cost spike | Cap prompt growth (context budget), check for retry loops re-sending the prompt, add per-trace cost guardrails. |
| F-INFRA-003 | Guard abort | Review guard rules hit rate vs true positives; if blocks are wrong, tune the guard; otherwise route the user to a safe fallback. |

Counterfactual replay (`forensiq.attribution.replay`) validates the candidate
fix before you page anyone: e.g. F-PROMPT-001 replayed with
`generate.context_limit = 16384` should resolve; F-RET-002 replayed with
`retrieve.query_rewrite` should lift hit scores, while `top_k 5→10` honestly
does **not** (marginal documents dilute the mean — see the simulated adapter's
documented effect model).
