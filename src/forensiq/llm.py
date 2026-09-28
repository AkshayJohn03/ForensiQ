"""LLM access behind a Protocol — the only place ForensiQ may touch a model.

* :class:`OpenAICompatClient` — async chat-completions over httpx against any
  OpenAI-compatible endpoint. Never constructed unless the user sets
  FORENSIQ_LLM_ENABLED + a key; tests never call it.
* :class:`EchoMockClient` — deterministic offline fallback: it "completes" the
  classification prompt by scoring candidate taxonomy ids against the evidence
  keywords embedded in the prompt itself. Zero network, zero randomness.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

import httpx


@runtime_checkable
class LLMClient(Protocol):
    """Any client that can complete a prompt. Async by design so the real
    provider path can batch/parallelize the ambiguous pass later."""

    async def complete(self, prompt: str) -> str: ...


# Keyword tables used by the offline fallback to score candidates.
TAXONOMY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "F-RET-001": ("empty", "hit_count=0", "no hits", "zero"),
    "F-RET-002": ("low score", "weak context", "hit_scores", "mean_score"),
    "F-PROMPT-001": ("length", "truncat", "prompt_chars", "limit"),
    "F-GEN-001": ("json", "parse", "format", "malformed"),
    "F-GEN-002": ("repeat", "loop", "n-gram"),
    "F-GEN-003": ("refus", "cannot answer", "decline", "abstain"),
    "F-TOOL-001": ("tool", "error", "exception"),
    "F-INFRA-001": ("timeout", "timed out"),
    "F-INFRA-002": ("cost", "spend", "spike"),
    "F-INFRA-003": ("guard", "blocked", "policy", "aborted"),
}


class OpenAICompatClient:
    """Async OpenAI-compatible chat-completions client.

    Callers must construct this deliberately (settings gate it); ForensiQ
    never builds it implicitly. All provider errors propagate to the caller
    — the classifier's fallback handles them.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 20.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    async def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 120,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
        return str(data["choices"][0]["message"]["content"])


class EchoMockClient:
    """Deterministic offline 'LLM'.

    Given the classifier's ambiguity prompt (which embeds candidate taxonomy
    ids plus the evidence summary), it picks the candidate whose keyword table
    best overlaps the evidence text and answers in the same structured format
    the real model is asked for. Deterministic, offline, test-friendly.
    """

    def __init__(self, keywords: dict[str, tuple[str, ...]] | None = None) -> None:
        self.keywords = keywords or TAXONOMY_KEYWORDS

    async def complete(self, prompt: str) -> str:
        low = prompt.lower()
        best_id: str | None = None
        best_score = 0
        for tax_id, kws in sorted(self.keywords.items()):
            if tax_id not in prompt:
                continue  # only candidates offered in the prompt
            score = sum(low.count(kw.lower()) for kw in kws)
            if score > best_score:
                best_id, best_score = tax_id, score
        if best_id is None or best_score == 0:
            return "TAXONOMY: UNKNOWN\nCONFIDENCE: 0.10\nREASON: no candidate matched the evidence"
        conf = min(0.75, 0.40 + 0.05 * best_score)
        return f"TAXONOMY: {best_id}\nCONFIDENCE: {conf:.2f}\nREASON: keyword match score {best_score}"


_TAX_RE = re.compile(r"F-[A-Z]+-\d{3}")
_CONF_RE = re.compile(r"CONFIDENCE:\s*([0-9.]+)")


def parse_llm_verdict(response: str) -> tuple[str | None, float | None]:
    """Parse 'TAXONOMY: F-XXX-NNN / CONFIDENCE: 0.xx' style responses."""
    tax = _TAX_RE.search(response)
    conf = _CONF_RE.search(response)
    tax_id = tax.group(0) if tax else None
    confidence = float(conf.group(1)) if conf and 0.0 <= float(conf.group(1)) <= 1.0 else None
    return tax_id, confidence
