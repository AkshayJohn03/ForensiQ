"""Runtime settings loaded from the environment (prefix FORENSIQ_).

Everything is optional: with no .env and no keys ForensiQ runs fully offline —
the LLM second pass degrades to the deterministic EchoMockClient and Langfuse
ingestion simply refuses to start instead of half-working.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class ForensiqSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FORENSIQ_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Rules-first classifier thresholds.
    low_score_threshold: float = 0.5
    cost_spike_factor: float = 3.0
    context_limit: int = 8192
    repetition_min_repeat: int = 5

    # Drift detection.
    alpha: float = 0.01
    baseline_days: int = 14
    window_days: int = 3

    # Optional LLM second pass (any OpenAI-compatible endpoint).
    llm_enabled: bool = False
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"

    # Langfuse REST pull.
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""


def load_settings() -> ForensiqSettings:
    return ForensiqSettings()
