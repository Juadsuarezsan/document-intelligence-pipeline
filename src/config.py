"""Environment-driven configuration.

All tunables live here and are read once through :func:`get_settings`, which is
``lru_cache``-d so the rest of the code can call it freely. Tests that mutate
environment variables call ``get_settings.cache_clear()``.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PINNED_MODEL = "claude-sonnet-4-5-20250929"
"""Dated model id used everywhere the application calls Claude."""


def package_version() -> str:
    """Return the installed package version, or ``0.0.0-dev`` outside an install.

    Returns:
        Version string declared in ``pyproject.toml``.
    """
    try:
        return version("document-intelligence")
    except PackageNotFoundError:
        return "0.0.0-dev"


class Settings(BaseSettings):
    """Typed view over the process environment (``.env`` is loaded if present)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    # Claude
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default=PINNED_MODEL, alias="ANTHROPIC_MODEL")
    anthropic_timeout_s: float = Field(default=30.0, alias="ANTHROPIC_TIMEOUT_S", gt=0)
    anthropic_max_attempts: int = Field(default=3, alias="ANTHROPIC_MAX_ATTEMPTS", ge=1, le=10)
    price_input_per_mtok: float = Field(default=3.0, alias="PRICE_INPUT_PER_MTOK", ge=0)
    price_output_per_mtok: float = Field(default=15.0, alias="PRICE_OUTPUT_PER_MTOK", ge=0)

    # Parsing
    tesseract_cmd: str = Field(default="/usr/bin/tesseract", alias="TESSERACT_CMD")
    ocr_min_chars_per_page: int = Field(default=20, alias="OCR_MIN_CHARS_PER_PAGE", ge=0)
    use_claude_vision_fallback: bool = Field(default=True, alias="USE_CLAUDE_VISION_FALLBACK")
    use_unstructured: bool = Field(default=False, alias="USE_UNSTRUCTURED")

    # Routing thresholds
    confidence_auto_approve: float = Field(default=0.90, alias="CONFIDENCE_AUTO_APPROVE")
    confidence_human_review: float = Field(default=0.70, alias="CONFIDENCE_HUMAN_REVIEW")

    # API limits and security
    cors_allow_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:5500", alias="CORS_ALLOW_ORIGINS"
    )
    rate_limit: str = Field(default="30/minute", alias="RATE_LIMIT")
    max_pdf_bytes: int = Field(default=10 * 1024 * 1024, alias="MAX_PDF_BYTES", gt=0)
    max_text_chars: int = Field(default=200_000, alias="MAX_TEXT_CHARS", gt=0)
    redact_pii: bool = Field(default=True, alias="REDACT_PII")

    # Storage
    database_url: str | None = Field(default=None, alias="DATABASE_URL")

    # Observability
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="document-intelligence", alias="LANGSMITH_PROJECT")
    langchain_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")

    @property
    def cors_origins(self) -> list[str]:
        """Parsed list of allowed CORS origins (empty entries dropped)."""
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    @property
    def llm_enabled(self) -> bool:
        """Whether Claude-backed paths are active (an API key is configured)."""
        return bool(self.anthropic_api_key)

    @property
    def langsmith_enabled(self) -> bool:
        """Whether LangSmith tracing should be wired (key present and flag on)."""
        return bool(self.langsmith_api_key) and self.langchain_tracing_v2


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached :class:`Settings` instance."""
    return Settings()
