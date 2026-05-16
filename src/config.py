from __future__ import annotations
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-sonnet-4-5", alias="ANTHROPIC_MODEL")
    tesseract_cmd: str = Field(default="/usr/bin/tesseract", alias="TESSERACT_CMD")
    use_claude_vision_fallback: bool = Field(default=True, alias="USE_CLAUDE_VISION_FALLBACK")
    confidence_auto_approve: float = Field(default=0.90, alias="CONFIDENCE_AUTO_APPROVE")
    confidence_human_review: float = Field(default=0.70, alias="CONFIDENCE_HUMAN_REVIEW")

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
