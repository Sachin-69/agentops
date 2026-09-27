"""Application configuration loaded from environment / .env file."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the AgentOpsAI platform."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Provider selection
    llm_provider: str = "none"  # none | openai | anthropic | ollama

    # OpenAI
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_chat_model: str = "claude-3-5-sonnet-20240620"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.1"

    # Retrieval
    docs_dir: str = "docs"
    top_k: int = 4
    min_relevance: float = 0.12
    chunk_size: int = 800
    chunk_overlap: int = 150

    refusal_message: str = (
        "I'm sorry, but I can only answer questions based on the provided "
        "support documentation, and I couldn't find anything relevant to your "
        "question. Please rephrase or contact a human support agent."
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
