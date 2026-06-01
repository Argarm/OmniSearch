from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_yaml(path: str | None = None) -> dict[str, Any]:
    settings_path = Path(path or os.getenv("SETTINGS_PATH") or "config/settings.yaml")
    if settings_path.exists():
        with settings_path.open() as f:
            return yaml.safe_load(f) or {}
    return {}


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    model_name: str = "BAAI/bge-large-en-v1.5"
    device: str = "cpu"
    batch_size: int = 32
    query_prefix: str = "Represent this sentence for searching relevant passages: "


class ChunkingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHUNKING_")

    chunk_size: int = 800
    chunk_overlap: int = 150
    separators: list[str] = Field(default=["\n\n", "\n", ". ", " ", ""])


class QdrantSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QDRANT_")

    url: str = "http://localhost:6333"
    api_key: str = ""
    collection_name: str = "omnisearch"
    vector_size: int = 1024
    distance: str = "Cosine"
    on_disk_payload: bool = True


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_")

    top_k: int = 6
    score_threshold: float = 0.35


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_")

    model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    temperature: float = 0.0
    max_tokens: int = 2048
    streaming: bool = True

    model_config = SettingsConfigDict(env_prefix="LLM_", populate_by_name=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # External API keys (environment only — never in YAML)
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", alias="OPENAI_BASE_URL"
    )
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    # App settings
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    cors_origins: list[str] = Field(
        default=["http://localhost:8501", "http://frontend:8501"]
    )
    backend_url: str = "http://localhost:8000"

    # Sub-settings (populated from YAML, overridable by env)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)

    def __init__(self, **data: Any) -> None:
        yaml_data = _load_yaml()
        # Merge YAML into defaults; env vars take precedence (handled by pydantic-settings)
        for section, values in yaml_data.items():
            if section in ("embedding", "chunking", "qdrant", "retrieval", "llm"):
                if section not in data:
                    data[section] = values
        super().__init__(**data)


@lru_cache
def get_settings() -> Settings:
    return Settings()
