
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Environment configuration
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    """Central configuration for Talk2SQL."""

    # Application
    app_name: str = "Talk2SQL"
    debug: bool = False

    # Gemini configuration
    gemini_api_key_1: SecretStr
    gemini_api_key_2: SecretStr
    gemini_api_key_3: SecretStr

    gemini_model: str = "gemini-3.8-flash"

    # Ollama configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:7b"

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Application settings instance
settings = Settings()
