"""Ollama LLM provider implementation for Talk2SQL."""

from urllib.parse import urlparse

from langchain_ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel

from backend.config import Settings, settings
from backend.core.logging_config import get_logger
from backend.exceptions.custom_exceptions import (
    ConfigurationException,
    ModelProviderException,
)
from backend.llm.base import LLMProvider


class OllamaProvider(LLMProvider):
    """Ollama provider for locally hosted language models."""

    def __init__(self, config: Settings | None = None) -> None:
        self._config = config or settings
        self._logger = get_logger("llm.ollama")

        self._logger.info("Ollama provider initialized")

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "ollama"
    
    def _get_validated_base_url(self) -> str:
        """Validate and normalize the Ollama server URL."""

        base_url = self._config.ollama_base_url.strip().rstrip("/")

        parsed_url = urlparse(base_url)

        if (
            parsed_url.scheme not in ("http", "https")
            or not parsed_url.hostname
        ):
            self._logger.error(
                "Invalid Ollama server URL configuration"
            )

            raise ConfigurationException(
                "Ollama server URL must be a valid HTTP or HTTPS URL.",
                error_code="INVALID_OLLAMA_BASE_URL",
            )

        return base_url
    
    def get_chat_model(self) -> BaseChatModel:
        """Create and return a LangChain-compatible Ollama chat model."""

        model_name = self._config.ollama_model.strip()

        if not model_name:
            self._logger.error(
                "Ollama model name is not configured"
            )

            raise ConfigurationException(
                "Ollama model name is not configured.",
                error_code="OLLAMA_MODEL_NOT_CONFIGURED",
            )

        base_url = self._get_validated_base_url()

        try:
            chat_model = ChatOllama(
                model=model_name,
                base_url=base_url,
                temperature=0,
            )

        except Exception as exc:
            self._logger.error(
                "Ollama model construction failed. "
                "Exception type: %s",
                type(exc).__name__,
            )

            raise ModelProviderException(
                "Unable to initialize the Ollama chat model.",
                error_details=exc,
                error_code="OLLAMA_MODEL_INITIALIZATION_ERROR",
            ) from exc

        self._logger.info(
            "Ollama chat model object created successfully "
            "for model '%s'",
            model_name,
        )

        return chat_model