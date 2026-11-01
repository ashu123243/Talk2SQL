"""Gemini LLM provider implementation for Talk2SQL."""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel

from backend.config import Settings, settings
from backend.core.logging_config import get_logger
from backend.exceptions.custom_exceptions import (
    ConfigurationException,
    ModelProviderException,
)
from backend.llm.base import LLMProvider


class GeminiProvider(LLMProvider):
    """Gemini provider supporting multiple API key slots."""

    def __init__(self, config: Settings | None = None) -> None:
        self._config = config or settings
        self._logger = get_logger("llm.gemini")

        self._logger.info("Gemini provider initialized")

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "gemini"

    @property
    def available_key_numbers(self) -> tuple[int, ...]:
        """Return the configured Gemini API key slot numbers."""

        available_keys = []

        for key_number in (1, 2, 3):
            secret = getattr(
                self._config,
                f"gemini_api_key_{key_number}",
            )

            if secret.get_secret_value().strip():
                available_keys.append(key_number)

        return tuple(available_keys)
    
    
    def get_chat_model_for_key(
        self,
        key_number: int,
    ) -> BaseChatModel:
        """Create a Gemini chat model using a selected API key."""

        if isinstance(key_number, bool) or key_number not in (1, 2, 3):
            raise ConfigurationException(
                "Gemini API key slot must be 1, 2, or 3.",
                error_code="INVALID_GEMINI_KEY_SLOT",
            )

        secret = getattr(
            self._config,
            f"gemini_api_key_{key_number}",
        )

        api_key = secret.get_secret_value().strip()

        if not api_key:
            self._logger.error(
                "Gemini API key slot %s is not configured",
                key_number,
            )

            raise ConfigurationException(
                f"Gemini API key slot {key_number} is missing.",
                error_code=f"GEMINI_API_KEY_{key_number}_MISSING",
            )

        model_name = self._config.gemini_model.strip()

        if not model_name:
            raise ConfigurationException(
                "Gemini model name is not configured.",
                error_code="GEMINI_MODEL_NOT_CONFIGURED",
            )

        try:
            chat_model = ChatGoogleGenerativeAI(
                model=model_name,
                api_key=api_key,
                temperature=0,
                max_retries=0,
            )

        except Exception as exc:
            self._logger.error(
                "Gemini model construction failed for key slot %s. "
                "Exception type: %s",
                key_number,
                type(exc).__name__,
            )

            raise ModelProviderException(
                "Unable to initialize the Gemini chat model.",
                error_details=exc,
                error_code="GEMINI_MODEL_INITIALIZATION_ERROR",
            ) from exc

        self._logger.info(
            "Gemini chat model object created for key slot %s",
            key_number,
        )

        return chat_model

    def get_chat_model(self) -> BaseChatModel:
        """Return a Gemini chat model using the first configured key."""

        available_keys = self.available_key_numbers

        if not available_keys:
            self._logger.error(
                "No Gemini API keys are configured"
            )

            raise ConfigurationException(
                "No Gemini API keys are configured.",
                error_code="GEMINI_API_KEYS_MISSING",
            )

        first_key = available_keys[0]

        return self.get_chat_model_for_key(first_key)

    