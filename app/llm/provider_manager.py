"""Provider manager for Talk2SQL."""

from typing import Any

from app.core.config import Settings, get_settings
from app.core.logging_config import logger
from app.exceptions.custom_exceptions import CustomException
from app.llm.gemini_provider import GeminiProvider
from app.llm.ollama_provider import OllamaProvider

class ProviderManager:
    """Select an LLM provider and handle eligible fallback failures."""

    RECOVERABLE_STATUS_CODES = {
        408, 429, 500, 502, 503, 504
    }

    RECOVERABLE_ERROR_NAMES = {
        "timeout",
        "timeouterror",
        "timeoutexception",
        "connecterror",
        "connectionerror",
        "networkerror",
        "resourceexhausted",
        "serviceunavailable",
        "deadlineexceeded",
        "toomanyrequests",
        "internalservererror",
        "badgateway",
        "gatewaytimeout",
    }

    def __init__(self, settings:Settings | None=None,
                gemini_provider: GeminiProvider | None = None,
                ollama_provider: OllamaProvider | None = None,
    )->None:
        """Initialize settings and provider dependencies."""

        self.settings = settings or get_settings()

        self.gemini_provider = (
            gemini_provider
            if gemini_provider is not None
            else GeminiProvider(settings=self.settings)
        )

        self.ollama_provider = (
            ollama_provider
            if ollama_provider is not None
            else OllamaProvider(settings=self.settings)
        )


    @staticmethod
    def _get_exception_chain(exc: Exception) -> list[Exception]:
        """Return an exception and its chained causes."""

        exceptions: list[Exception] = []
        current: Exception | None = exc
        visited: set[int] = set()

        while current is not None and id(current) not in visited:
            visited.add(id(current))
            exceptions.append(current)

            cause = current.__cause__ or current.__context__

            if not isinstance(cause, Exception):
                break

            current = cause

        return exceptions



    @classmethod
    def _is_recoverable_gemini_error(cls, exc: Exception) -> bool:
        """Identify known recoverable network and API failures."""

        for error in cls._get_exception_chain(exc):
            error_name = type(error).__name__.lower()

            if error_name in cls.RECOVERABLE_ERROR_NAMES:
                return True

            status_code: Any = getattr(
                error,
                "status_code",
                None,
            )

            if status_code is None:
                response = getattr(error, "response", None)

                if response is not None:
                    status_code = getattr(
                        response,
                        "status_code",
                        None,
                    )

            if isinstance(status_code, int):
                if status_code in cls.RECOVERABLE_STATUS_CODES:
                    return True

            status_name = str(status_code).upper()

            recoverable_status_names = {
                "RESOURCE_EXHAUSTED",
                "UNAVAILABLE",
                "DEADLINE_EXCEEDED",
                "TOO_MANY_REQUESTS",
            }

            if status_name in recoverable_status_names:
                return True

        return False

    def generate_text(self, prompt: str) -> str:
        """Generate text using the selected provider and fallback."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        configured_provider = self.settings.llm_provider

        provider_name = (
            configured_provider.strip().lower()
            if isinstance(configured_provider, str)
            else ""
        )

        if provider_name == "ollama":
            try:
                return self.ollama_provider.generate(prompt)

            except Exception as exc:
                logger.error(
                    "Configured Ollama provider failed with %s.",
                    type(exc).__name__,
                )

                raise CustomException(
                    "The configured Ollama provider could not "
                    "generate a response.",
                    exc,
                ) from exc

        if provider_name != "gemini":
            logger.error("Unsupported LLM provider configured.")

            raise CustomException(
                "Unsupported LLM provider: "
                f"{provider_name or 'empty'}."
            )

        try:
            return self.gemini_provider.generate(prompt)

        except Exception as gemini_error:

            if not self._is_recoverable_gemini_error(gemini_error):
                logger.error(
                    "Gemini failed with non-recoverable error %s; "
                    "Ollama fallback will not be attempted.",
                    type(gemini_error).__name__,
                )

                raise CustomException(
                    "Gemini generation failed with a "
                    "non-recoverable error.",
                    gemini_error,
                ) from gemini_error

            logger.warning(
                "Gemini encountered a recoverable failure (%s). "
                "Attempting Ollama fallback.",
                type(gemini_error).__name__,
            )

            try:
                result = self.ollama_provider.generate(prompt)

                logger.info("Ollama fallback succeeded.")

                return result

            except Exception as ollama_error:
                logger.error(
                    "Both providers failed. Gemini error: %s; "
                    "Ollama error: %s.",
                    type(gemini_error).__name__,
                    type(ollama_error).__name__,
                )

                raise CustomException(
                    "Gemini failed with a recoverable error, "
                    "and Ollama fallback also failed.",
                    ollama_error,
                ) from ollama_error
