"""Centralized LLM provider management for Talk2SQL."""

from typing import Any

from langchain_core.messages import BaseMessage

from backend.core.logging_config import get_logger
from backend.exceptions.custom_exceptions import (
    ConfigurationException,
    ModelAuthenticationException,
    ModelProviderException,
    ModelRateLimitException,
    ModelUnavailableException,
)
from backend.llm.gemini_provider import GeminiProvider
from backend.llm.ollama_provider import OllamaProvider


class ProviderManager:
    """Manage Gemini key rotation and Ollama fallback."""

    def __init__(
        self,
        gemini_provider: GeminiProvider | None = None,
        ollama_provider: OllamaProvider | None = None,
    ) -> None:

        self.gemini_provider = gemini_provider or GeminiProvider()
        self.ollama_provider = ollama_provider or OllamaProvider()

        self._logger = get_logger("llm.provider_manager")

        self._logger.info("LLM Provider Manager initialized")

    @staticmethod
    def _get_exception_chain(
        exception: Exception,
    ):
        """Yield an exception and its related underlying exceptions."""

        visited = set()
        current = exception

        while current is not None and id(current) not in visited:
            visited.add(id(current))
            yield current

            if current.__cause__ is not None:
                current = current.__cause__

            elif current.__context__ is not None:
                current = current.__context__

            else:
                details = getattr(current, "error_details", None)

                if isinstance(details, Exception):
                    current = details
                else:
                    current = None

    @staticmethod
    def _get_status_code(exception: Exception) -> int | None:
        """Extract an HTTP-like status code from an exception."""

        for attribute in ("status_code", "status", "code"):
            value = getattr(exception, attribute, None)

            if value is None:
                continue

            if callable(value):
                try:
                    value = value()
                except Exception:
                    continue

            try:
                return int(value)
            except (TypeError, ValueError):
                continue

        return None

    def _classify_exception(
        self,
        exception: Exception,
    ) -> Exception:
        """Classify provider errors into Talk2SQL exception types."""

        exception_chain = list(
            self._get_exception_chain(exception)
        )

        # Preserve already classified Talk2SQL exceptions.
        for current in exception_chain:
            if isinstance(
                current,
                (
                    ConfigurationException,
                    ModelRateLimitException,
                    ModelAuthenticationException,
                    ModelUnavailableException,
                ),
            ):
                return current

        # Inspect all exception messages and types.
        for current in exception_chain:
            status_code = self._get_status_code(current)
            exception_name = type(current).__name__.lower()
            error_message = str(current).lower()

            # Quota and rate-limit detection.
            quota_indicators = (
                "quota exceeded",
                "quotaexceeded",
                "resource exhausted",
                "resource_exhausted",
                "rate limit",
                "ratelimit",
                "too many requests",
            )

            if (
                status_code == 429
                or any(
                    indicator in error_message
                    for indicator in quota_indicators
                )
                or "resourceexhausted" in exception_name
            ):
                return ModelRateLimitException(
                    "The model provider rate limit or quota was reached.",
                    error_details=exception,
                )

            # Authentication detection.
            authentication_indicators = (
                "invalid api key",
                "api key not valid",
                "unauthorized",
                "unauthenticated",
                "authentication failed",
                "permission denied",
                "forbidden",
            )

            if (
                status_code == 401
                or status_code == 403
                or any(
                    indicator in error_message
                    for indicator in authentication_indicators
                )
                or exception_name in (
                    "unauthenticated",
                    "permissiondenied",
                )
            ):
                return ModelAuthenticationException(
                    "The model provider rejected authentication.",
                    error_details=exception,
                )

            # Temporary availability and network failures.
            unavailable_indicators = (
                "connection refused",
                "connection error",
                "connection reset",
                "timed out",
                "timeout",
                "service unavailable",
                "server error",
                "temporarily unavailable",
            )

            if (
                status_code in (408, 500, 502, 503, 504)
                or isinstance(
                    current,
                    (TimeoutError, ConnectionError, OSError),
                )
                or any(
                    indicator in error_message
                    for indicator in unavailable_indicators
                )
                or exception_name in (
                    "serviceunavailable",
                    "deadlineexceeded",
                )
            ):
                return ModelUnavailableException(
                    "The model provider is temporarily unavailable.",
                    error_details=exception,
                )

            # Invalid request detection.
            invalid_request_indicators = (
                "invalid argument",
                "invalid request",
                "bad request",
                "malformed request",
            )

            if (
                status_code == 400
                or any(
                    indicator in error_message
                    for indicator in invalid_request_indicators
                )
                or exception_name in (
                    "badrequest",
                    "invalidargument",
                )
            ):
                return ModelProviderException(
                    "The model provider rejected the request.",
                    error_details=exception,
                    error_code="MODEL_INVALID_REQUEST",
                )

        # Unknown provider error.
        return ModelProviderException(
            "An unclassified model provider error occurred.",
            error_details=exception,
            error_code="MODEL_UNKNOWN_ERROR",
        )

    def _raise_if_non_recoverable(
        self,
        classified_exception: Exception,
        original_exception: Exception,
    ) -> None:
        """Stop fallback for errors that are not safely recoverable."""

        if isinstance(classified_exception, ConfigurationException):
            if classified_exception is original_exception:
                raise original_exception

            raise classified_exception from original_exception

        if isinstance(
            classified_exception,
            (
                ModelRateLimitException,
                ModelAuthenticationException,
                ModelUnavailableException,
            ),
        ):
            return

        if isinstance(classified_exception, ModelProviderException):
            raise classified_exception from original_exception

    def _invoke_gemini(
        self,
        messages: Any,
        **kwargs: Any,
    ) -> BaseMessage | None:
        """Try Gemini API keys in their configured priority order."""

        available_keys = self.gemini_provider.available_key_numbers

        if not available_keys:
            self._logger.warning(
                "No Gemini API keys are configured. "
                "Proceeding to Ollama."
            )
            return None

        for key_number in available_keys:

            self._logger.info(
                "Attempting Gemini API key slot %s",
                key_number,
            )

            try:
                chat_model = (
                    self.gemini_provider.get_chat_model_for_key(
                        key_number
                    )
                )

                response = chat_model.invoke(
                    messages,
                    **kwargs,
                )

                self._logger.info(
                    "Gemini invocation succeeded using key slot %s",
                    key_number,
                )

                return response

            except Exception as exc:
                classified_exception = self._classify_exception(exc)

                self._raise_if_non_recoverable(
                    classified_exception,
                    exc,
                )

                self._logger.warning(
                    "Gemini key slot %s failed. Error code: %s. "
                    "Trying the next available provider key.",
                    key_number,
                    getattr(
                        classified_exception,
                        "error_code",
                        "UNKNOWN",
                    ),
                )

        self._logger.warning(
            "All configured Gemini key attempts failed. "
            "Switching to Ollama."
        )

        return None

    def _invoke_ollama(
        self,
        messages: Any,
        **kwargs: Any,
    ) -> BaseMessage:
        """Invoke the local Ollama model as the final fallback."""

        self._logger.info("Starting Ollama fallback")

        try:
            chat_model = self.ollama_provider.get_chat_model()

            response = chat_model.invoke(
                messages,
                **kwargs,
            )

            self._logger.info(
                "Ollama invocation completed successfully"
            )

            return response

        except ConfigurationException:
            raise

        except Exception as exc:
            classified_exception = self._classify_exception(exc)

            self._logger.error(
                "Ollama invocation failed. Error code: %s",
                getattr(
                    classified_exception,
                    "error_code",
                    "UNKNOWN",
                ),
            )

            raise ModelProviderException(
                "All available LLM providers failed.",
                error_details=exc,
                error_code="ALL_PROVIDERS_FAILED",
            ) from exc

    def invoke(
        self,
        messages: Any,
        **kwargs: Any,
    ) -> BaseMessage:
        """Invoke Gemini with automatic key rotation and Ollama fallback."""

        self._logger.info(
            "Starting LLM request through Provider Manager"
        )

        gemini_response = self._invoke_gemini(
            messages,
            **kwargs,
        )

        if gemini_response is not None:
            return gemini_response

        self._logger.warning(
            "Gemini could not process the request. "
            "Using Ollama as the final fallback."
        )

        return self._invoke_ollama(
            messages,
            **kwargs,
        )
