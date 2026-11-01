
import traceback

from pathlib import Path


class Talk2SQLException(Exception):
    """Base exception for all Talk2SQL application errors."""

    default_error_code = "TALK2SQL_ERROR"
    retryable = False

    def __init__(
        self,
        error_message: str,
        error_details: Exception | None = None,
        error_code: str | None = None,
    ) -> None:

        self.original_message = error_message
        self.error_details = error_details

        self.error_code = (
            error_code or self.default_error_code
        )

        self.file_name = "Unknown"
        self.line_number = 0

        # Extract location from the original exception.
        if error_details is not None:

            tb = error_details.__traceback__

            if tb is not None:

                frames = traceback.extract_tb(tb)

                if frames:
                    last_frame = frames[-1]

                    self.file_name = Path(
                        last_frame.filename
                    ).name

                    self.line_number = last_frame.lineno

        detailed_message = (
            f"Error code: {self.error_code}\n"
            f"Error occurred in file: {self.file_name}\n"
            f"Line number: {self.line_number}\n"
            f"Error message: {self.original_message}"
        )

        super().__init__(detailed_message)


class ConfigurationException(Talk2SQLException):
    """Raised when application configuration is invalid."""

    default_error_code = "CONFIGURATION_ERROR"


class ModelProviderException(Talk2SQLException):
    """Base exception for model provider failures."""

    default_error_code = "MODEL_PROVIDER_ERROR"


class ModelRateLimitException(ModelProviderException):
    """Raised when a model provider reaches its rate limit."""

    default_error_code = "MODEL_RATE_LIMIT"
    retryable = True


class ModelAuthenticationException(ModelProviderException):
    """Raised when model authentication fails."""

    default_error_code = "MODEL_AUTHENTICATION_ERROR"


class ModelUnavailableException(ModelProviderException):
    """Raised when a model service is unavailable."""

    default_error_code = "MODEL_UNAVAILABLE"
    retryable = True


class DatabaseException(Talk2SQLException):
    """Raised when a database operation fails."""

    default_error_code = "DATABASE_ERROR"


class SQLValidationException(Talk2SQLException):
    """Raised when SQL validation fails."""

    default_error_code = "SQL_VALIDATION_ERROR"


class ResultVerificationException(Talk2SQLException):
    """Raised when result verification fails."""

    default_error_code = "RESULT_VERIFICATION_ERROR"


# Backward compatibility with the previous project.
CustomException = Talk2SQLException


__all__ = [
    "Talk2SQLException",
    "CustomException",
    "ConfigurationException",
    "ModelProviderException",
    "ModelRateLimitException",
    "ModelAuthenticationException",
    "ModelUnavailableException",
    "DatabaseException",
    "SQLValidationException",
    "ResultVerificationException",
]
