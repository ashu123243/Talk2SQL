
"""Gemini provider implementation for Talk2SQL."""

from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import Settings, get_settings
from app.core.logging_config import logger
from app.exceptions.custom_exceptions import CustomException
from app.llm.base import LLMProvider


class GeminiProvider(LLMProvider):
    """Generate text using the Gemini API."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the provider with application settings."""
        self.settings = settings or get_settings()

    def generate(self, prompt: str) -> str:
        """Generate text using Gemini and configured API keys."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        api_keys = [
            key.strip()
            for key in [
                self.settings.gemini_api_key_1,
                self.settings.gemini_api_key_2,
                self.settings.gemini_api_key_3,
            ]
            if isinstance(key, str) and key.strip()
        ]

        if not api_keys:
            logger.error("No Gemini API keys are configured.")
            raise CustomException("No Gemini API keys are configured.")

        last_error: Exception | None = None

        for api_key in api_keys:
            try:
                model = ChatGoogleGenerativeAI(
                    model=self.settings.gemini_model,
                    google_api_key=api_key,
                    temperature=0,
                )

                response = model.invoke(prompt)
                content = response.content

                if isinstance(content, str):
                    generated_text = content

                elif isinstance(content, list):
                    text_parts = [
                        block["text"]
                        for block in content
                        if isinstance(block, dict)
                        and block.get("type") == "text"
                        and isinstance(block.get("text"), str)
                    ]
                    generated_text = "\n".join(text_parts)

                else:
                    generated_text = ""

                if not generated_text.strip():
                    logger.error("Gemini returned an empty response.")
                    raise CustomException(
                        "Gemini returned an empty response."
                    )

                logger.info("Gemini generation succeeded.")
                return generated_text.strip()

            except CustomException:
                raise

            except Exception as exc:
                last_error = exc

                logger.warning(
                    "Gemini request failed with %s; "
                    "trying the next configured key.",
                    type(exc).__name__,
                )

        logger.error(
            "Gemini generation failed for all configured API keys."
        )

        raise CustomException(
            "Gemini is unavailable using the configured API keys.",
            last_error,
        ) from last_error
