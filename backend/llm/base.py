
"""Abstract base interface for Talk2SQL LLM providers."""

from abc import ABC, abstractmethod

from langchain_core.language_models.chat_models import BaseChatModel


class LLMProvider(ABC):
    """Common interface that all Talk2SQL LLM providers must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider's name."""
        raise NotImplementedError

    @abstractmethod
    def get_chat_model(self) -> BaseChatModel:
        """Return a LangChain-compatible chat model."""
        raise NotImplementedError
