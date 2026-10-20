"""Abstract interface for Talk2SQL LLM providers."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):

    """Define the common interface for all LLM providers."""
    
    @abstractmethod
    def generate(self, prompt:str)->str:
        """Generate text from a prompt.

        Args:
            prompt: The input text sent to the language model.

        Returns:
            The generated text as a string.
        """
        raise NotImplementedError(
            "Subclasses must implement the generate() method."
        )
            
