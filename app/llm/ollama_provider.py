"""Ollama provider implementation for Talk2SQL."""


from langchain_ollama import ChatOllama
from app.core.config import Settings,get_settings
from app.core.logging_config import logger
from app.exceptions.custom_exceptions import CustomException
from app.llm.base import LLMProvider


class OllamaProvider(LLMProvider):

    """Generate text using a locally hosted Ollama model."""
    
    
    def __init__(self, settings:Settings |None=None)->None:
        """Initialize the provider with application settings."""
        self.settings = settings or get_settings()

    def generate(self,prompt:str)->str:
        """Generate text using the configured Ollama model."""
        
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")
        
        base_url = self.settings.ollama_base_url
        model_name = self.settings.ollama_model

        if ( not isinstance(base_url, str) or not base_url.strip()
            or not isinstance(model_name, str) or not model_name.strip() ):
            logger.error("Ollama configuration is incomplete.")
            raise CustomException(
                "Ollama base URL and model name must be configured."
            )
            
        try:
            model=ChatOllama(
                model = model_name.strip(),
                base_url=base_url.strip(),
                temperature=0,
                num_predict=128,
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
            
            generated_text = generated_text.strip()

            if not generated_text:
                logger.error("Ollama returned an empty response.")
                raise CustomException(
                    "Ollama returned an empty response."
                )
            
            logger.info(
                "Ollama generation succeeded using model '%s'.",
                model_name.strip(),
            )

            return generated_text

        except CustomException:
            raise

        except Exception as exc:
            logger.error(
                "Ollama generation failed with %s.",
                type(exc).__name__,
            )

            raise CustomException(
                "Ollama could not generate a response. "
                "Check that Ollama is running and the configured "
                "model is available.",
                exc,
            ) from exc