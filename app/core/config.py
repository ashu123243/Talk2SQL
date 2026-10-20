from pydantic_settings import BaseSettings,SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    
    """Central configuration for Talk2SQL."""
    
    app_name:str ="Talk2SQL"
    app_env:str ="developement"
    debug:bool=True
    llm_provider :str ='gemini'
    
    gemini_model:str = 'gemini-3.8-flash'
    gemini_api_key_1: str = ""
    gemini_api_key_2: str = ""
    gemini_api_key_3: str = ""
    
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:7b"

    database_url: str = (
        "postgresql+psycopg://talk2sql_user:"
        "your_password@localhost:5432/talk2sql_db"
    )
    
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    sql_query_timeout: int = 30
    sql_max_rows: int = 100
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached application settings instance."""
    return Settings()