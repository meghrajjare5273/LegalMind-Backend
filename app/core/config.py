# app/core/config.py
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ENV: str = "development"
    GEMINI_API_KEY: str = Field(" ")
    LLM_MODEL: str = "gemini-1.5-flash"
    MAX_CHUNK_TOKENS: int = 1_000

    MAX_FILE_SIZE_MB: int = 10
    CACHE_TTL: int = 3600
    AI_ENHANCEMENT_ENABLED: bool = True
    
    class Config:
        env_file = ".env"

settings = Settings()
