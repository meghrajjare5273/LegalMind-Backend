# app/core/config.py
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ENV: str = "development"
    GEMINI_API_KEY: str = Field(..., env="GEMINI_API_KEY")
    PINECONE_API_KEY: str = Field(..., env="PINECONE_API_KEY")
    PINECONE_INDEX: str = "legal-mind-v2"
    EMB_MODEL: str = "gemini-embedding-exp-03-07"
    LLM_MODEL: str = "gemini-1.5-flash"
    MAX_CHUNK_TOKENS: int = 1_000

settings = Settings()          # import this anywhere
