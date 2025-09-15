# app/core/config.py
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    ENV: str = "development"
    
    # Change this to accept a comma-separated string of keys
    GEMINI_API_KEYS: str = Field("AIzaSyBcVvgdrzeggnBxx1QP3L63_heST1ctkB8,AIzaSyBEtbjYNA3AmPbe8DLUDpIUaWP9cAX0vu8,AIzaSyA_9dbei3n2RhpymDNW04RT-3_e5ro2muc,AIzaSyCWqH4CpR1EfWcmF-yiq26xrwxyooPcrDs", alias="GEMINI_API_KEYS") 
    
    LLM_MODEL: str = "gemini-1.5-flash"
    MAX_CHUNK_TOKENS: int = 1_000

    MAX_FILE_SIZE_MB: int = 10
    CACHE_TTL: int = 3600
    AI_ENHANCEMENT_ENABLED: bool = True
    
    # Add a property to automatically split the keys into a list
    @property
    def GEMINI_KEY_LIST(self) -> List[str]:
        if not self.GEMINI_API_KEYS:
            return []
        return [key.strip() for key in self.GEMINI_API_KEYS.split(',') if key.strip()]

    class Config:
        env_file = ".env"
        # Allow reading GEMINI_API_KEYS from the environment
        env_prefix = '' 

settings = Settings()