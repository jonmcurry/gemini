from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost:5432/claims_staging_db" # Made default more specific
    TEST_DATABASE_URL: Optional[str] = None

    # Memcached settings
    MEMCACHED_HOST: str = "localhost"
    MEMCACHED_PORT: int = 11211

    # Concurrency settings
    MAX_CONCURRENT_CLAIM_PROCESSING: int = 10 # Default value
    # Add other settings as needed later

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding='utf-8')

@lru_cache()
def get_settings():
    return Settings()
