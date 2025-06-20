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

    # ML Model settings
    ML_MODEL_PATH: Optional[str] = "models/dummy_claim_model.tflite"
    ML_FEATURE_COUNT: int = 7
    ML_APPROVAL_THRESHOLD: float = 0.8

    # Application Security Settings
    APP_ENCRYPTION_KEY: str = "must_be_32_bytes_long_for_aes256_key!"
    # IMPORTANT: This is a default development key.
    # It MUST be overridden by an environment variable in production for security.
    # The key should be a 32-byte (256-bit) cryptographically secure random string.

    # RVU Service Settings
    RVU_MEDICARE_CONVERSION_FACTOR: float = 38.87
    RVU_DEFAULT_VALUE: float = 1.00

    # Add other settings as needed later

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding='utf-8')

@lru_cache()
def get_settings():
    return Settings()
