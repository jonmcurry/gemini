from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field # Ensure Field is imported
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost:5432/claims_staging_db" # Made default more specific
    TEST_DATABASE_URL: Optional[str] = None

    # Database Connection Pool Settings
    DB_POOL_SIZE: int = Field(
        5,
        gt=0,
        description="The number of connections to keep persistently in the pool."
    )
    DB_MAX_OVERFLOW: int = Field(
        10,
        ge=0,
        description="The maximum number of connections that can be opened beyond DB_POOL_SIZE."
    )

    # Memcached settings
    MEMCACHED_HOSTS: str = "localhost" # Comma-separated list of hosts, e.g., "host1,host2,host3"
    MEMCACHED_PORT: int = 11211 # Port used for all listed hosts

    # Concurrency settings
    MAX_CONCURRENT_CLAIM_PROCESSING: int = 10 # Default value

    # ML Model settings
    ML_MODEL_PATH: Optional[str] = "models/dummy_claim_model.tflite"
    ML_FEATURE_COUNT: int = 7
    ML_APPROVAL_THRESHOLD: float = 0.8

    # ML Prediction Cache Settings
    ML_PREDICTION_CACHE_MAXSIZE: int = 10000 # Max number of entries in the prediction cache
    ML_PREDICTION_CACHE_TTL: int = 3600      # TTL for prediction cache entries in seconds (1 hour)

    # ML Feature Cache Settings
    ML_FEATURE_CACHE_MAXSIZE: int = Field(
        5000,
        description="Max number of entries in the ML feature extraction cache."
    )
    ML_FEATURE_CACHE_TTL: int = Field(
        3600,
        description="TTL for ML feature cache entries in seconds (1 hour)."
    )

    # ML A/B Testing Settings
    ML_CHALLENGER_MODEL_PATH: Optional[str] = Field(None, description="Optional path to a challenger ML model for A/B testing.")
    ML_AB_TEST_TRAFFIC_PERCENTAGE_TO_CHALLENGER: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Percentage of traffic (0.0 to 1.0) to route to the challenger model. Default 0.0 (off)."
    )
    ML_AB_TEST_CLAIM_ID_SALT: str = Field(
        "default_ab_salt",
        description="A salt for hashing claim IDs for consistent A/B test routing."
    )

    # Application Security Settings
    APP_ENCRYPTION_KEY: str = "must_be_32_bytes_long_for_aes256_key!"
    # IMPORTANT: This is a default development key.
    # It MUST be overridden by an environment variable in production for security.
    # The key should be a 32-byte (256-bit) cryptographically secure random string.

    # RVU Service Settings
    RVU_MEDICARE_CONVERSION_FACTOR: float = 38.87
    RVU_DEFAULT_VALUE: float = 1.00

    # Detailed Parallel Processing Configuration
    # Concurrency Limits
    VALIDATION_CONCURRENCY: int = 100
    RVU_CALCULATION_CONCURRENCY: int = 80
    TRANSFER_CONCURRENCY: int = 16  # For data transfer operations
    MAX_CONCURRENT_BATCHES: int = 2 # Max number of full batches processed simultaneously by the system

    # Batch Sizes for different operations
    FETCH_BATCH_SIZE: int = 50000       # For fetching initial claims from staging
    VALIDATION_BATCH_SIZE: int = 2000   # Batch size for validation stage
    RVU_BATCH_SIZE: int = 3000          # Batch size for RVU calculation stage
    TRANSFER_BATCH_SIZE: int = 5000     # Batch size for transferring data to production
    MAX_INGESTION_BATCH_SIZE: int = 10000 # Max claims allowed in a single ingestion API call

    # Fetch Retry Settings (new)
    MAX_FETCH_RETRIES: int = Field(3, description="Maximum number of retries for fetching claims in a batch.")
    FETCH_RETRY_DELAY_SECONDS: float = Field(5.0, description="Delay in seconds between fetch retries.")

    # Add other settings as needed later

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding='utf-8')

@lru_cache()
def get_settings():
    return Settings()
