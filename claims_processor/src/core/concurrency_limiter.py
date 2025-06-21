import asyncio
import structlog

from claims_processor.src.core.config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

# Global semaphore to limit concurrent batch processing tasks
# This instance will be shared across the application where imported.
BATCH_PROCESSING_SEMAPHORE = asyncio.Semaphore(settings.MAX_CONCURRENT_BATCHES)

logger.info("Concurrency limiter initialized.", max_concurrent_batches=settings.MAX_CONCURRENT_BATCHES)

def get_batch_processing_semaphore() -> asyncio.Semaphore:
    """Returns the global batch processing semaphore."""
    return BATCH_PROCESSING_SEMAPHORE
