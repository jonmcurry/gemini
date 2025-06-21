from typing import Callable # Import Callable
from sqlalchemy.ext.asyncio import AsyncSession # Import AsyncSession

from claims_processor.src.core.monitoring.audit_logger import AuditLogger
from claims_processor.src.core.database.db_session import AsyncSessionLocal
from claims_processor.src.core.monitoring.app_metrics import MetricsCollector
from claims_processor.src.core.security.encryption_service import EncryptionService # Added
from claims_processor.src.core.config.settings import get_settings # Added
from typing import Optional
import structlog

# Note: get_async_session_factory might need to be created or already exist in db_session.py
# If it doesn't exist, this subtask cannot complete this part without modifying db_session.py.
# Assuming get_async_session_factory returns a callable that yields an AsyncSession.

logger = structlog.get_logger(__name__) # Logger for dependency related messages

_audit_logger_instance: Optional[AuditLogger] = None
_metrics_collector_instance: Optional[MetricsCollector] = None
_encryption_service_instance: Optional[EncryptionService] = None # Added for EncryptionService singleton

def get_async_session_factory() -> Callable[[], AsyncSession]:
    """Returns the raw session factory callable."""
    return AsyncSessionLocal

def get_audit_logger() -> AuditLogger:
    global _audit_logger_instance
    if _audit_logger_instance is None:
        factory = get_async_session_factory()
        _audit_logger_instance = AuditLogger(db_session_factory=factory)
        logger.info("Default AuditLogger instance created.") # Added log
    return _audit_logger_instance

def get_metrics_collector() -> MetricsCollector:
    global _metrics_collector_instance
    if _metrics_collector_instance is None:
        _metrics_collector_instance = MetricsCollector()
        logger.info("Default MetricsCollector instance created.")
    return _metrics_collector_instance

def get_encryption_service() -> EncryptionService:
    global _encryption_service_instance
    if _encryption_service_instance is None:
        app_settings = get_settings()
        if not app_settings.APP_ENCRYPTION_KEY:
            logger.error("APP_ENCRYPTION_KEY is not set. EncryptionService cannot be initialized.")
            raise ValueError("APP_ENCRYPTION_KEY must be set for EncryptionService.")
        _encryption_service_instance = EncryptionService(key=app_settings.APP_ENCRYPTION_KEY)
        logger.info("Default EncryptionService instance created.")
    return _encryption_service_instance
