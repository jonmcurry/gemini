from typing import Callable # Import Callable
from sqlalchemy.ext.asyncio import AsyncSession # Import AsyncSession

from claims_processor.src.core.monitoring.audit_logger import AuditLogger
from claims_processor.src.core.database.db_session import AsyncSessionLocal # Import AsyncSessionLocal
# Note: get_async_session_factory might need to be created or already exist in db_session.py
# If it doesn't exist, this subtask cannot complete this part without modifying db_session.py.
# Assuming get_async_session_factory returns a callable that yields an AsyncSession.

_audit_logger_instance = None

def get_async_session_factory() -> Callable[[], AsyncSession]:
    """Returns the raw session factory callable."""
    return AsyncSessionLocal

def get_audit_logger() -> AuditLogger:
    global _audit_logger_instance
    if _audit_logger_instance is None:
        factory = get_async_session_factory()
        _audit_logger_instance = AuditLogger(db_session_factory=factory)
    return _audit_logger_instance
