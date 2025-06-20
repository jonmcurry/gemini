import hashlib
import structlog
from typing import Optional, Dict, Any, Callable
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone # Import for timestamp if manually setting

# Assuming AuditLogModel will be importable after Part 1
from claims_processor.src.core.database.models.audit_log_db import AuditLogModel

logger = structlog.get_logger(__name__)

class AuditLogger:
    def __init__(self, db_session_factory: Callable[[], AsyncSession]):
        """
        Initializes the AuditLogger.
        Args:
            db_session_factory: An asynchronous factory function that provides an AsyncSession.
        """
        self.db_session_factory = db_session_factory
        logger.info("AuditLogger initialized.")

    def _hash_identifier(self, identifier: str) -> str:
        """Hashes an identifier using SHA-256."""
        if not identifier:
            return ""
        return hashlib.sha256(identifier.encode('utf-8')).hexdigest()

    async def log_access(
        self,
        user_id: Optional[str],
        action: str,
        resource: Optional[str] = None,
        resource_id: Optional[str] = None,
        patient_id: Optional[str] = None, # Raw Patient ID
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        success: bool = True,
        failure_reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Logs an access event to the audit log.
        The 'timestamp' is handled by the database default.
        """
        patient_id_hashed = None
        if patient_id:
            patient_id_hashed = self._hash_identifier(patient_id)

        audit_entry = AuditLogModel(
            user_id=user_id,
            action=action,
            resource=resource,
            resource_id=resource_id,
            patient_id_hash=patient_id_hashed,
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            success=success,
            failure_reason=failure_reason,
            details=details
        )

        try:
            async with self.db_session_factory() as session: # type: AsyncSession
                async with session.begin():
                    session.add(audit_entry)
                # Commit is implicit with session.begin() context manager success
            logger.debug("Audit log entry successfully stored.", action=action, resource=resource, user_id=user_id)
        except Exception as e:
            logger.error("Failed to store audit log entry to database.",
                         action=action, resource=resource, user_id=user_id,
                         error=str(e), exc_info=True)
            # Depending on policy, might re-raise or just log the failure to store audit.
            # For now, just logging.

# Example of how to potentially provide an AuditLogger instance (e.g., via dependency injection)
# This part would typically be in a main application setup or dependency injection module.
# from claims_processor.src.core.database.db_session import get_async_session_factory
#
# _audit_logger_instance: Optional[AuditLogger] = None
#
# def get_audit_logger() -> AuditLogger:
#     global _audit_logger_instance
#     if _audit_logger_instance is None:
#         session_factory = get_async_session_factory() # Assuming this function exists
#         _audit_logger_instance = AuditLogger(db_session_factory=session_factory)
#     return _audit_logger_instance
