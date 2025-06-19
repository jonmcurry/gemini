from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone # To ensure timezone awareness if not handled by DB default
from typing import Optional, Dict, Any
import structlog

# Assuming AuditLogModel is in:
from ..database.models.audit_log_db import AuditLogModel

logger = structlog.get_logger(__name__)

class AuditLoggerService:
    def __init__(self, db_session: AsyncSession):
        self.db = db_session

    async def log_event(
        self,
        action: str,
        success: bool,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        resource_id: Optional[str] = None,
        patient_id_hash: Optional[str] = None, # Placeholder for now
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        failure_reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Creates and saves an audit log entry.
        """
        logger.debug(
            "Attempting to log audit event",
            action=action, user_id=user_id, resource=resource,
            resource_id=resource_id, success=success
        )

        audit_log_entry = AuditLogModel(
            # timestamp is server_default=func.now()
            user_id=user_id,
            action=action,
            resource=resource,
            resource_id=resource_id,
            patient_id_hash=patient_id_hash, # Actual hashing TBD
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            success=success,
            failure_reason=failure_reason if not success else None, # Only store reason if failed
            details=details
        )

        try:
            self.db.add(audit_log_entry)
            await self.db.commit()
            await self.db.refresh(audit_log_entry) # To get id, timestamp etc.
            logger.info(
                "Audit event logged successfully",
                audit_log_id=audit_log_entry.id, action=action, user_id=user_id, success=success
            )
        except Exception as e:
            # If audit logging fails, we should not let it break the main operation.
            # Log the error to standard application logs.
            # In a more robust system, might push to a fallback logging mechanism (e.g., file, different queue).
            logger.error(
                "Failed to save audit log to database",
                action=action, user_id=user_id, resource=resource, resource_id=resource_id,
                original_error=str(e),
                exc_info=True # Include stack trace for the audit logging failure
            )
            # Do not rollback self.db here, as the main operation might have already committed
            # or needs to commit. Audit logging should be as decoupled as possible from main transaction flow
            # unless it's critical for the transaction itself (which is rare for general audit).
            # This means if the session passed to AuditLoggerService is the same one used by the main
            # operation, and the main operation fails and rolls back, this audit log add will also be rolled back.
            # If audit must survive main op rollback, it needs its own session/transaction.
            # For now, assume it shares session and lives/dies with main op's transaction.
            #
            # It might be safer to attempt a rollback of just the audit entry if the session is shared
            # and the commit for audit is separate, but SQLAlchemy sessions don't quite work that way
            # without nested transactions or more complex session management.
            # If self.db.commit() here is THE commit for the whole operation, then this try/except
            # is mainly for logging the audit failure itself if the commit fails due to the audit entry.
            # If audit is "best effort" and main op commit is separate, this is fine.
            # Given the current structure, this commit is specific to the audit log.
            # If this commit fails, it will roll back only the audit log entry.
            # If the session `self.db` is shared and committed by an outer scope, then this commit
            # might be redundant or even cause issues if called mid-transaction.
            # For now, assuming this service gets a session and is responsible for its own commit for audit.
            # To make it truly independent, it would need its own AsyncSessionLocal().
            # This will be reviewed when integrating.
            # For this subtask, the provided structure is implemented.
            # A simple solution is to not commit here, but let the calling function commit the session
            # that this audit log entry was added to. Or make this service use its own session.
            # Let's assume for now the service manages its own unit of work for the audit log.
            # This implies the session passed might need careful handling or be a factory.
            # The current structure is okay if the session is committed by the caller *after* this.
            # However, the `await self.db.commit()` is here. This means it assumes it controls the commit for this entry.
            # This is a common point of complexity for audit logging.
            # For now, leaving as is per prompt.
