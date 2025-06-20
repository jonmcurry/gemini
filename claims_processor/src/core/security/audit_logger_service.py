import structlog
from typing import Optional, Dict, Any, Callable
from sqlalchemy.ext.asyncio import AsyncSession
# from datetime import datetime # Not strictly needed if AuditLogModel uses server_default for timestamp
import hashlib # Added for SHA-256 hashing

# Assuming AuditLogModel is correctly imported
from ..database.models.audit_log_db import AuditLogModel
# Assuming db_session.py provides a way to get a session, e.g., AsyncSessionLocal
# from ..database.db_session import AsyncSessionLocal # Example, actual factory might differ

logger = structlog.get_logger(__name__)

class AuditLoggerService:
    """
    Service for logging audit events to the database.
    """

    def __init__(self, db_session_factory: Callable[[], AsyncSession]): # Expects a factory that returns a session
        """
        Initializes the AuditLoggerService.

        Args:
            db_session_factory: An asynchronous factory function that provides an AsyncSession.
                                e.g., the `get_db_session` context manager producer from `db_session.py`
                                or `AsyncSessionLocal` directly if not using Depends-like context.
                                For a service, a factory that can be called to create a session is typical.
                                Let's assume it's a callable that yields an AsyncSession context manager.
                                Or more simply, a callable that returns a session directly for internal use.
                                The processor uses `async with self.db_session_factory() as session:`.
                                So this service should probably do the same.
        """
        self.db_session_factory = db_session_factory
        logger.info("AuditLoggerService initialized.")

    async def log_event(
        self,
        action: str,
        success: bool,
        resource: Optional[str] = None,
        resource_id: Optional[str] = None,
        user_id: Optional[str] = "system", # Default to 'system' for automated processes
        patient_id_to_hash: Optional[str] = None, # Raw Patient ID to be hashed
        failure_reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None # For user sessions via API
    ) -> bool:
        """
        Logs an audit event to the database.

        Args:
            action: Verb describing the action (e.g., 'PROCESS_BATCH_START', 'CREATE_CLAIM').
            success: Boolean indicating if the action was successful.
            resource: The type of resource affected (e.g., 'ClaimBatch', 'Claim').
            resource_id: The ID of the specific resource instance.
            user_id: Identifier for the user or system performing the action.
            patient_id_to_hash: Raw Patient ID. Hashing will be placeholder for now.
            failure_reason: Reason for failure if success is False.
            details: Additional structured information about the event (JSONB).
            client_ip: IP address of the client (for API calls).
            user_agent: User-Agent string of the client (for API calls).
            session_id: Session identifier (for API calls).

        Returns:
            True if the log was successfully saved, False otherwise.
        """

        # Placeholder for hashing patient_id. In a real system, use a consistent, secure hash.
        hashed_patient_id = None
        if patient_id_to_hash:
            # Using direct SHA-256. For enhanced security in a real system,
            # consider using a keyed hash (HMAC) or including a system-wide pepper/salt
            # that is managed via configuration and not stored with the hash.
            # The purpose here is to make the patient_id non-reversible in logs,
            # not necessarily for authentication or password storage.
            data_to_hash = patient_id_to_hash
            hashed_patient_id = hashlib.sha256(data_to_hash.encode('utf-8')).hexdigest()
            # SHA-256 hexdigest is 64 characters. AuditLogModel.patient_id_hash is String(255).

        log_entry_data = {
            # 'timestamp' has server_default=func.now() in AuditLogModel
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "resource_id": resource_id,
            "patient_id_hash": hashed_patient_id,
            "ip_address": client_ip,
            "user_agent": user_agent,
            "session_id": session_id,
            "success": success,
            "failure_reason": failure_reason if not success else None, # Only store reason if failed
            "details": details
        }

        try:
            # The db_session_factory is expected to be an async context manager provider
            # like the one used in ParallelClaimsProcessor.
            async with self.db_session_factory() as session: # type: AsyncSession
                async with session.begin(): # Start a transaction for the log entry
                    audit_log = AuditLogModel(**log_entry_data)
                    session.add(audit_log)
                    # The commit will happen when session.begin() exits if no error,
                    # or rollback if an error occurs.
                # If session.begin() is not used, then await session.commit() would be needed here.
                # Using session.begin() is cleaner for single operations.

            logger.debug("Audit event logged successfully.", action=action, resource=resource, resource_id=resource_id, user_id=user_id)
            return True
        except Exception as e:
            logger.error(
                "Failed to log audit event to database.",
                action=action,
                resource=resource,
                resource_id=resource_id,
                error=str(e),
                exc_info=True
            )
            return False

    # Placeholder for actual hashing function if needed internally
    # def _hash_patient_id(self, patient_id: str) -> str:
    #     # Replace with actual secure hashing (e.g., HMAC-SHA256 with a secret key)
    #     # For now, just a conceptual placeholder.
    #     if not patient_id:
    #         return None
    #     return f"hashed_{hashlib.sha256(patient_id.encode()).hexdigest()}"
