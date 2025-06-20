import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from typing import List, Dict, Any, Optional, Callable # Added Callable
import uuid # Added for generating fallback batch_id for audit

from ....api.models.ingestion_models import IngestionClaim, IngestionResponse # Added IngestionResponse
from ....ingestion.data_ingestion_service import DataIngestionService
# from ....core.security.audit_logger_service import AuditLoggerService # Old logger
from ....core.monitoring.audit_logger import AuditLogger # New AuditLogger
from ....api.dependencies import get_audit_logger # New dependency getter
from ....core.security.encryption_service import EncryptionService
from ....core.config.settings import get_settings


logger = structlog.get_logger(__name__)
router = APIRouter()

# --- Start Temporary Dependency Setup for Route Definition ---
# This would normally be in a central place like `main.py` or `core/dependencies.py`
# and `app.dependency_overrides` used in tests if needed.
# For this subtask, to make the route code runnable if tested in isolation by worker:
from ....core.database.db_session import AsyncSessionLocal # Direct import for factory


_encryption_service_instance: Optional[EncryptionService] = None
# _audit_logger_service_instance: Optional[AuditLoggerService] = None # Old instance
_data_ingestion_service_instance: Optional[DataIngestionService] = None

def temp_get_encryption_service():
    global _encryption_service_instance
    if _encryption_service_instance is None:
        _encryption_service_instance = EncryptionService(encryption_key=get_settings().APP_ENCRYPTION_KEY)
    return _encryption_service_instance

# def temp_get_audit_logger_service(): # Old temporary getter
#     global _audit_logger_service_instance
#     if _audit_logger_service_instance is None:
#         # AsyncSessionLocal itself is the factory that creates sessions.
#         _audit_logger_service_instance = AuditLoggerService(db_session_factory=AsyncSessionLocal)
#     return _audit_logger_service_instance

def temp_get_data_ingestion_service(): # Kept for DataIngestionService if it's still used temporarily
    global _data_ingestion_service_instance
    if _data_ingestion_service_instance is None:
        _data_ingestion_service_instance = DataIngestionService(
            db_session_factory=AsyncSessionLocal, # Pass the session factory
            encryption_service=temp_get_encryption_service() # Use temp getter for encryption service
        )
    return _data_ingestion_service_instance
# --- End Temporary Dependency Setup ---


@router.post("/submissions/claims_batch", status_code=202, response_model=IngestionResponse) # Added response_model
async def submit_claims_batch(
    request: Request,
    claims_batch: List[IngestionClaim],
    ingestion_batch_id: Optional[str] = None,
    audit_logger: AuditLogger = Depends(get_audit_logger), # Use new AuditLogger
    data_ingester: DataIngestionService = Depends(temp_get_data_ingestion_service) # Keep temp for this
) -> IngestionResponse: # Return type is IngestionResponse
    """
    API endpoint to submit a batch of claims for ingestion.
    The actual processing (validation, ML, RVU, transfer) is asynchronous.
    This endpoint stages the data.
    """
    app_settings = get_settings()
    if len(claims_batch) > app_settings.MAX_INGESTION_BATCH_SIZE:
        logger.warn(
            "Received claim submission batch exceeding max allowed size.",
            received_size=len(claims_batch),
            max_size=app_settings.MAX_INGESTION_BATCH_SIZE
        )
        # NOTE: Audit logging for this 413 error could be added here if desired.
        # Example:
        # await audit_logger.log_access(
        #     user_id="api_caller", action="SUBMIT_CLAIMS_BATCH_REJECTED_SIZE", resource="ClaimSubmission",
        #     ip_address=request.client.host if request.client else "unknown",
        #     user_agent=request.headers.get("user-agent"), success=False,
        #     failure_reason=f"Batch size {len(claims_batch)} exceeds max {app_settings.MAX_INGESTION_BATCH_SIZE}",
        #     details={"received_claims": len(claims_batch), "max_allowed": app_settings.MAX_INGESTION_BATCH_SIZE}
        # )
        raise HTTPException(
            status_code=413, # Payload Too Large
            detail=f"Batch size {len(claims_batch)} exceeds maximum allowed size of {app_settings.MAX_INGESTION_BATCH_SIZE} claims."
        )

    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent")
    action_name = "SUBMIT_CLAIMS_BATCH" # Consistent action name

    logger.info(f"Received claims submission batch request from {client_ip}",
                num_claims=len(claims_batch), provided_ingestion_batch_id=ingestion_batch_id)

    if not claims_batch: # This check is now after the size check; it's fine.
        audit_event_batch_id = ingestion_batch_id or f"empty_submission_{uuid.uuid4()}"
        await audit_logger.log_access(
            user_id="api_caller", # Replace with actual user_id if auth is available
            action=action_name,
            resource="ClaimSubmission",
            resource_id=audit_event_batch_id,
            ip_address=client_ip,
            user_agent=user_agent,
            success=False,
            failure_reason="No claims provided in batch.",
            details={"received_claims": 0, "provided_ingestion_batch_id": ingestion_batch_id}
        )
        raise HTTPException(status_code=400, detail="No claims provided in the batch.")

    ingestion_summary: Optional[IngestionResponse] = None
    try:
        ingestion_summary = await data_ingester.ingest_claims_batch(
            raw_claims_data=claims_batch,
            provided_batch_id=ingestion_batch_id
        )

        # Determine overall success for audit based on whether any claims failed staging
        success_status = ingestion_summary.failed_ingestion_claims == 0

        await audit_logger.log_access(
            user_id="api_caller", # Replace with actual user_id if auth is available
            action=action_name,
            resource="ClaimSubmission",
            resource_id=ingestion_summary.ingestion_batch_id,
            ip_address=client_ip,
            user_agent=user_agent,
            success=success_status,
            failure_reason=f"Failed: {ingestion_summary.failed_ingestion_claims}, Errors: {ingestion_summary.errors}" if not success_status else None,
            details={
                "received_claims": ingestion_summary.received_claims,
                "successfully_staged_claims": ingestion_summary.successfully_staged_claims,
                "failed_ingestion_claims": ingestion_summary.failed_ingestion_claims,
                "errors": ingestion_summary.errors # include errors in details
            }
        )
        return ingestion_summary

    except Exception as e:
        logger.error(f"Critical error in claims submission API: {e}", exc_info=True,
                     provided_ingestion_batch_id=ingestion_batch_id)

        # Use batch_id from partial summary if available, else generate one
        audit_event_batch_id = ingestion_summary.ingestion_batch_id if ingestion_summary else \
                               (ingestion_batch_id or f"error_submission_{uuid.uuid4()}")

        await audit_logger.log_access(
            user_id="api_caller", # Replace with actual user_id if auth is available
            action=action_name,
            resource="ClaimSubmission",
            resource_id=audit_event_batch_id,
            ip_address=client_ip,
            user_agent=user_agent,
            success=False,
            failure_reason=f"Internal server error: {str(e)}",
            details={"received_claims": len(claims_batch), "provided_ingestion_batch_id": ingestion_batch_id}
        )
        # Ensure an IngestionResponse-like structure is returned for 500 if possible,
        # or re-evaluate if HTTPException is sufficient and if client expects specific structure on error
        raise HTTPException(status_code=500, detail=f"Internal server error during claim submission: {str(e)}")

```
