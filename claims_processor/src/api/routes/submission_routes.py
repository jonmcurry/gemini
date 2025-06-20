import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from typing import List, Dict, Any, Optional, Callable # Added Callable
from sqlalchemy.ext.asyncio import AsyncSession # For type hint in temp factory

from ....api.models.ingestion_models import IngestionClaim
from ....ingestion.data_ingestion_service import DataIngestionService
from ....core.security.audit_logger_service import AuditLoggerService
from ....core.security.encryption_service import EncryptionService
from ....core.config.settings import get_settings

logger = structlog.get_logger(__name__)
router = APIRouter()

# --- Start Temporary Dependency Setup for Route Definition ---
# This would normally be in a central place like `main.py` or `core/dependencies.py`
# and `app.dependency_overrides` used in tests if needed.
# For this subtask, to make the route code runnable if tested in isolation by worker:
from ....core.database.db_session import AsyncSessionLocal # Direct import for factory

# Note: The type hint for db_session_factory in AuditLoggerService and DataIngestionService
# is Callable[[], AsyncSession]. AsyncSessionLocal is a sessionmaker instance,
# which when called, returns an AsyncSession. So it matches.

_encryption_service_instance: Optional[EncryptionService] = None
_audit_logger_service_instance: Optional[AuditLoggerService] = None
_data_ingestion_service_instance: Optional[DataIngestionService] = None

def temp_get_encryption_service():
    global _encryption_service_instance
    if _encryption_service_instance is None:
        _encryption_service_instance = EncryptionService(encryption_key=get_settings().APP_ENCRYPTION_KEY)
    return _encryption_service_instance

def temp_get_audit_logger_service():
    global _audit_logger_service_instance
    if _audit_logger_service_instance is None:
        # AsyncSessionLocal itself is the factory that creates sessions.
        _audit_logger_service_instance = AuditLoggerService(db_session_factory=AsyncSessionLocal)
    return _audit_logger_service_instance

def temp_get_data_ingestion_service():
    global _data_ingestion_service_instance
    if _data_ingestion_service_instance is None:
        _data_ingestion_service_instance = DataIngestionService(
            db_session_factory=AsyncSessionLocal, # Pass the session factory
            encryption_service=temp_get_encryption_service()
        )
    return _data_ingestion_service_instance
# --- End Temporary Dependency Setup ---


@router.post("/submissions/claims_batch", status_code=202)
async def submit_claims_batch(
    request: Request,
    claims_batch: List[IngestionClaim],
    ingestion_batch_id: Optional[str] = None,
    # Using temporary direct instantiation for this subtask.
    # In a real app, use FastAPI's `Depends` with properly configured providers.
    # audit_logger: AuditLoggerService = Depends(temp_get_audit_logger_service),
    # data_ingester: DataIngestionService = Depends(temp_get_data_ingestion_service)
) -> Dict[str, Any]:
    """
    API endpoint to submit a batch of claims for ingestion.
    The actual processing (validation, ML, RVU, transfer) is asynchronous.
    This endpoint stages the data.
    """
    # Direct instantiation for the subtask
    audit_logger = temp_get_audit_logger_service()
    data_ingester = temp_get_data_ingestion_service()

    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"Received claims submission batch request from {client_ip}",
                num_claims=len(claims_batch), provided_ingestion_batch_id=ingestion_batch_id)

    if not claims_batch:
        # Log audit event for received empty batch
        # Since data_ingester is not called, need to generate a batch_id for audit if none provided
        audit_event_batch_id = ingestion_batch_id or f"empty_submission_{str(uuid.uuid4())[:8]}"
        await audit_logger.log_event(
            action="SUBMIT_CLAIMS_BATCH_API", resource="ClaimBatchSubmission", success=False,
            resource_id=audit_event_batch_id,
            failure_reason="No claims provided in batch.", user_id="api_caller", client_ip=client_ip,
            details={"num_claims": 0, "provided_ingestion_batch_id": ingestion_batch_id}
        )
        raise HTTPException(status_code=400, detail="No claims provided in the batch.")

    try:
        ingestion_summary = await data_ingester.ingest_claims_batch(
            raw_claims_data=claims_batch,
            provided_batch_id=ingestion_batch_id
        )

        # Determine overall success for audit based on whether any claims failed staging
        # The API call itself is "successful" if it reached here and the service processed the request.
        # The content of ingestion_summary indicates partial or full data staging success.
        all_claims_staged_successfully = ingestion_summary.get("failed_ingestion_claims", 0) == 0

        await audit_logger.log_event(
            action="SUBMIT_CLAIMS_BATCH_API", resource="ClaimBatchSubmission",
            resource_id=ingestion_summary.get("ingestion_batch_id"), # Use ID from service summary
            success=True, # API call processed the request. Detailed success in summary.
            failure_reason=None if all_claims_staged_successfully else "Some claims failed staging, see details.",
            user_id="api_caller",
            client_ip=client_ip,
            details=ingestion_summary
        )

        # Return 202 Accepted, as the actual processing by ParallelClaimsProcessor is async later.
        # The summary from DataIngestionService indicates result of STAGING the data.
        return ingestion_summary

    except Exception as e:
        # This catches errors from DataIngestionService instantiation or unexpected errors in ingest_claims_batch
        logger.error(f"Critical error in claims submission API: {e}", exc_info=True,
                     provided_ingestion_batch_id=ingestion_batch_id)

        # Log audit event for critical failure
        # Generate a batch_id for audit if none provided or not available from a partial summary
        audit_event_batch_id = ingestion_batch_id or f"error_submission_{str(uuid.uuid4())[:8]}"
        await audit_logger.log_event(
            action="SUBMIT_CLAIMS_BATCH_API", resource="ClaimBatchSubmission",
            resource_id=audit_event_batch_id, success=False,
            failure_reason=f"Internal server error: {str(e)}", user_id="api_caller", client_ip=client_ip,
            details={"num_claims": len(claims_batch), "provided_ingestion_batch_id": ingestion_batch_id}
        )
        raise HTTPException(status_code=500, detail=f"Internal server error during claim submission: {str(e)}")

```
