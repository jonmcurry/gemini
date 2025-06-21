from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional, Callable
from fastapi import Query
import asyncio # Added for Semaphore

from ..models.claim_models import ClaimCreate, ClaimResponse, BatchProcessResponse
from ...core.database.db_session import get_db_session, AsyncSessionLocal
from ...core.database.models.claims_db import ClaimModel
from ...core.monitoring.audit_logger import AuditLogger
from ..dependencies import get_audit_logger
from ...core.monitoring.app_metrics import MetricsCollector
from ..dependencies import get_metrics_collector
from ...processing.claims_processing_service import ClaimProcessingService
from ...core.security.encryption_service import EncryptionService # Added
from ..dependencies import get_encryption_service # Added
from ...core.concurrency_limiter import get_batch_processing_semaphore
from ...core.config.settings import get_settings
import structlog

logger = structlog.get_logger(__name__)
router = APIRouter()

@router.post("/", response_model=ClaimResponse)
async def create_claim(
    request: Request,
    claim_data: ClaimCreate,
    db: AsyncSession = Depends(get_db_session),
    audit_logger: AuditLogger = Depends(get_audit_logger) # Use new AuditLogger
):
    logger.info("Received request to create claim", claim_id=claim_data.claim_id, facility_id=claim_data.facility_id)

    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    action_details = {"submitted_claim_id": claim_data.claim_id, "total_charges": float(claim_data.total_charges)}
    resource_id_for_audit = claim_data.claim_id # Use this for attempts before db_claim_instance.id is available

    # Log attempt (optional, but can be useful)
    # await audit_logger.log_access(
    #     user_id=None, action="CREATE_CLAIM_ATTEMPT", resource="Claim", resource_id=resource_id_for_audit,
    #     ip_address=client_ip, user_agent=user_agent, success=True, details=action_details)

    try:
        existing_claim_stmt = await db.execute(select(ClaimModel).where(ClaimModel.claim_id == claim_data.claim_id))
        if existing_claim_stmt.scalars().first() is not None:
            logger.warn("Claim ID already exists", claim_id=claim_data.claim_id)
            failure_reason_detail = f"Claim with ID {claim_data.claim_id} already exists."
            await audit_logger.log_access(
                user_id=None, action="CREATE_CLAIM_DUPLICATE", resource="Claim", resource_id=resource_id_for_audit,
                ip_address=client_ip, user_agent=user_agent, success=False,
                failure_reason=failure_reason_detail, details=action_details
            )
            raise HTTPException(status_code=409, detail=failure_reason_detail)

        claim_model_data = claim_data.model_dump(exclude_unset=True)
        db_claim_instance = ClaimModel(**claim_model_data)

        db.add(db_claim_instance)
        await db.commit()
        await db.refresh(db_claim_instance)
        logger.info("Claim saved to database successfully", claim_id=db_claim_instance.claim_id, db_id=db_claim_instance.id)

        await audit_logger.log_access(
            user_id=None, action="CREATE_CLAIM_SUCCESS", resource="Claim", resource_id=db_claim_instance.claim_id,
            ip_address=client_ip, user_agent=user_agent, success=True,
            details={"db_id": db_claim_instance.id, **action_details}
        )
        return db_claim_instance

    except HTTPException as http_exc:
        # For duplicate (409), it's already logged. For other HTTPExceptions:
        if http_exc.status_code != 409:
            await audit_logger.log_access(
                user_id=None, action="CREATE_CLAIM_ERROR", resource="Claim", resource_id=resource_id_for_audit,
                ip_address=client_ip, user_agent=user_agent, success=False,
                failure_reason=f"HTTPException: {http_exc.status_code} - {http_exc.detail}", details=action_details
            )
        raise http_exc

    except Exception as e:
        logger.error("Error saving claim to database (unexpected)", claim_id=claim_data.claim_id, error=str(e), exc_info=True)
        try:
            if db.is_active: await db.rollback()
        except Exception as rb_exc: logger.error("Error during rollback attempt", error=str(rb_exc), exc_info=True)

        failure_reason_detail = f"Unexpected error: {str(e)}"
        await audit_logger.log_access(
            user_id=None, action="CREATE_CLAIM_ERROR", resource="Claim", resource_id=resource_id_for_audit,
            ip_address=client_ip, user_agent=user_agent, success=False,
            failure_reason=failure_reason_detail, details=action_details
        )
        raise HTTPException(status_code=500, detail=f"Failed to save claim to database due to an unexpected error.")

# The older definition of run_batch_processing_background that doesn't take MetricsCollector or new AuditLogger
# has been removed as it was redundant.

# Updated definition of run_batch_processing_background
async def run_batch_processing_background(
    batch_size: int,
    client_ip: Optional[str],
    user_agent_header: Optional[str],
    metrics_collector_instance: MetricsCollector,
    audit_logger_instance: AuditLogger,
    encryption_service_instance: EncryptionService, # New argument
    processing_semaphore: asyncio.Semaphore
):
    logger.info("Background task started, using semaphore.", batch_size=batch_size, client_ip=client_ip)

    try:
        async with AsyncSessionLocal() as session:
            await audit_logger_instance.log_access(
                user_id="background_system", action="PROCESS_BATCH_CLAIMS_TASK_STARTED", resource="ClaimBatchInternal",
                resource_id=f"batch_size_{batch_size}", ip_address=client_ip, user_agent=user_agent_header,
                success=True, details={"message": "Background processing service initiated."}
            )

            service = ClaimProcessingService(
                db_session=session,
                metrics_collector=metrics_collector_instance,
                encryption_service=encryption_service_instance # New
            )
            result = await service.process_pending_claims_batch(batch_size_override=batch_size)
            logger.info("Background task processing finished by service", batch_size=batch_size, result=result)

            await audit_logger_instance.log_access(
                user_id="background_system", action="PROCESS_BATCH_CLAIMS_TASK_COMPLETED", resource="ClaimBatchInternal",
                resource_id=f"batch_size_{batch_size}", ip_address=client_ip, user_agent=user_agent_header,
                success=True, details=result
            )
    except Exception as e:
        logger.error("Error in background batch processing task execution", error=str(e), exc_info=True, batch_size=batch_size)
        try:
            # Ensure audit_logger_instance is available for logging failure even if session failed to init
            await audit_logger_instance.log_access(
                user_id="background_system", action="PROCESS_BATCH_CLAIMS_TASK_FAILED", resource="ClaimBatchInternal",
                resource_id=f"batch_size_{batch_size}", ip_address=client_ip, user_agent=user_agent_header,
                success=False, failure_reason=f"Background task error: {str(e)}",
                details={"message": "Background processing service failed during execution."}
            )
        except Exception as audit_err_e:
             logger.error("Critical: Failed to log batch processing task failure audit event", error=str(audit_err_e), exc_info=True)
    finally:
        processing_semaphore.release()
        logger.info("Batch processing permit released.", batch_size=batch_size)


@router.post("/process-batch/", summary="Trigger Batch Processing of Claims (Background)", response_model=BatchProcessResponse)
async def trigger_batch_processing_background_endpoint(
    request: Request,
    background_tasks: BackgroundTasks,
    batch_size: int = Query(100, gt=0, le=10000),
    metrics: MetricsCollector = Depends(get_metrics_collector),
    audit_logger: AuditLogger = Depends(get_audit_logger),
    encryption_service: EncryptionService = Depends(get_encryption_service) # New
):
    settings = get_settings()
    batch_semaphore = get_batch_processing_semaphore()
    client_ip = request.client.host if request.client else None
    user_agent_header = request.headers.get("user-agent")

    logger.info(f"Received request to trigger batch processing in background.", batch_size=batch_size, client_ip=client_ip)

    if batch_semaphore.locked():
        logger.warn("Max concurrent batch processing limit reached. Rejecting new request.",
                    max_batches=settings.MAX_CONCURRENT_BATCHES, current_value=batch_semaphore._value)
        await audit_logger.log_access(
            user_id=None, action="TRIGGER_BATCH_PROCESSING_REJECTED", resource="ClaimBatchTrigger",
            ip_address=client_ip, user_agent=user_agent_header, success=False,
            failure_reason="Max concurrent batch processing limit reached.",
            details={"requested_batch_size": batch_size, "limit": settings.MAX_CONCURRENT_BATCHES}
        )
        raise HTTPException(
            status_code=503, # Service Unavailable
            detail="The system is currently processing the maximum number of batches. Please try again later."
        )

    await batch_semaphore.acquire()
    logger.info("Batch processing permit acquired.", max_batches=settings.MAX_CONCURRENT_BATCHES, current_value=batch_semaphore._value)

    try:
        await audit_logger.log_access(
            user_id=None,
            action="TRIGGER_BATCH_PROCESSING_ACCEPTED", resource="ClaimBatchTrigger",
            ip_address=client_ip, user_agent=user_agent_header, success=True,
            details={"requested_batch_size": batch_size, "limit": settings.MAX_CONCURRENT_BATCHES}
        )

        background_tasks.add_task(
            run_batch_processing_background,
            batch_size,
            client_ip,
            user_agent_header,
            metrics,
            audit_logger,
            encryption_service, # Pass EncryptionService instance
            processing_semaphore=batch_semaphore
        )
        logger.info("Batch processing task added to background.", batch_size=batch_size)
        return BatchProcessResponse(message="Batch processing started in the background.", batch_size=batch_size)
    except Exception as e:
        # This block is to ensure semaphore is released if add_task itself fails, though unlikely.
        batch_semaphore.release()
        logger.error("Failed to add batch processing task to background.", error=str(e), exc_info=True)
        # Log critical failure to add task
        await audit_logger.log_access(
            user_id=None, action="TRIGGER_BATCH_PROCESSING_FAILED_TO_ADD", resource="ClaimBatchTrigger",
            ip_address=client_ip, user_agent=user_agent_header, success=False,
            failure_reason=f"Error adding task to background: {str(e)}",
            details={"requested_batch_size": batch_size}
        )
        raise HTTPException(status_code=500, detail="Failed to initiate batch processing.")
