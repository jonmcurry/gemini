from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks # Added Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional, Callable # Added Callable
from fastapi import Query # Added Query

from ..models.claim_models import ClaimCreate, ClaimResponse, BatchProcessResponse # Added BatchProcessResponse
from ...core.database.db_session import get_db_session, AsyncSessionLocal
from ...core.database.models.claims_db import ClaimModel
from ...core.security.audit_logger_service import AuditLoggerService # This is the old one, might need to switch to new AuditLogger if scope includes it
from ...core.monitoring.audit_logger import AuditLogger as NewAuditLogger # New one
from ..dependencies import get_audit_logger as get_new_audit_logger # Dependency for new one
from ...core.monitoring.app_metrics import MetricsCollector # Import MetricsCollector
from ..dependencies import get_metrics_collector # Dependency for MetricsCollector
from ...processing.claims_processing_service import ClaimProcessingService
import structlog

logger = structlog.get_logger(__name__)
router = APIRouter()

@router.post("/", response_model=ClaimResponse)
async def create_claim(
    request: Request, # Add request
    claim_data: ClaimCreate,
    db: AsyncSession = Depends(get_db_session)
):
    logger.info("Received request to create claim", claim_id=claim_data.claim_id, facility_id=claim_data.facility_id)

    audit_service = AuditLoggerService(db_session=db)
    # Ensure total_charges is serializable, e.g. float
    action_details = {"submitted_claim_id": claim_data.claim_id, "total_charges": float(claim_data.total_charges)}
    resource_id_for_audit = claim_data.claim_id
    db_claim_instance: Optional[ClaimModel] = None

    try:
        existing_claim_stmt = await db.execute(select(ClaimModel).where(ClaimModel.claim_id == claim_data.claim_id))
        if existing_claim_stmt.scalars().first() is not None:
            logger.warn("Claim ID already exists", claim_id=claim_data.claim_id)
            failure_reason_detail = f"HTTPException: 409 - Claim with ID {claim_data.claim_id} already exists."
            try:
                 await audit_service.log_event(
                    action="CREATE_CLAIM", success=False, resource="Claim", resource_id=str(resource_id_for_audit),
                    ip_address=request.client.host if request.client else None, user_agent=request.headers.get("user-agent"),
                    failure_reason=failure_reason_detail, details=action_details
                )
            except Exception as audit_e: logger.error("Failed to log audit for duplicate CREATE_CLAIM attempt", error=str(audit_e), exc_info=True)
            raise HTTPException(status_code=409, detail=f"Claim with ID {claim_data.claim_id} already exists.")

        claim_model_data = claim_data.model_dump(exclude_unset=True)
        db_claim_instance = ClaimModel(**claim_model_data) # Assign to db_claim_instance

        db.add(db_claim_instance)
        await db.commit()
        await db.refresh(db_claim_instance)
        logger.info("Claim saved to database successfully", claim_id=db_claim_instance.claim_id, db_id=db_claim_instance.id)

        try: # Nested try for audit log on success
            await audit_service.log_event(
                action="CREATE_CLAIM", success=True, resource="Claim", resource_id=str(db_claim_instance.claim_id), # Using claim_id from db_claim
                ip_address=request.client.host if request.client else None, user_agent=request.headers.get("user-agent"),
                details={"db_id": db_claim_instance.id, **action_details}
            )
        except Exception as audit_e: logger.error("Failed to log successful CREATE_CLAIM", error=str(audit_e), exc_info=True)

        return db_claim_instance

    except HTTPException as http_exc:
        if http_exc.status_code != 409: # Avoid double-logging for the known duplicate case
            try:
                await audit_service.log_event(
                    action="CREATE_CLAIM", success=False, resource="Claim", resource_id=str(resource_id_for_audit),
                    ip_address=request.client.host if request.client else None, user_agent=request.headers.get("user-agent"),
                    failure_reason=f"HTTPException: {http_exc.status_code} - {http_exc.detail}", details=action_details
                )
            except Exception as audit_e: logger.error("Failed to log failed CREATE_CLAIM (HTTPException)", error=str(audit_e), exc_info=True)
        raise http_exc

    except Exception as e:
        try: # Ensure rollback happens if session is active
            if db.is_active:
                await db.rollback()
        except Exception as rb_exc:
            logger.error("Error during rollback attempt", error=str(rb_exc), exc_info=True)

        logger.error("Error saving claim to database (unexpected)", claim_id=claim_data.claim_id, error=str(e), exc_info=True)
        failure_reason_detail = f"Unexpected error: {str(e)}"
        try:
            await audit_service.log_event(
                action="CREATE_CLAIM", success=False, resource="Claim", resource_id=str(resource_id_for_audit),
                ip_address=request.client.host if request.client else None, user_agent=request.headers.get("user-agent"),
                failure_reason=failure_reason_detail, details=action_details
            )
        except Exception as audit_e: logger.error("Failed to log failed CREATE_CLAIM (unexpected exception)", error=str(audit_e), exc_info=True)

        raise HTTPException(status_code=500, detail=f"Failed to save claim to database due to an unexpected error.")


# Helper function for background task
async def run_batch_processing_background(batch_size: int, client_ip: Optional[str], user_agent_header: Optional[str]):
    logger.info("Background task started: run_batch_processing_background", batch_size=batch_size, client_ip=client_ip)

    async with AsyncSessionLocal() as session:
        audit_service_bg = AuditLoggerService(db_session=session)
        try:
            await audit_service_bg.log_event(
                action="PROCESS_BATCH_CLAIMS_TASK_STARTED", success=True,
                resource="ClaimBatchInternal", resource_id=f"batch_size_{batch_size}",
                user_id="background_system", ip_address=client_ip, user_agent=user_agent_header,
                details={"message": "Background processing service initiated."}
            )

            service = ClaimProcessingService(db_session=session)
            result = await service.process_pending_claims_batch(batch_size=batch_size)
            logger.info("Background task processing finished by service", batch_size=batch_size, result=result)

            await audit_service_bg.log_event(
                action="PROCESS_BATCH_CLAIMS_TASK_COMPLETED", success=True,
                resource="ClaimBatchInternal", resource_id=f"batch_size_{batch_size}",
                user_id="background_system", ip_address=client_ip, user_agent=user_agent_header,
                details=result
            )
        except Exception as e:
            logger.error("Error in background batch processing task execution", error=str(e), exc_info=True, batch_size=batch_size)
            try:
        await audit_service_bg.log_event(
                    action="PROCESS_BATCH_CLAIMS_TASK_FAILED", success=False,
                    resource="ClaimBatchInternal", resource_id=f"batch_size_{batch_size}",
                    user_id="background_system", ip_address=client_ip, user_agent=user_agent_header,
                    failure_reason=f"Background task error: {str(e)}",
                    details={"message": "Background processing service failed during execution."}
                )
            except Exception as audit_err_e:
                 logger.error("Critical: Failed to log batch processing task failure audit event", error=str(audit_err_e), exc_info=True)

# Updated helper function for background task to accept MetricsCollector
async def run_batch_processing_background(
    batch_size: int,
    client_ip: Optional[str],
    user_agent_header: Optional[str],
    metrics_collector_instance: MetricsCollector, # Added MetricsCollector instance
    # new_audit_logger_instance: NewAuditLogger # If new audit logger is to be used here
):
    logger.info("Background task started: run_batch_processing_background", batch_size=batch_size, client_ip=client_ip)

    async with AsyncSessionLocal() as session: # For ClaimProcessingService and old AuditLoggerService
        # old_audit_service_bg = AuditLoggerService(db_session=session) # Keep old for now if not migrating this part
        # If migrating audit here, use new_audit_logger_instance passed in.

        try:
            # Example of logging with old audit service, adjust if migrating
            # await old_audit_service_bg.log_event(
            #     action="PROCESS_BATCH_CLAIMS_TASK_STARTED", success=True,
            #     resource="ClaimBatchInternal", resource_id=f"batch_size_{batch_size}",
            #     user_id="background_system", ip_address=client_ip, user_agent=user_agent_header,
            #     details={"message": "Background processing service initiated."}
            # )

            # Pass metrics_collector to ClaimProcessingService
            service = ClaimProcessingService(db_session=session, metrics_collector=metrics_collector_instance)
            result = await service.process_pending_claims_batch(batch_size=batch_size)
            logger.info("Background task processing finished by service", batch_size=batch_size, result=result)

            # await old_audit_service_bg.log_event(
            #     action="PROCESS_BATCH_CLAIMS_TASK_COMPLETED", success=True,
            #     resource="ClaimBatchInternal", resource_id=f"batch_size_{batch_size}",
            #     user_id="background_system", ip_address=client_ip, user_agent=user_agent_header,
            #     details=result
            # )
        except Exception as e:
            logger.error("Error in background batch processing task execution", error=str(e), exc_info=True, batch_size=batch_size)
            # try:
            #     await old_audit_service_bg.log_event(
            #         action="PROCESS_BATCH_CLAIMS_TASK_FAILED", success=False,
            #         resource="ClaimBatchInternal", resource_id=f"batch_size_{batch_size}",
            #         user_id="background_system", ip_address=client_ip, user_agent=user_agent_header,
            #         failure_reason=f"Background task error: {str(e)}",
            #         details={"message": "Background processing service failed during execution."}
            #     )
            # except Exception as audit_err_e:
            #      logger.error("Critical: Failed to log batch processing task failure audit event", error=str(audit_err_e), exc_info=True)


@router.post("/process-batch/", summary="Trigger Batch Processing of Claims (Background)", response_model=BatchProcessResponse)
async def trigger_batch_processing_background_endpoint( # Renamed for clarity
    request: Request,
    background_tasks: BackgroundTasks,
    batch_size: int = Query(100, gt=0, le=10000), # Added Query for validation
    metrics: MetricsCollector = Depends(get_metrics_collector), # Inject MetricsCollector
    # audit: NewAuditLogger = Depends(get_new_audit_logger) # Inject new AuditLogger if to be used here
    # old_audit_service: AuditLoggerService = Depends(temp_get_audit_logger_service) # Example if using old one for this endpoint's direct log
):
    client_ip = request.client.host if request.client else None
    user_agent_header = request.headers.get("user-agent")
    logger.info(f"Received request to trigger batch processing in background.", batch_size=batch_size, client_ip=client_ip)

    # Example of using old audit logger for the trigger event itself.
    # This part of audit logging can be migrated to NewAuditLogger if desired.
    # async with AsyncSessionLocal() as audit_db_session:
    #     audit_service_trigger = AuditLoggerService(db_session=audit_db_session) # Old service
    #     try:
    #         await audit_service_trigger.log_event(
    #             action="TRIGGER_BATCH_PROCESSING", success=True, resource="ClaimBatchTrigger",
    #             ip_address=client_ip, user_agent=user_agent_header,
    #             details={"requested_batch_size": batch_size}
    #         )
    #     except Exception as audit_exc:
    #         logger.error("Failed to log audit event for batch trigger action", error=str(audit_exc), exc_info=True)

    # Pass the obtained MetricsCollector instance to the background task
    background_tasks.add_task(run_batch_processing_background, batch_size, client_ip, user_agent_header, metrics)

    logger.info("Batch processing task added to background.", batch_size=batch_size)
    return BatchProcessResponse(message="Batch processing started in the background.", batch_size=batch_size)
