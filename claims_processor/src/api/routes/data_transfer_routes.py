import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from typing import Dict, Any, Optional

# Adjust relative paths based on actual file locations.
# Assuming 'data_transfer_routes.py' is in 'claims_processor/src/api/routes/'
from ...core.database.db_session import AsyncSessionLocal, AsyncSession
from ...processing.data_transfer_service import DataTransferService
# No AuditLoggerService import here yet, can be added if auditing the trigger directly here.

logger = structlog.get_logger(__name__)
router = APIRouter(
    prefix="/data-transfer", # This prefix is for routes within this router
    tags=["Data Transfer"]
)

async def run_data_transfer_background(limit: int, client_ip: Optional[str], user_agent_header: Optional[str]):
    """
    Background task wrapper for transferring claims to production.
    Manages its own database session.
    """
    logger.info(
        "Background task started: run_data_transfer_background",
        record_limit=limit, client_ip=client_ip, user_agent=user_agent_header
    )
    session: Optional[AsyncSession] = None # To ensure it's defined for finally block if needed
    try:
        async with AsyncSessionLocal() as session:
            # TODO: Optionally, log trigger of this background task to AuditLog using this session
            # audit_service = AuditLoggerService(db_session=session)
            # await audit_service.log_event(action="DATA_TRANSFER_TASK_STARTED", ...)

            data_transfer_service = DataTransferService(db_session=session)
            result = await data_transfer_service.transfer_claims_to_production(limit=limit)
            logger.info("Data transfer background task finished successfully by service.", result=result, record_limit=limit)

            # TODO: Optionally, log completion summary to AuditLog as well, using this session.
            # await audit_service.log_event(action="DATA_TRANSFER_TASK_COMPLETED", details=result, ...)

    except Exception as e:
        logger.error(
            "Error in data transfer background task execution",
            error=str(e), exc_info=True, record_limit=limit
        )
        # TODO: Handle/log failure more robustly (e.g., update a task status table or log to AuditLog).
        # async with AsyncSessionLocal() as error_session:
        #     audit_service_err = AuditLoggerService(db_session=error_session)
        #     await audit_service_err.log_event(action="DATA_TRANSFER_TASK_FAILED", failure_reason=str(e), ...)
    finally:
        logger.debug("Finished run_data_transfer_background execution attempt.")


@router.post(
    "/trigger-production-transfer/",
    summary="Trigger Data Transfer from Staging to Production",
    status_code=202 # HTTP 202 Accepted for background tasks
)
async def trigger_production_data_transfer(
    request: Request,
    background_tasks: BackgroundTasks,
    limit: int = 1000 # Example: make limit configurable via query param
) -> Dict[str, Any]:
    """
    Triggers a background task to transfer processed claims from the staging
    `claims` table to the `claims_production` table.
    """
    client_ip = request.client.host if request.client else "N/A"
    user_agent_header = request.headers.get("user-agent", "N/A")

    logger.info(
        "Received request to trigger data transfer to production in background.",
        record_limit=limit, client_ip=client_ip, user_agent=user_agent_header
    )

    # Audit logging for the trigger action (using its own session)
    # This part can be added if AuditLoggerService is imported and configured.
    # For now, focusing on the task trigger.
    # async with AsyncSessionLocal() as audit_session:
    #     audit_service = AuditLoggerService(db_session=audit_session)
    #     try:
    #         await audit_service.log_event(
    #             action="TRIGGER_DATA_TRANSFER_TO_PROD",
    #             success=True,
    #             user_id="api_user_placeholder", # Replace with actual user if available
    #             ip_address=client_ip,
    #             user_agent=user_agent_header,
    #             details={"requested_limit": limit}
    #         )
    #     except Exception as audit_exc:
    #         logger.error("Failed to log audit event for data transfer trigger", error=str(audit_exc), exc_info=True)


    background_tasks.add_task(
        run_data_transfer_background,
        limit=limit,
        client_ip=client_ip,
        user_agent_header=user_agent_header
    )

    return {
        "message": "Data transfer to production process started in the background.",
        "requested_transfer_limit": limit
    }
