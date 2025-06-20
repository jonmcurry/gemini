import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from typing import Dict, Any, Optional

# Adjust relative paths based on actual file locations.
# Assuming 'data_transfer_routes.py' is in 'claims_processor/src/api/routes/'
from ...core.database.db_session import AsyncSessionLocal, AsyncSession
from ...processing.data_transfer_service import DataTransferService
from ...core.monitoring.audit_logger import AuditLogger # Import new AuditLogger
from ..dependencies import get_audit_logger # Import new dependency getter

logger = structlog.get_logger(__name__)
router = APIRouter(
    # prefix="/data-transfer", # Prefix is already applied by main app router for this file
    tags=["Data Transfer"] # Tags are fine
)

async def run_data_transfer_background(
    limit: int,
    client_ip: Optional[str],
    user_agent_header: Optional[str],
    audit_logger_instance: AuditLogger # Added AuditLogger instance
):
    """
    Background task wrapper for transferring claims to production.
    Manages its own database session.
    """
    logger.info(
        "Background task started: run_data_transfer_background",
        record_limit=limit, client_ip=client_ip, user_agent=user_agent_header
    )
    session: Optional[AsyncSession] = None
    try:
        await audit_logger_instance.log_access(
            user_id="background_system", action="DATA_TRANSFER_TASK_STARTED", resource="DataTransferTask",
            resource_id=f"limit_{limit}", ip_address=client_ip, user_agent=user_agent_header,
            success=True, details={"message": "Data transfer background task initiated."}
        )

        async with AsyncSessionLocal() as session:
            data_transfer_service = DataTransferService(db_session=session)
            result = await data_transfer_service.transfer_claims_to_production(limit=limit)
            logger.info("Data transfer background task finished successfully by service.", result=result, record_limit=limit)

            await audit_logger_instance.log_access(
                user_id="background_system", action="DATA_TRANSFER_TASK_COMPLETED", resource="DataTransferTask",
                resource_id=f"limit_{limit}", ip_address=client_ip, user_agent=user_agent_header,
                success=True, details=result
            )

    except Exception as e:
        logger.error(
            "Error in data transfer background task execution",
            error=str(e), exc_info=True, record_limit=limit
        )
        try:
            await audit_logger_instance.log_access(
                user_id="background_system", action="DATA_TRANSFER_TASK_FAILED", resource="DataTransferTask",
                resource_id=f"limit_{limit}", ip_address=client_ip, user_agent=user_agent_header,
                success=False, failure_reason=f"Background task error: {str(e)}",
                details={"message": "Data transfer background task failed during execution."}
            )
        except Exception as audit_err_e:
            logger.error("Critical: Failed to log data transfer task failure audit event", error=str(audit_err_e), exc_info=True)
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
    limit: int = 1000, # Example: make limit configurable via query param
    audit_logger: AuditLogger = Depends(get_audit_logger) # Inject AuditLogger
) -> Dict[str, Any]:
    """
    Triggers a background task to transfer processed claims from the staging
    `claims` table to the `claims_production` table.
    """
    client_ip = request.client.host if request.client else None # Allow None
    user_agent_header = request.headers.get("user-agent")

    logger.info(
        "Received request to trigger data transfer to production in background.",
        record_limit=limit, client_ip=client_ip, user_agent=user_agent_header
    )

    await audit_logger.log_access(
        user_id=None, # Or derive from authenticated user if available
        action="TRIGGER_DATA_TRANSFER",
        resource="DataTransferTrigger",
        ip_address=client_ip,
        user_agent=user_agent_header,
        success=True,
        details={"requested_limit": limit}
    )

    background_tasks.add_task(
        run_data_transfer_background,
        limit=limit,
        client_ip=client_ip,
        user_agent_header=user_agent_header,
        audit_logger_instance=audit_logger # Pass the audit_logger instance
    )

    return {
        "message": "Data transfer to production process started in the background.",
        "requested_transfer_limit": limit
    }
