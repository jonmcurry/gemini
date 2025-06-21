from fastapi import FastAPI, Request, Depends # Added Request, Depends
from datetime import datetime, timezone
from .api.routes import claims_routes, submission_routes # Import the new router
from .monitoring.logging.logging_config import setup_logging
from .core.monitoring.audit_logger import AuditLogger
from .api.dependencies import get_audit_logger
import structlog

# Imports for DB warmup
from sqlalchemy import text
from .core.database.db_session import engine as async_engine # Use 'engine' as defined in db_session.py
from .core.config.settings import get_settings

setup_logging() # Initialize logging
logger = structlog.get_logger(__name__)

app = FastAPI()

# --- DB Pool Warmup ---
async def warmup_db_pool():
    logger.info("Application startup: warming up database connection pool...")
    app_settings = get_settings()
    # Determine number of connections for warmup
    warmup_count = 0
    if app_settings.DB_POOL_SIZE > 0: # DB_POOL_SIZE should exist in Settings
        warmup_count = min(app_settings.DB_POOL_SIZE, 3)

    if warmup_count == 0:
        logger.info("DB_POOL_SIZE is 0 or not configured for positive value. Skipping pool warmup.")
        return

    try:
        for i in range(warmup_count):
            async with async_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
                # No commit needed for SELECT 1
            logger.debug(f"DB warmup connection {i+1}/{warmup_count} successful.")
        logger.info(f"Database connection pool warmed up with {warmup_count} connections.")
    except Exception as e:
        logger.error(f"Error during database connection pool warmup: {e}", exc_info=True)

app.add_event_handler("startup", warmup_db_pool)
# --- End DB Pool Warmup ---


# Import for shutdown event
from .core.cache.cache_manager import close_global_cache_manager
# Import new router
from .api.routes import data_transfer_routes

@app.on_event("shutdown")
async def app_shutdown():
    logger.info("Application shutdown: closing global cache manager.")
    await close_global_cache_manager()

app.include_router(claims_routes.router, prefix="/api/v1/claims", tags=["Claims Processing"])
app.include_router(submission_routes.router, prefix="/api/v1", tags=["Claim Submissions"]) # Added new submission router
app.include_router(data_transfer_routes.router, prefix="/api/v1/data-transfer", tags=["Data Transfer"])

@app.get("/health")
async def health_check(
    request: Request,
    audit_logger: AuditLogger = Depends(get_audit_logger)
):
    logger.info("Health check accessed")
    response_data = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0"
    }

    await audit_logger.log_access(
        user_id=str(request.client.host if request.client else "unknown_host"), # Using client host as a stand-in
        action="HEALTH_CHECK",
        resource="System",
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        success=True
    )
    return response_data

# Prometheus metrics endpoint
from prometheus_client import generate_latest, REGISTRY
from prometheus_client.exposition import CONTENT_TYPE_LATEST
from fastapi import Response

@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Exposes Prometheus metrics.
    """
    logger.debug("Metrics endpoint called.")
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

# Readiness Probe
from sqlalchemy import text, AsyncSession # Added AsyncSession for Depends
from sqlalchemy.ext.asyncio import AsyncSession # Ensure this is the one used by get_db_session
from typing import Dict, Any
from .core.database.db_session import get_db_session # Added get_db_session for Depends
from .core.cache.cache_manager import get_cache_manager, CacheManager # Changed get_cache_service to get_cache_manager
import uuid
from .core.config.settings import get_settings, Settings # Add Settings for type hint and get_settings
from pathlib import Path # For file path checking


@app.get("/ready", tags=["Monitoring"])
async def readiness_check(
    db: AsyncSession = Depends(get_db_session),
    cache: CacheManager = Depends(get_cache_manager), # Changed get_cache_service to get_cache_manager
    # settings: Settings = Depends(get_settings) # Alternative: inject settings
    # audit_logger: AuditLogger = Depends(get_audit_logger) # Example if needed here
) -> Dict[str, Any]:
    # Add HTTPException import if not already present at the top of the file
    from fastapi import HTTPException

    checks = {
        "database": {"status": "unhealthy", "details": "Check not performed"},
        "cache": {"status": "unhealthy", "details": "Check not performed"},
        "ml_service": {"status": "unhealthy", "details": "Check not performed"}
    }
    app_settings = get_settings() # Call get_settings directly

    # Database check
    try:
        result = await db.execute(text("SELECT 1"))
        if result.scalar_one() == 1:
            checks["database"]["status"] = "healthy"
            checks["database"]["details"] = "Successfully connected and queried."
        else:
            checks["database"]["details"] = "Query executed but result was unexpected."
    except Exception as e:
        logger.error("Readiness check: Database connection failed", error=str(e), exc_info=False)
        checks["database"]["details"] = f"Connection failed: {str(e)}"

    # Cache check
    try:
        test_key = f"readiness_check_{uuid.uuid4()}"
        test_value = "healthy"
        await cache.set(test_key, test_value, ttl=5) # Use a short ttl
        retrieved_value = await cache.get(test_key)
        if retrieved_value == test_value:
            checks["cache"]["status"] = "healthy"
            checks["cache"]["details"] = "Successfully connected and performed SET/GET."
        else:
            checks["cache"]["details"] = "SET/GET operations did not return expected value."
            logger.warn("Readiness check: Cache SET/GET mismatch", expected=test_value, got=retrieved_value, key=test_key)
    except Exception as e:
        logger.error("Readiness check: Cache connection failed", error=str(e), exc_info=True)
        checks["cache"]["details"] = f"Connection failed: {str(e)}"

    # ML Service check (basic file existence)
    try:
        model_path_str = app_settings.ML_MODEL_PATH
        if model_path_str:
            # Assuming model_path_str is relative to the project root.
            # If main.py is in /app/claims_processor/src, and path is "models/model.tflite",
            # this needs to be relative to where the app runs from (usually project root /app).
            # Path(model_path_str) assumes CWD is project root.
            model_file = Path(model_path_str)
            if model_file.is_file():
                checks["ml_service"]["status"] = "healthy"
                checks["ml_service"]["details"] = f"Model file found at {model_path_str}."
            else:
                checks["ml_service"]["details"] = f"Model file not found at {model_path_str}."
                logger.warn(f"Readiness check: ML model file not found at {model_path_str}")
        else:
            checks["ml_service"]["status"] = "not_configured" # A distinct status if not configured
            checks["ml_service"]["details"] = "ML_MODEL_PATH is not configured in settings."
            logger.warn("Readiness check: ML_MODEL_PATH not configured.")
    except Exception as e:
        logger.error("Readiness check: ML service check failed", error=str(e), exc_info=True)
        checks["ml_service"]["details"] = f"ML service check failed: {str(e)}"

    # Determine overall readiness
    # Service is ready if database and cache are healthy.
    # ML service being 'not_configured' or 'unhealthy' might be acceptable depending on requirements.
    # For now, let's assume ML service being 'not_configured' is acceptable for readiness,
    # but an 'unhealthy' status (e.g. file not found when path is set) makes it not ready.

    database_ready = checks["database"]["status"] == "healthy"
    cache_ready = checks["cache"]["status"] == "healthy"
    ml_service_acceptable = checks["ml_service"]["status"] in ["healthy", "not_configured"]

    if not (database_ready and cache_ready and ml_service_acceptable):
        # Log the full checks details if any critical component is not ready
        logger.warn("Readiness check failed", overall_status=checks)
        raise HTTPException(status_code=503, detail=checks)

    logger.info("Readiness check successful", overall_status=checks)
    return checks
