from fastapi import FastAPI
from datetime import datetime, timezone
from .api.routes import claims_routes, submission_routes # Import the new router
from .core.logging_config import setup_logging
import structlog

setup_logging() # Initialize logging
logger = structlog.get_logger(__name__)

app = FastAPI()

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
async def health_check():
    logger.info("Health check accessed")
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0"
    }

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
from .core.cache.cache_manager import get_cache_service, CacheManager
import uuid
from .core.config.settings import get_settings, Settings # Add Settings for type hint and get_settings
from pathlib import Path # For file path checking


@app.get("/ready", tags=["Monitoring"])
async def readiness_check(
    db: AsyncSession = Depends(get_db_session),
    cache: CacheManager = Depends(get_cache_service)
    # settings: Settings = Depends(get_settings) # Alternative: inject settings
) -> Dict[str, Any]:
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
