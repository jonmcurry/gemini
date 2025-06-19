from fastapi import FastAPI
from datetime import datetime, timezone
from .api.routes import claims_routes # Import the new router
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

app.include_router(claims_routes.router, prefix="/api/v1/claims", tags=["Claims Processing"]) # Renamed tag for clarity
app.include_router(data_transfer_routes.router, prefix="/api/v1/data-transfer", tags=["Data Transfer"]) # Add new router

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
