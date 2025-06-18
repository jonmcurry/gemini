from fastapi import FastAPI
from datetime import datetime, timezone
from .api.routes import claims_routes # Import the new router
from .core.logging_config import setup_logging
import structlog

setup_logging() # Initialize logging
logger = structlog.get_logger(__name__)

app = FastAPI()

app.include_router(claims_routes.router, prefix="/api/v1/claims", tags=["claims"])

@app.get("/health")
async def health_check():
    logger.info("Health check accessed")
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0"
    }
