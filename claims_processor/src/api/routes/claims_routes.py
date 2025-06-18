from fastapi import APIRouter
from ..models.claim_models import Claim  # Adjusted import path
import structlog

logger = structlog.get_logger(__name__)
router = APIRouter()

@router.post("/")
async def create_claim(claim: Claim):
    logger.info("Claim received", claim_data=claim.model_dump())
    return {"message": "Claim received successfully", "claim_id": claim.claim_id}
