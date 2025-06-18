from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select # Required for the optional duplicate check

from ..models.claim_models import ClaimCreate, ClaimResponse # Updated model names
from ...core.database.db_session import get_db_session
from ...core.database.models.claims_db import ClaimModel
import structlog

logger = structlog.get_logger(__name__)
router = APIRouter()

@router.post("/", response_model=ClaimResponse) # Set response_model
async def create_claim(
    claim_data: ClaimCreate,
    db: AsyncSession = Depends(get_db_session)
):
    logger.info("Received request to create claim", claim_id=claim_data.claim_id, facility_id=claim_data.facility_id)

    # Optional: Check if claim_id already exists to prevent duplicates if DB doesn't handle it
    # This requires claim_id to be unique in the database, which it is.
    existing_claim_stmt = await db.execute(select(ClaimModel).where(ClaimModel.claim_id == claim_data.claim_id))
    if existing_claim_stmt.scalars().first() is not None:
        logger.warn("Claim ID already exists", claim_id=claim_data.claim_id)
        raise HTTPException(status_code=409, detail=f"Claim with ID {claim_data.claim_id} already exists.")

    # Create a dictionary of fields present in ClaimModel from claim_data
    # This ensures only fields defined in ClaimCreate that are also in ClaimModel are passed
    # model_dump() will include all fields from ClaimCreate, including optional ones if provided
    claim_model_data = claim_data.model_dump(exclude_unset=True) # exclude_unset=True is good for partial updates, but here we want all provided fields

    # Ensure all required fields for ClaimModel are present if not covered by defaults
    # For this example, ClaimCreate and ClaimModel have aligned required fields or defaults in ClaimModel
    db_claim = ClaimModel(**claim_model_data)

    try:
        db.add(db_claim)
        await db.commit()
        await db.refresh(db_claim)
        logger.info("Claim saved to database successfully", claim_id=db_claim.claim_id, db_id=db_claim.id)
        return db_claim
    except Exception as e: # Catch generic SQLAlchemyError or other DB errors
        await db.rollback()
        logger.error("Error saving claim to database", claim_id=claim_data.claim_id, error=str(e), exc_info=True)
        # Consider more specific error handling for production
        raise HTTPException(status_code=500, detail="Failed to save claim to database.")
