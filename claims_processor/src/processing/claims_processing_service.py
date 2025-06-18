from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import joinedload # For eager loading
import structlog
from typing import List # Import List
from datetime import datetime, timezone # Import datetime, timezone
from decimal import Decimal # Import Decimal

# Assuming ClaimModel is correctly imported from:
from ..core.database.models.claims_db import ClaimModel
# Import ProcessableClaim for data conversion and ClaimValidator for validation
from ..api.models.claim_models import ProcessableClaim # ProcessableClaimLineItem is part of ProcessableClaim
from .validation.claim_validator import ClaimValidator
from .rvu_service import RVUService # Import RVUService

logger = structlog.get_logger(__name__)

class ClaimProcessingService:
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.validator = ClaimValidator()
        self.rvu_service = RVUService() # Instantiate RVUService


    async def process_pending_claims_batch(self, batch_size: int = 100):
        """
        Orchestrates the processing of a batch of pending claims.
        1. Fetch pending claims.
        2. Validate claims.
        3. Calculate RVUs for valid claims.
        4. Update claim statuses and details in the database.
        """
        logger.info("Starting batch processing of claims", batch_size=batch_size)

        pending_db_claims_list = await self._fetch_pending_claims(batch_size)
        if not pending_db_claims_list:
            logger.info("No pending claims to process.")
            return {"message": "No pending claims to process.", "attempted_claims": 0, "validated_count": 0, "failed_validation_count": 0, "processed_for_rvu_count": 0}

        # Create a map for easy lookup of the original DB objects
        pending_db_claims_map = {claim.id: claim for claim in pending_db_claims_list}
        logger.info(f"Fetched {len(pending_db_claims_map)} claims for processing.")

        validated_count = 0
        failed_validation_count = 0 # Includes conversion errors and validation rule failures
        processed_for_rvu_count = 0

        for db_claim_id, original_db_claim in pending_db_claims_map.items():
            processable_claim: ProcessableClaim = None
            try:
                # Convert SQLAlchemy model to Pydantic model for validation and processing
                processable_claim = ProcessableClaim.model_validate(original_db_claim)
            except Exception as e:
                logger.error("Failed to convert DB claim to Pydantic model", claim_db_id=original_db_claim.id, error=str(e), exc_info=True)
                # Use original_db_claim for the update method as processable_claim might be None
                await self._update_claim_and_lines_in_db(original_db_claim, None, "conversion_error", validation_errors=[f"Pydantic conversion error: {str(e)}"])
                failed_validation_count +=1
                continue

            validation_errors = self.validator.validate_claim(processable_claim)
            if validation_errors:
                logger.warn("Claim validation failed", claim_id=processable_claim.claim_id, errors=validation_errors)
                await self._update_claim_and_lines_in_db(original_db_claim, processable_claim, "validation_failed", validation_errors=validation_errors)
                failed_validation_count += 1
                continue

            logger.info("Claim validation successful", claim_id=processable_claim.claim_id)
            validated_count += 1

            # RVU Calculation
            try:
                await self.rvu_service.calculate_rvu_for_claim(processable_claim, self.db) # Modifies processable_claim
                logger.info("RVU calculation successful for claim", claim_id=processable_claim.claim_id)
                # Persist changes (including RVUs on lines) and set status to processing_complete
                await self._update_claim_and_lines_in_db(original_db_claim, processable_claim, "processing_complete")
                processed_for_rvu_count += 1
            except Exception as e:
                logger.error("RVU calculation failed for claim", claim_id=processable_claim.claim_id, error=str(e), exc_info=True)
                # Mark claim as failed in RVU calculation step
                await self._update_claim_and_lines_in_db(original_db_claim, processable_claim, "rvu_calculation_failed", validation_errors=[f"RVU calculation error: {str(e)}"])
                # This claim passed validation but failed RVU. It's not counted in processed_for_rvu_count.
                # It was already counted in validated_count.

        logger.info("Batch processing finished.",
                    attempted_claims=len(pending_db_claims_map),
                    validated_count=validated_count,
                    failed_validation_count=failed_validation_count,
                    processed_for_rvu_count=processed_for_rvu_count)

        return {
            "message": "Batch processing finished.",
            "attempted_claims": len(pending_db_claims_map),
            "validated_count": validated_count,
            "failed_validation_count": failed_validation_count,
            "successfully_processed_count": processed_for_rvu_count # Claims that completed all steps including RVU
        }


    async def _fetch_pending_claims(self, batch_size: int) -> list[ClaimModel]:
        """
        Fetches a batch of claims with 'pending' status from the database.
        Orders by 'created_at' to process older claims first.
        Eager loads line_items.
        """
        logger.info("Fetching pending claims from database", batch_size=batch_size)
        stmt = (
            select(ClaimModel)
            .where(ClaimModel.processing_status == 'pending')
            .order_by(ClaimModel.created_at.asc()) # Process older claims first
            # .order_by(ClaimModel.priority.desc(), ClaimModel.created_at.asc()) # If priority field is used
            .limit(batch_size)
            .options(joinedload(ClaimModel.line_items)) # Eager load line items
        )

        result = await self.db.execute(stmt)
        claims = result.scalars().all() # Get a list of ClaimModel instances

        if claims:
            logger.info(f"Fetched {len(claims)} pending claims from the database.")
        else:
            logger.info("No pending claims found in the database.")
        return list(claims)

    # Old _update_claim_status_in_db is removed by replacing the whole file content,
    # effectively replacing it with _update_claim_and_lines_in_db or requiring its recreation if separate logic is needed.
    # For this subtask, we assume _update_claim_and_lines_in_db is the comprehensive method.

    async def _update_claim_and_lines_in_db(self,
                                         db_claim_to_update: ClaimModel,
                                         processed_pydantic_claim: ProcessableClaim, # Can be None if conversion failed
                                         new_status: str,
                                         validation_errors: List[str] = None):
        """
        Updates the claim and its line items in the database based on the processed Pydantic model.
        Also updates the claim's processing status.
        `db_claim_to_update` is the SQLAlchemy model instance from the session.
        `processed_pydantic_claim` is the Pydantic model that has undergone processing (e.g., RVU calculation).
        """
        if db_claim_to_update is None:
            logger.error("Cannot update claim in DB: SQLAlchemy model instance is None.")
            return

        logger.info("Updating claim and lines in DB", claim_db_id=db_claim_to_update.id, new_status=new_status)
        try:
            db_claim_to_update.processing_status = new_status

            if validation_errors: # Typically for "validation_failed", "conversion_error", or "rvu_calculation_failed"
                logger.warn("Storing/logging errors for claim", claim_id=db_claim_to_update.claim_id, errors=validation_errors)
                # Example: db_claim_to_update.error_details = {"errors": validation_errors} # If such a field exists

            # Only attempt to update line items if processing was successful up to that point
            # and we have a Pydantic representation of the claim with potentially updated line data.
            if new_status == "processing_complete" and processed_pydantic_claim:
                # Transfer RVU data from Pydantic model back to SQLAlchemy models
                # Ensure line items are sorted or matched correctly if their order could change.
                # Using a map for robustness, assuming line item IDs are stable and present.

                map_pydantic_line_items = {line.id: line for line in processed_pydantic_claim.line_items}

                for db_line_item in db_claim_to_update.line_items: # These are SQLAlchemy line_item objects
                    if db_line_item.id in map_pydantic_line_items:
                        pydantic_line = map_pydantic_line_items[db_line_item.id]
                        if pydantic_line.rvu_total is not None: # Check if RVU was calculated
                            db_line_item.rvu_total = pydantic_line.rvu_total
                        # If expected_reimbursement was also calculated and stored on pydantic_line:
                        # db_line_item.expected_reimbursement = pydantic_line.expected_reimbursement

                db_claim_to_update.processed_at = datetime.now(timezone.utc)

            self.db.add(db_claim_to_update) # Add to session to track changes to the main claim object
            # Line items are part of the db_claim_to_update's relationship and session, changes to them are also tracked.
            await self.db.commit()

            # Refresh to get DB-generated values like updated_at for the main claim and its lines
            await self.db.refresh(db_claim_to_update)
            if db_claim_to_update.line_items: # Check if line_items collection is loaded/exists
                for line_item_to_refresh in db_claim_to_update.line_items:
                    await self.db.refresh(line_item_to_refresh)

            logger.info("Claim and lines updated successfully in DB", claim_db_id=db_claim_to_update.id, new_status=db_claim_to_update.processing_status)

        except Exception as e:
            await self.db.rollback()
            logger.error("Failed to update claim and lines in DB", claim_db_id=db_claim_to_update.id if db_claim_to_update else "Unknown ID", error=str(e), exc_info=True)
