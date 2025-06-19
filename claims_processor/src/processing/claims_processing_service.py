import asyncio # Import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import joinedload
import structlog
from typing import List, Dict, Any # Import Dict, Any
from datetime import datetime, timezone
from decimal import Decimal

import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import joinedload
import structlog
from typing import List, Dict, Any, Optional # Added Optional
from datetime import datetime, timezone
from decimal import Decimal

from ..core.config.settings import get_settings
from ..core.database.models.claims_db import ClaimModel
from ..api.models.claim_models import ProcessableClaim
from .validation.claim_validator import ClaimValidator
from .rvu_service import RVUService
from ..core.cache.cache_manager import get_cache_manager
# Import ML components
from .ml_pipeline.feature_extractor import FeatureExtractor
from .ml_pipeline.optimized_predictor import OptimizedPredictor

logger = structlog.get_logger(__name__)

class ClaimProcessingService:
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.validator = ClaimValidator()

        cache_manager = get_cache_manager()
        self.rvu_service = RVUService(cache_manager=cache_manager)

        current_settings = get_settings() # Use a consistent variable name
        self.concurrent_processing_semaphore = asyncio.Semaphore(current_settings.MAX_CONCURRENT_CLAIM_PROCESSING)

        # Initialize ML components
        self.feature_extractor = FeatureExtractor() # Uses settings in its own __init__
        self.predictor = OptimizedPredictor(model_path=current_settings.ML_MODEL_PATH)

        logger.info("ClaimProcessingService initialized",
                    db_session_id=id(db_session),
                    cache_manager_id=id(cache_manager),
                    concurrency_limit=current_settings.MAX_CONCURRENT_CLAIM_PROCESSING,
                    ml_model_path=current_settings.ML_MODEL_PATH)


    async def process_pending_claims_batch(self, batch_size: int = 100):
        """
        Orchestrates the processing of a batch of pending claims.
        1. Fetch pending claims.
        2. Validate claims.
        3. Calculate RVUs for valid claims.
        4. Update claim statuses and details in the database.
        """
        logger.info("Starting batch processing of claims with concurrency", batch_size=batch_size)

        logger.info("Starting batch processing of claims with concurrency", batch_size=batch_size)

        fetched_db_claims = await self._fetch_pending_claims(batch_size)

        if not fetched_db_claims:
            logger.info("No pending claims to process.")
            return {
                "message": "No pending claims to process.", "attempted_claims": 0,
                "conversion_errors":0, "validation_failures": 0,
                "ml_approved_raw": 0, "ml_rejected_raw": 0, "ml_errors_raw": 0, # Raw ML decisions
                "stopped_by_ml_rejection": 0,
                "rvu_calculation_failures":0, "successfully_processed_count": 0,
                "other_exceptions":0
            }

        logger.info(f"Fetched {len(fetched_db_claims)} claims for concurrent processing.")

        tasks = [self._process_single_claim_concurrently(db_claim) for db_claim in fetched_db_claims]
        processing_results = await asyncio.gather(*tasks, return_exceptions=True)

        attempted_claims = len(fetched_db_claims)
        conversion_errors_count = 0
        failed_validation_count = 0
        successfully_processed_count = 0
        failed_rvu_count = 0

        # Counters for ML decision outcomes, regardless of whether they halted processing
        ml_approved_raw_count = 0
        ml_rejected_raw_count = 0
        ml_errors_raw_count = 0 # Errors during ML step or unparseable output

        stopped_by_ml_rejection_count = 0 # Specifically those that had status 'ml_rejected'
        other_exceptions_count = 0

        for result in processing_results:
            if isinstance(result, Exception):
                logger.error("Unhandled exception in concurrent claim processing task", error=str(result), exc_info=result)
                other_exceptions_count +=1
            elif isinstance(result, dict):
                status = result.get("status")
                ml_decision_from_result = result.get("ml_decision", "N/A")

                # Tally raw ML decisions
                if ml_decision_from_result == "ML_APPROVED": ml_approved_raw_count += 1
                elif ml_decision_from_result == "ML_REJECTED": ml_rejected_raw_count += 1
                elif ml_decision_from_result.startswith("ML_ERROR") or ml_decision_from_result == "ML_PROCESSING_ERROR": ml_errors_raw_count +=1

                # Tally based on final status
                if status == "conversion_error":
                    conversion_errors_count +=1
                elif status == "validation_failed":
                    failed_validation_count += 1
                elif status == "ml_rejected": # This is a terminal status for this claim in the flow
                    stopped_by_ml_rejection_count += 1
                elif status == "rvu_calculation_failed":
                    # This claim passed validation and ML (or ML error allowed it to proceed)
                    failed_rvu_count +=1
                elif status == "processing_complete":
                    # This claim passed validation, ML (or ML error allowed it to proceed), and RVU
                    successfully_processed_count += 1

        final_summary = {
            "message": "Concurrent batch processing finished.",
            "attempted_claims": attempted_claims,
            "conversion_errors": conversion_errors_count,
            "validation_failures": failed_validation_count,
            "ml_approved_raw": ml_approved_raw_count, # How many ML thought were good
            "ml_rejected_raw": ml_rejected_raw_count, # How many ML thought were bad
            "ml_errors_raw": ml_errors_raw_count,     # How many had issues in ML step
            "stopped_by_ml_rejection": stopped_by_ml_rejection_count, # Actual stops due to ML
            "rvu_calculation_failures": failed_rvu_count,
            "successfully_processed_count": successfully_processed_count,
            "other_exceptions" : other_exceptions_count
        }
        logger.info("Concurrent batch processing summary.", **final_summary)

        return final_summary


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


    async def _process_single_claim_concurrently(self, db_claim: ClaimModel) -> Dict[str, Any]:
        async with self.concurrent_processing_semaphore:
            logger.info("Starting processing for a single claim concurrently", claim_id=db_claim.claim_id, db_id=db_claim.id)
            processable_claim: ProcessableClaim = None
            ml_decision = "N/A" # Default ML decision
            ml_confidence_score = 0.0

            try:
                processable_claim = ProcessableClaim.model_validate(db_claim)
            except Exception as e:
                logger.error("Pydantic conversion error for claim", claim_db_id=db_claim.id, error=str(e), exc_info=True)
                await self._update_claim_and_lines_in_db(db_claim, None, "conversion_error", validation_errors=[f"Pydantic conversion error: {str(e)}"])
                return {"db_id": db_claim.id, "claim_id": db_claim.claim_id, "status": "conversion_error", "error": str(e), "ml_decision": ml_decision}

            validation_errors = self.validator.validate_claim(processable_claim)
            if validation_errors:
                logger.warn("Claim validation failed (concurrent)", claim_id=processable_claim.claim_id, errors=validation_errors)
                await self._update_claim_and_lines_in_db(db_claim, processable_claim, "validation_failed", validation_errors=validation_errors)
                return {"db_id": db_claim.id, "claim_id": processable_claim.claim_id, "status": "validation_failed", "errors": validation_errors, "ml_decision": ml_decision}

            logger.info("Claim validation successful (concurrent)", claim_id=processable_claim.claim_id)

            # --- ML Processing Step ---
            try:
                logger.debug("Starting ML feature extraction (stub)", claim_id=processable_claim.claim_id)
                features = self.feature_extractor.extract_features(processable_claim)
                logger.debug("Dummy features extracted", claim_id=processable_claim.claim_id, features_shape=features.shape)

                logger.debug("Starting ML prediction (stub)", claim_id=processable_claim.claim_id)
                prediction_scores = await self.predictor.predict_async(features)
                logger.debug("Dummy ML prediction scores received", claim_id=processable_claim.claim_id, scores=prediction_scores)

                if prediction_scores.ndim == 2 and prediction_scores.shape[0] > 0 and prediction_scores.shape[1] >= 2:
                    prob_approve = prediction_scores[0][1]
                    ml_confidence_score = float(prob_approve)

                    current_settings = get_settings()
                    if prob_approve >= current_settings.ML_APPROVAL_THRESHOLD:
                        ml_decision = "ML_APPROVED"
                        logger.info("Claim approved by ML model (stub)", claim_id=processable_claim.claim_id, score=prob_approve, threshold=current_settings.ML_APPROVAL_THRESHOLD)
                    else:
                        ml_decision = "ML_REJECTED"
                        logger.warn("Claim rejected by ML model (stub)", claim_id=processable_claim.claim_id, score=prob_approve, threshold=current_settings.ML_APPROVAL_THRESHOLD)
                else:
                    logger.warn("Unexpected ML prediction score format", claim_id=processable_claim.claim_id, scores_shape=prediction_scores.shape)
                    ml_decision = "ML_ERROR_UNEXPECTED_FORMAT"
            except Exception as ml_exc:
                logger.error("Error during ML processing step (stub)", claim_id=processable_claim.claim_id, error=str(ml_exc), exc_info=True)
                ml_decision = "ML_PROCESSING_ERROR"
            # --- End ML Processing Step ---

            try:
                await self.rvu_service.calculate_rvu_for_claim(processable_claim, self.db)
                logger.info("RVU calculation successful (concurrent)", claim_id=processable_claim.claim_id)
                await self._update_claim_and_lines_in_db(db_claim, processable_claim, "processing_complete",
                                                         ml_log_info={"ml_decision": ml_decision, "ml_score": ml_confidence_score})
                return {"db_id": db_claim.id, "claim_id": processable_claim.claim_id, "status": "processing_complete", "ml_decision": ml_decision}
            except Exception as e:
                logger.error("RVU calculation failed (concurrent)", claim_id=processable_claim.claim_id, error=str(e), exc_info=True)
                await self._update_claim_and_lines_in_db(db_claim, processable_claim, "rvu_calculation_failed",
                                                         validation_errors=[f"RVU calculation error: {str(e)}"],
                                                         ml_log_info={"ml_decision": ml_decision, "ml_score": ml_confidence_score})
                return {"db_id": db_claim.id, "claim_id": processable_claim.claim_id, "status": "rvu_calculation_failed", "error": str(e), "ml_decision": ml_decision}

    async def _update_claim_and_lines_in_db(self,
                                         db_claim_to_update: ClaimModel,
                                         processed_pydantic_claim: Optional[ProcessableClaim],
                                         new_status: str,
                                         validation_errors: List[str] = None,
                                         ml_log_info: Optional[Dict] = None): # New parameter
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
