import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import joinedload
import structlog
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from decimal import Decimal
import time # For measuring duration

from ..core.config.settings import get_settings
from ..core.database.models.claims_db import ClaimModel
from ..api.models.claim_models import ProcessableClaim
from .validation.claim_validator import ClaimValidator
from .rvu_service import RVUService
from ..core.cache.cache_manager import get_cache_manager
from .ml_pipeline.feature_extractor import FeatureExtractor
from .ml_pipeline.optimized_predictor import OptimizedPredictor

# Import Metric Objects
from ..core.monitoring.metrics import (
    CLAIMS_PROCESSED_TOTAL,
    CLAIM_PROCESSING_DURATION_SECONDS,
    CLAIMS_BATCH_PROCESSING_DURATION_SECONDS,
    CLAIMS_THROUGHPUT_LAST_BATCH,
    ML_PREDICTIONS_TOTAL,
    ML_PREDICTION_CONFIDENCE_SCORE
)

logger = structlog.get_logger(__name__)

class ClaimProcessingService:
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.validator = ClaimValidator()

        cache_manager = get_cache_manager()
        self.rvu_service = RVUService(cache_manager=cache_manager)

        current_settings = get_settings()
        self.concurrent_processing_semaphore = asyncio.Semaphore(current_settings.MAX_CONCURRENT_CLAIM_PROCESSING)

        self.feature_extractor = FeatureExtractor()
        self.predictor = OptimizedPredictor(model_path=current_settings.ML_MODEL_PATH)

        logger.info("ClaimProcessingService initialized",
                    db_session_id=id(db_session),
                    cache_manager_id=id(cache_manager),
                    concurrency_limit=current_settings.MAX_CONCURRENT_CLAIM_PROCESSING,
                    ml_model_path=current_settings.ML_MODEL_PATH)

    async def _process_single_claim_concurrently(self, db_claim: ClaimModel) -> Dict[str, Any]:
        start_time = time.monotonic()
        # Default status for metric if an unexpected error occurs before specific status is set
        status_for_metric: str = "unknown_processing_error"
        ml_decision_for_metric_label: str = "ML_NOT_REACHED" # For ML_PREDICTIONS_TOTAL label
        ml_confidence_score_for_metric: float = 0.0

        # Return dictionary structure
        result_dict = {
            "db_id": db_claim.id,
            "claim_id": db_claim.claim_id, # Use original claim_id from db_claim for consistency in return
            "status": status_for_metric, # Will be updated
            "ml_decision": ml_decision_for_metric_label, # Will be updated
            "errors": None, # Placeholder for errors
            "error_detail_str": None # Placeholder for error string
        }

        try:
            async with self.concurrent_processing_semaphore:
                logger.debug("Starting processing for a single claim concurrently", claim_id=db_claim.claim_id, db_id=db_claim.id)
                processable_claim: ProcessableClaim

                # 1. Pydantic Conversion
                try:
                    processable_claim = ProcessableClaim.model_validate(db_claim)
                    result_dict["claim_id"] = processable_claim.claim_id # Update with Pydantic model's claim_id if needed
                except Exception as e:
                    status_for_metric = "conversion_error"
                    logger.error("Pydantic conversion error for claim", claim_db_id=db_claim.id, error=str(e), exc_info=True)
                    await self._update_claim_and_lines_in_db(db_claim, None, status_for_metric, validation_errors=[f"Pydantic conversion error: {str(e)}"])
                    result_dict.update({"status": status_for_metric, "error_detail_str": str(e)})
                    return result_dict # Early exit

                # 2. Validation
                validation_errors = self.validator.validate_claim(processable_claim)
                if validation_errors:
                    status_for_metric = "validation_failed"
                    logger.warn("Claim validation failed (concurrent)", claim_id=processable_claim.claim_id, errors=validation_errors)
                    await self._update_claim_and_lines_in_db(db_claim, processable_claim, status_for_metric, validation_errors=validation_errors)
                    result_dict.update({"status": status_for_metric, "errors": validation_errors})
                    return result_dict # Early exit

                logger.info("Claim validation successful (concurrent)", claim_id=processable_claim.claim_id)

                # 3. ML Processing Step
                try:
                    features = self.feature_extractor.extract_features(processable_claim)
                    prediction_scores = await self.predictor.predict_async(features)

                    if prediction_scores.ndim == 2 and prediction_scores.shape[0] > 0 and prediction_scores.shape[1] >= 2:
                        prob_approve = prediction_scores[0][1]
                        ml_confidence_score_for_metric = float(prob_approve)
                        ML_PREDICTION_CONFIDENCE_SCORE.observe(ml_confidence_score_for_metric)

                        current_settings = get_settings()
                        if prob_approve >= current_settings.ML_APPROVAL_THRESHOLD:
                            ml_decision_for_metric_label = "ML_APPROVED"
                            logger.info("Claim approved by ML model", claim_id=processable_claim.claim_id, score=ml_confidence_score_for_metric)
                        else:
                            ml_decision_for_metric_label = "ML_REJECTED"
                            logger.warn("Claim rejected by ML model", claim_id=processable_claim.claim_id, score=ml_confidence_score_for_metric)
                    else:
                        ml_decision_for_metric_label = "ML_ERROR_UNEXPECTED_FORMAT"
                        logger.warn("Unexpected ML prediction score format", claim_id=processable_claim.claim_id, scores_shape=prediction_scores.shape)
                except Exception as ml_exc:
                    ml_decision_for_metric_label = "ML_PROCESSING_ERROR"
                    logger.error("Error during ML processing step", claim_id=processable_claim.claim_id, error=str(ml_exc), exc_info=True)

                ML_PREDICTIONS_TOTAL.labels(ml_decision_outcome=ml_decision_for_metric_label).inc()
                processable_claim.ml_score = ml_confidence_score_for_metric
                processable_claim.ml_derived_decision = ml_decision_for_metric_label
                result_dict["ml_decision"] = ml_decision_for_metric_label

                if ml_decision_for_metric_label == "ML_REJECTED":
                    status_for_metric = "ml_rejected"
                    await self._update_claim_and_lines_in_db(
                        db_claim, processable_claim, status_for_metric,
                        ml_log_info={"ml_decision": ml_decision_for_metric_label, "ml_score": ml_confidence_score_for_metric}
                    )
                    result_dict["status"] = status_for_metric
                    return result_dict # Early exit

                # 4. RVU Calculation
                try:
                    await self.rvu_service.calculate_rvu_for_claim(processable_claim, self.db)
                    status_for_metric = "processing_complete"
                    logger.info("RVU calculation successful (concurrent)", claim_id=processable_claim.claim_id)
                    await self._update_claim_and_lines_in_db(
                        db_claim, processable_claim, status_for_metric,
                        ml_log_info={"ml_decision": ml_decision_for_metric_label, "ml_score": ml_confidence_score_for_metric}
                    )
                    result_dict["status"] = status_for_metric
                    return result_dict # Success exit
                except Exception as e_rvu:
                    status_for_metric = "rvu_calculation_failed"
                    logger.error("RVU calculation failed (concurrent)", claim_id=processable_claim.claim_id, error=str(e_rvu), exc_info=True)
                    await self._update_claim_and_lines_in_db(
                        db_claim, processable_claim, status_for_metric,
                        validation_errors=[f"RVU calculation error: {str(e_rvu)}"], # Using validation_errors to pass error string
                        ml_log_info={"ml_decision": ml_decision_for_metric_label, "ml_score": ml_confidence_score_for_metric}
                    )
                    result_dict.update({"status": status_for_metric, "error_detail_str": str(e_rvu)})
                    return result_dict # RVU failure exit

        except Exception as final_e: # Catch any other unexpected error during the whole guarded process
            status_for_metric = "unknown_processing_error" # Or could be more specific if identifiable
            logger.error("Unhandled error in single claim processing under semaphore", claim_id=db_claim.claim_id, error=str(final_e), exc_info=True)
            # Attempt to update DB with an error status if possible, but processable_claim might not be defined
            # This path is tricky; the error might be before processable_claim is made.
            # For now, this error won't update DB status here, but will be counted in batch summary.
            result_dict.update({"status": status_for_metric, "error_detail_str": str(final_e), "ml_decision": ml_decision_for_metric_label})
            # We still increment CLAIMS_PROCESSED_TOTAL in finally, so this status is important.
            raise # Re-raise to be caught by asyncio.gather's return_exceptions=True

        finally:
            duration = time.monotonic() - start_time
            CLAIM_PROCESSING_DURATION_SECONDS.observe(duration)
            CLAIMS_PROCESSED_TOTAL.labels(final_status=status_for_metric).inc() # Increment based on determined status

    async def process_pending_claims_batch(self, batch_size: int = 100):
        batch_start_time = time.monotonic()
        logger.info("Starting batch processing of claims with concurrency", batch_size=batch_size)

        fetched_db_claims = await self._fetch_pending_claims(batch_size)

        if not fetched_db_claims:
            logger.info("No pending claims to process.")
            # Ensure all keys are present in the return dictionary, even for no claims
            summary = {
                "message": "No pending claims to process.", "attempted_claims": 0,
                "conversion_errors":0, "validation_failures": 0,
                "ml_approved_raw": 0, "ml_rejected_raw": 0, "ml_errors_raw": 0,
                "stopped_by_ml_rejection": 0,
                "rvu_calculation_failures":0, "successfully_processed_count": 0,
                "other_exceptions":0
            }
            CLAIMS_BATCH_PROCESSING_DURATION_SECONDS.observe(time.monotonic() - batch_start_time)
            CLAIMS_THROUGHPUT_LAST_BATCH.set(0)
            logger.info("Concurrent batch processing summary (no claims).", **summary)
            return summary

        logger.info(f"Fetched {len(fetched_db_claims)} claims for concurrent processing.")

        tasks = [self._process_single_claim_concurrently(db_claim) for db_claim in fetched_db_claims]
        processing_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Initialize counters
        attempted_claims = len(fetched_db_claims)
        conversion_errors_count = 0
        failed_validation_count = 0
        successfully_processed_count = 0
        failed_rvu_count = 0
        ml_approved_raw_count = 0
        ml_rejected_raw_count = 0
        ml_errors_raw_count = 0
        stopped_by_ml_rejection_count = 0
        other_exceptions_count = 0

        for result in processing_results:
            if isinstance(result, Exception): # Unhandled exception from _process_single_claim_concurrently
                logger.error("Task failed with unhandled exception in asyncio.gather results", exception_type=type(result).__name__, error_detail=str(result))
                other_exceptions_count +=1
            elif isinstance(result, dict):
                status = result.get("status")
                ml_decision_from_result = result.get("ml_decision", "N/A")

                if ml_decision_from_result == "ML_APPROVED": ml_approved_raw_count += 1
                elif ml_decision_from_result == "ML_REJECTED": ml_rejected_raw_count += 1
                elif ml_decision_from_result.startswith("ML_ERROR") or ml_decision_from_result == "ML_PROCESSING_ERROR": ml_errors_raw_count +=1

                if status == "conversion_error": conversion_errors_count +=1
                elif status == "validation_failed": failed_validation_count += 1
                elif status == "ml_rejected": stopped_by_ml_rejection_count += 1
                elif status == "rvu_calculation_failed": failed_rvu_count +=1
                elif status == "processing_complete": successfully_processed_count += 1
                # "unknown_processing_error" from task will be caught by `isinstance(result, Exception)` if re-raised,
                # or if returned as dict, would need specific handling here. Assuming it's re-raised from task.

        batch_duration_seconds = time.monotonic() - batch_start_time
        CLAIMS_BATCH_PROCESSING_DURATION_SECONDS.observe(batch_duration_seconds)

        if batch_duration_seconds > 0:
            throughput = successfully_processed_count / batch_duration_seconds
            CLAIMS_THROUGHPUT_LAST_BATCH.set(throughput)
        else:
            CLAIMS_THROUGHPUT_LAST_BATCH.set(0)

        final_summary = {
            "message": "Concurrent batch processing finished.",
            "attempted_claims": attempted_claims,
            "conversion_errors": conversion_errors_count,
            "validation_failures": failed_validation_count,
            "ml_approved_raw": ml_approved_raw_count,
            "ml_rejected_raw": ml_rejected_raw_count,
            "ml_errors_raw": ml_errors_raw_count,
            "stopped_by_ml_rejection": stopped_by_ml_rejection_count,
            "rvu_calculation_failures": failed_rvu_count,
            "successfully_processed_count": successfully_processed_count,
            "other_exceptions" : other_exceptions_count,
            "batch_duration_seconds": round(batch_duration_seconds, 2),
            "throughput_claims_per_second": CLAIMS_THROUGHPUT_LAST_BATCH._value.get() # For logging current Gauge value
        }
        logger.info("Concurrent batch processing summary.", **final_summary)

        return final_summary

    async def _fetch_pending_claims(self, batch_size: int) -> list[ClaimModel]:
        # ... (definition as before) ...
        logger.info("Fetching pending claims from database", batch_size=batch_size)
        stmt = (
            select(ClaimModel)
            .where(ClaimModel.processing_status == 'pending')
            .order_by(ClaimModel.created_at.asc())
            .limit(batch_size)
            .options(joinedload(ClaimModel.line_items))
        )
        result = await self.db.execute(stmt)
        claims = result.scalars().all()
        if claims:
            logger.info(f"Fetched {len(claims)} pending claims from the database.")
        else:
            logger.info("No pending claims found in the database.")
        return list(claims)

    async def _update_claim_and_lines_in_db(self,
                                         db_claim_to_update: ClaimModel,
                                         processed_pydantic_claim: Optional[ProcessableClaim],
                                         new_status: str,
                                         validation_errors: List[str] = None,
                                         ml_log_info: Optional[Dict] = None):
        # ... (definition as before) ...
        if db_claim_to_update is None:
            logger.error("Cannot update claim in DB: SQLAlchemy model instance is None.")
            return

        logger.info("Updating claim and lines in DB", claim_db_id=db_claim_to_update.id, new_status=new_status, ml_info=ml_log_info or "N/A")
        try:
            db_claim_to_update.processing_status = new_status

            if validation_errors:
                logger.warn("Storing/logging errors for claim", claim_id=db_claim_to_update.claim_id, errors=validation_errors)

            if new_status == "processing_complete" and processed_pydantic_claim:
                map_pydantic_line_items = {line.id: line for line in processed_pydantic_claim.line_items}
                for db_line_item in db_claim_to_update.line_items:
                    if db_line_item.id in map_pydantic_line_items:
                        pydantic_line = map_pydantic_line_items[db_line_item.id]
                        if pydantic_line.rvu_total is not None:
                            db_line_item.rvu_total = pydantic_line.rvu_total
                db_claim_to_update.processed_at = datetime.now(timezone.utc)

            self.db.add(db_claim_to_update)
            await self.db.commit()

            await self.db.refresh(db_claim_to_update)
            if db_claim_to_update.line_items:
                for line_item_to_refresh in db_claim_to_update.line_items:
                    await self.db.refresh(line_item_to_refresh)

            logger.info("Claim and lines updated successfully in DB", claim_db_id=db_claim_to_update.id, new_status=db_claim_to_update.processing_status)

        except Exception as e:
            await self.db.rollback()
            logger.error("Failed to update claim and lines in DB", claim_db_id=db_claim_to_update.id if db_claim_to_update else "Unknown ID", error=str(e), exc_info=True)
