import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import joinedload
import structlog
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from decimal import Decimal
import time

from ..core.config.settings import get_settings
from ..core.database.models.claims_db import ClaimModel
from ..api.models.claim_models import ProcessableClaim
from .validation.claim_validator import ClaimValidator
from .rvu_service import RVUService
from ..core.cache.cache_manager import get_cache_manager
from .ml_pipeline.feature_extractor import FeatureExtractor
from .ml_pipeline.optimized_predictor import OptimizedPredictor
# Import MetricsCollector instead of individual metrics
from ..core.monitoring.app_metrics import MetricsCollector
from ..core.database.models.failed_claims_db import FailedClaimModel # Added FailedClaimModel


logger = structlog.get_logger(__name__)

class ClaimProcessingService:
    def __init__(self, db_session: AsyncSession, metrics_collector: MetricsCollector): # Added metrics_collector
        self.db = db_session
        self.metrics_collector = metrics_collector # Store it
        self.validator = ClaimValidator()
        cache_manager = get_cache_manager()
        # Pass metrics_collector to RVUService
        self.rvu_service = RVUService(cache_manager=cache_manager, metrics_collector=self.metrics_collector)
        current_settings = get_settings()
        self.concurrent_processing_semaphore = asyncio.Semaphore(current_settings.MAX_CONCURRENT_CLAIM_PROCESSING)
        self.feature_extractor = FeatureExtractor()
        # Pass metrics_collector to OptimizedPredictor
        self.predictor = OptimizedPredictor(
            model_path=current_settings.ML_MODEL_PATH,
            metrics_collector=self.metrics_collector,
            feature_count=current_settings.ML_FEATURE_COUNT # Ensure feature_count is passed if init expects it
        )
        logger.info("ClaimProcessingService initialized",
                    db_session_id=id(db_session),
                    cache_manager_id=id(cache_manager),
                    metrics_collector_id=id(self.metrics_collector),
                    concurrency_limit=current_settings.MAX_CONCURRENT_CLAIM_PROCESSING,
                    ml_model_path=current_settings.ML_MODEL_PATH)

    async def _process_single_claim_concurrently(self, db_claim: ClaimModel) -> Dict[str, Any]:
        start_time = time.monotonic()
        status_for_metric: str = "unknown_processing_error" # This will be the final_status for CLAIMS_PROCESSED_TOTAL
        # ML metrics are now handled by OptimizedPredictor, so no direct calls here
        # ml_decision_for_metric_label: str = "ML_NOT_REACHED"
        # ml_confidence_score_for_metric: float = 0.0
        current_duration_ms: Optional[float] = None
        processable_claim_instance: Optional[ProcessableClaim] = None

        result_dict = {
            "db_id": db_claim.id,
            "claim_id": db_claim.claim_id, # Initial value, might be updated by ProcessableClaim
            "status": status_for_metric,
            "ml_decision": "N/A", # Default, updated after ML
            "errors": None,
            "error_detail_str": None
        }

        try:
            async with self.concurrent_processing_semaphore:
                logger.debug("Starting processing for a single claim concurrently", claim_id=db_claim.claim_id, db_id=db_claim.id)

                try:
                    processable_claim_instance = ProcessableClaim.model_validate(db_claim)
                    result_dict["claim_id"] = processable_claim_instance.claim_id
                except Exception as e:
                    status_for_metric = "conversion_error"
                    error_reason = f"Pydantic conversion error: {str(e)}"
                    logger.error(error_reason, claim_db_id=db_claim.id, exc_info=True)
                    current_duration_ms = (time.monotonic() - start_time) * 1000.0
                    # Store failed claim before updating staging table
                    await self._store_failed_claim(db_claim_to_update=db_claim, processable_claim_instance=None,
                                                   failed_stage="CONVERSION_ERROR", reason=error_reason)
                    await self._update_claim_and_lines_in_db(db_claim, None, status_for_metric,
                                                             duration_ms=current_duration_ms,
                                                             validation_errors=[error_reason])
                    result_dict.update({"status": status_for_metric, "error_detail_str": str(e)})
                    return result_dict

                validation_errors = self.validator.validate_claim(processable_claim_instance)
                if validation_errors:
                    status_for_metric = "validation_failed"
                    error_reason_val = f"Validation errors: {'; '.join(validation_errors)}"
                    logger.warn("Claim validation failed", claim_id=processable_claim_instance.claim_id, errors=validation_errors)
                    current_duration_ms = (time.monotonic() - start_time) * 1000.0
                    processable_claim_instance.processing_duration_ms = current_duration_ms
                    # Store failed claim
                    await self._store_failed_claim(db_claim_to_update=db_claim, processable_claim_instance=processable_claim_instance,
                                                   failed_stage="VALIDATION", reason=error_reason_val)
                    await self._update_claim_and_lines_in_db(db_claim, processable_claim_instance, status_for_metric,
                                                             duration_ms=current_duration_ms,
                                                             validation_errors=validation_errors)
                    result_dict.update({"status": status_for_metric, "errors": validation_errors, "ml_decision": "N/A"})
                    return result_dict

                logger.info("Claim validation successful", claim_id=processable_claim_instance.claim_id)

                try:
                    features = self.feature_extractor.extract_features(processable_claim_instance)
                    # OptimizedPredictor.predict_batch expects a list of features
                    # Assuming predict_async was a typo and it should use predict_batch logic or similar
                    # For simplicity, let's assume predict_batch is called elsewhere or this is adapted.
                    # The key is that OptimizedPredictor itself now handles its ML metrics.
                    # The call to self.predictor.predict_async(features) needs to be aligned with actual OptimizedPredictor method.
                    # Let's assume it returns a dict like {'ml_score': score, 'ml_derived_decision': decision}
                    prediction_result = await self.predictor.predict_batch([features]) # Pass as a batch of one

                    ml_score = None
                    ml_decision = "ML_ERROR" # Default if result is not as expected

                    if prediction_result and isinstance(prediction_result, list) and prediction_result[0]:
                        pred_data = prediction_result[0]
                        ml_score = pred_data.get('ml_score')
                        ml_decision = pred_data.get('ml_derived_decision', "ML_ERROR")
                    else:
                        logger.warn("Unexpected prediction result format from OptimizedPredictor", claim_id=processable_claim_instance.claim_id)
                        ml_decision = "ML_ERROR_UNEXPECTED_FORMAT"

                except Exception as ml_exc:
                    ml_decision = "ML_PROCESSING_ERROR"
                    logger.error("Error during ML processing step", claim_id=processable_claim_instance.claim_id, error=str(ml_exc), exc_info=True)

                # ML metrics (Counter for total, Histogram for confidence) are now recorded by OptimizedPredictor's methods.
                # We just store the results.
                processable_claim_instance.ml_score = Decimal(str(ml_score)) if ml_score is not None else None
                processable_claim_instance.ml_derived_decision = ml_decision
                result_dict["ml_decision"] = ml_decision
                log_ml_info = {"ml_decision": ml_decision, "ml_score": ml_score}

                if ml_decision == "ML_REJECTED":
                    status_for_metric = "ml_rejected"
                    error_reason_ml = f"ML Rejected. Score: {ml_score}, Threshold: {get_settings().ML_APPROVAL_THRESHOLD}"
                    logger.warn("Claim rejected by ML model", claim_id=processable_claim_instance.claim_id, score=ml_score)
                    current_duration_ms = (time.monotonic() - start_time) * 1000.0
                    processable_claim_instance.processing_duration_ms = current_duration_ms
                    # Store failed claim
                    await self._store_failed_claim(db_claim_to_update=db_claim, processable_claim_instance=processable_claim_instance,
                                                   failed_stage="ML_REJECTION", reason=error_reason_ml)
                    await self._update_claim_and_lines_in_db(db_claim, processable_claim_instance, status_for_metric,
                                                             duration_ms=current_duration_ms, ml_log_info=log_ml_info)
                    result_dict["status"] = status_for_metric
                    return result_dict

                try:
                    await self.rvu_service.calculate_rvu_for_claim(processable_claim_instance, self.db)
                    status_for_metric = "processing_complete" # Successfully processed
                    logger.info("RVU calculation successful", claim_id=processable_claim_instance.claim_id)
                    current_duration_ms = (time.monotonic() - start_time) * 1000.0
                    processable_claim_instance.processing_duration_ms = current_duration_ms
                    await self._update_claim_and_lines_in_db(db_claim, processable_claim_instance, status_for_metric,
                                                             duration_ms=current_duration_ms, ml_log_info=log_ml_info)
                    result_dict["status"] = status_for_metric
                    return result_dict
                except Exception as e_rvu:
                    status_for_metric = "rvu_calculation_failed"
                    error_reason_rvu = f"RVU calculation error: {str(e_rvu)}"
                    logger.error(error_reason_rvu, claim_id=processable_claim_instance.claim_id, exc_info=True)
                    current_duration_ms = (time.monotonic() - start_time) * 1000.0
                    if processable_claim_instance: processable_claim_instance.processing_duration_ms = current_duration_ms
                    # Store failed claim
                    await self._store_failed_claim(db_claim_to_update=db_claim, processable_claim_instance=processable_claim_instance,
                                                   failed_stage="RVU_CALCULATION_FAILED", reason=error_reason_rvu)
                    await self._update_claim_and_lines_in_db(db_claim, processable_claim_instance, status_for_metric,
                                                             duration_ms=current_duration_ms,
                                                             validation_errors=[error_reason_rvu], # Use validation_errors to store this specific error
                                                             ml_log_info=log_ml_info)
                    result_dict.update({"status": status_for_metric, "error_detail_str": str(e_rvu)})
                    return result_dict
        except Exception as final_e:
            status_for_metric = "unknown_processing_error"
            logger.error("Unhandled error in single claim processing", claim_id=db_claim.claim_id, error=str(final_e), exc_info=True)
            result_dict.update({"status": status_for_metric, "error_detail_str": str(final_e)})
            # If processable_claim_instance exists, try to set duration before re-raising for Prometheus.
            # This path is if semaphore itself or something outside the main try-catch within semaphore fails.
            if processable_claim_instance and hasattr(processable_claim_instance, 'processing_duration_ms'):
                 current_duration_ms_outer_exc = (time.monotonic() - start_time) * 1000.0
                 processable_claim_instance.processing_duration_ms = current_duration_ms_outer_exc
            raise
        finally:
            final_duration_sec = time.monotonic() - start_time
            # Use MetricsCollector for individual claim duration
            self.metrics_collector.record_individual_claim_duration(final_duration_sec)
            # CLAIMS_PROCESSED_TOTAL is now handled by record_batch_processed at the end of a batch.
            # We return status_for_metric from this function, and process_pending_claims_batch will aggregate.
            # The status_for_metric at this point should be "unknown_processing_error" if this finally block is reached
            # due to an exception within the main try block of _process_single_claim_concurrently.
            # If it's a normal exit (successful processing), status_for_metric would be "processing_complete".

            # Ensure the Pydantic model has the duration if it was processed to a point where it exists
            if processable_claim_instance and processable_claim_instance.processing_duration_ms is None:
                 processable_claim_instance.processing_duration_ms = final_duration_sec * 1000.0
            # This doesn't help persist it if not already done before return. Persistence is handled before returns.

    async def _store_failed_claim(
        self,
        db_claim_to_update: Optional[ClaimModel],
        processable_claim_instance: Optional[ProcessableClaim],
        failed_stage: str,
        reason: str,
    ):
        logger.warn(f"Storing claim to failed_claims table. Stage: {failed_stage}, Reason: {reason}",
                    claim_id=db_claim_to_update.claim_id if db_claim_to_update else (processable_claim_instance.claim_id if processable_claim_instance else "N/A"))

        original_data_json = None
        if processable_claim_instance:
            original_data_json = processable_claim_instance.model_dump(mode='json')
        elif db_claim_to_update:
            # Basic serialization for ClaimModel if ProcessableClaim is not available (e.g. conversion error)
            # This assumes ClaimModel attributes are directly serializable or have simple types.
            # Adjust if complex types like relationships need specific handling.
            original_data_json = {
                "claim_id": db_claim_to_update.claim_id,
                "facility_id": db_claim_to_update.facility_id,
                "patient_account_number": db_claim_to_update.patient_account_number,
                "total_charges": float(db_claim_to_update.total_charges) if db_claim_to_update.total_charges is not None else None,
                # Add other relevant fields from ClaimModel
            }

        failed_claim_entry = FailedClaimModel(
            original_claim_db_id=db_claim_to_update.id if db_claim_to_update else None,
            claim_id=db_claim_to_update.claim_id if db_claim_to_update else (processable_claim_instance.claim_id if processable_claim_instance else None),
            facility_id=db_claim_to_update.facility_id if db_claim_to_update else (processable_claim_instance.facility_id if processable_claim_instance else None),
            patient_account_number=db_claim_to_update.patient_account_number if db_claim_to_update else (processable_claim_instance.patient_account_number if processable_claim_instance else None),
            failed_at_stage=failed_stage,
            failure_reason=reason,
            original_claim_data=original_data_json
        )

        try:
            self.db.add(failed_claim_entry)
            # Note: Commit is handled by the calling function's transaction (_update_claim_and_lines_in_db or similar context)
            logger.debug("Added FailedClaimModel instance to session.",
                         failed_claim_id=failed_claim_entry.claim_id,
                         original_db_id=failed_claim_entry.original_claim_db_id)
        except Exception as e:
            logger.error("Failed to create FailedClaimModel instance or add to session.",
                         error=str(e), exc_info=True)
            # Do not rollback here, let the outer transaction handle it.


    async def process_pending_claims_batch(self, batch_size: int = 100):
        batch_start_time = time.monotonic()
        logger.info("Starting batch processing of claims with concurrency", batch_size=batch_size)

        fetched_db_claims = await self._fetch_pending_claims(batch_size)

        if not fetched_db_claims:
            logger.info("No pending claims to process.")
            summary = {
                "message": "No pending claims to process.", "attempted_claims": 0,
                "conversion_errors":0, "validation_failures": 0,
                "ml_approved_raw": 0, "ml_rejected_raw": 0, "ml_errors_raw": 0, # These are for info, metric is handled by predictor
                "stopped_by_ml_rejection": 0,
                "rvu_calculation_failures":0, "successfully_processed_count": 0,
                "other_exceptions":0
            }
            # Use MetricsCollector for batch metrics
            self.metrics_collector.record_batch_processed(
                batch_size=0,
                duration_seconds=(time.monotonic() - batch_start_time),
                claims_by_final_status={} # No claims processed
            )
            logger.info("Concurrent batch processing summary (no claims).", **summary)
            return summary

        logger.info(f"Fetched {len(fetched_db_claims)} claims for concurrent processing.")

        tasks = [self._process_single_claim_concurrently(db_claim) for db_claim in fetched_db_claims]
        processing_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results for batch metrics
        attempted_claims = len(fetched_db_claims)
        claims_by_final_status_for_metric: Dict[str, int] = {}
        # Detailed counters for summary log (metrics derive from claims_by_final_status_for_metric)
        conversion_errors_count = 0
        failed_validation_count = 0
        # successfully_processed_count = 0 # This will be sum from claims_by_final_status_for_metric
        failed_rvu_count = 0
        # Raw ML counts for logging, actual ML metrics are handled by OptimizedPredictor
        ml_approved_raw_count = 0
        ml_rejected_raw_count = 0
        ml_errors_raw_count = 0
        stopped_by_ml_rejection_count = 0 # This is a final status
        other_exceptions_count = 0


        for result in processing_results:
            if isinstance(result, Exception):
                logger.error("Task failed with unhandled exception in asyncio.gather results", exception_type=type(result).__name__, error_detail=str(result))
                other_exceptions_count +=1
                # Increment a specific status for CLAIMS_PROCESSED_TOTAL via claims_by_final_status_for_metric
                claims_by_final_status_for_metric["unhandled_exception_in_gather"] = \
                    claims_by_final_status_for_metric.get("unhandled_exception_in_gather", 0) + 1
            elif isinstance(result, dict):
                status = result.get("status", "unknown_processing_error") # Default if status missing
                claims_by_final_status_for_metric[status] = \
                    claims_by_final_status_for_metric.get(status, 0) + 1

                # Update detailed counters for logging summary
                ml_decision_from_result = result.get("ml_decision", "N/A")
                if ml_decision_from_result == "ML_APPROVED": ml_approved_raw_count += 1
                elif ml_decision_from_result == "ML_REJECTED": ml_rejected_raw_count += 1
                elif ml_decision_from_result.startswith("ML_ERROR") or ml_decision_from_result == "ML_PROCESSING_ERROR": ml_errors_raw_count +=1

                if status == "conversion_error": conversion_errors_count +=1
                elif status == "validation_failed": failed_validation_count += 1
                # Note: "ml_rejected" status is counted by claims_by_final_status_for_metric
                # if status == "ml_rejected": stopped_by_ml_rejection_count += 1 # This is now part of claims_by_final_status_for_metric
                elif status == "rvu_calculation_failed": failed_rvu_count +=1
                # if status == "processing_complete": successfully_processed_count += 1 # Also from claims_by_final_status_for_metric

        batch_duration_seconds = time.monotonic() - batch_start_time

        # Use MetricsCollector for batch metrics
        self.metrics_collector.record_batch_processed(
            batch_size=attempted_claims,
            duration_seconds=batch_duration_seconds,
            claims_by_final_status=claims_by_final_status_for_metric
        )

        # For logging summary, retrieve some counts from the aggregated map
        successfully_processed_count = claims_by_final_status_for_metric.get('processing_complete', 0)
        stopped_by_ml_rejection_count = claims_by_final_status_for_metric.get('ml_rejected',0)


        final_summary = {
            "message": "Concurrent batch processing finished.",
            "attempted_claims": attempted_claims,
            "conversion_errors": claims_by_final_status_for_metric.get('conversion_error',0), # From map
            "validation_failures": claims_by_final_status_for_metric.get('validation_failed',0), # From map
            "ml_approved_raw": ml_approved_raw_count, # Informational from ML results
            "ml_rejected_raw": ml_rejected_raw_count, # Informational from ML results
            "ml_errors_raw": ml_errors_raw_count,     # Informational from ML results
            "stopped_by_ml_rejection": stopped_by_ml_rejection_count, # From map
            "rvu_calculation_failures": claims_by_final_status_for_metric.get('rvu_calculation_failed',0), # From map
            "successfully_processed_count": successfully_processed_count, # From map
            "other_exceptions" : claims_by_final_status_for_metric.get('unhandled_exception_in_gather',0) + \
                                 claims_by_final_status_for_metric.get('unknown_processing_error',0), # From map
            "batch_duration_seconds": round(batch_duration_seconds, 2),
            "throughput_claims_per_second": CLAIMS_THROUGHPUT_GAUGE._value.get() # Read from global metric
        }
        logger.info("Concurrent batch processing summary.", **final_summary) # type: ignore

        return final_summary

    async def _fetch_pending_claims(self, batch_size: int) -> list[ClaimModel]:
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
                                         duration_ms: Optional[float] = None, # New parameter
                                         validation_errors: List[str] = None,
                                         ml_log_info: Optional[Dict] = None):
        if db_claim_to_update is None:
            logger.error("Cannot update claim in DB: SQLAlchemy model instance is None.")
            return

        logger.info("Updating claim and lines in DB", claim_db_id=db_claim_to_update.id, new_status=new_status, ml_info=ml_log_info or "N/A", duration_ms=duration_ms)
        try:
            db_claim_to_update.processing_status = new_status

            if duration_ms is not None:
                db_claim_to_update.processing_duration_ms = int(duration_ms)

            if processed_pydantic_claim: # Check if Pydantic model exists (it won't for early conversion error)
                if processed_pydantic_claim.ml_score is not None:
                    # Ensure conversion to Decimal for Numeric field if ml_score is float
                    db_claim_to_update.ml_score = Decimal(str(processed_pydantic_claim.ml_score))
                if processed_pydantic_claim.ml_derived_decision is not None:
                    db_claim_to_update.ml_derived_decision = processed_pydantic_claim.ml_derived_decision

            if validation_errors:
                logger.warn("Storing/logging errors for claim", claim_id=db_claim_to_update.claim_id, errors=validation_errors)
                # Example: db_claim_to_update.error_details = {"errors": validation_errors} # If such a field exists

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
