import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import joinedload
import structlog
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from decimal import Decimal
import time
import hashlib # For A/B testing consistent hashing
from sqlalchemy.exc import OperationalError # For retry logic

from ..core.config.settings import get_settings
from ..core.database.models.claims_db import ClaimModel, ClaimLineItemModel # Added ClaimLineItemModel
from ..api.models.claim_models import ProcessableClaim
from .validation.claim_validator import ClaimValidator
from .rvu_service import RVUService
from ..core.cache.cache_manager import get_cache_manager
from .ml_pipeline.feature_extractor import FeatureExtractor
from .ml_pipeline.optimized_predictor import OptimizedPredictor
from ..core.monitoring.app_metrics import MetricsCollector
from ..core.database.models.failed_claims_db import FailedClaimModel


logger = structlog.get_logger(__name__)

class ClaimProcessingService:
    def __init__(self, db_session: AsyncSession, metrics_collector: MetricsCollector):
        self.db = db_session
        self.metrics_collector = metrics_collector
        self.settings = get_settings()
        self.validator = ClaimValidator()
        cache_manager = get_cache_manager()
        self.rvu_service = RVUService(cache_manager=cache_manager, metrics_collector=self.metrics_collector)

        self.validation_ml_semaphore = asyncio.Semaphore(self.settings.VALIDATION_CONCURRENCY)
        self.rvu_semaphore = asyncio.Semaphore(self.settings.RVU_CALCULATION_CONCURRENCY)

        self.feature_extractor = FeatureExtractor(metrics_collector=self.metrics_collector)

        # Primary Predictor
        self.primary_predictor = OptimizedPredictor(
            model_path=self.settings.ML_MODEL_PATH,
            metrics_collector=self.metrics_collector,
            feature_count=self.settings.ML_FEATURE_COUNT
        )

        # Challenger Predictor for A/B Testing
        self.challenger_predictor: Optional[OptimizedPredictor] = None
        if self.settings.ML_CHALLENGER_MODEL_PATH and \
           self.settings.ML_AB_TEST_TRAFFIC_PERCENTAGE_TO_CHALLENGER > 0:
            logger.info("Challenger model path configured for A/B testing.",
                        path=self.settings.ML_CHALLENGER_MODEL_PATH,
                        traffic_percentage=self.settings.ML_AB_TEST_TRAFFIC_PERCENTAGE_TO_CHALLENGER)
            try:
                # Ensure the challenger model file exists before attempting to load
                # This check is simplified here; OptimizedPredictor itself logs if file not found.
                self.challenger_predictor = OptimizedPredictor(
                    model_path=self.settings.ML_CHALLENGER_MODEL_PATH,
                    metrics_collector=self.metrics_collector,
                    feature_count=self.settings.ML_FEATURE_COUNT # Assume same feature count
                )
                logger.info("Challenger OptimizedPredictor instantiated successfully.")
            except Exception as e:
                logger.error("Failed to instantiate Challenger OptimizedPredictor. A/B test will be inactive.",
                             path=self.settings.ML_CHALLENGER_MODEL_PATH, error=str(e), exc_info=True)
                self.challenger_predictor = None
        else:
            logger.info("No valid challenger model path or A/B traffic percentage configured. A/B testing inactive.")

        logger.info("ClaimProcessingService initialized",
                    db_session_id=id(db_session),
                    cache_manager_id=id(cache_manager), # type: ignore
                    metrics_collector_id=id(self.metrics_collector),
                    validation_concurrency=self.settings.VALIDATION_CONCURRENCY,
                    rvu_concurrency=self.settings.RVU_CALCULATION_CONCURRENCY,
                    primary_ml_model_path=self.settings.ML_MODEL_PATH,
                    challenger_ml_model_path=self.settings.ML_CHALLENGER_MODEL_PATH if self.challenger_predictor else "N/A")

    async def _perform_validation_and_ml(self, processable_claim: ProcessableClaim, db_claim_model_for_failed: ClaimModel) -> ProcessableClaim:
        async with self.validation_ml_semaphore:
            try:
                logger.debug("Performing validation for claim processing stage", claim_id=processable_claim.claim_id)
                validation_errors = self.validator.validate_claim(processable_claim)
                if validation_errors:
                    error_reason_val = f"Validation errors: {'; '.join(validation_errors)}"
                    logger.warn("Claim validation failed", claim_id=processable_claim.claim_id, errors=validation_errors)
                    processable_claim.processing_status = "validation_failed"
                    await self._store_failed_claim(db_claim_to_update=db_claim_model_for_failed,
                                                   processable_claim_instance=processable_claim,
                                                   failed_stage="VALIDATION", reason=error_reason_val)
                    return processable_claim

                logger.info("Claim validation successful", claim_id=processable_claim.claim_id)

                # ML Prediction Logic (with A/B test routing)
                selected_predictor = self.primary_predictor
                model_version_tag = f"control:{self.settings.ML_MODEL_PATH.split('/')[-1]}" if self.settings.ML_MODEL_PATH else "control:unknown"

                if self.challenger_predictor and \
                   self.settings.ML_CHALLENGER_MODEL_PATH and \
                   self.settings.ML_AB_TEST_TRAFFIC_PERCENTAGE_TO_CHALLENGER > 0:

                    combined_id_salt = f"{processable_claim.claim_id}-{self.settings.ML_AB_TEST_CLAIM_ID_SALT}"
                    hashed_string = hashlib.sha256(combined_id_salt.encode('utf-8')).hexdigest()
                    hash_val_percent = int(hashed_string[:4], 16) % 100

                    if hash_val_percent < (self.settings.ML_AB_TEST_TRAFFIC_PERCENTAGE_TO_CHALLENGER * 100):
                        selected_predictor = self.challenger_predictor
                        model_version_tag = f"challenger:{self.settings.ML_CHALLENGER_MODEL_PATH.split('/')[-1]}"
                        logger.debug("Routing claim to challenger model for A/B test", claim_id=processable_claim.claim_id, percentage_hash_val=hash_val_percent)
                    else:
                        logger.debug("Routing claim to primary model for A/B test", claim_id=processable_claim.claim_id, percentage_hash_val=hash_val_percent)

                processable_claim.ml_model_version_used = model_version_tag

                logger.debug("Performing ML prediction", claim_id=processable_claim.claim_id, model_used=model_version_tag)
                features = self.feature_extractor.extract_features(processable_claim)
                prediction_result_list = await selected_predictor.predict_batch([features])

                ml_score = None
                ml_decision = "ML_ERROR"

                if prediction_result_list and isinstance(prediction_result_list, list) and prediction_result_list[0]:
                    pred_data = prediction_result_list[0]
                    if "error" in pred_data:
                        ml_decision = pred_data.get("ml_derived_decision", "ML_ERROR_PREDICTOR")
                        logger.warn("ML prediction failed (predictor error)", claim_id=processable_claim.claim_id, predictor_error=pred_data["error"])
                    else:
                        ml_score = pred_data.get('ml_score')
                        ml_decision = pred_data.get('ml_derived_decision', "ML_ERROR")
                else:
                    logger.warn("Unexpected or empty prediction result format from OptimizedPredictor", claim_id=processable_claim.claim_id)
                    ml_decision = "ML_ERROR_UNEXPECTED_FORMAT"

                processable_claim.ml_score = Decimal(str(ml_score)) if ml_score is not None else None
                processable_claim.ml_derived_decision = ml_decision

                if ml_decision == "ML_REJECTED":
                    processable_claim.processing_status = "ml_rejected"
                    error_reason_ml = f"ML Rejected. Score: {ml_score}, Threshold: {self.settings.ML_APPROVAL_THRESHOLD}"
                    logger.warn("Claim rejected by ML model", claim_id=processable_claim.claim_id, score=ml_score)
                    await self._store_failed_claim(db_claim_to_update=db_claim_model_for_failed,
                                                   processable_claim_instance=processable_claim,
                                                   failed_stage="ML_REJECTION", reason=error_reason_ml)
                    return processable_claim

                if not ml_decision.startswith("ML_ERROR"):
                     processable_claim.processing_status = "ml_complete"
                else:
                    processable_claim.processing_status = "ml_error"
                    error_reason_ml_err = f"ML processing error: {ml_decision}"
                    await self._store_failed_claim(db_claim_to_update=db_claim_model_for_failed,
                                                   processable_claim_instance=processable_claim,
                                                   failed_stage="ML_ERROR", reason=error_reason_ml_err)
                    return processable_claim

            except Exception as e:
                logger.error("Unhandled error during Validation/ML stage", claim_id=processable_claim.claim_id, error=str(e), exc_info=True)
                processable_claim.processing_status = "validation_ml_error"
                await self._store_failed_claim(db_claim_to_update=db_claim_model_for_failed,
                                               processable_claim_instance=processable_claim,
                                               failed_stage="VALIDATION_ML_ERROR", reason=str(e))
        return processable_claim

    async def _perform_rvu_calculation(self, processable_claim: ProcessableClaim, db_claim_model_for_failed: ClaimModel) -> ProcessableClaim:
        async with self.rvu_semaphore:
            try:
                logger.debug("Performing RVU calculation", claim_id=processable_claim.claim_id)
                await self.rvu_service.calculate_rvu_for_claim(processable_claim, self.db)
                logger.info("RVU calculation successful", claim_id=processable_claim.claim_id)
                processable_claim.processing_status = "processing_complete"
            except Exception as e_rvu:
                error_reason_rvu = f"RVU calculation error: {str(e_rvu)}"
                logger.error(error_reason_rvu, claim_id=processable_claim.claim_id, exc_info=True)
                processable_claim.processing_status = "rvu_calculation_failed"
                await self._store_failed_claim(db_claim_to_update=db_claim_model_for_failed,
                                               processable_claim_instance=processable_claim,
                                               failed_stage="RVU_CALCULATION_FAILED", reason=error_reason_rvu)
        return processable_claim

    async def _process_single_claim_concurrently(self, db_claim: ClaimModel) -> Any: # Return type can be ProcessableClaim or Dict
        start_time = time.monotonic()
        processable_claim_instance: Optional[ProcessableClaim] = None
        # final_status will be tracked on processable_claim_instance.processing_status directly

        try:
            logger.debug("Orchestrating single claim processing", claim_id=db_claim.claim_id, db_id=db_claim.id)

            # 1. Pydantic Conversion
            try:
                processable_claim_instance = ProcessableClaim.model_validate(db_claim)
                if processable_claim_instance.processing_status == 'pending': # Default status from model_validate
                    processable_claim_instance.processing_status = "converted_to_pydantic"
            except Exception as e:
                error_reason = f"Pydantic conversion error: {str(e)}"
                logger.error(error_reason, claim_db_id=db_claim.id, claim_id=db_claim.claim_id, exc_info=True)
                # Do not call _store_failed_claim here directly as it needs a session.
                # The batch processor will handle storing this failure.
                # No _update_claim_and_lines_in_db call.
                return {
                    "error": "conversion_error",
                    "original_db_id": db_claim.id,
                    "claim_id": db_claim.claim_id,
                    "reason": error_reason,
                    "processing_duration_ms": (time.monotonic() - start_time) * 1000.0
                }

            # 2. Perform Validation and ML
            # These methods will update processable_claim_instance.processing_status
            # and call _store_failed_claim (which now only adds to session).
            processable_claim_instance = await self._perform_validation_and_ml(processable_claim_instance, db_claim)

            if processable_claim_instance.processing_status in ["validation_failed", "ml_rejected", "ml_error", "validation_ml_error"]:
                # No DB update call here. Status is set on processable_claim_instance.
                # Duration will be set in finally block.
                return processable_claim_instance

            # 3. Perform RVU Calculation (only if status is "ml_complete")
            if processable_claim_instance.processing_status == "ml_complete":
                 processable_claim_instance = await self._perform_rvu_calculation(processable_claim_instance, db_claim)
                 # Status updated within _perform_rvu_calculation

            # 4. Final state of processable_claim_instance is returned. No final DB update here.
            # Duration will be set in finally block.
            return processable_claim_instance

        except Exception as final_e:
            logger.error("Unhandled error in orchestrating single claim processing (_process_single_claim_concurrently)",
                         claim_id=db_claim.claim_id, error=str(final_e), exc_info=True)
            if processable_claim_instance:
                processable_claim_instance.processing_status = "unknown_orchestration_error"
                if not processable_claim_instance.ml_model_version_used: # Ensure this is set if possible
                    processable_claim_instance.ml_model_version_used = "unknown_due_to_error"
                await self._store_failed_claim(db_claim_to_update=db_claim,
                                               processable_claim_instance=processable_claim_instance,
                                               failed_stage="ORCHESTRATION_ERROR", reason=str(final_e))
                # No _update_claim_and_lines_in_db call.
                return processable_claim_instance
            else:
                # Very rare: error happened after Pydantic conversion but before processable_claim_instance was robustly set
                # or if the error is in the return handling itself.
                # The batch processor will need to create a FailedClaimModel for this.
                return {
                    "error": "unknown_orchestration_error",
                    "original_db_id": db_claim.id,
                    "claim_id": db_claim.claim_id,
                    "reason": str(final_e),
                    "processing_duration_ms": (time.monotonic() - start_time) * 1000.0
                }
        finally:
            # This duration is for the individual claim processing attempt.
            # It should be set on the ProcessableClaim if available.
            final_duration_ms = (time.monotonic() - start_time) * 1000.0
            if processable_claim_instance: # Check if it's a ProcessableClaim instance
                if isinstance(processable_claim_instance, ProcessableClaim):
                    processable_claim_instance.processing_duration_ms = final_duration_ms

            # Metrics collector should still record the duration of the attempt.
            self.metrics_collector.record_individual_claim_duration(final_duration_ms / 1000.0) # Expects seconds

    async def _store_failed_claim(
        self,
        db_claim_to_update: Optional[ClaimModel], # Original ClaimModel from DB
        processable_claim_instance: Optional[ProcessableClaim], # Pydantic model, potentially partially updated
        failed_stage: str,
        reason: str,
    ):
        logger.warn(f"Storing claim to failed_claims table. Stage: {failed_stage}, Reason: {reason}",
                    claim_id=db_claim_to_update.claim_id if db_claim_to_update else (processable_claim_instance.claim_id if processable_claim_instance else "N/A"))

        original_data_json = None
        # Prioritize data from processable_claim_instance if available, as it's the "working copy"
        if processable_claim_instance:
            # Ensure ml_model_version_used is in the dump if set
            dump_data = processable_claim_instance.model_dump(mode='json')
            if not dump_data.get("ml_model_version_used") and processable_claim_instance.ml_model_version_used:
                dump_data["ml_model_version_used"] = processable_claim_instance.ml_model_version_used
            original_data_json = dump_data
        elif db_claim_to_update: # Fallback to basic info from db_claim if pydantic model failed early
            original_data_json = {
                "claim_id": db_claim_to_update.claim_id,
                "facility_id": db_claim_to_update.facility_id,
                "patient_account_number": db_claim_to_update.patient_account_number,
                "total_charges": float(db_claim_to_update.total_charges) if db_claim_to_update.total_charges is not None else None,
                # Add other key fields that might be useful for context
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
            logger.debug("Added FailedClaimModel instance to session.",
                         failed_claim_id=failed_claim_entry.claim_id,
                         original_db_id=failed_claim_entry.original_claim_db_id)
        except Exception as e:
            logger.error("Failed to create FailedClaimModel instance or add to session.",
                         error=str(e), exc_info=True)

    async def process_pending_claims_batch(self, batch_size_override: Optional[int] = None):
        effective_batch_size = batch_size_override if batch_size_override is not None else self.settings.FETCH_BATCH_SIZE
        batch_start_time = time.monotonic()
        logger.info("Starting batch processing of claims with concurrency", batch_size=effective_batch_size)

        fetched_db_claims = await self._fetch_pending_claims(effective_batch_size)
        attempted_claims = len(fetched_db_claims)

        if not fetched_db_claims:
            logger.info("No pending claims to process.")
            # Existing summary and metrics logic for no claims is fine.
            summary = {
                "message": "No pending claims to process.", "attempted_claims": 0,
                "conversion_errors":0, "validation_failures": 0, "ml_errors_raw":0,
                "ml_approved_raw": 0, "ml_rejected_raw": 0,
                "rvu_calculation_failures":0, "successfully_processed_count": 0,
                "unhandled_orchestration_error":0, "db_commit_errors":0
            }
            self.metrics_collector.record_batch_processed(
                batch_size=0, duration_seconds=(time.monotonic() - batch_start_time), claims_by_final_status={}
            )
            logger.info("Concurrent batch processing summary (no claims).", **summary)
            return summary

        logger.info(f"Fetched {attempted_claims} claims for concurrent processing.")
        original_claims_map = {c.id: c for c in fetched_db_claims}

        tasks = [self._process_single_claim_concurrently(db_claim) for db_claim in fetched_db_claims]
        processing_results = await asyncio.gather(*tasks, return_exceptions=True)

        successful_claim_updates = []
        successful_line_item_updates = []
        # failed_claim_updates are for claims that failed pydantic conversion, need minimal update
        failed_claim_updates_for_conversion_error = []

        claims_by_final_status_for_metric: Dict[str, int] = {}
        ml_approved_raw_count = 0
        ml_rejected_raw_count = 0
        ml_errors_raw_count = 0
        db_commit_errors_count = 0 # For final summary

        for i, result in enumerate(processing_results):
            original_db_claim_for_result = fetched_db_claims[i] # Assuming results are in order

            if isinstance(result, dict) and result.get("error") == "conversion_error":
                original_db_id = result["original_db_id"]
                original_claim_for_failure = original_claims_map.get(original_db_id)
                if original_claim_for_failure:
                    await self._store_failed_claim(
                        db_claim_to_update=original_claim_for_failure,
                        processable_claim_instance=None,
                        failed_stage="CONVERSION_ERROR",
                        reason=result["reason"]
                    )
                    failed_claim_updates_for_conversion_error.append({
                        'id': original_db_id,
                        'processing_status': 'conversion_error',
                        'processing_duration_ms': int(result["processing_duration_ms"]) if result["processing_duration_ms"] is not None else None,
                        'updated_at': datetime.now(timezone.utc)
                    })
                claims_by_final_status_for_metric["conversion_error"] = claims_by_final_status_for_metric.get("conversion_error", 0) + 1

            elif isinstance(result, ProcessableClaim):
                processed_claim = result
                status = processed_claim.processing_status
                claims_by_final_status_for_metric[status] = claims_by_final_status_for_metric.get(status, 0) + 1

                if processed_claim.ml_derived_decision == "ML_APPROVED": ml_approved_raw_count +=1
                elif processed_claim.ml_derived_decision == "ML_REJECTED": ml_rejected_raw_count +=1
                elif processed_claim.ml_derived_decision and processed_claim.ml_derived_decision.startswith("ML_ERROR"): ml_errors_raw_count +=1

                claim_update_dict = {
                    'id': processed_claim.id, # This is original_db_claim.id
                    'processing_status': status,
                    'ml_score': processed_claim.ml_score,
                    'ml_derived_decision': processed_claim.ml_derived_decision,
                    'ml_model_version_used': processed_claim.ml_model_version_used,
                    'processing_duration_ms': int(processed_claim.processing_duration_ms) if processed_claim.processing_duration_ms is not None else None,
                    'processed_at': datetime.now(timezone.utc) if status == 'processing_complete' else None,
                    'updated_at': datetime.now(timezone.utc)
                }
                successful_claim_updates.append(claim_update_dict)

                if status == 'processing_complete' and processed_claim.line_items:
                    for line_item in processed_claim.line_items:
                        if line_item.rvu_total is not None and line_item.id is not None: # Ensure line_item has an ID
                            line_item_update_dict = {
                                'id': line_item.id,
                                'rvu_total': line_item.rvu_total,
                                'updated_at': datetime.now(timezone.utc)
                            }
                            successful_line_item_updates.append(line_item_update_dict)

            elif isinstance(result, Exception):
                logger.error("Unhandled exception from _process_single_claim_concurrently",
                             claim_id=original_db_claim_for_result.claim_id, error=str(result), exc_info=result)
                claims_by_final_status_for_metric["unhandled_orchestration_error"] = claims_by_final_status_for_metric.get("unhandled_orchestration_error", 0) + 1
                # Optionally, create a FailedClaimModel for these as well, if the original_db_claim_for_result can be reliably determined
                await self._store_failed_claim(
                    db_claim_to_update=original_db_claim_for_result,
                    processable_claim_instance=None, # No processable instance
                    failed_stage="UNHANDLED_ORCHESTRATION_ERROR",
                    reason=str(result)
                )
                # Update the original claim's status to reflect this error
                failed_claim_updates_for_conversion_error.append({ # Reusing this list for simplicity for now
                    'id': original_db_claim_for_result.id,
                    'processing_status': 'unhandled_orchestration_error',
                    'updated_at': datetime.now(timezone.utc)
                })


        # Perform Batch Database Operations
        max_retries = 3
        base_delay = 0.1
        commit_successful = False
        for attempt in range(max_retries):
            try:
                async with self.db.begin(): # Starts a transaction
                    if failed_claim_updates_for_conversion_error: # Claims that failed Pydantic conversion
                        await self.db.bulk_update_mappings(ClaimModel, failed_claim_updates_for_conversion_error)
                        logger.info(f"Bulk updated {len(failed_claim_updates_for_conversion_error)} claims with conversion/orchestration errors.")

                    if successful_claim_updates: # Claims that went through processing (successfully or failed later)
                        await self.db.bulk_update_mappings(ClaimModel, successful_claim_updates)
                        logger.info(f"Bulk updated {len(successful_claim_updates)} successfully processed claims.")

                    if successful_line_item_updates:
                        await self.db.bulk_update_mappings(ClaimLineItemModel, successful_line_item_updates)
                        logger.info(f"Bulk updated {len(successful_line_item_updates)} line items.")

                    # FailedClaimModel instances (added via _store_failed_claim) will be committed here.
                    logger.info("Committing all staged changes (ClaimModel, ClaimLineItemModel, FailedClaimModel).")
                commit_successful = True
                break # Exit retry loop on success
            except OperationalError as oe:
                logger.warn(f"Attempt {attempt + 1} of {max_retries}: OperationalError during DB commit. Retrying...", error=str(oe))
                if attempt + 1 == max_retries:
                    logger.error(f"All {max_retries} retries failed for OperationalError during batch update.", exc_info=True)
                    db_commit_errors_count = attempted_claims # Or be more specific if possible
                    # Update status for all claims in this batch to a db_error status if commit fails ultimately
                    for status_key in list(claims_by_final_status_for_metric.keys()): # Iterate over copy of keys
                        count = claims_by_final_status_for_metric.pop(status_key)
                        claims_by_final_status_for_metric["db_commit_error"] = \
                            claims_by_final_status_for_metric.get("db_commit_error",0) + count
                    break # Break from retry, error handled by summary
            except Exception as e:
                logger.error(f"Non-retryable error during batch DB update: {str(e)}", exc_info=True)
                db_commit_errors_count = attempted_claims # Or be more specific
                for status_key in list(claims_by_final_status_for_metric.keys()): # Iterate over copy of keys
                    count = claims_by_final_status_for_metric.pop(status_key)
                    claims_by_final_status_for_metric["db_commit_error"] = \
                        claims_by_final_status_for_metric.get("db_commit_error",0) + count
                break # Break from retry, error handled by summary

        if not commit_successful:
             logger.error("Batch processing failed to commit to database after retries or due to non-retryable error.")
             # Summary will reflect failures based on claims_by_final_status_for_metric which was updated in except blocks.


        batch_duration_seconds = time.monotonic() - batch_start_time
        self.metrics_collector.record_batch_processed(
            batch_size=attempted_claims,
            duration_seconds=batch_duration_seconds,
            claims_by_final_status=claims_by_final_status_for_metric
        )

        # Construct final summary using claims_by_final_status_for_metric
        # This summary will now more accurately reflect outcomes including DB errors.
        final_summary = {
            "message": "Batch processing finished.",
            "attempted_claims": attempted_claims,
            "by_status": claims_by_final_status_for_metric,
            "ml_approved_raw": ml_approved_raw_count,
            "ml_rejected_raw": ml_rejected_raw_count,
            "ml_errors_raw": ml_errors_raw_count,
            "db_commit_errors": db_commit_errors_count, # From the loop
            "batch_duration_seconds": round(batch_duration_seconds, 2),
            "throughput_claims_per_second": CLAIMS_THROUGHPUT_GAUGE._value.get() if CLAIMS_THROUGHPUT_GAUGE._value else 0.0
        }
        logger.info("Concurrent batch processing summary.", **final_summary)
        return final_summary

    async def _fetch_pending_claims(self, effective_batch_size: int) -> list[ClaimModel]:
        logger.info("Fetching pending claims from database", batch_size=effective_batch_size)
        stmt = (
            select(ClaimModel)
            .where(ClaimModel.processing_status == 'pending')
            .order_by(ClaimModel.priority.desc(), ClaimModel.created_at.asc())
            .limit(effective_batch_size)
            .options(joinedload(ClaimModel.line_items))
        )
        result = await self.db.execute(stmt)
        claims = result.scalars().all()
        if claims:
            logger.info(f"Fetched {len(claims)} pending claims from the database.")
        else:
            logger.info("No pending claims found in the database.")
        return list(claims)

```
