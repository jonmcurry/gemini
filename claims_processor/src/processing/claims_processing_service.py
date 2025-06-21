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
from ..core.database.models.claims_db import ClaimModel
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

        self.feature_extractor = FeatureExtractor()

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

    async def _process_single_claim_concurrently(self, db_claim: ClaimModel) -> Dict[str, Any]:
        start_time = time.monotonic()
        processable_claim_instance: Optional[ProcessableClaim] = None
        final_status: str = "unknown_initial_error"

        try:
            logger.debug("Orchestrating single claim processing", claim_id=db_claim.claim_id, db_id=db_claim.id)
            # 1. Pydantic Conversion
            try:
                processable_claim_instance = ProcessableClaim.model_validate(db_claim)
                if processable_claim_instance.processing_status == 'pending':
                    processable_claim_instance.processing_status = "converted_to_pydantic"
                final_status = processable_claim_instance.processing_status
            except Exception as e:
                final_status = "conversion_error"
                error_reason = f"Pydantic conversion error: {str(e)}"
                logger.error(error_reason, claim_db_id=db_claim.id, exc_info=True)
                await self._store_failed_claim(db_claim_to_update=db_claim, processable_claim_instance=None,
                                               failed_stage="CONVERSION_ERROR", reason=error_reason)
                current_duration_ms = (time.monotonic() - start_time) * 1000.0
                await self._update_claim_and_lines_in_db(db_claim, None, final_status,
                                                         duration_ms=current_duration_ms, validation_errors=[error_reason])
                return {"db_id": db_claim.id, "claim_id": db_claim.claim_id, "status": final_status, "ml_decision": "N/A", "errors": [error_reason], "error_detail_str": str(e)}

            # 2. Perform Validation and ML
            processable_claim_instance = await self._perform_validation_and_ml(processable_claim_instance, db_claim)
            final_status = processable_claim_instance.processing_status

            if final_status in ["validation_failed", "ml_rejected", "ml_error", "validation_ml_error"]:
                current_duration_ms = (time.monotonic() - start_time) * 1000.0
                processable_claim_instance.processing_duration_ms = current_duration_ms
                await self._update_claim_and_lines_in_db(db_claim, processable_claim_instance, final_status, duration_ms=current_duration_ms)
                # Extract validation errors if any from the ProcessableClaim if the validator adds them there.
                # For now, assuming error details are in FailedClaimModel.
                # errors_list = getattr(processable_claim_instance, 'validation_errors_list', None)
                return {"db_id": db_claim.id, "claim_id": processable_claim_instance.claim_id, "status": final_status,
                        "ml_decision": processable_claim_instance.ml_derived_decision,
                        "ml_model_version_used": processable_claim_instance.ml_model_version_used,
                        "errors": None} # Errors logged to FailedClaimModel

            # 3. Perform RVU Calculation (only if status is "ml_complete")
            if final_status == "ml_complete":
                 processable_claim_instance = await self._perform_rvu_calculation(processable_claim_instance, db_claim)
                 final_status = processable_claim_instance.processing_status

            # 4. Final DB Update
            current_duration_ms = (time.monotonic() - start_time) * 1000.0
            processable_claim_instance.processing_duration_ms = current_duration_ms
            await self._update_claim_and_lines_in_db(db_claim, processable_claim_instance, final_status, duration_ms=current_duration_ms)

            return {"db_id": db_claim.id, "claim_id": processable_claim_instance.claim_id, "status": final_status,
                    "ml_decision": processable_claim_instance.ml_derived_decision,
                    "ml_model_version_used": processable_claim_instance.ml_model_version_used,
                    "errors": None if final_status == "processing_complete" else ["Processing ended with status: " + final_status]}

        except Exception as final_e:
            final_status = "unknown_orchestration_error"
            logger.error("Unhandled error in orchestrating single claim processing (_process_single_claim_concurrently)",
                         claim_id=db_claim.claim_id, error=str(final_e), exc_info=True)
            if processable_claim_instance:
                processable_claim_instance.processing_status = final_status
                # Ensure ml_model_version_used is set even in error if it was determined
                if not processable_claim_instance.ml_model_version_used:
                    processable_claim_instance.ml_model_version_used = "unknown_due_to_error"
            else: # If processable_claim_instance is None due to very early error
                # We can't set ml_model_version_used on it. _store_failed_claim will handle None.
                pass

            await self._store_failed_claim(db_claim_to_update=db_claim,
                                           processable_claim_instance=processable_claim_instance,
                                           failed_stage="ORCHESTRATION_ERROR", reason=str(final_e))
            current_duration_ms = (time.monotonic() - start_time) * 1000.0
            if processable_claim_instance: processable_claim_instance.processing_duration_ms = current_duration_ms
            await self._update_claim_and_lines_in_db(db_claim, processable_claim_instance, final_status,
                                                     duration_ms=current_duration_ms)

            return {"db_id": db_claim.id, "claim_id": db_claim.claim_id, "status": final_status,
                    "ml_decision": processable_claim_instance.ml_derived_decision if processable_claim_instance else "N/A",
                    "ml_model_version_used": processable_claim_instance.ml_model_version_used if processable_claim_instance else "unknown_due_to_error",
                    "error_detail_str": str(final_e)}
        finally:
            final_duration_sec = time.monotonic() - start_time
            self.metrics_collector.record_individual_claim_duration(final_duration_sec)
            if processable_claim_instance and processable_claim_instance.processing_duration_ms is None:
                 processable_claim_instance.processing_duration_ms = final_duration_sec * 1000.0

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

        if not fetched_db_claims:
            logger.info("No pending claims to process.")
            summary = {
                "message": "No pending claims to process.", "attempted_claims": 0,
                "conversion_errors":0, "validation_failures": 0,
                "ml_approved_raw": 0, "ml_rejected_raw": 0, "ml_errors_raw": 0,
                "stopped_by_ml_rejection": 0,
                "rvu_calculation_failures":0, "successfully_processed_count": 0,
                "other_exceptions":0
            }
            self.metrics_collector.record_batch_processed(
                batch_size=0,
                duration_seconds=(time.monotonic() - batch_start_time),
                claims_by_final_status={}
            )
            logger.info("Concurrent batch processing summary (no claims).", **summary)
            return summary

        logger.info(f"Fetched {len(fetched_db_claims)} claims for concurrent processing.")

        tasks = [self._process_single_claim_concurrently(db_claim) for db_claim in fetched_db_claims]
        processing_results = await asyncio.gather(*tasks, return_exceptions=True)

        attempted_claims = len(fetched_db_claims)
        claims_by_final_status_for_metric: Dict[str, int] = {}

        ml_approved_raw_count = 0
        ml_rejected_raw_count = 0
        ml_errors_raw_count = 0

        for result in processing_results:
            if isinstance(result, Exception):
                status = "unhandled_orchestration_error"
                claims_by_final_status_for_metric[status] = claims_by_final_status_for_metric.get(status, 0) + 1
            elif isinstance(result, dict):
                status = result.get("status", "unknown_processing_error")
                claims_by_final_status_for_metric[status] = claims_by_final_status_for_metric.get(status, 0) + 1

                ml_decision_from_result = result.get("ml_decision", "N/A")
                if ml_decision_from_result == "ML_APPROVED": ml_approved_raw_count += 1
                elif ml_decision_from_result == "ML_REJECTED": ml_rejected_raw_count += 1
                elif ml_decision_from_result.startswith("ML_ERROR"): ml_errors_raw_count +=1

        batch_duration_seconds = time.monotonic() - batch_start_time

        self.metrics_collector.record_batch_processed(
            batch_size=attempted_claims,
            duration_seconds=batch_duration_seconds,
            claims_by_final_status=claims_by_final_status_for_metric
        )

        final_summary = {
            "message": "Concurrent batch processing finished.",
            "attempted_claims": attempted_claims,
            "by_status": claims_by_final_status_for_metric,
            "ml_approved_raw": ml_approved_raw_count,
            "ml_rejected_raw": ml_rejected_raw_count,
            "ml_errors_raw": ml_errors_raw_count,
            "batch_duration_seconds": round(batch_duration_seconds, 2),
            "throughput_claims_per_second": CLAIMS_THROUGHPUT_GAUGE._value.get()
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

    async def _update_claim_and_lines_in_db(self,
                                         db_claim_to_update: ClaimModel,
                                         processed_pydantic_claim: Optional[ProcessableClaim],
                                         new_status: str,
                                         duration_ms: Optional[float] = None,
                                         validation_errors: List[str] = None,
                                         ml_log_info: Optional[Dict] = None):
        if db_claim_to_update is None:
            logger.error("Cannot update claim in DB: SQLAlchemy model instance is None.")
            return

        logger.info("Updating claim and lines in DB", claim_db_id=db_claim_to_update.id, new_status=new_status, duration_ms=duration_ms)

        db_claim_to_update.processing_status = new_status
        if duration_ms is not None:
            db_claim_to_update.processing_duration_ms = int(duration_ms)
        if processed_pydantic_claim:
            if processed_pydantic_claim.ml_score is not None:
                db_claim_to_update.ml_score = Decimal(str(processed_pydantic_claim.ml_score))
            if processed_pydantic_claim.ml_derived_decision is not None:
                db_claim_to_update.ml_derived_decision = processed_pydantic_claim.ml_derived_decision
            if processed_pydantic_claim.ml_model_version_used is not None: # Persist model version used
                db_claim_to_update.ml_model_version_used = processed_pydantic_claim.ml_model_version_used

            if new_status == "processing_complete" and processed_pydantic_claim.line_items:
                map_pydantic_line_items = {line.id: line for line in processed_pydantic_claim.line_items}
                for db_line_item in db_claim_to_update.line_items:
                    if db_line_item.id in map_pydantic_line_items:
                        pydantic_line = map_pydantic_line_items[db_line_item.id]
                        if pydantic_line.rvu_total is not None:
                            db_line_item.rvu_total = pydantic_line.rvu_total
            db_claim_to_update.processed_at = datetime.now(timezone.utc)

        max_retries = 3
        base_delay = 0.1

        for attempt in range(max_retries):
            try:
                self.db.add(db_claim_to_update)
                await self.db.commit()

                logger.info(f"Attempt {attempt + 1}: Claim and lines updated successfully in DB",
                            claim_db_id=db_claim_to_update.id, new_status=db_claim_to_update.processing_status)

                await self.db.refresh(db_claim_to_update)
                if db_claim_to_update.line_items:
                    for line_item_to_refresh in db_claim_to_update.line_items:
                        await self.db.refresh(line_item_to_refresh)
                return
            except OperationalError as oe:
                logger.warn(f"Attempt {attempt + 1} of {max_retries}: OperationalError during DB commit. Retrying...",
                            claim_db_id=db_claim_to_update.id, error=str(oe))
                await self.db.rollback()
                if attempt + 1 == max_retries:
                    logger.error(f"All {max_retries} retries failed for OperationalError.", claim_db_id=db_claim_to_update.id)
                    raise

                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} (or non-retryable error): Failed to update claim and lines in DB.",
                             claim_db_id=db_claim_to_update.id if db_claim_to_update else "Unknown ID",
                             error=str(e), exc_info=True)
                await self.db.rollback()
                raise

        logger.error("Fell through retry loop in _update_claim_and_lines_in_db without success or re-raising error.",
                     claim_db_id=db_claim_to_update.id if db_claim_to_update else "Unknown ID")
        raise Exception(f"Failed to update claim {db_claim_to_update.id if db_claim_to_update else 'Unknown ID'} after all retries, but no specific exception was propagated.")

```
