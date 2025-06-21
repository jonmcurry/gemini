import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.orm import joinedload
import structlog
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, date
from decimal import Decimal
import time
import hashlib
from sqlalchemy.exc import OperationalError

from ..core.config.settings import get_settings
from ..core.database.models.claims_db import ClaimModel, ClaimLineItemModel
from ..api.models.claim_models import ProcessableClaim, ProcessableClaimLineItem
from .validation.claim_validator import ClaimValidator
from .rvu_service import RVUService
from ..core.cache.cache_manager import get_cache_manager
from .ml_pipeline.feature_extractor import FeatureExtractor
from .ml_pipeline.optimized_predictor import OptimizedPredictor
from ..core.monitoring.app_metrics import MetricsCollector
from ..core.database.models.failed_claims_db import FailedClaimModel
from ..core.security.encryption_service import EncryptionService


logger = structlog.get_logger(__name__)

class ClaimProcessingService:
    def __init__(self, db_session: AsyncSession, metrics_collector: MetricsCollector, encryption_service: EncryptionService):
        self.db = db_session
        self.metrics_collector = metrics_collector
        self.encryption_service = encryption_service
        self.settings = get_settings()
        self.validator = ClaimValidator()
        cache_manager = get_cache_manager()
        self.rvu_service = RVUService(cache_manager=cache_manager, metrics_collector=self.metrics_collector)

        self.validation_ml_semaphore = asyncio.Semaphore(self.settings.VALIDATION_CONCURRENCY)
        self.rvu_semaphore = asyncio.Semaphore(self.settings.RVU_CALCULATION_CONCURRENCY)

        self.feature_extractor = FeatureExtractor(metrics_collector=self.metrics_collector)

        self.primary_predictor = OptimizedPredictor(
            model_path=self.settings.ML_MODEL_PATH,
            metrics_collector=self.metrics_collector,
            feature_count=self.settings.ML_FEATURE_COUNT
        )

        self.challenger_predictor: Optional[OptimizedPredictor] = None
        if self.settings.ML_CHALLENGER_MODEL_PATH and \
           self.settings.ML_AB_TEST_TRAFFIC_PERCENTAGE_TO_CHALLENGER > 0:
            logger.info("Challenger model path configured for A/B testing.",
                        path=self.settings.ML_CHALLENGER_MODEL_PATH,
                        traffic_percentage=self.settings.ML_AB_TEST_TRAFFIC_PERCENTAGE_TO_CHALLENGER)
            try:
                self.challenger_predictor = OptimizedPredictor(
                    model_path=self.settings.ML_CHALLENGER_MODEL_PATH,
                    metrics_collector=self.metrics_collector,
                    feature_count=self.settings.ML_FEATURE_COUNT
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
                    cache_manager_id=id(cache_manager),
                    metrics_collector_id=id(self.metrics_collector),
                    encryption_service_id=id(self.encryption_service),
                    validation_concurrency=self.settings.VALIDATION_CONCURRENCY,
                    rvu_concurrency=self.settings.RVU_CALCULATION_CONCURRENCY,
                    primary_ml_model_path=self.settings.ML_MODEL_PATH,
                    challenger_ml_model_path=self.settings.ML_CHALLENGER_MODEL_PATH if self.challenger_predictor else "N/A")

    async def _perform_validation_stage(self, processable_claim: ProcessableClaim, db_claim_model_for_failed: ClaimModel) -> ProcessableClaim:
        async with self.validation_ml_semaphore:
            try:
                logger.debug("Performing validation for claim", claim_id=processable_claim.claim_id, current_status=processable_claim.processing_status)
                validation_errors = self.validator.validate_claim(processable_claim)
                if validation_errors:
                    error_reason_val = f"Validation errors: {'; '.join(validation_errors)}"
                    logger.warn("Claim validation failed", claim_id=processable_claim.claim_id, errors=validation_errors)
                    processable_claim.processing_status = "validation_failed"
                    await self._store_failed_claim(
                        db_claim_to_update=db_claim_model_for_failed,
                        processable_claim_instance=processable_claim,
                        failed_stage="VALIDATION",
                        reason=error_reason_val
                    )
                    return processable_claim
                else:
                    logger.info("Claim validation successful", claim_id=processable_claim.claim_id)
                    processable_claim.processing_status = "validation_complete"
            except Exception as e:
                logger.error("Unhandled error during Validation stage", claim_id=processable_claim.claim_id, error=str(e), exc_info=True)
                processable_claim.processing_status = "validation_error_internal"
                await self._store_failed_claim(
                    db_claim_to_update=db_claim_model_for_failed,
                    processable_claim_instance=processable_claim,
                    failed_stage="VALIDATION_INTERNAL_ERROR",
                    reason=str(e)
                )
        return processable_claim

    async def _perform_ml_stage(self, processable_claim: ProcessableClaim, db_claim_model_for_failed: ClaimModel) -> ProcessableClaim: # Method renamed
        async with self.validation_ml_semaphore: # Still uses the same semaphore
            try:
                logger.debug("Performing ML prediction stage", claim_id=processable_claim.claim_id, current_status=processable_claim.processing_status)

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
                logger.error("Unhandled error during ML stage", claim_id=processable_claim.claim_id, error=str(e), exc_info=True)
                processable_claim.processing_status = "ml_error_internal"
                await self._store_failed_claim(
                    db_claim_to_update=db_claim_model_for_failed,
                    processable_claim_instance=processable_claim,
                    failed_stage="ML_INTERNAL_ERROR",
                    reason=str(e)
                )
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

    async def _process_single_claim_concurrently(self, db_claim: ClaimModel) -> Any:
        start_time = time.monotonic()
        processable_claim_instance: Optional[ProcessableClaim] = None

        try:
            logger.debug("Orchestrating single claim processing", claim_id=db_claim.claim_id, db_id=db_claim.id)

            # 1. PII Decryption and Pydantic Conversion
            try:
                claim_data_for_pydantic = {c.name: getattr(db_claim, c.name) for c in db_claim.__table__.columns}
                if db_claim.patient_date_of_birth:
                    dec_dob_str = self.encryption_service.decrypt(db_claim.patient_date_of_birth)
                    claim_data_for_pydantic['patient_date_of_birth'] = date.fromisoformat(dec_dob_str) if dec_dob_str else None
                else:
                    claim_data_for_pydantic['patient_date_of_birth'] = None
                if db_claim.medical_record_number:
                    claim_data_for_pydantic['medical_record_number'] = self.encryption_service.decrypt(db_claim.medical_record_number)
                if getattr(db_claim, 'subscriber_id', None):
                    claim_data_for_pydantic['subscriber_id'] = self.encryption_service.decrypt(db_claim.subscriber_id)
                else:
                    claim_data_for_pydantic['subscriber_id'] = None

                line_items_pydantic = []
                if db_claim.line_items:
                    for li_db in db_claim.line_items:
                        line_items_pydantic.append(ProcessableClaimLineItem.model_validate(li_db))
                claim_data_for_pydantic['line_items'] = line_items_pydantic

                processable_claim_instance = ProcessableClaim(**claim_data_for_pydantic)
                if processable_claim_instance.processing_status == 'pending':
                    processable_claim_instance.processing_status = "converted_to_pydantic"
            except Exception as e:
                error_reason = f"Data conversion/decryption error: {str(e)}"
                logger.error(error_reason, claim_db_id=db_claim.id, claim_id=db_claim.claim_id, exc_info=True)
                return {
                    "error": "conversion_error",
                    "original_db_id": db_claim.id,
                    "claim_id": db_claim.claim_id,
                    "reason": error_reason,
                    "processing_duration_ms": (time.monotonic() - start_time) * 1000.0
                }

            # --- Orchestration of Stages ---
            # 2. Perform Validation Stage
            val_stage_start_time = time.perf_counter()
            processable_claim_instance = await self._perform_validation_stage(processable_claim_instance, db_claim)
            val_stage_duration = time.perf_counter() - val_stage_start_time
            self.metrics_collector.record_validation_stage_duration(val_stage_duration)

            if processable_claim_instance.processing_status != "validation_complete":
                logger.info("Claim processing halted after validation stage.",
                            claim_id=processable_claim_instance.claim_id, status=processable_claim_instance.processing_status)
                return processable_claim_instance

            # 3. Perform ML Stage (if validation succeeded)
            ml_stage_start_time = time.perf_counter()
            processable_claim_instance = await self._perform_ml_stage(processable_claim_instance, db_claim)
            ml_stage_duration = time.perf_counter() - ml_stage_start_time
            self.metrics_collector.record_ml_stage_duration(ml_stage_duration)

            if processable_claim_instance.processing_status not in ["ml_complete"]:
                logger.info("Claim processing halted after ML stage.",
                            claim_id=processable_claim_instance.claim_id, status=processable_claim_instance.processing_status)
                return processable_claim_instance

            # 4. Perform RVU Calculation (if ML is complete)
            if processable_claim_instance.processing_status == "ml_complete":
                 rvu_stage_start_time = time.perf_counter()
                 processable_claim_instance = await self._perform_rvu_calculation(processable_claim_instance, db_claim)
                 rvu_stage_duration = time.perf_counter() - rvu_stage_start_time
                 self.metrics_collector.record_rvu_stage_duration(rvu_stage_duration)

            # 5. Final state of processable_claim_instance is returned.
            return processable_claim_instance

        except Exception as final_e:
            logger.error("Unhandled error in orchestrating single claim processing (_process_single_claim_concurrently)",
                         claim_id=db_claim.claim_id, error=str(final_e), exc_info=True)
            if processable_claim_instance:
                processable_claim_instance.processing_status = "unknown_orchestration_error"
                if not processable_claim_instance.ml_model_version_used:
                    processable_claim_instance.ml_model_version_used = "unknown_due_to_error"
                await self._store_failed_claim(db_claim_to_update=db_claim,
                                               processable_claim_instance=processable_claim_instance,
                                               failed_stage="ORCHESTRATION_ERROR", reason=str(final_e))
                return processable_claim_instance
            else:
                return {
                    "error": "unknown_orchestration_error",
                    "original_db_id": db_claim.id,
                    "claim_id": db_claim.claim_id,
                    "reason": str(final_e),
                    "processing_duration_ms": (time.monotonic() - start_time) * 1000.0
                }
        finally:
            final_duration_ms = (time.monotonic() - start_time) * 1000.0
            if processable_claim_instance:
                if isinstance(processable_claim_instance, ProcessableClaim):
                    processable_claim_instance.processing_duration_ms = final_duration_ms
            self.metrics_collector.record_individual_claim_duration(final_duration_ms / 1000.0)

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
            dump_data = processable_claim_instance.model_dump(mode='json')
            if not dump_data.get("ml_model_version_used") and processable_claim_instance.ml_model_version_used:
                dump_data["ml_model_version_used"] = processable_claim_instance.ml_model_version_used
            original_data_json = dump_data
        elif db_claim_to_update:
            original_data_json = {
                "claim_id": db_claim_to_update.claim_id,
                "facility_id": db_claim_to_update.facility_id,
                "patient_account_number": db_claim_to_update.patient_account_number,
                "total_charges": float(db_claim_to_update.total_charges) if db_claim_to_update.total_charges is not None else None,
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

        # Outer retry loop for fetching claims
        max_fetch_retries = self.settings.MAX_FETCH_RETRIES if hasattr(self.settings, 'MAX_FETCH_RETRIES') else 3
        fetch_retry_delay = self.settings.FETCH_RETRY_DELAY_SECONDS if hasattr(self.settings, 'FETCH_RETRY_DELAY_SECONDS') else 5.0 # Use FETCH_RETRY_DELAY_SECONDS

        fetched_db_claims = []
        current_batch_id_for_update = "" # Initialize
        for attempt in range(max_fetch_retries):
            try:
                if attempt == 0:
                     current_batch_id_for_update = str(asyncio.current_task().get_name()) + "_" + str(time.time_ns())

                fetched_db_claims = await self._fetch_pending_claims(effective_batch_size, current_batch_id_for_update)
                break
            except OperationalError as oe:
                logger.warn(f"Fetch attempt {attempt + 1} failed with OperationalError: {oe}. Retrying after {fetch_retry_delay}s...")
                if attempt + 1 == max_fetch_retries:
                    logger.error("All fetch attempts failed due to OperationalError. Aborting batch.", exc_info=True)
                    self.metrics_collector.record_batch_processed(
                        batch_size=0,
                        duration_seconds=(time.monotonic() - batch_start_time),
                        claims_by_final_status={"fetch_error": effective_batch_size}
                    )
                    return {
                        "message": "Batch processing aborted: Failed to fetch claims after multiple retries.",
                        "attempted_claims": 0, "successfully_processed_count": 0, "db_commit_errors": 0,
                        "by_status": {"fetch_error": effective_batch_size},
                         "error_detail": str(oe)
                    }
                await asyncio.sleep(fetch_retry_delay)

        attempted_claims = len(fetched_db_claims)
        if not fetched_db_claims:
            logger.info("No pending claims to process after fetch attempts.")
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
            logger.info("Concurrent batch processing summary (no claims fetched).", **summary)
            return summary

        logger.info(f"Fetched {attempted_claims} claims for concurrent processing with batch_id {current_batch_id_for_update}.")
        original_claims_map = {c.id: c for c in fetched_db_claims}
        tasks = [self._process_single_claim_concurrently(db_claim) for db_claim in fetched_db_claims]
        processing_results = await asyncio.gather(*tasks, return_exceptions=True)

        successful_claim_updates = []
        successful_line_item_updates = []
        failed_claim_updates_for_conversion_error = []
        claims_by_final_status_for_metric: Dict[str, int] = {}
        ml_approved_raw_count = 0
        ml_rejected_raw_count = 0
        ml_errors_raw_count = 0
        db_commit_errors_count = 0

        for i, result in enumerate(processing_results):
            original_db_claim_for_result = fetched_db_claims[i]
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
                    'id': processed_claim.id,
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
                        if line_item.rvu_total is not None and line_item.id is not None:
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
                await self._store_failed_claim(
                    db_claim_to_update=original_db_claim_for_result,
                    processable_claim_instance=None,
                    failed_stage="UNHANDLED_ORCHESTRATION_ERROR",
                    reason=str(result)
                )
                failed_claim_updates_for_conversion_error.append({
                    'id': original_db_claim_for_result.id,
                    'processing_status': 'unhandled_orchestration_error',
                    'updated_at': datetime.now(timezone.utc)
                })

        max_db_retries = 3 # Should be from settings ideally
        db_base_delay = 0.1
        commit_successful = False
        for attempt in range(max_db_retries):
            try:
                async with self.db.begin():
                    if failed_claim_updates_for_conversion_error:
                        await self.db.bulk_update_mappings(ClaimModel, failed_claim_updates_for_conversion_error)
                        logger.info(f"Bulk updated {len(failed_claim_updates_for_conversion_error)} claims with conversion/orchestration errors.")
                    if successful_claim_updates:
                        await self.db.bulk_update_mappings(ClaimModel, successful_claim_updates)
                        logger.info(f"Bulk updated {len(successful_claim_updates)} successfully processed claims.")
                    if successful_line_item_updates:
                        await self.db.bulk_update_mappings(ClaimLineItemModel, successful_line_item_updates)
                        logger.info(f"Bulk updated {len(successful_line_item_updates)} line items.")
                    logger.info("Committing all staged changes (ClaimModel, ClaimLineItemModel, FailedClaimModel).")
                commit_successful = True
                break
            except OperationalError as oe:
                logger.warn(f"Attempt {attempt + 1} of {max_db_retries}: OperationalError during DB commit. Retrying...", error=str(oe))
                if attempt + 1 == max_db_retries:
                    logger.error(f"All {max_db_retries} retries failed for OperationalError during batch update.", exc_info=True)
                    db_commit_errors_count = attempted_claims
                    for status_key in list(claims_by_final_status_for_metric.keys()):
                        count = claims_by_final_status_for_metric.pop(status_key)
                        claims_by_final_status_for_metric["db_commit_error"] = \
                            claims_by_final_status_for_metric.get("db_commit_error",0) + count
                    break
            except Exception as e:
                logger.error(f"Non-retryable error during batch DB update: {str(e)}", exc_info=True)
                db_commit_errors_count = attempted_claims
                for status_key in list(claims_by_final_status_for_metric.keys()):
                    count = claims_by_final_status_for_metric.pop(status_key)
                    claims_by_final_status_for_metric["db_commit_error"] = \
                        claims_by_final_status_for_metric.get("db_commit_error",0) + count
                break

        if not commit_successful:
             logger.error("Batch processing failed to commit to database after retries or due to non-retryable error.")

        batch_duration_seconds = time.monotonic() - batch_start_time
        self.metrics_collector.record_batch_processed(
            batch_size=attempted_claims,
            duration_seconds=batch_duration_seconds,
            claims_by_final_status=claims_by_final_status_for_metric
        )

        final_summary = {
            "message": "Batch processing finished.",
            "attempted_claims": attempted_claims,
            "by_status": claims_by_final_status_for_metric,
            "ml_approved_raw": ml_approved_raw_count,
            "ml_rejected_raw": ml_rejected_raw_count,
            "ml_errors_raw": ml_errors_raw_count,
            "db_commit_errors": db_commit_errors_count,
            "batch_duration_seconds": round(batch_duration_seconds, 2),
            "throughput_claims_per_second": CLAIMS_THROUGHPUT_GAUGE._value.get() if hasattr(self, 'CLAIMS_THROUGHPUT_GAUGE') and CLAIMS_THROUGHPUT_GAUGE._value else 0.0
        }
        logger.info("Concurrent batch processing summary.", **final_summary)
        return final_summary

    async def _fetch_pending_claims(self, effective_batch_size: int, current_batch_id_for_update: str) -> List[ClaimModel]:
        logger.info("Fetching pending claims with SELECT FOR UPDATE SKIP LOCKED",
                    limit=effective_batch_size, batch_id_to_set=current_batch_id_for_update)

        try:
            select_ids_stmt = (
                select(ClaimModel.id)
                .where(ClaimModel.processing_status == 'pending')
                .order_by(ClaimModel.priority.desc(), ClaimModel.created_at.asc())
                .limit(effective_batch_size)
                .with_for_update(skip_locked=True)
            )
            claim_db_ids_result = await self.db.execute(select_ids_stmt)
            claim_db_ids = [row[0] for row in claim_db_ids_result.fetchall()]

            if not claim_db_ids:
                logger.info("No pending claims found to fetch with SKIP LOCKED.")
                return []

            update_stmt = (
                update(ClaimModel)
                .where(ClaimModel.id.in_(claim_db_ids))
                .values(processing_status='processing', batch_id=current_batch_id_for_update)
                .execution_options(synchronize_session=False)
            )
            await self.db.execute(update_stmt)

            logger.info(f"Marked {len(claim_db_ids)} claims as 'processing' with batch_id {current_batch_id_for_update}.")

            stmt = (
                select(ClaimModel)
                .where(ClaimModel.id.in_(claim_db_ids))
                .options(joinedload(ClaimModel.line_items))
                .order_by(ClaimModel.priority.desc(), ClaimModel.created_at.asc())
            )
            result = await self.db.execute(stmt)
            fetched_db_claims = list(result.scalars().all())

            if fetched_db_claims:
                logger.info(f"Fetched {len(fetched_db_claims)} full claim objects for processing.")
            else:
                logger.warn("Fetched 0 full claim objects after marking IDs as processing.", count_ids_marked=len(claim_db_ids))

            return fetched_db_claims

        except OperationalError:
            logger.error("Database OperationalError during _fetch_pending_claims", exc_info=True)
            raise
        except Exception as e:
            logger.error("Unexpected error during _fetch_pending_claims", error=str(e), exc_info=True)
            raise
