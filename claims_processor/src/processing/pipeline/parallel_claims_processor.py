import structlog
from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.orm import joinedload, selectinload
import numpy as np
from sqlalchemy.sql import insert
from decimal import Decimal
from datetime import datetime, timezone
import time

from ....core.cache.cache_manager import CacheManager
from ....api.models.claim_models import ProcessableClaim
from ....core.database.models.claims_db import ClaimModel
from ....core.database.models.claims_production_db import ClaimsProductionModel
from ....core.database.models.failed_claims_db import FailedClaimModel
from ..validation.claim_validator import ClaimValidator
from ..rvu_service import RVUService
from ..ml_pipeline.feature_extractor import FeatureExtractor
from ..ml_pipeline.optimized_predictor import OptimizedPredictor
from ....core.monitoring.app_metrics import MetricsCollector # Import MetricsCollector

logger = structlog.get_logger(__name__)

class ParallelClaimsProcessor:
    def __init__(self,
                 db_session_factory: Any,
                 claim_validator: ClaimValidator,
                 rvu_service: RVUService,
                 feature_extractor: FeatureExtractor,
                 optimized_predictor: OptimizedPredictor,
                 metrics_collector: MetricsCollector):
        self.db_session_factory = db_session_factory
        self.validator = claim_validator
        self.rvu_service = rvu_service
        self.feature_extractor = feature_extractor
        self.predictor = optimized_predictor # Assumed to be already configured with its own MetricsCollector if needed internally
        self.metrics_collector = metrics_collector
        logger.info("ParallelClaimsProcessor initialized with all services including MetricsCollector.")

    async def _fetch_claims_parallel(self, session: AsyncSession, batch_id: Optional[str] = None, limit: int = 1000) -> List[ProcessableClaim]:
        logger.info("Fetching pending claims for processing", batch_id=batch_id, limit=limit)
        try:
            with self.metrics_collector.time_db_query('fetch_pending_claim_ids'):
                select_ids_stmt = (
                    select(ClaimModel.id)
                    .where(ClaimModel.processing_status == 'pending')
                    .order_by(ClaimModel.created_at.asc())
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
                claim_db_ids_result = await session.execute(select_ids_stmt)
                claim_db_ids = [row[0] for row in claim_db_ids_result.fetchall()]

            if not claim_db_ids:
                logger.info("No pending claims found to fetch.")
                return []

            update_values = {'processing_status': 'processing'}
            if batch_id: update_values['batch_id'] = batch_id

            with self.metrics_collector.time_db_query('update_fetched_claims_status'):
                update_stmt = (
                    update(ClaimModel)
                    .where(ClaimModel.id.in_(claim_db_ids))
                    .values(**update_values)
                    .execution_options(synchronize_session=False)
                )
                await session.execute(update_stmt)

            with self.metrics_collector.time_db_query('fetch_full_claim_data'):
                stmt = (
                    select(ClaimModel)
                    .where(ClaimModel.id.in_(claim_db_ids))
                    .options(selectinload(ClaimModel.line_items))
                    .order_by(ClaimModel.created_at.asc())
                )
                result = await session.execute(stmt)
                claim_db_models = result.scalars().unique().all()

            processable_claims: List[ProcessableClaim] = []
            for claim_db in claim_db_models:
                try:
                    p_claim = ProcessableClaim.model_validate(claim_db)
                    processable_claims.append(p_claim)
                except Exception as e:
                    logger.error(f"Error converting ClaimModel to ProcessableClaim for claim_id {claim_db.claim_id}",
                                 db_id=claim_db.id, error=str(e), exc_info=True)
            logger.info(f"Fetched and prepared {len(processable_claims)} claims for processing.", batch_id=batch_id)
            return processable_claims
        except Exception as e:
            logger.error("Database error during claim fetching and preparation", error=str(e), exc_info=True, batch_id=batch_id)
            raise

    async def _validate_claims_parallel(self, claims_data: List[ProcessableClaim]) -> Tuple[List[ProcessableClaim], List[ProcessableClaim]]:
        logger.info(f"Validating {len(claims_data)} claims.")
        valid_claims_list: List[ProcessableClaim] = []
        invalid_claims_list: List[ProcessableClaim] = []
        for claim_idx, claim in enumerate(claims_data):
            if (claim_idx + 1) % 100 == 0:
                logger.debug(f"Validation progress for batch '{claim.batch_id}': {claim_idx + 1}/{len(claims_data)} claims checked.")
            validation_errors = self.validator.validate_claim(claim)
            if not validation_errors:
                valid_claims_list.append(claim)
            else:
                setattr(claim, 'validation_errors_detail', validation_errors)
                logger.info(
                    "Claim failed validation", claim_id=claim.claim_id, db_claim_id=claim.id,
                    batch_id=claim.batch_id, errors=validation_errors
                )
                invalid_claims_list.append(claim)
        batch_id_for_log = claims_data[0].batch_id if claims_data and claims_data[0].batch_id else 'N/A'
        logger.info(f"Validation complete for batch '{batch_id_for_log}'. Valid: {len(valid_claims_list)}, Invalid: {len(invalid_claims_list)}")
        return valid_claims_list, invalid_claims_list

    async def _calculate_rvus_for_claims(self, session: AsyncSession, claims: List[ProcessableClaim]) -> None:
        if not claims: logger.info("No claims for RVU calculation."); return
        logger.info(f"Calculating RVUs for {len(claims)} claims.")
        processed_count = 0
        for claim_idx, claim in enumerate(claims):
            if (claim_idx + 1) % 100 == 0: logger.debug(f"RVU progress for batch '{claim.batch_id}': {claim_idx + 1}/{len(claims)}.")
            try:
                await self.rvu_service.calculate_rvu_for_claim(claim, session)
                processed_count += 1
            except Exception as e: logger.error("Error during RVU calculation", claim_id=claim.claim_id, error=str(e), exc_info=True)
        logger.info(f"RVU calculation attempted for {len(claims)}. Processed calls to service: {processed_count}.")

    async def _apply_ml_predictions(self, claims: List[ProcessableClaim]) -> None:
        if not claims: logger.info("No claims for ML prediction."); return
        logger.info(f"Applying ML predictions for {len(claims)} claims.")
        features_batch: List[np.ndarray] = []; claims_with_features: List[ProcessableClaim] = []
        for claim in claims: # Iterate through the input 'claims' list
            try:
                features = self.feature_extractor.extract_features(claim)
                if features is not None:
                    features_batch.append(features)
                    claims_with_features.append(claim) # Only add claim if features were extracted
                else:
                    claim.ml_derived_decision = "ML_SKIPPED_NO_FEATURES"; claim.ml_score = None
                    # Metrics for skips/errors during feature extraction are not recorded here,
                    # but could be if MetricsCollector was used directly.
                    # OptimizedPredictor will record metrics for predictions it's asked to make.
            except Exception as e:
                logger.error("Feature extraction error", claim_id=claim.claim_id, error=str(e), exc_info=True) # Corrected: 'e=e' to 'error=str(e)'
                claim.ml_derived_decision = "ML_SKIPPED_EXTRACTION_ERROR"; claim.ml_score = None

        if not features_batch: logger.info("No features extracted successfully for ML prediction batch."); return

        try:
            # OptimizedPredictor.predict_batch will internally record ML_PREDICTIONS_TOTAL and ML_INFERENCE_DURATION
            prediction_results = await self.predictor.predict_batch(features_batch) # Pass the list of feature arrays

            if len(prediction_results) == len(claims_with_features):
                for claim_obj, prediction_dict in zip(claims_with_features, prediction_results):
                    claim_obj.ml_score = prediction_dict.get('ml_score')
                    claim_obj.ml_derived_decision = prediction_dict.get('ml_derived_decision', "ML_ERROR")

                    if "error" in prediction_dict: # This implies predictor had an issue for this specific sample
                        logger.warn("ML prediction failed for claim as reported by predictor",
                                    claim_id=claim_obj.claim_id, error_detail=prediction_dict["error"])
                    else:
                        logger.debug("ML prediction results applied to claim", claim_id=claim_obj.claim_id,
                                     score=claim_obj.ml_score, decision=claim_obj.ml_derived_decision)
            else:
                logger.error(f"Mismatch between claims sent for prediction ({len(claims_with_features)}) and results ({len(prediction_results)}).")
                for c in claims_with_features:
                    c.ml_derived_decision = "ML_PREDICTION_BATCH_ERROR"; c.ml_score = None
        except Exception as e:
            logger.error("Error during call to OptimizedPredictor.predict_batch", error=str(e), exc_info=True)
            for c in claims_with_features:
                c.ml_derived_decision = "ML_PREDICTION_BATCH_ERROR"; c.ml_score = None
        logger.info(f"ML prediction application completed for {len(claims_with_features)} claims that had features.")


    async def _transfer_claims_to_production(self, session: AsyncSession, claims: List[ProcessableClaim], metrics: Optional[Dict]=None) -> int:
        if not claims: logger.info("No claims for transfer."); return 0
        logger.info(f"Preparing {len(claims)} claims for production transfer.")
        throughput = metrics.get('throughput_achieved') if metrics else None; insert_list = []
        for claim in claims:
            data = {
                "id": claim.id, "claim_id": claim.claim_id, "facility_id": claim.facility_id,
                "patient_account_number": claim.patient_account_number, "patient_first_name": claim.patient_first_name,
                "patient_last_name": claim.patient_last_name, "patient_date_of_birth": claim.patient_date_of_birth,
                "service_from_date": claim.service_from_date, "service_to_date": claim.service_to_date,
                "total_charges": claim.total_charges,
                "ml_prediction_score": Decimal(str(claim.ml_score)) if claim.ml_score is not None else None,
                "processing_duration_ms": int(claim.processing_duration_ms) if claim.processing_duration_ms is not None else None,
                "throughput_achieved": Decimal(str(throughput)) if throughput is not None else None,
            }
            risk_cat = "UNKNOWN" # Default risk category
            # Ensure score is Decimal for comparison, handling None or non-Decimal types gracefully
            score_for_risk_eval: Optional[Decimal] = None
            if claim.ml_score is not None:
                try: score_for_risk_eval = Decimal(str(claim.ml_score))
                except: logger.warn("Could not convert ml_score to Decimal for risk category", value=claim.ml_score)

            if score_for_risk_eval is not None:
                if score_for_risk_eval >= Decimal("0.8"): risk_cat = "LOW"
                elif score_for_risk_eval >= Decimal("0.5"): risk_cat = "MEDIUM"
                else: risk_cat = "HIGH"
            elif claim.ml_derived_decision: # Fallback to decision if score is None but decision exists
                if "ERROR" in claim.ml_derived_decision.upper(): risk_cat = "ERROR_IN_ML"
                elif "SKIPPED" in claim.ml_derived_decision.upper(): risk_cat = "ML_SKIPPED"
            data["risk_category"] = risk_cat
            insert_list.append(data)
        if not insert_list: logger.warn("No data mapped for prod transfer."); return 0
        try:
            with self.metrics_collector.time_db_query('transfer_to_production_insert'):
                await session.execute(insert(ClaimsProductionModel).values(insert_list))
            return len(insert_list)
        except Exception as e: logger.error("DB error during prod insert", error=str(e), exc_info=True); return 0 # Corrected: error=str(e)

    async def _update_staging_claims_status(self, session: AsyncSession, ids: List[int], status: str, transferred: bool = False) -> int:
        if not ids: logger.info("No IDs for staging update."); return 0
        logger.info(f"Updating {len(ids)} staging claims to status '{status}'. Transferred: {transferred}")
        values_to_set = {"processing_status": status, "processed_at": datetime.now(timezone.utc)}
        if transferred: values_to_set["transferred_to_prod_at"] = datetime.now(timezone.utc)
        try:
            query_name = f"update_staging_status_{status}{'_transferred' if transferred else ''}"
            with self.metrics_collector.time_db_query(query_name):
                stmt = update(ClaimModel).where(ClaimModel.id.in_(ids)).values(**values_to_set).execution_options(synchronize_session=False)
                res = await session.execute(stmt)
            updated_count = res.rowcount if res else 0
            if updated_count != len(ids): logger.warn(f"Staging update count mismatch for status '{status}'", expected=len(ids), actual=updated_count)
            return updated_count
        except Exception as e: logger.error(f"DB error updating staging to '{status}'", error=str(e), exc_info=True); return 0 # Corrected: error=str(e)

    async def _route_failed_claims(self, session: AsyncSession, failed_info: List[Tuple[ProcessableClaim, str, str]]) -> int:
        if not failed_info: logger.debug("No failed claims to route."); return 0
        logger.info(f"Routing {len(failed_info)} failed claims.")
        insert_data = []
        for claim, reason, stage in failed_info:
            try: original_data = claim.model_dump()
            except Exception as e: original_data = {"error": f"Serialization fail: {str(e)}", "claim_id": getattr(claim, 'claim_id', 'N/A')}
            insert_data.append({
                "original_claim_db_id": claim.id, "claim_id": claim.claim_id, "facility_id": claim.facility_id,
                "patient_account_number": claim.patient_account_number, "failed_at_stage": stage,
                "failure_reason": reason, "original_claim_data": original_data
            })
        if not insert_data: logger.warn("No data mapped for failed_claims table."); return 0
        try:
            with self.metrics_collector.time_db_query('route_failed_claims_insert'):
                await session.execute(insert(FailedClaimModel).values(insert_data))
            return len(insert_data)
        except Exception as e: logger.error("DB error inserting into failed_claims", error=str(e), exc_info=True); return 0 # Corrected: error=str(e)

    async def process_claims_parallel(self, batch_id: Optional[str] = None, limit: int = 1000) -> Dict[str, Any]:
        batch_start_time = time.perf_counter()
        logger.info("Starting parallel claims processing pipeline", batch_id=batch_id, limit=limit)
        summary = {
            "batch_id": batch_id, "attempted_fetch_limit": limit, "fetched_count": 0,
            "validation_passed_count": 0, "validation_failed_count": 0,
            "ml_prediction_attempted_count": 0,
            "ml_approved_raw": 0, "ml_rejected_raw": 0, "ml_errors_raw": 0, # Added from previous summary
            "stopped_by_ml_rejection": 0, # Renamed from ml_rejected_count for clarity
            "rvu_calculation_completed_count": 0, "transferred_to_prod_count": 0,
            "staging_updated_transferred_count": 0, "staging_updated_failed_count": 0,
            "failed_claims_routed_count": 0, "error": None
        }
        # These lists will store (ProcessableClaim, reason_str, stage_str) for _route_failed_claims
        claims_to_route_as_failed: List[Tuple[ProcessableClaim, str, str]] = []
        # This list will store (ProcessableClaim, status_str) for _set_final_status_for_staging_claims
        claims_to_update_final_status_in_staging: List[Tuple[ProcessableClaim, str]] = []


        if not callable(self.db_session_factory):
            summary["error"] = "DB session factory not configured."; logger.error(summary["error"]); return summary

        try:
            async with self.db_session_factory() as session:
                fetched_claims = await self._fetch_claims_parallel(session, batch_id, limit)
                summary["fetched_count"] = len(fetched_claims)
                if not fetched_claims:
                    self.metrics_collector.record_batch_processed(len(fetched_claims), time.perf_counter() - batch_start_time, {})
                    return summary

                valid_for_ml, invalid_for_ml = await self._validate_claims_parallel(fetched_claims)
                summary["validation_passed_count"] = len(valid_for_ml)
                summary["validation_failed_count"] = len(invalid_for_ml)
                for claim in invalid_for_ml:
                    reason = "Validation errors: " + str(getattr(claim, 'validation_errors_detail', 'See logs'))
                    claims_to_route_as_failed.append((claim, reason, "VALIDATION_FAILED"))
                    claims_to_update_final_status_in_staging.append((claim, "VALIDATION_FAILED"))

                claims_for_rvu = []
                if valid_for_ml:
                    await self._apply_ml_predictions(valid_for_ml) # This now sets ml_score, ml_derived_decision on claims
                    summary["ml_prediction_attempted_count"] = len(valid_for_ml)

                    for claim in valid_for_ml: # Tally raw ML decisions from Pydantic models
                        if claim.ml_derived_decision == "ML_APPROVED": summary["ml_approved_raw"] += 1
                        elif claim.ml_derived_decision == "ML_REJECTED": summary["ml_rejected_raw"] += 1
                        elif claim.ml_derived_decision and ("ERROR" in claim.ml_derived_decision.upper() or "SKIPPED" in claim.ml_derived_decision.upper()):
                            summary["ml_errors_raw"] += 1

                        if claim.ml_derived_decision == "ML_REJECTED":
                            summary["stopped_by_ml_rejection"] +=1
                            claims_to_route_as_failed.append((claim, f"ML Rejected: score {claim.ml_score}", "ML_REJECTED"))
                            claims_to_update_final_status_in_staging.append((claim, "ML_REJECTED"))
                        elif claim.ml_derived_decision and ("ERROR" in claim.ml_derived_decision.upper() or "SKIPPED" in claim.ml_derived_decision.upper()):
                            # If ML errors/skips should also stop processing and be routed
                            claims_to_route_as_failed.append((claim, f"ML Error/Skipped: {claim.ml_derived_decision}", "ML_PROCESSING_ERROR"))
                            claims_to_update_final_status_in_staging.append((claim, "ML_PROCESSING_ERROR"))
                        else: # ML_APPROVED (or unhandled cases that proceed)
                            claims_for_rvu.append(claim)
                    if summary["ml_rejected_count"] > 0 or summary["ml_errors_raw"] > 0: # Use combined count for logging
                        logger.info(f"{summary['ml_rejected_count'] + summary['ml_errors_raw']} claims will not proceed further due to ML outcome.", batch_id=batch_id)

                claims_for_transfer = []
                if claims_for_rvu:
                    await self._calculate_rvus_for_claims(session, claims_for_rvu)
                    summary["rvu_calculation_completed_count"] = len(claims_for_rvu)
                    claims_for_transfer = claims_for_rvu # Assuming RVU doesn't filter out claims currently

                batch_duration_for_metrics = time.perf_counter() - batch_start_time # Duration up to this point
                batch_throughput = 0
                if batch_duration_for_metrics > 0 and summary["fetched_count"] > 0:
                    batch_throughput = summary["fetched_count"] / batch_duration_for_metrics

                batch_metrics_for_transfer = {"throughput_achieved": Decimal(str(round(batch_throughput,2)))}
                for claim in claims_for_transfer: # Set processing_duration_ms for claims to be transferred
                    claim.processing_duration_ms = batch_duration_for_metrics * 1000.0

                if claims_for_transfer:
                    inserted_to_prod_count = await self._transfer_claims_to_production(session, claims_for_transfer, batch_metrics_for_transfer)
                    summary["transferred_to_prod_count"] = inserted_to_prod_count

                    successfully_transferred_claims = claims_for_transfer[:inserted_to_prod_count] # Assuming order is preserved and all attempted were these
                    failed_to_transfer_claims = claims_for_transfer[inserted_to_prod_count:]

                    if successfully_transferred_claims:
                        ids_transferred = [c.id for c in successfully_transferred_claims]
                        updated_staging_transferred = await self._update_staging_claims_status(session, ids_transferred, "completed_transferred", transferred=True)
                        summary["staging_updated_transferred_count"] = updated_staging_transferred

                    for claim in failed_to_transfer_claims:
                        claims_to_route_as_failed.append((claim, "Failed during transfer to production step.", "TRANSFER_FAILED"))
                        claims_to_update_final_status_in_staging.append((claim, "TRANSFER_FAILED"))

                if claims_to_update_final_status_in_staging:
                    updated_failed_staging_count = await self._set_final_status_for_staging_claims(session, claims_to_update_final_status_in_staging)
                    summary["staging_updated_failed_count"] = updated_failed_staging_count

                if claims_to_route_as_failed:
                    # Pass only (claim, reason) tuples to _route_failed_claims
                    route_tuples = [(claim, reason) for claim, reason, _stage in claims_to_route_as_failed]
                    # The stage recorded in failed_claims will be the one from the tuple, not a generic one.
                    # This requires _route_failed_claims to accept stage per claim or this method to group by stage.
                    # For simplicity, let's assume _route_failed_claims can take List[Tuple[ProcessableClaim, str]]
                    # and a single 'failed_at_stage' for the batch of failures being routed.
                    # This means we might call it multiple times per stage or pass a generic stage.
                    # The current _route_failed_claims takes a single 'failed_at_stage'.
                    # Let's group them and call:

                    failed_by_stage_for_routing: Dict[str, List[Tuple[ProcessableClaim,str]]] = {}
                    for claim, reason, stage_name in claims_to_route_as_failed:
                        if stage_name not in failed_by_stage_for_routing:
                            failed_by_stage_for_routing[stage_name] = []
                        failed_by_stage_for_routing[stage_name].append((claim, reason))

                    total_routed_count = 0
                    for stage_name, claims_tuples_for_stage in failed_by_stage_for_routing.items():
                        total_routed_count += await self._route_failed_claims(session, claims_tuples_for_stage, stage_name)
                    summary["failed_claims_routed_count"] = total_routed_count

                await session.commit()
                logger.info(f"Main processing batch committed.", **summary)
        except Exception as e:
            logger.error("Critical error in parallel claims processing pipeline", batch_id=batch_id, error=str(e), exc_info=True)
            summary["error"] = str(e)

        batch_duration_seconds_final = time.perf_counter() - batch_start_time
        final_status_counts_for_metric = {
            "completed_transferred": summary.get("staging_updated_transferred_count", 0),
            "validation_failed": summary.get("validation_failed_count", 0), # This is total failed validation
            "ml_rejected": summary.get("stopped_by_ml_rejection", 0),
            "conversion_error": summary.get("conversion_errors",0), # Need a conversion_errors in summary
            "rvu_calculation_failed": summary.get("rvu_calculation_failures",0), # Need rvu_calculation_failures in summary
            "transfer_failed": len(claims_transfer_error_tuples), # Need transfer_errors in summary
            "ml_processing_error": summary.get("ml_errors_raw", 0) # Assuming ml_errors_raw includes processing errors
        }
        # Filter out zero counts for cleaner metric labels
        final_status_counts_for_metric = {k: v for k,v in final_status_counts_for_metric.items() if v > 0}

        self.metrics_collector.record_batch_processed(
            batch_size=summary["fetched_count"],
            duration_seconds=batch_duration_seconds_final,
            claims_by_final_status=final_status_counts_for_metric
        )
        logger.info(f"Batch processing fully complete.", **summary)
        return summary
