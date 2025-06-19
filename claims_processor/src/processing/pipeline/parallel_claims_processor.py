import structlog
from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.orm import joinedload, selectinload
import numpy as np
from sqlalchemy.sql import insert
from decimal import Decimal
from datetime import datetime, timezone

from ....core.cache.cache_manager import CacheManager
from ....api.models.claim_models import ProcessableClaim
from ....core.database.models.claims_db import ClaimModel
from ....core.database.models.claims_production_db import ClaimsProductionModel
from ....core.database.models.failed_claims_db import FailedClaimModel
from ..validation.claim_validator import ClaimValidator
from ..rvu_service import RVUService
from ..ml_pipeline.feature_extractor import FeatureExtractor
from ..ml_pipeline.optimized_predictor import OptimizedPredictor

logger = structlog.get_logger(__name__)

class ParallelClaimsProcessor:
    def __init__(self,
                 db_session_factory: Any,
                 claim_validator: ClaimValidator,
                 rvu_service: RVUService,
                 feature_extractor: FeatureExtractor,
                 optimized_predictor: OptimizedPredictor):
        self.db_session_factory = db_session_factory
        self.validator = claim_validator
        self.rvu_service = rvu_service
        self.feature_extractor = feature_extractor
        self.predictor = optimized_predictor
        logger.info("ParallelClaimsProcessor initialized with validator, RVU service, FeatureExtractor, and OptimizedPredictor.")

    async def _fetch_claims_parallel(self, session: AsyncSession, batch_id: Optional[str] = None, limit: int = 1000) -> List[ProcessableClaim]:
        # ... (Existing implementation) ...
        logger.info("Fetching pending claims for processing", batch_id=batch_id, limit=limit)
        try:
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
            update_stmt = (
                update(ClaimModel)
                .where(ClaimModel.id.in_(claim_db_ids))
                .values(**update_values)
                .execution_options(synchronize_session=False)
            )
            await session.execute(update_stmt)
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
        # ... (Existing implementation) ...
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
                setattr(claim, 'validation_errors_detail', validation_errors) # Attach errors for routing
                logger.info(
                    "Claim failed validation", claim_id=claim.claim_id, db_claim_id=claim.id,
                    batch_id=claim.batch_id, errors=validation_errors
                )
                invalid_claims_list.append(claim)
        batch_id_for_log = claims_data[0].batch_id if claims_data and claims_data[0].batch_id else 'N/A'
        logger.info(f"Validation complete for batch '{batch_id_for_log}'. Valid: {len(valid_claims_list)}, Invalid: {len(invalid_claims_list)}")
        return valid_claims_list, invalid_claims_list

    async def _calculate_rvus_for_claims(self, session: AsyncSession, claims: List[ProcessableClaim]) -> None:
        # ... (Existing implementation) ...
        if not claims: logger.info("No claims for RVU calculation."); return
        logger.info(f"Calculating RVUs for {len(claims)} claims.")
        processed_count = 0
        for claim_idx, claim in enumerate(claims):
            if (claim_idx + 1) % 100 == 0: logger.debug(f"RVU progress for batch '{claim.batch_id}': {claim_idx + 1}/{len(claims)}.")
            try:
                await self.rvu_service.calculate_rvu_for_claim(claim, session)
                processed_count += 1
            except Exception as e: logger.error("Error during RVU calculation", claim_id=claim.claim_id, error=str(e), exc_info=True)
        logger.info(f"RVU calculation attempted for {len(claims)}. Processed calls: {processed_count}.")

    async def _apply_ml_predictions(self, claims: List[ProcessableClaim]) -> None:
        # ... (Existing implementation) ...
        if not claims: logger.info("No claims for ML prediction."); return
        logger.info(f"Applying ML predictions for {len(claims)} claims.")
        features_batch: List[np.ndarray] = []; claims_with_features: List[ProcessableClaim] = []
        for claim in claims:
            try:
                features = self.feature_extractor.extract_features(claim)
                if features is not None: features_batch.append(features); claims_with_features.append(claim)
                else: claim.ml_derived_decision = "ML_SKIPPED_NO_FEATURES"; claim.ml_score = None
            except Exception as e: logger.error("Feature extraction error", claim_id=claim.claim_id, e=e); claim.ml_derived_decision = "ML_SKIPPED_EXTRACTION_ERROR"; claim.ml_score = None
        if not features_batch: logger.info("No features extracted for ML."); return
        try:
            results = await self.predictor.predict_batch(features_batch)
            if len(results) == len(claims_with_features):
                for claim, res_dict in zip(claims_with_features, results):
                    if "error" in res_dict: claim.ml_derived_decision = "ML_PREDICTION_ERROR"; claim.ml_score = None
                    else: claim.ml_score = res_dict.get('ml_score'); claim.ml_derived_decision = res_dict.get('ml_derived_decision')
            else: logger.error("ML prediction result count mismatch."); [setattr(c, 'ml_derived_decision', "ML_PREDICTION_BATCH_ERROR") for c in claims_with_features]
        except Exception as e: logger.error("ML batch prediction error", e=e); [setattr(c, 'ml_derived_decision', "ML_PREDICTION_BATCH_ERROR") for c in claims_with_features]
        logger.info(f"ML prediction application completed for {len(claims_with_features)} claims.")

    async def _transfer_claims_to_production(self, session: AsyncSession, claims: List[ProcessableClaim], metrics: Optional[Dict]=None) -> int:
        # ... (Existing implementation, ensure it uses `claims` not `claims_to_transfer`) ...
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
            if claim.ml_score is not None: score = Decimal(str(claim.ml_score))
            elif claim.ml_derived_decision and "ERROR" in claim.ml_derived_decision.upper(): data["risk_category"] = "ERROR_IN_ML"
            elif "SKIPPED" in (claim.ml_derived_decision or ""): data["risk_category"] = "ML_SKIPPED"
            else: data["risk_category"] = "UNKNOWN"
            if 'score' in locals() : data["risk_category"] = "LOW" if score >= Decimal("0.8") else ("MEDIUM" if score >= Decimal("0.5") else "HIGH") # type: ignore
            insert_list.append(data)
        if not insert_list: logger.warn("No data mapped for prod transfer."); return 0
        try: await session.execute(insert(ClaimsProductionModel).values(insert_list)); return len(insert_list)
        except Exception as e: logger.error("DB error during prod insert", e=e); return 0

    async def _update_staging_claims_status(self, session: AsyncSession, ids: List[int], status: str, transferred: bool = False) -> int:
        if not ids: logger.info("No IDs for staging update."); return 0
        logger.info(f"Updating {len(ids)} staging claims to status '{status}'. Transferred: {transferred}")
        values_to_set = {"processing_status": status, "processed_at": datetime.now(timezone.utc)}
        if transferred: values_to_set["transferred_to_prod_at"] = datetime.now(timezone.utc)
        try:
            stmt = update(ClaimModel).where(ClaimModel.id.in_(ids)).values(**values_to_set).execution_options(synchronize_session=False)
            res = await session.execute(stmt); updated_count = res.rowcount if res else 0
            if updated_count != len(ids): logger.warn(f"Staging update count mismatch for status '{status}'", expected=len(ids), actual=updated_count)
            return updated_count
        except Exception as e: logger.error(f"DB error updating staging to '{status}'", e=e); return 0

    async def _route_failed_claims(self, session: AsyncSession, failed_info: List[Tuple[ProcessableClaim, str, str]]) -> int: # claim, reason, stage
        if not failed_info: logger.debug("No failed claims to route."); return 0
        logger.info(f"Routing {len(failed_info)} failed claims.")
        insert_data = []
        for claim, reason, stage in failed_info:
            try: original_data = claim.model_dump()
            except Exception as e: original_data = {"error": f"Serialization fail: {e}", "claim_id": getattr(claim, 'claim_id', 'N/A')}
            insert_data.append({
                "original_claim_db_id": claim.id, "claim_id": claim.claim_id, "facility_id": claim.facility_id,
                "patient_account_number": claim.patient_account_number, "failed_at_stage": stage,
                "failure_reason": reason, "original_claim_data": original_data
            })
        if not insert_data: logger.warn("No data mapped for failed_claims table."); return 0
        try: await session.execute(insert(FailedClaimModel).values(insert_data)); return len(insert_data)
        except Exception as e: logger.error("DB error inserting into failed_claims", e=e); return 0

    async def process_claims_parallel(self, batch_id: Optional[str] = None, limit: int = 1000) -> Dict[str, Any]:
        logger.info("Starting parallel claims processing pipeline", batch_id=batch_id, limit=limit)
        summary = {
            "batch_id": batch_id, "attempted_fetch_limit": limit, "fetched_count": 0,
            "validation_passed_count": 0, "validation_failed_count": 0,
            "ml_prediction_attempted_count": 0, "ml_rejected_count": 0,
            "rvu_calculation_completed_count": 0, "transferred_to_prod_count": 0,
            "staging_updated_transferred_count": 0, "staging_updated_failed_count": 0,
            "failed_claims_routed_count": 0, "error": None
        }
        claims_failed_validation: List[Tuple[ProcessableClaim, str]] = []
        claims_rejected_by_ml: List[Tuple[ProcessableClaim, str]] = []
        claims_failed_rvu: List[Tuple[ProcessableClaim, str]] = [] # If RVU can fail per claim
        claims_failed_transfer: List[Tuple[ProcessableClaim, str]] = []


        if not callable(self.db_session_factory):
            summary["error"] = "DB session factory not configured."; logger.error(summary["error"]); return summary

        try:
            async with self.db_session_factory() as session:
                fetched_claims = await self._fetch_claims_parallel(session, batch_id, limit)
                summary["fetched_count"] = len(fetched_claims)
                if not fetched_claims: return summary

                valid_for_ml, invalid_for_ml = await self._validate_claims_parallel(fetched_claims)
                summary["validation_passed_count"] = len(valid_for_ml)
                summary["validation_failed_count"] = len(invalid_for_ml)
                for claim in invalid_for_ml: claims_failed_validation.append((claim, "Validation errors: " + str(getattr(claim, 'validation_errors_detail', 'See logs'))))

                claims_for_rvu = []
                if valid_for_ml:
                    await self._apply_ml_predictions(valid_for_ml)
                    summary["ml_prediction_attempted_count"] = len(valid_for_ml)
                    for claim in valid_for_ml:
                        if claim.ml_derived_decision == "ML_REJECTED": summary["ml_rejected_count"] +=1; claims_rejected_by_ml.append((claim, f"ML Rejected: score {claim.ml_score}"))
                        elif claim.ml_derived_decision and ("ERROR" in claim.ml_derived_decision.upper() or "SKIPPED" in claim.ml_derived_decision.upper()):
                             claims_rejected_by_ml.append((claim, f"ML Error: {claim.ml_derived_decision}")) # Count as rejected from main flow
                        else: claims_for_rvu.append(claim)

                claims_for_transfer = []
                if claims_for_rvu:
                    await self._calculate_rvus_for_claims(session, claims_for_rvu)
                    summary["rvu_calculation_completed_count"] = len(claims_for_rvu)
                    claims_for_transfer = claims_for_rvu # Assuming RVU doesn't filter out claims, only logs errors

                if claims_for_transfer:
                    inserted_to_prod_count = await self._transfer_claims_to_production(session, claims_for_transfer, {})
                    summary["transferred_to_prod_count"] = inserted_to_prod_count
                    if inserted_to_prod_count > 0:
                        ids_transferred = [c.id for c in claims_for_transfer[:inserted_to_prod_count]] # Assumes order and full success for list slice
                        updated_staging_transferred = await self._update_staging_claims_status(session, ids_transferred, "completed_transferred", transferred=True)
                        summary["staging_updated_transferred_count"] = updated_staging_transferred
                        if inserted_to_prod_count < len(claims_for_transfer): # Some failed transfer
                             for claim in claims_for_transfer[inserted_to_prod_count:]: claims_failed_transfer.append((claim, "Failed during transfer to production step."))
                    elif len(claims_for_transfer) > 0 : # Attempted but none inserted
                        for claim in claims_for_transfer: claims_failed_transfer.append((claim, "Failed during transfer to production step (batch error)."))


                # Consolidate all claims that didn't make it to 'completed_transferred'
                final_failed_claims_to_route: List[Tuple[ProcessableClaim, str, str]] = []
                for claim, reason in claims_failed_validation: final_failed_claims_to_route.append((claim, reason, "VALIDATION_FAILED"))
                for claim, reason in claims_rejected_by_ml: final_failed_claims_to_route.append((claim, reason, "ML_REJECTED_OR_ERROR"))
                # TODO: Capture claims failing RVU if _calculate_rvus_for_claims is modified to return them
                for claim, reason in claims_failed_transfer: final_failed_claims_to_route.append((claim, reason, "TRANSFER_FAILED"))

                if final_failed_claims_to_route:
                    # Update statuses for these terminal non-transferred states
                    # This is complex if reasons map to different statuses. Group them.
                    failed_map_for_status_update: Dict[str, List[ProcessableClaim]] = {}
                    for claim, reason, stage_name in final_failed_claims_to_route:
                        final_status = stage_name # Use stage name as the status for now
                        if final_status not in failed_map_for_status_update: failed_map_for_status_update[final_status] = []
                        failed_map_for_status_update[final_status].append(claim)

                    total_failed_staging_updated = 0
                    for status_val, claims_list in failed_map_for_status_update.items():
                        ids_to_update = [c.id for c in claims_list]
                        if ids_to_update:
                            total_failed_staging_updated += await self._update_staging_claims_status(session, ids_to_update, status_val, transferred=False)
                    summary["staging_updated_failed_count"] = total_failed_staging_updated

                    # Route to failed_claims table (pass original list with specific reasons)
                    routed_count = await self._route_failed_claims(session, [(c,r) for c,r,s in final_failed_claims_to_route], "PIPELINE_FAILURE") # Generic stage for routing log
                    summary["failed_claims_routed_count"] = routed_count

                await session.commit()
                logger.info(f"Main processing stages complete for batch and session committed.", **summary)
        except Exception as e:
            logger.error("Error during parallel claims processing pipeline", batch_id=batch_id, error=str(e), exc_info=True)
            summary["error"] = str(e)
        return summary
