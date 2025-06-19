import structlog
from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.orm import joinedload, selectinload
import numpy as np
from sqlalchemy.sql import insert
from decimal import Decimal
from datetime import datetime, timezone # Ensure datetime, timezone are imported

from ....core.cache.cache_manager import CacheManager
from ....api.models.claim_models import ProcessableClaim
from ....core.database.models.claims_db import ClaimModel
from ....core.database.models.claims_production_db import ClaimsProductionModel
from ..validation.claim_validator import ClaimValidator
from ..rvu_service import RVUService
from ..ml_pipeline.feature_extractor import FeatureExtractor
from ..ml_pipeline.optimized_predictor import OptimizedPredictor

logger = structlog.get_logger(__name__)

class ParallelClaimsProcessor:
    """
    Processes healthcare claims in parallel, from fetching through validation,
    RVU calculation, ML prediction, and finally transfer to a production database.
    """

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
            if batch_id:
                update_values['batch_id'] = batch_id

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
                logger.info(
                    "Claim failed validation", claim_id=claim.claim_id, db_claim_id=claim.id,
                    batch_id=claim.batch_id, errors=validation_errors
                )
                invalid_claims_list.append(claim)
        batch_id_for_log = claims_data[0].batch_id if claims_data and claims_data[0].batch_id else 'N/A'
        logger.info(f"Validation complete for batch '{batch_id_for_log}'. Valid: {len(valid_claims_list)}, Invalid: {len(invalid_claims_list)}")
        return valid_claims_list, invalid_claims_list

    async def _calculate_rvus_for_claims(self, session: AsyncSession, claims: List[ProcessableClaim]) -> None:
        if not claims:
            logger.info("No claims provided for RVU calculation.")
            return
        logger.info(f"Calculating RVUs for {len(claims)} claims.")
        processed_count = 0
        for claim_idx, claim in enumerate(claims):
            if (claim_idx + 1) % 100 == 0:
                logger.debug(f"RVU calculation progress for batch '{claim.batch_id}': {claim_idx + 1}/{len(claims)} claims processed.")
            try:
                await self.rvu_service.calculate_rvu_for_claim(claim, session)
                processed_count += 1
            except Exception as e:
                logger.error(
                    "Error during RVU calculation for claim", claim_id=claim.claim_id, db_claim_id=claim.id,
                    batch_id=claim.batch_id, error=str(e), exc_info=True
                )
        logger.info(f"RVU calculation attempted for {len(claims)} claims. Processed calls to service: {processed_count}.")

    async def _apply_ml_predictions(self, claims: List[ProcessableClaim]) -> None:
        if not claims:
            logger.info("No claims provided for ML prediction.")
            return
        logger.info(f"Applying ML predictions for {len(claims)} claims.")
        features_batch_for_predictor: List[np.ndarray] = []
        claims_with_features: List[ProcessableClaim] = []
        for claim_idx, claim in enumerate(claims):
            if (claim_idx + 1) % 50 == 0:
                logger.info(f"ML Feature Extraction progress: {claim_idx + 1}/{len(claims)} claims processed.")
            try:
                features_array_2d = self.feature_extractor.extract_features(claim)
                if features_array_2d is not None:
                    features_batch_for_predictor.append(features_array_2d)
                    claims_with_features.append(claim)
                else:
                    logger.warn("Feature extraction failed or returned None for claim", claim_id=claim.claim_id, db_claim_id=claim.id)
                    claim.ml_derived_decision = "ML_SKIPPED_NO_FEATURES"; claim.ml_score = None
            except Exception as e:
                logger.error("Error during feature extraction for claim", claim_id=claim.claim_id, error=str(e), exc_info=True)
                claim.ml_derived_decision = "ML_SKIPPED_EXTRACTION_ERROR"; claim.ml_score = None
        if not features_batch_for_predictor:
            logger.info("No claims had features successfully extracted for ML prediction.")
            return
        logger.info(f"Sending batch of {len(features_batch_for_predictor)} feature sets to OptimizedPredictor.")
        try:
            prediction_results = await self.predictor.predict_batch(features_batch_for_predictor)
            if len(prediction_results) == len(claims_with_features):
                for claim_obj, prediction_dict in zip(claims_with_features, prediction_results):
                    if "error" in prediction_dict:
                        logger.warn("ML prediction failed for claim", claim_id=claim_obj.claim_id, error=prediction_dict["error"])
                        claim_obj.ml_derived_decision = "ML_PREDICTION_ERROR"; claim_obj.ml_score = None
                    else:
                        claim_obj.ml_score = prediction_dict.get('ml_score')
                        claim_obj.ml_derived_decision = prediction_dict.get('ml_derived_decision')
                        logger.debug("ML prediction applied", claim_id=claim_obj.claim_id, score=claim_obj.ml_score, decision=claim_obj.ml_derived_decision)
            else:
                logger.error(f"Mismatch: claims sent ({len(claims_with_features)}) vs results received ({len(prediction_results)}).")
                for claim_obj in claims_with_features:
                    claim_obj.ml_derived_decision = "ML_PREDICTION_BATCH_ERROR"; claim_obj.ml_score = None
        except Exception as e:
            logger.error("Error during ML batch prediction", error=str(e), exc_info=True)
            for claim_obj in claims_with_features:
                claim_obj.ml_derived_decision = "ML_PREDICTION_BATCH_ERROR"; claim_obj.ml_score = None
        logger.info(f"ML prediction application completed for {len(claims_with_features)} claims.")

    async def _transfer_claims_to_production(
        self,
        session: AsyncSession,
        claims_to_transfer: List[ProcessableClaim],
        batch_processing_metrics: Optional[Dict[str, Any]] = None
    ) -> int:
        # ... (Existing implementation from prior step) ...
        if not claims_to_transfer:
            logger.info("No claims provided for transfer to production.")
            return 0
        logger.info(f"Preparing {len(claims_to_transfer)} claims for transfer to production table.")
        throughput_for_batch = None
        if batch_processing_metrics and 'throughput_achieved' in batch_processing_metrics:
            try:
                throughput_for_batch = Decimal(str(batch_processing_metrics['throughput_achieved']))
            except Exception:
                logger.warn("Could not parse 'throughput_achieved' for batch", exc_info=True)

        insert_data_list = []
        for claim in claims_to_transfer:
            data = {
                "id": claim.id, "claim_id": claim.claim_id, "facility_id": claim.facility_id,
                "patient_account_number": claim.patient_account_number,
                "patient_first_name": claim.patient_first_name, "patient_last_name": claim.patient_last_name,
                "patient_date_of_birth": claim.patient_date_of_birth,
                "service_from_date": claim.service_from_date, "service_to_date": claim.service_to_date,
                "total_charges": claim.total_charges,
                "ml_prediction_score": Decimal(str(claim.ml_score)) if claim.ml_score is not None else None,
                # risk_category is derived based on ml_score or ml_derived_decision
                "processing_duration_ms": int(claim.processing_duration_ms) if claim.processing_duration_ms is not None else None,
                "throughput_achieved": throughput_for_batch,
            }
            if claim.ml_score is not None:
                score = Decimal(str(claim.ml_score))
                if score >= Decimal("0.8"): data["risk_category"] = "LOW"
                elif score >= Decimal("0.5"): data["risk_category"] = "MEDIUM"
                else: data["risk_category"] = "HIGH"
            elif claim.ml_derived_decision and "ERROR" in claim.ml_derived_decision.upper(): # Check if decision indicates error
                 data["risk_category"] = "ERROR_IN_ML"
            elif claim.ml_derived_decision == "ML_SKIPPED_NO_FEATURES" or claim.ml_derived_decision == "ML_SKIPPED_EXTRACTION_ERROR":
                 data["risk_category"] = "ML_SKIPPED"
            else:
                 data["risk_category"] = "UNKNOWN"
            insert_data_list.append(data)

        if not insert_data_list:
            logger.warn("No data mapped for production transfer.")
            return 0
        try:
            stmt = insert(ClaimsProductionModel).values(insert_data_list)
            await session.execute(stmt)
            logger.info(f"Successfully prepared bulk insert for {len(insert_data_list)} claims into production table. Commit pending on main session.")
            return len(insert_data_list)
        except Exception as e:
            logger.error("Database error during bulk insert preparation to claims_production", error=str(e), exc_info=True)
            return 0

    async def _update_staging_claims_status(
        self,
        session: AsyncSession,
        transferred_claim_db_ids: List[int],
    ) -> int:
        if not transferred_claim_db_ids:
            logger.info("No claim IDs provided for staging status update.")
            return 0
        logger.info(f"Updating staging status for {len(transferred_claim_db_ids)} transferred claims.")
        try:
            update_stmt = (
                update(ClaimModel)
                .where(ClaimModel.id.in_(transferred_claim_db_ids))
                .values(
                    processing_status='completed_transferred',
                    transferred_to_prod_at=datetime.now(timezone.utc),
                    processed_at=datetime.now(timezone.utc)
                )
                .execution_options(synchronize_session=False)
            )
            result = await session.execute(update_stmt)
            updated_count = result.rowcount
            logger.info(f"Successfully updated status for {updated_count} claims in staging table.")
            if updated_count != len(transferred_claim_db_ids):
                logger.warn(
                    f"Mismatch in expected ({len(transferred_claim_db_ids)}) vs actual ({updated_count}) "
                    "staging claims updated. Some IDs might not have been found or matched (already updated?)."
                )
            return updated_count
        except Exception as e:
            logger.error(
                "Database error during staging claims status update.",
                error=str(e),
                num_ids_to_update=len(transferred_claim_db_ids),
                exc_info=True
            )
            return 0

    async def process_claims_parallel(self, batch_id: Optional[str] = None, limit: int = 1000) -> Dict[str, Any]:
        logger.info("Starting parallel claims processing pipeline", batch_id=batch_id, limit=limit)
        summary = {
            "batch_id": batch_id, "attempted_fetch_limit": limit,
            "fetched_count": 0, "validation_passed_count": 0,
            "validation_failed_count": 0,
            "ml_prediction_attempted_count": 0,
            "ml_rejected_count": 0,
            "rvu_calculation_completed_count": 0,
            "transferred_to_prod_count": 0,
            "staging_updated_count": 0, # New summary key
            "error": None
        }
        if not callable(self.db_session_factory):
            logger.error("db_session_factory is not callable. Cannot proceed.")
            summary["error"] = "DB session factory not configured."
            return summary

        try:
            async with self.db_session_factory() as session:
                fetched_claims = await self._fetch_claims_parallel(session, batch_id, limit)
                summary["fetched_count"] = len(fetched_claims)
                if not fetched_claims:
                    logger.info("No claims fetched, ending process.", batch_id=batch_id)
                    return summary

                valid_claims, invalid_claims = await self._validate_claims_parallel(fetched_claims)
                summary["validation_passed_count"] = len(valid_claims)
                summary["validation_failed_count"] = len(invalid_claims)
                if invalid_claims:
                    logger.warn(f"{len(invalid_claims)} claims failed validation.",
                                batch_id=batch_id,
                                first_few_invalid_ids=[c.claim_id for c in invalid_claims[:3]])

                claims_for_further_processing = []
                if valid_claims:
                    await self._apply_ml_predictions(valid_claims)
                    summary["ml_prediction_attempted_count"] = len(valid_claims)
                    ml_rejected_this_batch = 0
                    for claim in valid_claims:
                        if claim.ml_derived_decision != "ML_REJECTED":
                            claims_for_further_processing.append(claim)
                        else:
                            ml_rejected_this_batch += 1
                    summary["ml_rejected_count"] = ml_rejected_this_batch
                    if ml_rejected_this_batch > 0:
                        logger.info(f"{ml_rejected_this_batch} claims rejected by ML, will not proceed further.", batch_id=batch_id)

                if claims_for_further_processing:
                    await self._calculate_rvus_for_claims(session, claims_for_further_processing)
                    summary["rvu_calculation_completed_count"] = len(claims_for_further_processing)

                    batch_metrics = {"throughput_achieved": None}

                    successfully_inserted_to_prod_count = await self._transfer_claims_to_production(
                        session, claims_for_further_processing, batch_metrics
                    )
                    summary["transferred_to_prod_count"] = successfully_inserted_to_prod_count

                    if successfully_inserted_to_prod_count > 0:
                        ids_of_transferred_claims = [
                            claim.id for claim in claims_for_further_processing # All these were attempted for transfer
                        ]
                        # Filter based on actual number inserted if there was a partial failure logic in transfer
                        # Current _transfer_claims_to_production returns 0 on any failure, or len(all_attempted) on success.
                        # So, if successfully_inserted_to_prod_count > 0, it implies all in claims_for_further_processing were part of the batch.
                        if successfully_inserted_to_prod_count == len(claims_for_further_processing):
                            updated_staging_count = await self._update_staging_claims_status(session, ids_of_transferred_claims)
                            summary["staging_updated_count"] = updated_staging_count
                            if updated_staging_count != successfully_inserted_to_prod_count:
                                 logger.warn("Mismatch between successfully inserted to prod and updated in staging",
                                             inserted_prod=successfully_inserted_to_prod_count, updated_staging=updated_staging_count)
                        else:
                            logger.warn("Partial success from _transfer_claims_to_production not fully handled for staging update logic.",
                                        inserted_prod=successfully_inserted_to_prod_count,
                                        candidates_for_transfer=len(claims_for_further_processing))
                            summary["staging_updated_count"] = 0 # Or handle more gracefully
                    else:
                        summary["staging_updated_count"] = 0
                else:
                    logger.info("No claims eligible for RVU calculation or transfer after ML stage.", batch_id=batch_id)
                    summary["rvu_calculation_completed_count"] = 0
                    summary["transferred_to_prod_count"] = 0
                    summary["staging_updated_count"] = 0

                await session.commit()
                logger.info(f"Main processing stages complete for batch and session committed.", **summary)
        except Exception as e:
            logger.error("Error during parallel claims processing pipeline",
                         batch_id=batch_id, error=str(e), exc_info=True)
            summary["error"] = str(e)
        return summary
