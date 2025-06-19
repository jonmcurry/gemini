import structlog
from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.orm import joinedload, selectinload

from ....core.cache.cache_manager import CacheManager
from ....api.models.claim_models import ProcessableClaim
from ....core.database.models.claims_db import ClaimModel
from ..validation.claim_validator import ClaimValidator
from ..rvu_service import RVUService
from ..ml_pipeline.feature_extractor import FeatureExtractor
from ..ml_pipeline.optimized_predictor import OptimizedPredictor
import numpy as np

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
                    "Claim failed validation",
                    claim_id=claim.claim_id,
                    db_claim_id=claim.id,
                    batch_id=claim.batch_id,
                    errors=validation_errors
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
                    "Error during RVU calculation for claim",
                    claim_id=claim.claim_id, db_claim_id=claim.id,
                    batch_id=claim.batch_id, error=str(e), exc_info=True
                )
        logger.info(f"RVU calculation attempted for {len(claims)} claims. Successfully processed (or attempted with errors): {processed_count}.")

    async def _apply_ml_predictions(self, claims: List[ProcessableClaim]) -> None:
        """
        Applies ML predictions to a list of claims using FeatureExtractor and OptimizedPredictor.
        Modifies claims in-place with ml_score and ml_derived_decision.
        """
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
                    # OptimizedPredictor.predict_batch expects a list of 1D arrays if it processes them one by one,
                    # or a single 2D array if it can batch predict.
                    # The current OptimizedPredictor.predict_batch iterates a list of 1D arrays.
                    # FeatureExtractor returns (1, N). We need to ensure this matches.
                    # The implemented OptimizedPredictor.predict_batch handles reshaping (1,N) to (N,) internally.
                    # So we can append the (1,N) array.
                    features_batch_for_predictor.append(features_array_2d)
                    claims_with_features.append(claim)
                else:
                    logger.warn("Feature extraction failed or returned None for claim", claim_id=claim.claim_id, db_claim_id=claim.id)
                    claim.ml_derived_decision = "ML_SKIPPED_NO_FEATURES"
                    claim.ml_score = None

            except Exception as e:
                logger.error("Error during feature extraction for claim", claim_id=claim.claim_id, error=str(e), exc_info=True)
                claim.ml_derived_decision = "ML_SKIPPED_EXTRACTION_ERROR"
                claim.ml_score = None

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
                        claim_obj.ml_derived_decision = "ML_PREDICTION_ERROR"
                        claim_obj.ml_score = None
                    else:
                        claim_obj.ml_score = prediction_dict.get('ml_score')
                        claim_obj.ml_derived_decision = prediction_dict.get('ml_derived_decision')
                        logger.debug("ML prediction applied to claim", claim_id=claim_obj.claim_id, score=claim_obj.ml_score, decision=claim_obj.ml_derived_decision)
            else:
                logger.error(
                    f"Mismatch between number of claims sent for prediction ({len(claims_with_features)}) "
                    f"and number of results received ({len(prediction_results)}). Cannot map results."
                )
                for claim_obj in claims_with_features:
                    claim_obj.ml_derived_decision = "ML_PREDICTION_BATCH_ERROR"
                    claim_obj.ml_score = None

        except Exception as e:
            logger.error("Error during ML batch prediction", error=str(e), exc_info=True)
            for claim_obj in claims_with_features:
                claim_obj.ml_derived_decision = "ML_PREDICTION_BATCH_ERROR"
                claim_obj.ml_score = None

        logger.info(f"ML prediction application completed for {len(claims_with_features)} claims.")

    async def process_claims_parallel(self, batch_id: Optional[str] = None, limit: int = 1000) -> Dict[str, Any]:
        logger.info("Starting parallel claims processing pipeline", batch_id=batch_id, limit=limit)
        summary = {
            "batch_id": batch_id, "attempted_fetch_limit": limit,
            "fetched_count": 0, "validation_passed_count": 0,
            "validation_failed_count": 0,
            "ml_prediction_attempted_count": 0, # New summary key
            "rvu_calculation_completed_count": 0,
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

                if valid_claims:
                    await self._apply_ml_predictions(valid_claims) # Call ML prediction stage
                    summary["ml_prediction_attempted_count"] = len(valid_claims)

                    # TODO: Add logic to filter claims based on ML decision if needed before RVU calc
                    # For now, all valid_claims (post-validation) proceed to RVU regardless of ML outcome.

                    await self._calculate_rvus_for_claims(session, valid_claims)
                    summary["rvu_calculation_completed_count"] = len(valid_claims)

                logger.info(f"Main processing stages (fetch, validate, ML, RVU calc) complete for batch.", **summary)

        except Exception as e:
            logger.error("Error during parallel claims processing pipeline",
                         batch_id=batch_id, error=str(e), exc_info=True)
            summary["error"] = str(e)

        return summary
