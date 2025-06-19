import structlog
from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.orm import joinedload, selectinload

from ....core.cache.cache_manager import CacheManager # Retain for type hint if needed by services passed in
from ....api.models.claim_models import ProcessableClaim
from ....core.database.models.claims_db import ClaimModel
from ..validation.claim_validator import ClaimValidator
from ..rvu_service import RVUService # New import

logger = structlog.get_logger(__name__)

class ParallelClaimsProcessor:
    """
    Processes healthcare claims in parallel, from fetching through validation,
    RVU calculation, ML prediction, and finally transfer to a production database.
    """

    def __init__(self,
                 db_session_factory: Any,
                 claim_validator: ClaimValidator,
                 rvu_service: RVUService):
        """
        Initializes the ParallelClaimsProcessor.

        Args:
            db_session_factory: An asynchronous factory function that provides an AsyncSession.
            claim_validator: An instance of ClaimValidator.
            rvu_service: An instance of RVUService (pre-configured with CacheManager).
        """
        self.db_session_factory = db_session_factory
        self.validator = claim_validator
        self.rvu_service = rvu_service
        logger.info("ParallelClaimsProcessor initialized with validator and RVU service.")

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
        """
        Calculates RVUs for a list of claims using RVUService.
        Modifies claims in-place.
        """
        if not claims:
            logger.info("No claims provided for RVU calculation.")
            return

        logger.info(f"Calculating RVUs for {len(claims)} claims.")
        processed_count = 0
        for claim_idx, claim in enumerate(claims):
            if (claim_idx + 1) % 100 == 0:
                logger.debug(f"RVU calculation progress for batch '{claim.batch_id}': {claim_idx + 1}/{len(claims)} claims processed.")
            try:
                await self.rvu_service.calculate_rvu_for_claim(claim, session) # Pass session
                processed_count += 1
            except Exception as e:
                logger.error(
                    "Error during RVU calculation for claim",
                    claim_id=claim.claim_id,
                    db_claim_id=claim.id,
                    batch_id=claim.batch_id,
                    error=str(e),
                    exc_info=True
                )
        logger.info(f"RVU calculation attempted for {len(claims)} claims. Successfully processed (or attempted with errors): {processed_count}.")


    async def process_claims_parallel(self, batch_id: Optional[str] = None, limit: int = 1000) -> Dict[str, Any]:
        logger.info("Starting parallel claims processing pipeline", batch_id=batch_id, limit=limit)
        summary = {
            "batch_id": batch_id, "attempted_fetch_limit": limit,
            "fetched_count": 0, "validation_passed_count": 0,
            "validation_failed_count": 0, "rvu_calculation_completed_count": 0, # New counter
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
                    await self._calculate_rvus_for_claims(session, valid_claims)
                    summary["rvu_calculation_completed_count"] = len(valid_claims) # Assumes all valid claims had RVU calc attempted

                logger.info(f"Main processing stages (fetch, validate, RVU calc) complete for batch.", **summary)

        except Exception as e:
            logger.error("Error during parallel claims processing pipeline",
                         batch_id=batch_id, error=str(e), exc_info=True)
            summary["error"] = str(e)

        return summary
