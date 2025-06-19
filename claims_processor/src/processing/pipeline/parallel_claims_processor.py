import structlog
from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.orm import joinedload, selectinload

from ....core.cache.cache_manager import CacheManager
from ....api.models.claim_models import ProcessableClaim # ProcessableClaim now has batch_id
from ....core.database.models.claims_db import ClaimModel
from ..validation.claim_validator import ClaimValidator # New import

logger = structlog.get_logger(__name__)

class ParallelClaimsProcessor:
    """
    Processes healthcare claims in parallel, from fetching through validation,
    RVU calculation, ML prediction, and finally transfer to a production database.
    (Note: This is a high-level orchestrator, actual parallelism per stage might vary)
    """

    def __init__(self, db_session_factory: Any, cache_manager: CacheManager):
        """
        Initializes the ParallelClaimsProcessor.

        Args:
            db_session_factory: An asynchronous factory function that provides an AsyncSession.
            cache_manager: An instance of CacheManager for caching operations.
        """
        self.db_session_factory = db_session_factory
        self.cache_manager = cache_manager
        self.validator = ClaimValidator() # Instantiate ClaimValidator
        logger.info("ParallelClaimsProcessor initialized with ClaimValidator.")

    async def _fetch_claims_parallel(self, session: AsyncSession, batch_id: Optional[str] = None, limit: int = 1000) -> List[ProcessableClaim]:
        """
        Fetches a batch of 'pending' claims and their line items from the staging database,
        updates their status to 'processing', and assigns them the given batch_id.
        Orders by created_at.
        """
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
            if batch_id: # Only assign batch_id if provided
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
        """
        Validates a list of claims using ClaimValidator.
        Separates claims into valid and invalid lists.
        """
        logger.info(f"Validating {len(claims_data)} claims.")

        valid_claims_list: List[ProcessableClaim] = []
        invalid_claims_list: List[ProcessableClaim] = []

        for claim_idx, claim in enumerate(claims_data):
            # Log progress periodically for large batches
            if (claim_idx + 1) % 100 == 0:
                logger.debug(f"Validation progress for batch '{claim.batch_id}': {claim_idx + 1}/{len(claims_data)} claims checked.")

            validation_errors = self.validator.validate_claim(claim) # Use the instance's validator
            if not validation_errors:
                valid_claims_list.append(claim)
            else:
                logger.info( # Changed to info for failed validation as it's a key outcome
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

    async def process_claims_parallel(self, batch_id: Optional[str] = None, limit: int = 1000) -> Dict[str, Any]:
        """
        Main processing pipeline for a batch of claims.
        Orchestrates fetching, validation, and other processing stages.
        Returns a summary of processing.
        """
        logger.info("Starting parallel claims processing pipeline", batch_id=batch_id, limit=limit)

        summary = {
            "batch_id": batch_id,
            "attempted_fetch_limit": limit,
            "fetched_count": 0,
            "validation_passed_count": 0,
            "validation_failed_count": 0,
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
                    logger.warn(f"{len(invalid_claims)} claims failed validation.", batch_id=batch_id, first_few_invalid_ids=[c.claim_id for c in invalid_claims[:3]])

                logger.info(f"Initial processing stages complete for batch.",
                            batch_id=batch_id,
                            fetched=summary['fetched_count'],
                            valid=summary['validation_passed_count'],
                            invalid=summary['validation_failed_count'])

        except Exception as e:
            logger.error("Error during parallel claims processing pipeline",
                         batch_id=batch_id, error=str(e), exc_info=True)
            summary["error"] = str(e)

        return summary
