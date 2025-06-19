from sqlalchemy.ext.asyncio import AsyncSession
import structlog
from typing import List, Dict, Any # For type hints

# Required models will be imported later when implementing logic
# from ..core.database.models.claims_db import ClaimModel
# from ..core.database.models.claims_production_db import ClaimsProductionModel
# from ..api.models.claim_models import ProcessableClaim

from sqlalchemy import select
from ..core.database.models.claims_db import ClaimModel
from ..core.database.models.claims_production_db import ClaimsProductionModel
from decimal import Decimal
from sqlalchemy.sql import insert, update # Added update
from datetime import datetime, timezone # Added datetime, timezone

logger = structlog.get_logger(__name__)

class DataTransferService:
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        logger.info("DataTransferService initialized.")

    async def transfer_claims_to_production(self, limit: int = 1000) -> Dict[str, Any]:
        logger.info("Starting transfer of claims to production", record_limit=limit)

        staging_claims_to_transfer = await self._select_claims_from_staging(limit)

        if not staging_claims_to_transfer:
            logger.info("No claims found in staging ready for transfer to production.")
            return {"message": "No claims to transfer.", "transferred_count": 0, "selected_from_staging": 0}

        production_claim_records = self._map_staging_to_production_records(staging_claims_to_transfer)

        if not production_claim_records:
            logger.warn("Mapping resulted in zero production records, though staging claims were selected.",
                        num_staging_claims=len(staging_claims_to_transfer))
            return {"message": "Mapping failed to produce records for transfer.",
                    "selected_from_staging": len(staging_claims_to_transfer),
                    "transferred_count": 0}


        successfully_inserted_count = await self._bulk_insert_to_production(production_claim_records)


        if successfully_inserted_count > 0:
            # Assuming successfully_inserted_count implies all records in production_claim_records were inserted
            # due to the transactional nature of _bulk_insert_to_production.
            if successfully_inserted_count == len(production_claim_records):
                 await self._update_staging_claims_after_transfer(staging_claims_to_transfer)
            else:
                # This case suggests a mismatch or an issue if _bulk_insert_to_production logic changes
                # to allow partial inserts (which it currently doesn't explicitly signal well).
                logger.warn(f"Mismatch in inserted count ({successfully_inserted_count}) vs mapped count ({len(production_claim_records)}). Staging records not updated to prevent inconsistency.")
        elif len(production_claim_records) > 0: # If we attempted to insert but got 0 success
            logger.warn("No records were inserted into production table (insert step failed). Staging records not updated.")
        # else: No records to map/insert, so no update needed.

        logger.info(
            "Data transfer to production processing step finished.",
            selected_from_staging=len(staging_claims_to_transfer),
            mapped_to_production_format=len(production_claim_records),
            successfully_transferred_to_prod_db=successfully_inserted_count
        )

        return {
            "message": "Data transfer process step finished.",
            "selected_from_staging": len(staging_claims_to_transfer),
            "mapped_to_production_format": len(production_claim_records),
            "successfully_transferred": successfully_inserted_count
        }

    async def _select_claims_from_staging(self, limit: int) -> List[ClaimModel]:
        """
        Selects claims from the staging 'claims' table that are ready for transfer.

        Criteria:
        - processing_status is 'processing_complete'.
        - transferred_to_prod_at IS NULL.
        Orders by created_at to process older claims first.
        """
        logger.info("Selecting claims from staging for production transfer", record_limit=limit)

        stmt = (
            select(ClaimModel)
            .where(
                ClaimModel.processing_status == 'processing_complete',
                ClaimModel.transferred_to_prod_at.is_(None)
            )
            .order_by(ClaimModel.created_at.asc())
            .limit(limit)
            # .options(joinedload(ClaimModel.line_items)) # Not needed for now
        )

        result = await self.db.execute(stmt)
        claims_to_transfer = list(result.scalars().all())

        if claims_to_transfer:
            logger.info(f"Found {len(claims_to_transfer)} claims in staging ready for transfer.")
        else:
            logger.info("No claims in staging are currently ready for transfer to production.")

        return claims_to_transfer

    def _map_staging_to_production_records(self, staging_claims: List[ClaimModel]) -> List[Dict[str, Any]]:
        """
        Maps a list of staging ClaimModel instances to a list of dictionaries
        suitable for bulk insertion into the ClaimsProductionModel table.
        """
        logger.info(f"Mapping {len(staging_claims)} staging claims to production record format.")
        production_records: List[Dict[str, Any]] = []

        for stag_claim in staging_claims:
            prod_rec = {
                "id": stag_claim.id,
                "claim_id": stag_claim.claim_id,
                "facility_id": stag_claim.facility_id,
                "patient_account_number": stag_claim.patient_account_number,
                "patient_first_name": stag_claim.patient_first_name,
                "patient_last_name": stag_claim.patient_last_name,
                "patient_date_of_birth": stag_claim.patient_date_of_birth,
                "service_from_date": stag_claim.service_from_date,
                "service_to_date": stag_claim.service_to_date,
                "total_charges": stag_claim.total_charges,

                "ml_prediction_score": stag_claim.ml_score,
                "processing_duration_ms": stag_claim.processing_duration_ms,
                "throughput_achieved": None # Placeholder, typically a batch-level metric
            }

            # Placeholder for risk_category logic
            if stag_claim.ml_score is not None:
                # Ensure ml_score is Decimal for comparison, though it should be from DB
                ml_score_decimal = Decimal(str(stag_claim.ml_score)) if not isinstance(stag_claim.ml_score, Decimal) else stag_claim.ml_score
                if ml_score_decimal >= Decimal("0.8"):
                    prod_rec["risk_category"] = "LOW"
                elif ml_score_decimal >= Decimal("0.5"):
                    prod_rec["risk_category"] = "MEDIUM"
                else:
                    prod_rec["risk_category"] = "HIGH"
            else:
                prod_rec["risk_category"] = "UNKNOWN"

            production_records.append(prod_rec)
            logger.debug(f"Mapped staging claim ID {stag_claim.claim_id} to production format.",
                         production_record_preview={k: v for k, v in prod_rec.items() if k in ['claim_id', 'ml_prediction_score', 'risk_category']})

        logger.info(f"Successfully mapped {len(production_records)} claims.")
        return production_records

    async def _bulk_insert_to_production(self, production_records: List[Dict[str, Any]]) -> int:
        """
        Bulk inserts records into the ClaimsProductionModel table.
        Returns the count of successfully inserted records.
        Uses SQLAlchemy Core insert for efficiency with the existing session.
        """
        if not production_records:
            return 0

        logger.info(f"Attempting to bulk insert {len(production_records)} records into claims_production.")

        try:
            await self.db.execute(
                insert(ClaimsProductionModel.__table__),
                production_records
            )
            await self.db.commit()
            logger.info(f"Successfully bulk inserted {len(production_records)} records into claims_production.")
            return len(production_records)

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to bulk insert records into claims_production: {e}", exc_info=True)
            return 0

    async def _update_staging_claims_after_transfer(self, transferred_staging_claims: List[ClaimModel]):
        """
        Updates claims in the staging table after they have been successfully transferred.
        Sets the 'transferred_to_prod_at' timestamp.
        """
        if not transferred_staging_claims:
            return

        logger.info(f"Updating {len(transferred_staging_claims)} claims in staging table as 'transferred'.")

        claim_ids_to_update = [claim.id for claim in transferred_staging_claims]

        if not claim_ids_to_update:
            return

        try:
            stmt = (
                update(ClaimModel.__table__)
                .where(ClaimModel.id.in_(claim_ids_to_update))
                .values(
                    transferred_to_prod_at=datetime.now(timezone.utc),
                    # Optionally, also update status if there's a distinct "transferred" status
                    # processing_status="transferred_to_production"
                )
                .execution_options(synchronize_session=False)
            )

            result = await self.db.execute(stmt)
            await self.db.commit()

            logger.info(f"Successfully updated {result.rowcount if result else 'N/A'} staging claims. Claim IDs: {claim_ids_to_update}")

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to update staging claims after transfer: {e}", exc_info=True)
