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
from sqlalchemy.sql import insert, update
from sqlalchemy import text
from datetime import datetime, timezone
# from sqlalchemy.dialects.postgresql import insert as pg_insert # Removed as no longer used
from typing import Optional
from ..core.config.settings import get_settings
from sqlalchemy.exc import OperationalError # Added for retry logic
import asyncio # Added for asyncio.sleep

logger = structlog.get_logger(__name__)

class DataTransferService:
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.settings = get_settings() # Store settings if needed by other methods or for consistency
        logger.info("DataTransferService initialized.")

    async def transfer_claims_to_production(self, limit: Optional[int] = None) -> Dict[str, Any]:
        effective_limit = limit if limit is not None else self.settings.TRANSFER_BATCH_SIZE
        logger.info("Starting transfer of claims to production", record_limit=effective_limit)

        # Define retry parameters (consider moving to settings if more widely used)
        max_retries = self.settings.MAX_DB_RETRIES if hasattr(self.settings, 'MAX_DB_RETRIES') else 3
        retry_delay = self.settings.DB_RETRY_DELAY if hasattr(self.settings, 'DB_RETRY_DELAY') else 1.0

        staging_claims_to_transfer: List[ClaimModel] = []
        production_claim_records: List[Dict[str, Any]] = []
        successfully_inserted_count = 0

        for attempt in range(max_retries):
            try:
                async with self.db.begin(): # Single transaction for all operations in this attempt
                    staging_claims_to_transfer = await self._select_claims_from_staging(effective_limit)

                    if not staging_claims_to_transfer:
                        logger.info("No claims found in staging ready for transfer on attempt %d.", attempt + 1)
                        # No need to break here, as the outer summary handles empty list.
                        # The transaction will simply commit nothing.
                        successfully_inserted_count = 0 # Ensure it's reset for this path
                        production_claim_records = [] # Ensure it's reset
                        break # Break retry loop, as no claims means nothing to retry for fetch.

                    production_claim_records = self._map_staging_to_production_records(staging_claims_to_transfer)

                    if not production_claim_records:
                        logger.warn("Mapping resulted in zero production records on attempt %d.", attempt + 1,
                                    num_staging_claims=len(staging_claims_to_transfer))
                        successfully_inserted_count = 0 # Ensure reset
                        break # Break retry loop, mapping failure is not a DB operational error.

                    # Replace _bulk_insert_to_production with _upsert_to_production_with_copy
                    successfully_inserted_count = await self._upsert_to_production_with_copy(production_claim_records)

                    if successfully_inserted_count > 0:
                        if successfully_inserted_count == len(production_claim_records):
                            await self._update_staging_claims_after_transfer(staging_claims_to_transfer)
                        else:
                            logger.warn(
                                f"Attempt {attempt + 1}: Mismatch in upserted count ({successfully_inserted_count}) vs "
                                f"mapped count ({len(production_claim_records)}). "
                                "Staging records not updated to prevent inconsistency. This may require manual review."
                            )
                            # This is a partial success, but the transaction will commit what was done.
                            # Further partial inserts are complex to handle robustly without more info.
                            # For now, we commit this partial success and log the warning.
                    elif len(production_claim_records) > 0:
                        logger.warn("Attempt %d: No records were upserted into production (upsert step returned 0). Staging records not updated.", attempt + 1)

                logger.info(f"Attempt {attempt + 1}: Transaction successful.")
                break # Success, exit retry loop

            except OperationalError as oe:
                logger.warn(f"Attempt {attempt + 1} of {max_retries}: OperationalError during data transfer transaction. Retrying in {retry_delay}s...", error=str(oe))
                if attempt + 1 == max_retries:
                    logger.error("All attempts failed for data transfer due to OperationalError.", exc_info=True)
                    # Reset counts as the final transaction failed
                    successfully_inserted_count = 0
                    # staging_claims_to_transfer and production_claim_records will hold values from last failed attempt's try block
                    # This is okay for final summary context.
                    return {
                        "message": "Data transfer process failed after multiple retries.",
                        "selected_from_staging": len(staging_claims_to_transfer), # From last attempt
                        "mapped_to_production_format": len(production_claim_records), # From last attempt
                        "successfully_transferred": 0,
                        "error": str(oe)
                    }
                await asyncio.sleep(retry_delay)
            except Exception as e:
                logger.error(f"Critical non-retryable error during data transfer transaction: {e}", exc_info=True)
                successfully_inserted_count = 0
                return {
                    "message": "Data transfer process failed due to a non-retryable critical error.",
                    "selected_from_staging": len(staging_claims_to_transfer),
                    "mapped_to_production_format": len(production_claim_records),
                    "successfully_transferred": 0,
                    "error": str(e)
                }

        # Final logging and return based on the outcome of the loop
        if not staging_claims_to_transfer and successfully_inserted_count == 0: # Handles case where _select_claims_from_staging was empty
             logger.info("No claims found in staging ready for transfer.")
             return {"message": "No claims to transfer.", "transferred_count": 0, "selected_from_staging": 0}

        logger.info(
            "Data transfer to production finished.",
            selected_from_staging=len(staging_claims_to_transfer),
            mapped_to_production_format=len(production_claim_records),
            successfully_transferred_to_prod_db=successfully_inserted_count
        )
        return {
            "message": "Data transfer process finished.",
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
                "ml_model_version_used": stag_claim.ml_model_version_used, # New field
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

    def _format_data_for_copy(
        self,
        production_records: List[Dict[str, Any]],
        column_names: List[str]
    ) -> List[tuple]:
        """
        Formats a list of dictionaries into a list of tuples for asyncpg's copy_records_to_table.
        The order of values in each tuple must match the order of column_names.
        """
        if not production_records:
            return []

        formatted_records = []
        for record_dict in production_records:
            # Ensure all column_names are present in the dict, or handle missing ones (e.g. default to None)
            # For robustness, get with default None if a key might be missing, though mapped records should be consistent.
            record_tuple = tuple(record_dict.get(col_name) for col_name in column_names)
            formatted_records.append(record_tuple)

        # Log a sample for verification if needed, but be careful with PII if any
        # if formatted_records:
        #     logger.debug("Sample record formatted for COPY", sample_tuple=formatted_records[0], column_order=column_names)

        return formatted_records

    async def _upsert_to_production_with_copy(
        self,
        production_records: List[Dict[str, Any]]
    ) -> int:
        """
        Upserts records into claims_production using a temporary table and COPY,
        followed by an INSERT ... ON CONFLICT ... SELECT statement.
        This method assumes it is called within an existing SQLAlchemy session transaction
        started by the caller (e.g. process_pending_claims_batch's retry loop).
        """
        if not production_records:
            logger.info("No production records to upsert with COPY.")
            return 0

        # Column order for COPY and INSERT...SELECT
        # Must match ClaimsProductionModel and the output of _map_staging_to_production_records
        column_names = [
            "id", "claim_id", "facility_id", "patient_account_number",
            "patient_first_name", "patient_last_name", "patient_date_of_birth",
            "service_from_date", "service_to_date", "total_charges",
            "ml_prediction_score", "risk_category", "processing_duration_ms",
            "throughput_achieved", "ml_model_version_used"
            # created_at and updated_at are handled by server_default or ON UPDATE triggers in ClaimsProductionModel
        ]

        temp_table_name = f"temp_claims_upsert_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"


        try:
            # Get raw asyncpg connection. The main transaction is managed by the caller.
            async_sqla_conn = await self.db.get_bind() # get_bind() gives the Engine or Connection
            raw_pg_conn = await async_sqla_conn.get_raw_connection()


            create_temp_table_sql = f"""
            CREATE TEMP TABLE {temp_table_name} (
                LIKE claims_production INCLUDING DEFAULTS
            ) ON COMMIT DROP;
            """
            await raw_pg_conn.execute(create_temp_table_sql)
            logger.debug(f"Temporary table {temp_table_name} created for COPY operation.")

            formatted_records_for_copy = self._format_data_for_copy(production_records, column_names)
            if not formatted_records_for_copy:
                logger.info("No records to COPY after formatting.")
                # No need to close raw_pg_conn here, it's managed by SQLAlchemy session
                return 0

            await raw_pg_conn.copy_records_to_table(
                temp_table_name,
                records=formatted_records_for_copy,
                columns=column_names,
                timeout=60
            )
            logger.debug(f"Successfully copied {len(formatted_records_for_copy)} records to {temp_table_name}.")

            insert_cols_str = ", ".join(f'"{col}"' for col in column_names) # Quote column names
            update_set_parts = []
            for col in column_names:
                if col != "id": # 'id' is the conflict target
                    update_set_parts.append(f'"{col}" = EXCLUDED."{col}"')

            # Add explicit updated_at for ON CONFLICT case
            update_set_parts.append('"updated_at" = NOW()')
            update_set_str = ", ".join(update_set_parts)

            upsert_sql = f"""
            INSERT INTO claims_production ({insert_cols_str})
            SELECT {insert_cols_str} FROM {temp_table_name}
            ON CONFLICT (id) DO UPDATE
            SET {update_set_str};
            """
            status_message = await raw_pg_conn.execute(upsert_sql)
            logger.debug(f"Upsert from temp table completed. Status: {status_message}")

            affected_rows = 0
            if status_message:
                parts = status_message.split()
                if len(parts) > 0: # Typically "INSERT 0 N" or "UPDATE N"
                    try:
                        # For INSERT ... ON CONFLICT, status is "INSERT oid rows"
                        # where rows is the count of rows inserted OR updated.
                        affected_rows = int(parts[-1])
                    except ValueError:
                        logger.warn(f"Could not parse row count from status_message: {status_message}")

            # Temporary table is dropped ON COMMIT.
            # No need to close raw_pg_conn, SQLAlchemy session handles the underlying connection lifecycle.
            return affected_rows

        except Exception as e:
            logger.error(f"Error during COPY-based upsert to production: {e}", exc_info=True)
            # No rollback here, as the method assumes it's part of a larger transaction
            # managed by the caller (process_pending_claims_batch's retry loop)
            raise

    # _bulk_insert_to_production method removed as it's superseded by _upsert_to_production_with_copy
    # and its only call site was updated.

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
            # await self.db.commit() # Removed: Commit managed by caller

            logger.info(f"Staging claims update statement executed. Rows matched: {result.rowcount if result else 'N/A'}. Claim IDs: {claim_ids_to_update}")
            # The actual commit will happen in the calling method's transaction.

        except Exception as e:
            # await self.db.rollback() # Removed: Rollback managed by caller
            logger.error(f"Failed to execute update for staging claims after transfer: {e}", exc_info=True)
            raise # Re-raise to trigger rollback in the calling transaction
