import structlog
from typing import List, Optional, Dict, Any, Callable
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import uuid4

from ..api.models.ingestion_models import IngestionClaim # IngestionClaimLineItem is used by IngestionClaim
from ..core.database.models.claims_db import ClaimModel, ClaimLineItemModel
from ..core.security.encryption_service import EncryptionService # For encrypting PII

logger = structlog.get_logger(__name__)

class DataIngestionService:
    """
    Service for ingesting raw claims data and saving it to staging tables.
    """

    def __init__(self,
                 db_session_factory: Callable[[], AsyncSession],
                 encryption_service: EncryptionService):
        """
        Initializes the DataIngestionService.

        Args:
            db_session_factory: An asynchronous factory function that provides an AsyncSession context manager.
            encryption_service: Service for encrypting PII data.
        """
        self.db_session_factory = db_session_factory
        self.encryption_service = encryption_service
        logger.info("DataIngestionService initialized with EncryptionService.")

    async def ingest_claims_batch(
        self,
        raw_claims_data: List[IngestionClaim],
        provided_batch_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ingests a batch of raw claims data, encrypts PII, maps to DB models,
        and saves it to the staging database with 'pending' status.
        Assumes ClaimModel.id and ClaimLineItemModel.id are auto-incrementing.

        Args:
            raw_claims_data: A list of IngestionClaim Pydantic models.
            provided_batch_id: An optional batch identifier for this ingestion.

        Returns:
            A dictionary summarizing the ingestion result.
        """
        received_claims_count = len(raw_claims_data)
        successfully_staged_count = 0
        failed_ingestion_count = 0 # Counts claims that failed mapping/encryption before DB attempt
        errors_detail: List[Dict[str, Any]] = []

        if not raw_claims_data:
            logger.info("No raw claims data provided for ingestion.")
            return {
                "ingestion_batch_id": provided_batch_id,
                "received_claims": 0,
                "successfully_staged_claims": 0,
                "failed_ingestion_claims": 0,
                "errors": [{"error_type": "NoDataError", "error_message": "No data provided"}]
            }

        current_ingestion_batch_id = provided_batch_id if provided_batch_id else str(uuid4())
        logger.info(f"Starting ingestion for batch_id: {current_ingestion_batch_id}, number of claims received: {received_claims_count}")

        claims_to_add_to_db: List[ClaimModel] = []

        for idx, ingestion_claim in enumerate(raw_claims_data):
            claim_business_id = ingestion_claim.claim_id or f"at_index_{idx}"
            try:
                encrypted_dob_str: Optional[str] = None
                if ingestion_claim.patient_date_of_birth:
                    dob_iso_str = ingestion_claim.patient_date_of_birth.isoformat()
                    encrypted_dob_str = self.encryption_service.encrypt(dob_iso_str)
                    if not encrypted_dob_str:
                        # Log and raise to stop processing this claim, mark as failed
                        logger.error("Failed to encrypt date of birth", claim_id=claim_business_id)
                        raise ValueError("Date of birth encryption failed.")

                encrypted_mrn_str: Optional[str] = None
                if ingestion_claim.medical_record_number:
                    encrypted_mrn_str = self.encryption_service.encrypt(ingestion_claim.medical_record_number)
                    if not encrypted_mrn_str:
                        logger.error("Failed to encrypt medical record number", claim_id=claim_business_id)
                        raise ValueError("Medical record number encryption failed.")

                db_claim = ClaimModel(
                    # ClaimModel.id is auto-incrementing, not set here.
                    claim_id=ingestion_claim.claim_id,
                    facility_id=ingestion_claim.facility_id,
                    patient_account_number=ingestion_claim.patient_account_number,

                    medical_record_number=encrypted_mrn_str,
                    patient_first_name=ingestion_claim.patient_first_name,
                    patient_last_name=ingestion_claim.patient_last_name,
                    patient_date_of_birth=encrypted_dob_str, # Storing encrypted string

                    financial_class=ingestion_claim.financial_class,
                    insurance_type=ingestion_claim.insurance_type,
                    insurance_plan_id=ingestion_claim.insurance_plan_id,

                    service_from_date=ingestion_claim.service_from_date,
                    service_to_date=ingestion_claim.service_to_date,
                    total_charges=ingestion_claim.total_charges,

                    processing_status='pending',
                    batch_id=ingestion_claim.ingestion_batch_id or current_ingestion_batch_id,
                    # created_at, updated_at have server_default in DB model
                )

                db_claim.line_items = []
                for line_item_ingest in ingestion_claim.line_items:
                    db_line_item = ClaimLineItemModel(
                        # ClaimLineItemModel.id is auto-incrementing
                        # claim_db_id will be set by SQLAlchemy relationship when db_claim is added to session and flushed
                        line_number=line_item_ingest.line_number,
                        service_date=line_item_ingest.service_date,
                        procedure_code=line_item_ingest.procedure_code,
                        units=line_item_ingest.units,
                        charge_amount=line_item_ingest.charge_amount,
                        procedure_description=line_item_ingest.procedure_description,
                        rendering_provider_npi=line_item_ingest.rendering_provider_npi
                        # rvu_total is nullable, processed later
                        # created_at, updated_at have server_default
                    )
                    db_claim.line_items.append(db_line_item)

                claims_to_add_to_db.append(db_claim)
            except Exception as e:
                error_message = f"Error processing raw claim '{claim_business_id}': {str(e)}"
                logger.error(error_message, batch_id=current_ingestion_batch_id, exc_info=True)
                failed_ingestion_count += 1
                errors_detail.append({"claim_id": claim_business_id, "error": str(e), "error_type": type(e).__name__})

        db_operation_failed_for_all_mapped = False
        if claims_to_add_to_db:
            try:
                async with self.db_session_factory() as session: # type: AsyncSession
                    async with session.begin():
                        session.add_all(claims_to_add_to_db)
                    # Commit is implicit with session.begin() context manager success
                successfully_staged_count = len(claims_to_add_to_db)
                logger.info(f"Successfully staged {successfully_staged_count} claims for ingestion_batch_id: {current_ingestion_batch_id}.")
            except Exception as e:
                db_operation_failed_for_all_mapped = True
                error_message = f"Database error during bulk ingestion for batch_id '{current_ingestion_batch_id}': {str(e)}"
                logger.error(error_message, exc_info=True)
                # All claims in this specific DB transaction attempt are considered failed
                failed_ingestion_count += len(claims_to_add_to_db) # Add these to ones that failed prior to DB
                successfully_staged_count = 0
                errors_detail.append({"batch_db_error": str(e), "error_type": type(e).__name__, "num_claims_in_failed_db_batch": len(claims_to_add_to_db)})
        elif received_claims_count > 0 : # No claims made it to DB stage, but some were received
             logger.warn(f"No claims were successfully mapped/encrypted for DB insertion from batch {current_ingestion_batch_id}.")
             if not errors_detail: # Should ideally have errors if all failed before DB
                 errors_detail.append({"batch_error": "All claims failed before DB stage, but no specific errors captured per claim.", "error_type": "UnknownPreDBError"})


        # Final reconciliation of failed_ingestion_count if DB op failed for all mapped claims
        if db_operation_failed_for_all_mapped:
             # If DB op failed, all successfully mapped claims become failed.
             # The failed_ingestion_count should reflect total received if DB op fails for all that made it that far.
             # The ones that failed before mapping (failed_ingestion_count) are already counted.
             # The ones that were mapped but failed DB are len(claims_to_add_to_db).
             # So total failed = failed_ingestion_count (pre-DB) + len(claims_to_add_to_db) (failed DB)
             # This is already correctly calculated by `failed_ingestion_count += len(claims_to_add_to_db)`
             pass


        return {
            "ingestion_batch_id": current_ingestion_batch_id,
            "received_claims": received_claims_count,
            "successfully_staged_claims": successfully_staged_count,
            "failed_ingestion_claims": failed_ingestion_count, # Total claims that could not be staged
            "errors": errors_detail
        }
```
