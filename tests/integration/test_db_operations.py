import pytest
from fastapi import status # For status codes
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from datetime import datetime, date, timedelta, timezone
from decimal import Decimal

# Models and Services needed for tests
from claims_processor.src.core.database.models.claims_db import ClaimModel, ClaimLineItemModel
from claims_processor.src.api.models.claim_models import ClaimResponse # Can be used if checking API response structure for single claim
from claims_processor.src.processing.rvu_service import MOCK_RVU_DATA # For verifying RVU calculations
from claims_processor.src.api.routes.claims_routes import run_batch_processing_background # Function to test directly

@pytest.mark.asyncio
async def test_create_and_retrieve_claim(client, db_session: AsyncSession):
    sample_claim_data = {
        "claim_id": "DBTEST001",
        "facility_id": "FAC002",
        "patient_account_number": "PAT002",
        "service_from_date": "2023-03-01",
        "service_to_date": "2023-03-05",
        "total_charges": 250.75,
        "patient_first_name": "DBTest",
        "patient_last_name": "User"
    }
    response = client.post("/api/v1/claims/", json=sample_claim_data)
    assert response.status_code == status.HTTP_200_OK, response.text
    response_data = response.json()
    assert response_data["claim_id"] == sample_claim_data["claim_id"]
    db_id = response_data["id"]

    stmt = select(ClaimModel).where(ClaimModel.id == db_id)
    result = await db_session.execute(stmt)
    retrieved_claim_db = result.scalars().first()
    assert retrieved_claim_db is not None
    assert retrieved_claim_db.claim_id == sample_claim_data["claim_id"]

    duplicate_response = client.post("/api/v1/claims/", json=sample_claim_data)
    assert duplicate_response.status_code == status.HTTP_409_CONFLICT, duplicate_response.text

@pytest.mark.asyncio
async def test_health_check_separate(client):
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == "healthy"

# --- Tests for Batch Processing Logic (Directly calling the background task function) ---

@pytest.mark.asyncio
async def test_process_batch_no_pending_claims_direct(db_session: AsyncSession):
    update_stmt = update(ClaimModel).where(ClaimModel.processing_status == 'pending').values(processing_status='processed_test_setup')
    await db_session.execute(update_stmt)
    await db_session.commit()

    result = await run_batch_processing_background(batch_size=10)

    assert result["message"] == "No pending claims to process."
    assert result["attempted_claims"] == 0
    assert result.get("conversion_errors", 0) == 0
    assert result.get("validation_failures", 0) == 0
    assert result.get("rvu_calculation_failures", 0) == 0
    assert result.get("successfully_processed_count", 0) == 0
    assert result.get("other_exceptions", 0) == 0

@pytest.mark.asyncio
async def test_process_batch_valid_claims_direct(db_session: AsyncSession):
    claim_data_1 = {
        "claim_id": "BATCH_VALID_001", "facility_id": "F_BATCH", "patient_account_number": "P_BATCH_001",
        "service_from_date": date(2023, 4, 1), "service_to_date": date(2023, 4, 2),
        "total_charges": Decimal("200.00"), "processing_status": "pending",
        "created_at": datetime.now(timezone.utc) - timedelta(minutes=10)
    }
    claim_1_line_1 = ClaimLineItemModel(line_number=1, service_date=date(2023,4,1), procedure_code="99213", units=1, charge_amount=Decimal("200.00"))
    db_claim_1 = ClaimModel(**claim_data_1, line_items=[claim_1_line_1])

    claim_data_2 = {
        "claim_id": "BATCH_VALID_002", "facility_id": "F_BATCH", "patient_account_number": "P_BATCH_002",
        "service_from_date": date(2023, 4, 3), "service_to_date": date(2023, 4, 4),
        "total_charges": Decimal("350.00"), "processing_status": "pending",
        "created_at": datetime.now(timezone.utc) - timedelta(minutes=5)
    }
    claim_2_line_1 = ClaimLineItemModel(line_number=1, service_date=date(2023,4,3), procedure_code="99214", units=1, charge_amount=Decimal("350.00"))
    db_claim_2 = ClaimModel(**claim_data_2, line_items=[claim_2_line_1])

    db_session.add_all([db_claim_1, db_claim_2])
    await db_session.commit()
    claim_1_id = db_claim_1.id
    claim_2_id = db_claim_2.id

    result = await run_batch_processing_background(batch_size=5)

    assert result["attempted_claims"] == 2
    assert result["conversion_errors"] == 0
    assert result["validation_failures"] == 0
    assert result["rvu_calculation_failures"] == 0
    assert result["successfully_processed_count"] == 2
    assert result["other_exceptions"] == 0

    refreshed_claim_1 = await db_session.get(ClaimModel, claim_1_id)
    refreshed_claim_2 = await db_session.get(ClaimModel, claim_2_id)

    assert refreshed_claim_1.processing_status == "processing_complete"
    assert refreshed_claim_1.line_items[0].rvu_total == MOCK_RVU_DATA["99213"] * Decimal("1")
    assert refreshed_claim_1.processed_at is not None

    assert refreshed_claim_2.processing_status == "processing_complete"
    assert refreshed_claim_2.line_items[0].rvu_total == MOCK_RVU_DATA["99214"] * Decimal("1")
    assert refreshed_claim_2.processed_at is not None

@pytest.mark.asyncio
async def test_process_batch_invalid_claim_direct(db_session: AsyncSession):
    invalid_claim_data = {
        "claim_id": "BATCH_INVALID_001", "facility_id": "F_BATCH_INV", "patient_account_number": "P_BATCH_INV_001",
        "service_from_date": date(2023, 5, 5), "service_to_date": date(2023, 5, 1), # Invalid date order
        "total_charges": Decimal("100.00"), "processing_status": "pending"
    } # No line items - also an error
    db_invalid_claim = ClaimModel(**invalid_claim_data, line_items=[])
    db_session.add(db_invalid_claim)
    await db_session.commit()
    invalid_claim_id = db_invalid_claim.id

    result = await run_batch_processing_background(batch_size=5)

    assert result["attempted_claims"] == 1
    assert result["conversion_errors"] == 0
    assert result["validation_failures"] == 1
    assert result["rvu_calculation_failures"] == 0
    assert result["successfully_processed_count"] == 0
    assert result["other_exceptions"] == 0

    refreshed_invalid_claim = await db_session.get(ClaimModel, invalid_claim_id)
    assert refreshed_invalid_claim.processing_status == "validation_failed"
    assert refreshed_invalid_claim.processed_at is None

@pytest.mark.asyncio
async def test_process_batch_mixed_claims_direct(db_session: AsyncSession):
    valid_claim_data = {
        "claim_id": "BATCH_MIXED_VALID", "facility_id": "F_MIXED", "patient_account_number": "P_MIXED_V",
        "service_from_date": date(2023, 6, 1), "service_to_date": date(2023, 6, 2),
        "total_charges": Decimal("50.00"), "processing_status": "pending"
    }
    valid_line = ClaimLineItemModel(line_number=1, service_date=date(2023,6,1), procedure_code="80053", units=1, charge_amount=Decimal("50.00"))
    db_valid_claim = ClaimModel(**valid_claim_data, line_items=[valid_line])

    invalid_claim_data = { # Invalid: total_charges must be positive
        "claim_id": "BATCH_MIXED_INVALID", "facility_id": "F_MIXED", "patient_account_number": "P_MIXED_I",
        "service_from_date": date(2023, 6, 3), "service_to_date": date(2023, 6, 4),
        "total_charges": Decimal("0.00"), "processing_status": "pending"
    }
    invalid_line = ClaimLineItemModel(line_number=1, service_date=date(2023,6,3), procedure_code="99213", units=1, charge_amount=Decimal("0.00"))
    db_invalid_claim = ClaimModel(**invalid_claim_data, line_items=[invalid_line])

    db_session.add_all([db_valid_claim, db_invalid_claim])
    await db_session.commit()
    valid_claim_id = db_valid_claim.id
    invalid_claim_id = db_invalid_claim.id

    result = await run_batch_processing_background(batch_size=5)

    assert result["attempted_claims"] == 2
    assert result["conversion_errors"] == 0
    assert result["validation_failures"] == 1
    assert result["rvu_calculation_failures"] == 0
    assert result["successfully_processed_count"] == 1
    assert result["other_exceptions"] == 0

    refreshed_valid_claim = await db_session.get(ClaimModel, valid_claim_id)
    refreshed_invalid_claim = await db_session.get(ClaimModel, invalid_claim_id)

    assert refreshed_valid_claim.processing_status == "processing_complete"
    assert refreshed_invalid_claim.processing_status == "validation_failed"

# --- Test for the API endpoint itself (light check) ---
@pytest.mark.asyncio
async def test_trigger_batch_api_response(client): # Uses the http client
    response = client.post("/api/v1/claims/process-batch/", params={"batch_size": 5})
    assert response.status_code == status.HTTP_200_OK, response.text
    response_data = response.json()
    assert response_data["message"] == "Batch processing started in the background."
    assert response_data["batch_size"] == 5
    # This test does NOT check for completion or outcome of the background task itself.
    # That is covered by the _direct tests above.
