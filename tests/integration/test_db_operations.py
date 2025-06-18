import pytest
from httpx import AsyncClient # Not used here, TestClient is sync but calls async code
from fastapi import status

from claims_processor.src.api.models.claim_models import ClaimResponse # To validate response
from claims_processor.src.core.database.models.claims_db import ClaimModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update # Added update
from datetime import datetime, date, timedelta, timezone # ensure datetime components are imported
from decimal import Decimal # ensure Decimal is imported
import pytest # ensure pytest is imported

# Import MOCK_RVU_DATA for verification
from claims_processor.src.processing.rvu_service import MOCK_RVU_DATA


@pytest.mark.asyncio
async def test_create_and_retrieve_claim(client, db_session: AsyncSession): # client fixture from conftest
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

    # 1. Create claim via API
    response = client.post("/api/v1/claims/", json=sample_claim_data)

    assert response.status_code == status.HTTP_200_OK, response.text
    response_data = response.json()
    assert response_data["claim_id"] == sample_claim_data["claim_id"]
    assert response_data["facility_id"] == sample_claim_data["facility_id"]
    assert response_data["total_charges"] == sample_claim_data["total_charges"]
    assert response_data["patient_first_name"] == sample_claim_data["patient_first_name"]
    assert "id" in response_data
    db_id = response_data["id"]

    # 2. Verify in database directly (using the test session)
    stmt = select(ClaimModel).where(ClaimModel.id == db_id)
    result = await db_session.execute(stmt)
    retrieved_claim_db = result.scalars().first()

    assert retrieved_claim_db is not None
    assert retrieved_claim_db.claim_id == sample_claim_data["claim_id"]
    assert retrieved_claim_db.facility_id == sample_claim_data["facility_id"]
    assert float(retrieved_claim_db.total_charges) == sample_claim_data["total_charges"]
    assert retrieved_claim_db.patient_first_name == sample_claim_data["patient_first_name"]

    # 3. Test duplicate claim_id submission
    duplicate_response = client.post("/api/v1/claims/", json=sample_claim_data)
    assert duplicate_response.status_code == status.HTTP_409_CONFLICT, duplicate_response.text

@pytest.mark.asyncio
async def test_health_check_separate(client):
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == "healthy"

# New tests for batch processing endpoint

@pytest.mark.asyncio
async def test_process_batch_api_no_pending_claims(client, db_session: AsyncSession):
    # Explicitly update any existing claims to a non-pending status
    # Using sqlalchemy.update for broader effect, though specific test DBs should be clean.
    update_stmt = update(ClaimModel).where(ClaimModel.processing_status == 'pending').values(processing_status='processed_test_setup')
    await db_session.execute(update_stmt)
    await db_session.commit()

    response = client.post("/api/v1/claims/process-batch/", params={"batch_size": 10})
    assert response.status_code == status.HTTP_200_OK, response.text
    result = response.json()
    assert result["message"] == "No pending claims to process."
    assert result["attempted_claims"] == 0

@pytest.mark.asyncio
async def test_process_batch_api_valid_claims(client, db_session: AsyncSession):
    # 1. Seed database with a few valid pending claims
    claim_data_1 = {
        "claim_id": "BATCH_VALID_001", "facility_id": "F_BATCH", "patient_account_number": "P_BATCH_001",
        "service_from_date": date(2023, 4, 1), "service_to_date": date(2023, 4, 2),
        "total_charges": Decimal("200.00"), "processing_status": "pending",
        "created_at": datetime.now(timezone.utc) - timedelta(minutes=10) # Older
    }
    claim_1_line_1 = ClaimLineItemModel(line_number=1, service_date=date(2023,4,1), procedure_code="99213", units=1, charge_amount=Decimal("200.00"))
    db_claim_1 = ClaimModel(**claim_data_1, line_items=[claim_1_line_1])

    claim_data_2 = {
        "claim_id": "BATCH_VALID_002", "facility_id": "F_BATCH", "patient_account_number": "P_BATCH_002",
        "service_from_date": date(2023, 4, 3), "service_to_date": date(2023, 4, 4),
        "total_charges": Decimal("350.00"), "processing_status": "pending",
        "created_at": datetime.now(timezone.utc) - timedelta(minutes=5) # Newer
    }
    claim_2_line_1 = ClaimLineItemModel(line_number=1, service_date=date(2023,4,3), procedure_code="99214", units=1, charge_amount=Decimal("350.00"))
    db_claim_2 = ClaimModel(**claim_data_2, line_items=[claim_2_line_1])

    db_session.add_all([db_claim_1, db_claim_2])
    await db_session.commit()
    claim_1_id = db_claim_1.id # Ensure IDs are captured after commit if not before
    claim_2_id = db_claim_2.id


    # 2. Call the batch processing API
    response = client.post("/api/v1/claims/process-batch/", params={"batch_size": 5})
    assert response.status_code == status.HTTP_200_OK, response.text
    result = response.json()
    assert result["attempted_claims"] == 2
    assert result["validated_count"] == 2
    assert result["failed_validation_count"] == 0
    assert result["successfully_processed_count"] == 2

    # 3. Verify claims in DB by re-fetching or refreshing
    # Refreshing existing instances from the session
    refreshed_claim_1 = await db_session.get(ClaimModel, claim_1_id)
    refreshed_claim_2 = await db_session.get(ClaimModel, claim_2_id)

    assert refreshed_claim_1 is not None
    assert refreshed_claim_1.processing_status == "processing_complete"
    assert len(refreshed_claim_1.line_items) == 1 # Ensure line items are loaded/available
    assert refreshed_claim_1.line_items[0].rvu_total is not None
    assert refreshed_claim_1.line_items[0].rvu_total == MOCK_RVU_DATA["99213"] * Decimal("1")
    assert refreshed_claim_1.processed_at is not None

    assert refreshed_claim_2 is not None
    assert refreshed_claim_2.processing_status == "processing_complete"
    assert len(refreshed_claim_2.line_items) == 1
    assert refreshed_claim_2.line_items[0].rvu_total is not None
    assert refreshed_claim_2.line_items[0].rvu_total == MOCK_RVU_DATA["99214"] * Decimal("1")
    assert refreshed_claim_2.processed_at is not None


@pytest.mark.asyncio
async def test_process_batch_api_invalid_claim(client, db_session: AsyncSession):
    invalid_claim_data = {
        "claim_id": "BATCH_INVALID_001", "facility_id": "F_BATCH_INV", "patient_account_number": "P_BATCH_INV_001",
        "service_from_date": date(2023, 5, 5),
        "service_to_date": date(2023, 5, 1), # Invalid: from_date after to_date
        "total_charges": Decimal("100.00"), "processing_status": "pending"
    }
    db_invalid_claim = ClaimModel(**invalid_claim_data, line_items=[]) # No line items also an error
    db_session.add(db_invalid_claim)
    await db_session.commit()
    invalid_claim_id = db_invalid_claim.id

    response = client.post("/api/v1/claims/process-batch/", params={"batch_size": 5})
    assert response.status_code == status.HTTP_200_OK, response.text
    result = response.json()
    assert result["attempted_claims"] == 1
    assert result["validated_count"] == 0
    assert result["failed_validation_count"] == 1
    assert result["successfully_processed_count"] == 0

    refreshed_invalid_claim = await db_session.get(ClaimModel, invalid_claim_id)
    assert refreshed_invalid_claim.processing_status == "validation_failed"
    assert refreshed_invalid_claim.processed_at is None

@pytest.mark.asyncio
async def test_process_batch_api_mixed_claims(client, db_session: AsyncSession):
    valid_claim_data = {
        "claim_id": "BATCH_MIXED_VALID", "facility_id": "F_MIXED", "patient_account_number": "P_MIXED_V",
        "service_from_date": date(2023, 6, 1), "service_to_date": date(2023, 6, 2),
        "total_charges": Decimal("50.00"), "processing_status": "pending"
    }
    valid_line = ClaimLineItemModel(line_number=1, service_date=date(2023,6,1), procedure_code="80053", units=1, charge_amount=Decimal("50.00"))
    db_valid_claim = ClaimModel(**valid_claim_data, line_items=[valid_line])

    invalid_claim_data = {
        "claim_id": "BATCH_MIXED_INVALID", "facility_id": "F_MIXED", "patient_account_number": "P_MIXED_I",
        "service_from_date": date(2023, 6, 3), "service_to_date": date(2023, 6, 4),
        "total_charges": Decimal("0.00"), "processing_status": "pending" # Invalid: total_charges must be positive
    }
    # Add a line item to avoid "no line items" error, focusing on "total_charges" error
    invalid_line = ClaimLineItemModel(line_number=1, service_date=date(2023,6,3), procedure_code="99213", units=1, charge_amount=Decimal("0.00"))
    db_invalid_claim = ClaimModel(**invalid_claim_data, line_items=[invalid_line])

    db_session.add_all([db_valid_claim, db_invalid_claim])
    await db_session.commit()
    valid_claim_id = db_valid_claim.id
    invalid_claim_id = db_invalid_claim.id

    response = client.post("/api/v1/claims/process-batch/", params={"batch_size": 5})
    assert response.status_code == status.HTTP_200_OK, response.text
    result = response.json()
    assert result["attempted_claims"] == 2
    assert result["validated_count"] == 1
    assert result["failed_validation_count"] == 1
    assert result["successfully_processed_count"] == 1

    refreshed_valid_claim = await db_session.get(ClaimModel, valid_claim_id)
    refreshed_invalid_claim = await db_session.get(ClaimModel, invalid_claim_id)

    assert refreshed_valid_claim.processing_status == "processing_complete"
    assert len(refreshed_valid_claim.line_items) == 1
    assert refreshed_valid_claim.line_items[0].rvu_total is not None
    assert refreshed_invalid_claim.processing_status == "validation_failed"
