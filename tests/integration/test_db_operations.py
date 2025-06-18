import pytest
from httpx import AsyncClient # Not used here, TestClient is sync but calls async code
from fastapi import status

from claims_processor.src.api.models.claim_models import ClaimResponse # To validate response
from claims_processor.src.core.database.models.claims_db import ClaimModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select


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
        # "patient_date_of_birth": "1990-01-01" # Example if this field is added
    }

    # 1. Create claim via API
    response = client.post("/api/v1/claims/", json=sample_claim_data)

    assert response.status_code == status.HTTP_200_OK, response.text
    response_data = response.json()
    assert response_data["claim_id"] == sample_claim_data["claim_id"]
    assert response_data["facility_id"] == sample_claim_data["facility_id"]
    assert response_data["total_charges"] == sample_claim_data["total_charges"]
    assert response_data["patient_first_name"] == sample_claim_data["patient_first_name"] # Check added field
    assert "id" in response_data
    db_id = response_data["id"]

    # 2. Verify in database directly (using the test session)
    # Use await for async db_session operations
    stmt = select(ClaimModel).where(ClaimModel.id == db_id)
    result = await db_session.execute(stmt)
    retrieved_claim_db = result.scalars().first()

    assert retrieved_claim_db is not None
    assert retrieved_claim_db.claim_id == sample_claim_data["claim_id"]
    assert retrieved_claim_db.facility_id == sample_claim_data["facility_id"]
    assert float(retrieved_claim_db.total_charges) == sample_claim_data["total_charges"] # Cast Decimal to float for comparison
    assert retrieved_claim_db.patient_first_name == sample_claim_data["patient_first_name"]

    # 3. Test duplicate claim_id submission
    duplicate_response = client.post("/api/v1/claims/", json=sample_claim_data) # Using the same data
    assert duplicate_response.status_code == status.HTTP_409_CONFLICT, duplicate_response.text

    # 4. Test retrieval via a GET endpoint (if one exists) - Placeholder
    #    response_get = client.get(f"/api/v1/claims/{db_id}")
    #    assert response_get.status_code == status.HTTP_200_OK
    #    assert response_get.json()["claim_id"] == sample_claim_data["claim_id"]

@pytest.mark.asyncio
async def test_health_check_separate(client): # Example of another test using the client
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == "healthy"
