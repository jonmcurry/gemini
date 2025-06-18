from fastapi.testing import TestClient
from claims_processor.src.main import app # Assuming main.py is in src
import pytest # It's good practice to use pytest for more features, though not strictly required for TestClient

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data

def test_submit_claim():
    sample_claim = {
        "claim_id": "C001",
        "facility_id": "F001",
        "patient_account_number": "P001",
        "service_from_date": "2023-01-15",
        "service_to_date": "2023-01-20",
        "total_charges": 1500.75
    }
    response = client.post("/api/v1/claims/", json=sample_claim) # Added trailing slash to match router
    assert response.status_code == 200
    assert response.json() == {"message": "Claim received successfully", "claim_id": "C001"}

# Basic test to check if pytest is running (can be removed later)
def test_example():
    assert 1 == 1
