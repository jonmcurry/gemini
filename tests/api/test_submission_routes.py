import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from typing import List, Dict, Any, Optional

# Assuming main.py creates the FastAPI app instance
# The path to main.app needs to be correct relative to where pytest is run from
# If tests are run from project root, this should work:
from claims_processor.src.main import app
from claims_processor.src.api.models.ingestion_models import IngestionClaim, IngestionClaimLineItem # For request body
from claims_processor.src.ingestion.data_ingestion_service import DataIngestionService
from claims_processor.src.core.security.audit_logger_service import AuditLoggerService

client = TestClient(app)

# --- Mocked Service Fixtures ---
@pytest.fixture
def mock_data_ingestion_service() -> AsyncMock:
    # Using AsyncMock for the service itself if its methods are async
    mock_service = AsyncMock(spec=DataIngestionService)
    # Configure common return values for its methods here or in each test
    mock_service.ingest_claims_batch = AsyncMock()
    return mock_service

@pytest.fixture
def mock_audit_logger_service() -> AsyncMock:
    mock_service = AsyncMock(spec=AuditLoggerService)
    mock_service.log_event = AsyncMock()
    return mock_service

# --- Test Cases ---
def test_submit_claims_batch_success(
    mock_data_ingestion_service: AsyncMock,
    mock_audit_logger_service: AsyncMock
):
    # Patch the temporary factory functions in submission_routes to return our mocks
    with patch('claims_processor.src.api.routes.submission_routes.temp_get_data_ingestion_service', return_value=mock_data_ingestion_service), \
         patch('claims_processor.src.api.routes.submission_routes.temp_get_audit_logger_service', return_value=mock_audit_logger_service):

        sample_claim_data = {
            "claim_id": "CLAIM001", "facility_id": "FAC001", "patient_account_number": "PAT001",
            "service_from_date": "2023-01-01", "service_to_date": "2023-01-02",
            "total_charges": 150.75,
            "line_items": [{
                "line_number": 1, "service_date": "2023-01-01", "procedure_code": "P123",
                "units": 1, "charge_amount": 150.75
            }]
        }
        # Mock the return value of the ingestion service
        mock_ingestion_summary = {
            "ingestion_batch_id": "test_batch_success_id", "received_claims": 1,
            "successfully_staged_claims": 1, "failed_ingestion_claims": 0, "errors": []
        }
        mock_data_ingestion_service.ingest_claims_batch.return_value = mock_ingestion_summary

        response = client.post("/api/v1/submissions/claims_batch?ingestion_batch_id=test_batch_success_id", json=[sample_claim_data])

        assert response.status_code == 202
        assert response.json() == mock_ingestion_summary

        mock_data_ingestion_service.ingest_claims_batch.assert_called_once()
        # Can add more detailed check on args passed to ingest_claims_batch if needed

        # Audit log should be called once after successful call to DataIngestionService
        assert mock_audit_logger_service.log_event.call_count == 1
        audit_call_args = mock_audit_logger_service.log_event.call_args_list[0].kwargs # Get the first (and only) call
        assert audit_call_args['action'] == "SUBMIT_CLAIMS_BATCH_API"
        assert audit_call_args['success'] is True # API call itself was successful in processing the request
        assert audit_call_args['details'] == mock_ingestion_summary


def test_submit_claims_batch_empty_payload(
    mock_data_ingestion_service: AsyncMock,
    mock_audit_logger_service: AsyncMock
):
    with patch('claims_processor.src.api.routes.submission_routes.temp_get_data_ingestion_service', return_value=mock_data_ingestion_service), \
         patch('claims_processor.src.api.routes.submission_routes.temp_get_audit_logger_service', return_value=mock_audit_logger_service):

        response = client.post("/api/v1/submissions/claims_batch", json=[])

        assert response.status_code == 400
        assert "No claims provided in the batch" in response.json()["detail"]

        mock_data_ingestion_service.ingest_claims_batch.assert_not_called()
        # Audit log for the failed API attempt
        mock_audit_logger_service.log_event.assert_called_once()
        audit_call_args = mock_audit_logger_service.log_event.call_args.kwargs
        assert audit_call_args['action'] == "SUBMIT_CLAIMS_BATCH_API"
        assert audit_call_args['success'] is False
        assert audit_call_args['failure_reason'] == "No claims provided in batch."


def test_submit_claims_batch_pydantic_validation_error():
    # No need to mock services, FastAPI/Pydantic handles this before route code
    invalid_payload = [{
        "claim_id": "CLAIM002", # Missing other required fields like facility_id, etc.
        "line_items": [{"line_number": "not_an_int"}] # Invalid type for line_number
    }]
    response = client.post("/api/v1/submissions/claims_batch", json=invalid_payload)
    assert response.status_code == 422 # Unprocessable Entity

def test_submit_claims_batch_ingestion_service_exception(
    mock_data_ingestion_service: AsyncMock,
    mock_audit_logger_service: AsyncMock
):
    with patch('claims_processor.src.api.routes.submission_routes.temp_get_data_ingestion_service', return_value=mock_data_ingestion_service), \
         patch('claims_processor.src.api.routes.submission_routes.temp_get_audit_logger_service', return_value=mock_audit_logger_service):

        sample_claim_data = { # Valid structure, but service will fail
            "claim_id": "CLAIM_FAIL", "facility_id": "FAC_FAIL", "patient_account_number": "PAT_FAIL",
            "service_from_date": "2023-02-01", "service_to_date": "2023-02-02",
            "total_charges": 200.00,
            "line_items": [{"line_number": 1, "service_date": "2023-02-01", "procedure_code": "P456", "units": 1, "charge_amount": 200.00}]
        }

        # Mock the ingestion service to raise an unexpected error
        mock_data_ingestion_service.ingest_claims_batch.side_effect = Exception("Simulated service error")

        response = client.post("/api/v1/submissions/claims_batch", json=[sample_claim_data])

        assert response.status_code == 500
        assert "Internal server error during claim submission: Simulated service error" in response.json()["detail"]

        mock_data_ingestion_service.ingest_claims_batch.assert_called_once()

        # Audit log for the failed API attempt (critical error)
        mock_audit_logger_service.log_event.assert_called_once() # Only one call in this case (the critical error one)
        audit_call_args = mock_audit_logger_service.log_event.call_args.kwargs
        assert audit_call_args['action'] == "SUBMIT_CLAIMS_BATCH_API"
        assert audit_call_args['success'] is False
        assert "Internal server error: Simulated service error" in audit_call_args['failure_reason']
```
