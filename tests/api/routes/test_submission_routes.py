import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
from decimal import Decimal
from datetime import date

from claims_processor.src.main import app # Main FastAPI application instance
from claims_processor.src.api.models.ingestion_models import IngestionResponse, IngestionClaim, IngestionClaimLineItem
from claims_processor.src.core.config.settings import Settings # For mock_get_settings

@pytest.fixture
def client(): # Removed self
    return TestClient(app)

@pytest.fixture
def mock_data_ingestion_service_instance(): # This will be the instance returned by the patched factory
    service_instance = MagicMock()
    service_instance.ingest_claims_batch = AsyncMock()
    return service_instance

@pytest.fixture(autouse=True) # Ensure it's autouse if all tests in this file need it
def patch_get_data_ingestion_service_for_submission(mock_data_ingestion_service_instance: MagicMock): # Renamed and updated target
    # This patches the centralized `get_data_ingestion_service` function in dependencies.py
    with patch("claims_processor.src.api.dependencies.get_data_ingestion_service",
                 return_value=mock_data_ingestion_service_instance) as mock_getter:
        yield mock_getter

@pytest.fixture
def mock_audit_logger_instance(): # This will be the instance returned by the patched factory
    logger_instance = MagicMock()
    logger_instance.log_access = AsyncMock()
    return logger_instance

@pytest.fixture
def mock_get_audit_logger(mock_audit_logger_instance):
    # This patches the `get_audit_logger` function in dependencies.py, which is imported by submission_routes.py
    with patch("claims_processor.src.api.dependencies.get_audit_logger", return_value=mock_audit_logger_instance) as mock_factory:
        yield mock_factory

@pytest.fixture
def mock_get_settings_for_submission():
    # Patches get_settings specifically for submission_routes.py
    mock_settings_obj = Settings(MAX_INGESTION_BATCH_SIZE=2)
    with patch("claims_processor.src.api.routes.submission_routes.get_settings", return_value=mock_settings_obj) as mock_settings_func:
        yield mock_settings_func


def create_valid_claim_data(count=1) -> list:
    claims = []
    for i in range(count):
        claims.append({
            "claim_id": f"CLAIM_TEST_{i+1}",
            "facility_id": "FAC_TEST",
            "patient_account_number": f"PAT_ACC_{i+1}",
            "service_from_date": "2024-01-01",
            "service_to_date": "2024-01-02",
            "total_charges": "150.75",
            "line_items": [
                {
                    "line_number": 1,
                    "service_date": "2024-01-01",
                    "procedure_code": "P001",
                    "units": 1,
                    "charge_amount": "150.75"
                }
            ]
        })
    return claims

@pytest.mark.asyncio
async def test_submit_claims_batch_success(
    client: TestClient,
    patch_get_data_ingestion_service_for_submission: MagicMock, # Use new fixture name
    mock_data_ingestion_service_instance: MagicMock,
    mock_get_audit_logger: MagicMock,
    mock_audit_logger_instance: MagicMock, # The actual instance
    mock_get_settings_for_submission: MagicMock
):
    valid_payload = create_valid_claim_data(1)
    mock_response_dict = { # DataIngestionService returns a dict, which matches IngestionResponse fields
        "ingestion_batch_id": "test_batch_id",
        "received_claims": 1,
        "successfully_staged_claims": 1,
        "failed_ingestion_claims": 0,
        "errors": []
    }
    # Configure the instance returned by the patched factory
    mock_data_ingestion_service_instance.ingest_claims_batch.return_value = IngestionResponse(**mock_response_dict)

    response = client.post("/api/v1/submissions/claims_batch", json=valid_payload)

    assert response.status_code == 202
    response_json = response.json()
    assert response_json["ingestion_batch_id"] == "test_batch_id"
    assert response_json["successfully_staged_claims"] == 1
    mock_data_ingestion_service_instance.ingest_claims_batch.assert_called_once()

    # Check that the first argument passed to ingest_claims_batch has Pydantic IngestionClaim instances
    call_args = mock_data_ingestion_service_instance.ingest_claims_batch.call_args
    assert len(call_args.kwargs['raw_claims_data']) == 1
    assert isinstance(call_args.kwargs['raw_claims_data'][0], IngestionClaim)

    mock_audit_logger_instance.log_access.assert_called_once()
    audit_args, audit_kwargs = mock_audit_logger_instance.log_access.call_args
    assert audit_kwargs.get('action') == "SUBMIT_CLAIMS_BATCH"
    assert audit_kwargs.get('success') is True


def test_submit_claims_batch_empty_list(
    client: TestClient,
    patch_get_data_ingestion_service_for_submission: MagicMock, # Use new fixture name
    mock_data_ingestion_service_instance: MagicMock,
    mock_get_audit_logger: MagicMock,
    mock_audit_logger_instance: MagicMock,
    mock_get_settings_for_submission: MagicMock
):
    response = client.post("/api/v1/submissions/claims_batch", json=[])

    assert response.status_code == 400 # Endpoint raises 400 for empty batch
    json_response = response.json()
    assert "No claims provided in the batch." in json_response["detail"]

    mock_data_ingestion_service_instance.ingest_claims_batch.assert_not_called()
    mock_audit_logger_instance.log_access.assert_called_once()
    audit_args, audit_kwargs = mock_audit_logger_instance.log_access.call_args
    assert audit_kwargs.get('action') == "SUBMIT_CLAIMS_BATCH"
    assert audit_kwargs.get('success') is False
    assert "No claims provided in batch." in audit_kwargs.get('failure_reason', "")


def test_submit_claims_batch_payload_too_large(
    client: TestClient,
    patch_get_data_ingestion_service_for_submission: MagicMock, # Use new fixture name
    mock_data_ingestion_service_instance: MagicMock,
    mock_get_audit_logger: MagicMock,
    mock_audit_logger_instance: MagicMock,
    mock_get_settings_for_submission: MagicMock # This fixture sets MAX_INGESTION_BATCH_SIZE to 2
):
    invalid_payload = create_valid_claim_data(3) # More than MAX_INGESTION_BATCH_SIZE of 2

    response = client.post("/api/v1/submissions/claims_batch", json=invalid_payload)

    assert response.status_code == 413
    assert "exceeds maximum allowed size" in response.json()["detail"]
    mock_data_ingestion_service_instance.ingest_claims_batch.assert_not_called()
    # Audit logger is NOT called if size check fails early, as per current endpoint logic.
    # If an audit log for "payload too large" was desired, it would need to be added before the HTTPException.
    mock_audit_logger_instance.log_access.assert_not_called()


def test_submit_claims_batch_pydantic_validation_error(client: TestClient, mock_get_settings_for_submission: MagicMock):
    # This test doesn't need data_ingestion_service or audit_logger mocks if Pydantic validation fails before endpoint code runs them.
    invalid_payload = [{
        "service_from_date": "2024-01-01",
        "service_to_date": "2024-01-02",
        "total_charges": "150.75",
        "line_items": []
    }]
    response = client.post("/api/v1/submissions/claims_batch", json=invalid_payload)
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_submit_claims_batch_service_exception(
    client: TestClient,
    patch_get_data_ingestion_service_for_submission: MagicMock, # Use new fixture name
    mock_data_ingestion_service_instance: MagicMock,
    mock_get_audit_logger: MagicMock,
    mock_audit_logger_instance: MagicMock,
    mock_get_settings_for_submission: MagicMock
):
    valid_payload = create_valid_claim_data(1)
    mock_data_ingestion_service_instance.ingest_claims_batch.side_effect = Exception("Internal service error")

    response = client.post("/api/v1/submissions/claims_batch", json=valid_payload)

    assert response.status_code == 500
    assert "Internal server error during claim submission" in response.json()["detail"] # Adjusted to match actual endpoint error
    mock_data_ingestion_service_instance.ingest_claims_batch.assert_called_once()

    mock_audit_logger_instance.log_access.assert_called_once()
    audit_args, audit_kwargs = mock_audit_logger_instance.log_access.call_args
    assert audit_kwargs.get('success') is False
    assert "Internal server error: Internal service error" in audit_kwargs.get('failure_reason', "")

```
