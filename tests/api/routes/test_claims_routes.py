import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
from sqlalchemy.ext.asyncio import AsyncSession # For type hinting mock
from fastapi import HTTPException # For raising in mocks if needed
from decimal import Decimal # For payload
from datetime import date # For payload

from claims_processor.src.main import app
from claims_processor.src.api.models.claim_models import ClaimCreate, ClaimResponse, BatchProcessResponse # Added BatchProcessResponse
from claims_processor.src.core.database.models.claims_db import ClaimModel
from claims_processor.src.core.monitoring.audit_logger import AuditLogger
from claims_processor.src.core.monitoring.app_metrics import MetricsCollector # Added MetricsCollector
from fastapi import BackgroundTasks # Added BackgroundTasks for patching

@pytest.fixture
def client():
    return TestClient(app)

# Mock for database session
@pytest.fixture
def mock_db_session_for_claims_routes():
    session = AsyncMock(spec=AsyncSession)
    execute_result = AsyncMock()
    # Configure the chain of calls: session.execute().scalars().first()
    scalars_mock = AsyncMock()
    scalars_mock.first.return_value = None # Default: claim not found
    execute_result.scalars = MagicMock(return_value=scalars_mock)
    session.execute.return_value = execute_result

    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.refresh = AsyncMock()
    session.add = MagicMock()
    return session

@pytest.fixture(autouse=True)
def mock_get_db_session_for_claims_routes(mock_db_session_for_claims_routes):
    # This patches get_db_session specifically where it's imported in claims_routes.py
    with patch("claims_processor.src.api.routes.claims_routes.get_db_session", return_value=mock_db_session_for_claims_routes) as mock_get_session:
        yield mock_get_session

# Mock for AuditLogger
@pytest.fixture
def mock_audit_logger_for_claims_routes_instance(): # The instance that will be returned
    logger_instance = MagicMock(spec=AuditLogger)
    logger_instance.log_access = AsyncMock()
    return logger_instance

@pytest.fixture(autouse=True)
def mock_get_audit_logger_for_claims_routes(mock_audit_logger_for_claims_routes_instance):
    # claims_routes.py imports get_audit_logger from ..dependencies
    # The actual name used in claims_routes for the import is 'get_new_audit_logger'
    # However, the dependency injection uses the original name from dependencies.py if not aliased in Depends()
    # The dependency is `audit_logger: AuditLogger = Depends(get_audit_logger)` in create_claim after recent changes.
    # So, we patch where FastAPI looks for it: the actual function in dependencies.py used by the route.
    # If claims_routes.py does `from ..dependencies import get_audit_logger`, this is the path.
    with patch("claims_processor.src.api.dependencies.get_audit_logger", return_value=mock_audit_logger_for_claims_routes_instance) as mock_get_logger:
        yield mock_get_logger

# Mock for MetricsCollector
@pytest.fixture
def mock_metrics_collector_for_claims_routes_instance(): # The instance
    collector_instance = MagicMock(spec=MetricsCollector)
    return collector_instance

@pytest.fixture(autouse=True)
def mock_get_metrics_collector_for_claims_routes(mock_metrics_collector_for_claims_routes_instance):
    # Patches where FastAPI's Depends() will look for get_metrics_collector
    with patch("claims_processor.src.api.dependencies.get_metrics_collector", return_value=mock_metrics_collector_for_claims_routes_instance) as mock_get_collector:
        yield mock_get_collector

# Helper for valid payload
def valid_claim_create_payload(**overrides):
    payload = {
        "claim_id": "TEST_CLAIM_001",
        "facility_id": "FAC001",
        "patient_account_number": "PAT001",
        "service_from_date": "2024-01-10", # Will be parsed to date by Pydantic
        "service_to_date": "2024-01-12",   # Will be parsed to date
        "total_charges": 1250.75           # ClaimCreate model expects float
    }
    payload.update(overrides)
    return payload

# --- Tests for POST /api/v1/claims/ (create_claim) ---

def test_create_claim_success(client: TestClient, mock_db_session_for_claims_routes: MagicMock, mock_audit_logger_for_claims_routes_instance: MagicMock):
    payload = valid_claim_create_payload()

    # Simulate claim not found initially
    mock_db_session_for_claims_routes.execute.return_value.scalars.return_value.first.return_value = None

    # Simulate refresh populating the ID and other DB defaults
    async def mock_refresh_effect(model_instance):
        model_instance.id = 1 # Simulate DB assigning an ID
        model_instance.created_at = datetime.now(timezone.utc)
        model_instance.updated_at = datetime.now(timezone.utc)
        model_instance.processing_status = "pending" # Default from model
        return None
    mock_db_session_for_claims_routes.refresh.side_effect = mock_refresh_effect

    response = client.post("/api/v1/claims/", json=payload)

    assert response.status_code == 200 # Assuming success is 200 OK
    response_json = response.json()
    assert response_json["claim_id"] == payload["claim_id"]
    assert response_json["id"] == 1 # From refresh mock
    assert response_json["processing_status"] == "pending"

    mock_db_session_for_claims_routes.add.assert_called_once()
    mock_db_session_for_claims_routes.commit.assert_called_once()
    mock_audit_logger_for_claims_routes_instance.log_access.assert_called_once()
    audit_args, audit_kwargs = mock_audit_logger_for_claims_routes_instance.log_access.call_args
    assert audit_kwargs.get('action') == "CREATE_CLAIM_SUCCESS"
    assert audit_kwargs.get('success') is True

def test_create_claim_duplicate_claim_id(client: TestClient, mock_db_session_for_claims_routes: MagicMock, mock_audit_logger_for_claims_routes_instance: MagicMock):
    payload = valid_claim_create_payload(claim_id="DUPLICATE_001")

    # Simulate claim already exists
    mock_db_session_for_claims_routes.execute.return_value.scalars.return_value.first.return_value = ClaimModel(claim_id="DUPLICATE_001")

    response = client.post("/api/v1/claims/", json=payload)

    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]

    mock_db_session_for_claims_routes.commit.assert_not_called()
    mock_audit_logger_for_claims_routes_instance.log_access.assert_called_once()
    audit_args, audit_kwargs = mock_audit_logger_for_claims_routes_instance.log_access.call_args
    assert audit_kwargs.get('action') == "CREATE_CLAIM_DUPLICATE"
    assert audit_kwargs.get('success') is False
    assert "already exists" in audit_kwargs.get('failure_reason', "")

def test_create_claim_pydantic_validation_error(client: TestClient):
    invalid_payload = valid_claim_create_payload(total_charges="not_a_number") # Invalid type

    response = client.post("/api/v1/claims/", json=invalid_payload)
    assert response.status_code == 422 # Unprocessable Entity

    # Audit logger should not be called if Pydantic validation fails before endpoint logic
    # This depends on whether get_audit_logger is resolved before Pydantic validation error is raised.
    # Typically, FastAPI would raise 422 before endpoint code runs if body parsing fails.
    # mock_audit_logger_for_claims_routes_instance.log_access.assert_not_called() # This might be too strict, depends on FastAPI internals

@pytest.mark.asyncio
async def test_create_claim_db_commit_error(client: TestClient, mock_db_session_for_claims_routes: MagicMock, mock_audit_logger_for_claims_routes_instance: MagicMock):
    payload = valid_claim_create_payload(claim_id="COMMIT_ERROR_001")

    mock_db_session_for_claims_routes.execute.return_value.scalars.return_value.first.return_value = None # Claim does not exist
    mock_db_session_for_claims_routes.commit.side_effect = Exception("DB commit error")

    response = client.post("/api/v1/claims/", json=payload)

    assert response.status_code == 500
    assert "Failed to save claim" in response.json()["detail"]

    mock_db_session_for_claims_routes.add.assert_called_once()
    mock_db_session_for_claims_routes.commit.assert_called_once()
    mock_db_session_for_claims_routes.rollback.assert_called_once() # Check rollback was called

    mock_audit_logger_for_claims_routes_instance.log_access.assert_called_once()
    audit_args, audit_kwargs = mock_audit_logger_for_claims_routes_instance.log_access.call_args
    assert audit_kwargs.get('action') == "CREATE_CLAIM_ERROR"
    assert audit_kwargs.get('success') is False
    assert "DB commit error" in audit_kwargs.get('failure_reason', "")

# --- Tests for POST /api/v1/claims/process-batch/ (trigger_batch_processing_background_endpoint) ---

@patch("fastapi.BackgroundTasks.add_task") # Patch BackgroundTasks.add_task
def test_trigger_batch_processing_success(
    mock_add_task: MagicMock,
    client: TestClient,
    mock_audit_logger_for_claims_routes_instance: MagicMock, # Used by the endpoint directly
    mock_metrics_collector_for_claims_routes_instance: MagicMock # Passed to background task
):
    response = client.post("/api/v1/claims/process-batch/?batch_size=55")

    assert response.status_code == 202
    assert response.json() == {"message": "Batch processing started in the background.", "batch_size": 55}

    mock_add_task.assert_called_once()
    args, kwargs = mock_add_task.call_args

    # args[0] should be the function: run_batch_processing_background
    # from claims_processor.src.api.routes.claims_routes import run_batch_processing_background
    # This is tricky to assert directly without importing it here. Check by name.
    assert args[0].__name__ == "run_batch_processing_background"

    # Check other args passed to run_batch_processing_background
    assert args[1] == 55  # batch_size
    # args[2] is client_ip (can be None or string), args[3] is user_agent_header (can be None or string)
    # args[4] should be the metrics_collector instance
    assert args[4] == mock_metrics_collector_for_claims_routes_instance
    # args[5] should be the audit_logger instance
    assert args[5] == mock_audit_logger_for_claims_routes_instance

    # Check that the endpoint's direct audit log call was made
    mock_audit_logger_for_claims_routes_instance.log_access.assert_called_once()
    audit_call_args, audit_call_kwargs = mock_audit_logger_for_claims_routes_instance.log_access.call_args
    assert audit_call_kwargs.get('action') == "TRIGGER_BATCH_PROCESSING"
    assert audit_call_kwargs.get('success') is True
    assert audit_call_kwargs.get('details') == {"requested_batch_size": 55}


def test_trigger_batch_processing_invalid_batch_size_too_small(client: TestClient):
    response = client.post("/api/v1/claims/process-batch/?batch_size=0")
    assert response.status_code == 422 # FastAPI/Pydantic validation error from Query(gt=0)

def test_trigger_batch_processing_invalid_batch_size_too_large(client: TestClient):
    response = client.post("/api/v1/claims/process-batch/?batch_size=20000") # Assuming le=10000
    assert response.status_code == 422

def test_trigger_batch_processing_invalid_batch_size_non_integer(client: TestClient):
    response = client.post("/api/v1/claims/process-batch/?batch_size=abc")
    assert response.status_code == 422
```
