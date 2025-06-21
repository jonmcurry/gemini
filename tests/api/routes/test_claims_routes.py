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
from claims_processor.src.core.monitoring.app_metrics import MetricsCollector
from claims_processor.src.core.config.settings import Settings # Added for mocking settings
from fastapi import BackgroundTasks
import asyncio # Added for Semaphore

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

# --- Fixtures for Semaphore Testing ---
@pytest.fixture
def mock_app_settings():
    """Provides a Settings object with MAX_CONCURRENT_BATCHES=1 for semaphore testing."""
    return Settings(MAX_CONCURRENT_BATCHES=1)

@pytest.fixture(autouse=True)
def patch_settings_for_claims_routes(mock_app_settings: Settings):
    """Patches get_settings used within claims_routes module."""
    with patch("claims_processor.src.api.routes.claims_routes.get_settings", return_value=mock_app_settings) as mock_settings_getter:
        yield mock_settings_getter

@pytest.fixture
def mock_batch_semaphore(mock_app_settings: Settings): # Depends on mock_app_settings to ensure consistency
    """Provides a real asyncio.Semaphore instance for testing."""
    return asyncio.Semaphore(mock_app_settings.MAX_CONCURRENT_BATCHES)

@pytest.fixture(autouse=True)
def patch_semaphore_getter_for_claims_routes(mock_batch_semaphore: asyncio.Semaphore):
    """Patches get_batch_processing_semaphore used within claims_routes module."""
    with patch("claims_processor.src.api.routes.claims_routes.get_batch_processing_semaphore", return_value=mock_batch_semaphore) as mock_sem_getter:
        yield mock_sem_getter

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
    # args[6] should be the processing_semaphore instance
    # This requires mock_batch_semaphore to be passed or accessible
    # For now, let's assume the test setup makes mock_batch_semaphore available if needed for direct comparison
    # or rely on the patch ensuring it's the one from the fixture.
    # The patch 'patch_semaphore_getter_for_claims_routes' ensures the semaphore is the one from 'mock_batch_semaphore'
    # So, args[6] should be the same object as the one returned by mock_batch_semaphore fixture.
    # This assertion is tricky as we don't have mock_batch_semaphore in this test's scope directly.
    # We trust the patching. If we need to assert this, we'd pass mock_batch_semaphore to this test.

    # Check that the endpoint's direct audit log call was made for "ACCEPTED"
    # This depends on where it's called vs other potential audit calls.
    # If semaphore is acquired, "TRIGGER_BATCH_PROCESSING_ACCEPTED" is logged.
    accepted_audit_call_found = False
    for call_item in mock_audit_logger_for_claims_routes_instance.log_access.call_args_list:
        _, item_kwargs = call_item
        if item_kwargs.get('action') == "TRIGGER_BATCH_PROCESSING_ACCEPTED":
            assert item_kwargs.get('success') is True
            assert item_kwargs.get('details') == {"requested_batch_size": 55, "limit": 1} # Assuming limit is 1 from mock_app_settings
            accepted_audit_call_found = True
            break
    assert accepted_audit_call_found, "Audit log for TRIGGER_BATCH_PROCESSING_ACCEPTED not found."


def test_trigger_batch_processing_invalid_batch_size_too_small(client: TestClient):
    response = client.post("/api/v1/claims/process-batch/?batch_size=0")
    assert response.status_code == 422 # FastAPI/Pydantic validation error from Query(gt=0)

def test_trigger_batch_processing_invalid_batch_size_too_large(client: TestClient):
    response = client.post("/api/v1/claims/process-batch/?batch_size=20000") # Assuming le=10000
    assert response.status_code == 422

def test_trigger_batch_processing_invalid_batch_size_non_integer(client: TestClient):
    response = client.post("/api/v1/claims/process-batch/?batch_size=abc")
    assert response.status_code == 422

# --- Tests for Semaphore Logic in /process-batch/ ---

@pytest.mark.asyncio
async def test_trigger_batch_processing_semaphore_available_and_release_on_completion(
    client: TestClient,
    mock_batch_semaphore: asyncio.Semaphore, # Injected by fixture
    mock_app_settings: Settings, # To check limit
    mock_audit_logger_for_claims_routes_instance: MagicMock
):
    # Semaphore starts with 1 permit (from mock_app_settings.MAX_CONCURRENT_BATCHES)
    assert mock_batch_semaphore._value == 1 # Check initial state (internal, but useful for testing)

    # Mock the actual background task function to simulate its execution and semaphore release
    mock_runner = AsyncMock()
    async def side_effect_runner(*args, processing_semaphore: asyncio.Semaphore, **kwargs):
        assert processing_semaphore == mock_batch_semaphore # Ensure correct semaphore passed
        # Simulate work
        await asyncio.sleep(0.01)
        # Simulate successful completion and release
        processing_semaphore.release()
        return {"message": "Mock task done"}

    mock_runner.side_effect = side_effect_runner

    with patch("claims_processor.src.api.routes.claims_routes.run_batch_processing_background", new=mock_runner) as mock_run_batch_bg:
        response = client.post("/api/v1/claims/process-batch/?batch_size=10")

    assert response.status_code == 202 # Task accepted
    # Immediately after call, before task "finishes", semaphore should be locked (value 0)
    assert mock_batch_semaphore.locked() is True
    assert mock_batch_semaphore._value == 0

    mock_run_batch_bg.assert_called_once()
    call_args, call_kwargs = mock_run_batch_bg.call_args
    assert call_kwargs.get("processing_semaphore") == mock_batch_semaphore

    # Wait for the mocked background task to "complete" and release the semaphore
    await asyncio.sleep(0.05) # Give time for the mocked task's side effect to run

    assert mock_batch_semaphore.locked() is False # Semaphore should be released
    assert mock_batch_semaphore._value == 1

    # Verify audit log for acceptance
    accepted_audit_call_found = False
    for call_item in mock_audit_logger_for_claims_routes_instance.log_access.call_args_list:
        _, item_kwargs = call_item
        if item_kwargs.get('action') == "TRIGGER_BATCH_PROCESSING_ACCEPTED":
            accepted_audit_call_found = True
            break
    assert accepted_audit_call_found


@pytest.mark.asyncio
async def test_trigger_batch_processing_semaphore_unavailable(
    client: TestClient,
    mock_batch_semaphore: asyncio.Semaphore,
    mock_app_settings: Settings, # To check limit
    mock_audit_logger_for_claims_routes_instance: MagicMock
):
    # Acquire the only permit
    await mock_batch_semaphore.acquire()
    assert mock_batch_semaphore.locked() is True

    with patch("fastapi.BackgroundTasks.add_task") as mock_add_task: # To ensure task is not added
        response = client.post("/api/v1/claims/process-batch/?batch_size=20")

    assert response.status_code == 503 # Service Unavailable
    assert "system is currently processing the maximum number of batches" in response.json()["detail"]
    mock_add_task.assert_not_called() # Task should not have been added

    # Verify audit log for rejection
    rejected_audit_call_found = False
    for call_item in mock_audit_logger_for_claims_routes_instance.log_access.call_args_list:
        _, item_kwargs = call_item
        if item_kwargs.get('action') == "TRIGGER_BATCH_PROCESSING_REJECTED":
            assert item_kwargs.get('success') is False
            assert item_kwargs.get('failure_reason') == "Max concurrent batch processing limit reached."
            assert item_kwargs.get('details') == {"requested_batch_size": 20, "limit": mock_app_settings.MAX_CONCURRENT_BATCHES}
            rejected_audit_call_found = True
            break
    assert rejected_audit_call_found

    # Release the semaphore for cleanup / other tests
    mock_batch_semaphore.release()
    assert mock_batch_semaphore.locked() is False


@pytest.mark.asyncio
async def test_trigger_batch_processing_semaphore_release_on_task_failure(
    client: TestClient,
    mock_batch_semaphore: asyncio.Semaphore,
    mock_app_settings: Settings
):
    assert mock_batch_semaphore._value == 1 # Initial state

    # Mock the background task to raise an exception but still release semaphore (due to finally block in real code)
    mock_runner_fail = AsyncMock()
    async def side_effect_runner_fail(*args, processing_semaphore: asyncio.Semaphore, **kwargs):
        assert processing_semaphore == mock_batch_semaphore
        try:
            raise ValueError("Simulated task failure")
        finally:
            # This simulates the 'finally' block in the actual run_batch_processing_background
            processing_semaphore.release()

    mock_runner_fail.side_effect = side_effect_runner_fail

    with patch("claims_processor.src.api.routes.claims_routes.run_batch_processing_background", new=mock_runner_fail) as mock_run_batch_bg_fail:
        response = client.post("/api/v1/claims/process-batch/?batch_size=30")

    assert response.status_code == 202 # Endpoint still accepts the task
    assert mock_batch_semaphore.locked() is True # Acquired by endpoint

    mock_run_batch_bg_fail.assert_called_once()

    # Wait for the mocked background task to "fail" and release the semaphore
    await asyncio.sleep(0.05)

    assert mock_batch_semaphore.locked() is False # Semaphore should be released by the finally block
    assert mock_batch_semaphore._value == 1


# Test for the unlikely case where add_task itself fails
@patch("fastapi.BackgroundTasks.add_task", side_effect=Exception("Failed to add task"))
@pytest.mark.asyncio
async def test_trigger_batch_processing_add_task_fails_releases_semaphore(
    mock_add_task_fails: MagicMock,
    client: TestClient,
    mock_batch_semaphore: asyncio.Semaphore,
    mock_audit_logger_for_claims_routes_instance: MagicMock
):
    assert mock_batch_semaphore._value == 1 # Initial state

    response = client.post("/api/v1/claims/process-batch/?batch_size=40")

    assert response.status_code == 500 # Endpoint should catch this and return 500
    assert "Failed to initiate batch processing" in response.json()["detail"]

    # Semaphore should have been acquired by the endpoint, then released by its error handling
    assert mock_batch_semaphore.locked() is False
    assert mock_batch_semaphore._value == 1

    # Check audit log for failure to add
    failed_to_add_audit_call_found = False
    for call_item in mock_audit_logger_for_claims_routes_instance.log_access.call_args_list:
        _, item_kwargs = call_item
        if item_kwargs.get('action') == "TRIGGER_BATCH_PROCESSING_FAILED_TO_ADD":
            assert item_kwargs.get('success') is False
            assert "Error adding task to background: Failed to add task" in item_kwargs.get('failure_reason')
            failed_to_add_audit_call_found = True
            break
    assert failed_to_add_audit_call_found, "Audit log for TRIGGER_BATCH_PROCESSING_FAILED_TO_ADD not found."

```
