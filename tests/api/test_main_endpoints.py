import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock # MagicMock for general, AsyncMock for async methods

from claims_processor.src.main import app # FastAPI app instance
from claims_processor.src.core.config.settings import Settings # To mock settings object
# CacheManager for type hinting, not strictly needed for mocking if using spec with MagicMock
from claims_processor.src.core.cache.cache_manager import CacheManager
from sqlalchemy.ext.asyncio import AsyncSession # For type hinting mock session

client = TestClient(app)

# --- Test Fixtures for Mocked Dependencies ---
@pytest.fixture
def mock_db_dep():
    mock_db_instance = MagicMock(spec=AsyncSession) # Use MagicMock for the session object itself
    mock_execute_result = MagicMock() # Mock for the result of execute()
    mock_execute_result.scalar_one.return_value = 1 # Simulate successful DB query
    mock_db_instance.execute = AsyncMock(return_value=mock_execute_result) # execute() is async
    with patch("claims_processor.src.main.get_db_session", return_value=mock_db_instance) as mock_dep:
        yield mock_dep, mock_db_instance

@pytest.fixture
def mock_cache_dep():
    mock_cache_instance = MagicMock(spec=CacheManager) # Use MagicMock for the CacheManager instance
    mock_cache_instance.set = AsyncMock(return_value=None) # set() is async
    mock_cache_instance.get = AsyncMock(return_value="healthy") # get() is async, simulate successful get
    with patch("claims_processor.src.main.get_cache_service", return_value=mock_cache_instance) as mock_dep:
        yield mock_dep, mock_cache_instance

@pytest.fixture
def mock_settings_dep():
    # Default healthy settings: ML model path configured and file exists
    mock_settings_instance = Settings(ML_MODEL_PATH="fake/model.tflite", _env_file=None) # _env_file=None to prevent .env load
    with patch("claims_processor.src.main.get_settings", return_value=mock_settings_instance) as mock_dep:
        yield mock_dep, mock_settings_instance

@pytest.fixture
def mock_path_is_file():
    with patch("claims_processor.src.main.Path.is_file", return_value=True) as mock_method: # Default: file exists
        yield mock_method

# --- Test Cases ---

def test_ready_all_healthy_model_configured_exists(
    mock_db_dep, mock_cache_dep, mock_settings_dep, mock_path_is_file
):
    # mock_path_is_file already defaults to True via fixture
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["database"]["status"] == "healthy"
    assert data["cache"]["status"] == "healthy"
    assert data["ml_service"]["status"] == "healthy"

def test_ready_all_healthy_model_not_configured(
    mock_db_dep, mock_cache_dep, mock_settings_dep, mock_path_is_file
):
    mock_get_settings, mock_settings_instance = mock_settings_dep
    mock_settings_instance.ML_MODEL_PATH = None # Override for this test
    mock_get_settings.return_value = mock_settings_instance

    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["database"]["status"] == "healthy"
    assert data["cache"]["status"] == "healthy"
    assert data["ml_service"]["status"] == "not_configured"

def test_ready_db_unhealthy(
    mock_db_dep, mock_cache_dep, mock_settings_dep, mock_path_is_file
):
    _, mock_db_instance = mock_db_dep
    mock_db_instance.execute.side_effect = Exception("DB Connection Error")

    response = client.get("/ready")
    assert response.status_code == 503
    data = response.json()["detail"] # HTTPException stores details in "detail"
    assert data["database"]["status"] == "unhealthy"
    assert "DB Connection Error" in data["database"]["details"]
    assert data["cache"]["status"] == "healthy"
    assert data["ml_service"]["status"] == "healthy"

def test_ready_cache_unhealthy_set_fails(
    mock_db_dep, mock_cache_dep, mock_settings_dep, mock_path_is_file
):
    _, mock_cache_instance = mock_cache_dep
    mock_cache_instance.set.side_effect = Exception("Cache SET Error")

    response = client.get("/ready")
    assert response.status_code == 503
    data = response.json()["detail"]
    assert data["cache"]["status"] == "unhealthy"
    assert "Cache SET Error" in data["cache"]["details"]
    assert data["database"]["status"] == "healthy"
    assert data["ml_service"]["status"] == "healthy"

def test_ready_cache_unhealthy_get_mismatch(
    mock_db_dep, mock_cache_dep, mock_settings_dep, mock_path_is_file
):
    _, mock_cache_instance = mock_cache_dep
    mock_cache_instance.get.return_value = "wrong_value" # Simulate GET returning unexpected value

    response = client.get("/ready")
    assert response.status_code == 503
    data = response.json()["detail"]
    assert data["cache"]["status"] == "unhealthy"
    assert "SET/GET operations did not return expected value" in data["cache"]["details"]
    assert data["database"]["status"] == "healthy"
    assert data["ml_service"]["status"] == "healthy"

def test_ready_ml_service_unhealthy_file_not_found(
    mock_db_dep, mock_cache_dep, mock_settings_dep, mock_path_is_file
):
    # Settings fixture already configures ML_MODEL_PATH to "fake/model.tflite"
    mock_path_is_file.return_value = False # Simulate model file does not exist

    response = client.get("/ready")
    assert response.status_code == 503
    data = response.json()["detail"]
    assert data["ml_service"]["status"] == "unhealthy"
    assert "Model file not found" in data["ml_service"]["details"]
    assert data["database"]["status"] == "healthy"
    assert data["cache"]["status"] == "healthy"

def test_ready_all_unhealthy(
    mock_db_dep, mock_cache_dep, mock_settings_dep, mock_path_is_file
):
    _, mock_db_instance = mock_db_dep
    mock_db_instance.execute.side_effect = Exception("DB Error")

    _, mock_cache_instance = mock_cache_dep
    mock_cache_instance.set.side_effect = Exception("Cache Error")

    mock_path_is_file.return_value = False # ML model file not found

    response = client.get("/ready")
    assert response.status_code == 503
    data = response.json()["detail"]
    assert data["database"]["status"] == "unhealthy"
    assert data["cache"]["status"] == "unhealthy"
    assert data["ml_service"]["status"] == "unhealthy"
