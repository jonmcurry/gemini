import pytest
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
from typing import List, Dict, Any, Optional
import time # For mocking time for TTL tests
from cachetools import TTLCache # For type hinting
import mlflow # Added
import tempfile # Added

# Conditional import for testing
TFLITE_INSTALLED_FOR_TEST = False
try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
    TFLITE_INSTALLED_FOR_TEST = True
except ImportError:
    import structlog # Import structlog here if needed for the dummy class
    class TFLiteInterpreter: # type: ignore
        def __init__(self, model_path=None, model_content=None, experimental_delegates=None, num_threads=None):
            self.model_path = model_path
        def allocate_tensors(self): pass
        def get_input_details(self): return [{'shape': np.array([1, 7], dtype=np.int32), 'index': 0, 'dtype': np.float32}] # Default, adjust if needed
        def get_output_details(self): return [{'shape': np.array([1, 2], dtype=np.int32), 'index': 0, 'dtype': np.float32}]
        def set_tensor(self, tensor_index, value): pass
        def invoke(self): pass
        def get_tensor(self, tensor_index): return np.array([[0.25, 0.75]], dtype=np.float32) # Default mock output for single item
    logger_for_dummy = structlog.get_logger(__name__)
    logger_for_dummy.warn("Dummy TFLiteInterpreter created as tflite_runtime is not installed.")


from claims_processor.src.processing.ml_pipeline.optimized_predictor import OptimizedPredictor
from claims_processor.src.core.config.settings import Settings
from claims_processor.src.core.monitoring.app_metrics import MetricsCollector

# Use settings for model path and feature count consistently
# These will be overridden by mock_settings_fixture in tests needing specific settings.
BASE_SETTINGS_FOR_TESTS = Settings() # Use real settings as a base for defaults
MODEL_PATH_SETTING = BASE_SETTINGS_FOR_TESTS.ML_MODEL_PATH or "dummy_model.tflite"
FEATURE_COUNT_SETTING = BASE_SETTINGS_FOR_TESTS.ML_FEATURE_COUNT
APPROVAL_THRESHOLD_SETTING = BASE_SETTINGS_FOR_TESTS.ML_APPROVAL_THRESHOLD

@pytest.fixture
def mock_metrics_collector() -> MagicMock:
    collector = MagicMock(spec=MetricsCollector)
    collector.record_ml_prediction = MagicMock()
    collector.record_cache_operation = MagicMock()
    collector.record_ml_inference_duration = MagicMock()
    return collector

@pytest.fixture
def mock_settings( # Default mock settings, can be overridden in tests
    tmp_path: Path # Pytest fixture for temporary directory
) -> Settings:
    # Create a dummy model file so Path.is_file() check passes during init
    dummy_model_file = tmp_path / "dummy_model.tflite"
    dummy_model_file.touch()

    settings = Settings(
        ML_MODEL_PATH=str(dummy_model_file), # Use the path to the dummy file
        ML_FEATURE_COUNT=FEATURE_COUNT_SETTING,
        ML_APPROVAL_THRESHOLD=APPROVAL_THRESHOLD_SETTING,
        ML_PREDICTION_CACHE_MAXSIZE=128, # Default for most tests
        ML_PREDICTION_CACHE_TTL=3600     # Default for most tests
    )
    return settings

@pytest.fixture
def optimized_predictor(mock_metrics_collector: MagicMock, mock_settings: Settings) -> OptimizedPredictor:
    # Patch get_settings call within OptimizedPredictor's __init__
    with patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.get_settings', return_value=mock_settings):
        # Ensure TFLITE_AVAILABLE is True for these tests to attempt loading the interpreter
        with patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE', True):
            # Mock the TFLite Interpreter constructor itself
            with patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.Interpreter') as mock_interpreter_constructor:
                mock_interpreter_instance = MagicMock(spec=TFLiteInterpreter)
                mock_interpreter_instance.get_input_details.return_value = [{'shape': np.array([-1, mock_settings.ML_FEATURE_COUNT], dtype=np.int32), 'index': 0, 'dtype': np.float32}] # Dynamic batch size
                mock_interpreter_instance.get_output_details.return_value = [{'shape': np.array([-1, 2], dtype=np.int32), 'index': 1, 'dtype': np.float32}] # Dynamic batch size, 2 outputs
                mock_interpreter_constructor.return_value = mock_interpreter_instance

                # OptimizedPredictor __init__ no longer takes model_path or feature_count directly
                predictor = OptimizedPredictor(
                    metrics_collector=mock_metrics_collector
                )
                # Store the mocked interpreter instance on the predictor for easy access in tests
                predictor.mock_interpreter_instance = mock_interpreter_instance
                return predictor


# --- Initialization Tests ---
@patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.get_settings')
def test_init_tflite_not_available(mock_get_settings_func: MagicMock, mock_metrics_collector: MagicMock, mock_settings: Settings): # mock_settings is the fixture result
    # Ensure this test uses the specific mock_settings fixture if needed for ML_MODEL_PATH
    mock_get_settings_func.return_value = mock_settings

    with patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE', False):
        # __init__ signature changed
        predictor = OptimizedPredictor(metrics_collector=mock_metrics_collector)
        assert predictor.interpreter is None
        assert isinstance(predictor.prediction_cache, TTLCache) # Cache should still init


@patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.get_settings')
@patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE', True)
@patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.Interpreter')
@patch('mlflow.set_tracking_uri')
@patch('mlflow.artifacts.download_artifacts')
@patch('tempfile.TemporaryDirectory') # Keep this mocked, even if not directly used by this test path for cleanup check
def test_init_mlflow_download_fails_fallback_success(
    mock_temp_dir_constructor: MagicMock, # Mocked but not expected to retain instance
    mock_download_artifacts: MagicMock,
    mock_set_tracking_uri: MagicMock,
    mock_interpreter_constructor: MagicMock,
    mock_tflite_available_patch: MagicMock,
    mock_get_settings_func: MagicMock,
    mock_metrics_collector: MagicMock,
    tmp_path: Path
):
    fallback_model_path = tmp_path / "fallback_model.tflite"
    fallback_model_path.touch()

    mock_settings_obj = Settings(
        MLFLOW_TRACKING_URI="http://fake-mlflow:5000",
        ML_MODEL_NAME_IN_REGISTRY="test_model_fallback",
        ML_MODEL_VERSION_OR_STAGE="Production",
        ML_MODEL_PATH=str(fallback_model_path), # Valid fallback
        ML_FEATURE_COUNT=FEATURE_COUNT_SETTING,
        ML_APPROVAL_THRESHOLD=0.7,
        ML_PREDICTION_CACHE_MAXSIZE=10,
        ML_PREDICTION_CACHE_TTL=60
    )
    mock_get_settings_func.return_value = mock_settings_obj

    mock_download_artifacts.side_effect = Exception("MLflow download failed")

    # Mock for TemporaryDirectory in case it's created then cleaned up
    mock_temp_dir_instance = MagicMock(spec=tempfile.TemporaryDirectory)
    mock_temp_dir_instance.name = str(tmp_path / "mlflow_temp_failed")
    mock_temp_dir_constructor.return_value = mock_temp_dir_instance


    mock_interpreter_instance = MagicMock(spec=TFLiteInterpreter)
    mock_interpreter_instance.get_input_details.return_value = [{'shape': np.array([-1, mock_settings_obj.ML_FEATURE_COUNT])}]
    mock_interpreter_instance.get_output_details.return_value = [{'shape': np.array([-1, 2])}]
    mock_interpreter_constructor.return_value = mock_interpreter_instance

    predictor = OptimizedPredictor(metrics_collector=mock_metrics_collector)

    mock_set_tracking_uri.assert_called_once_with(mock_settings_obj.MLFLOW_TRACKING_URI)
    mock_download_artifacts.assert_called_once()
    # Interpreter should be called with the fallback path
    mock_interpreter_constructor.assert_called_once_with(model_path=str(fallback_model_path))
    assert predictor.model_path == str(fallback_model_path)
    assert predictor.interpreter == mock_interpreter_instance
    assert predictor.temp_model_dir is None # Should be cleaned up or not set successfully
    # Check if cleanup was called on the mock_temp_dir_instance if it was instantiated by the code
    # This depends on whether TemporaryDirectory() is called before download_artifacts raises exception.
    # Based on current code, it is. So, cleanup should be called.
    mock_temp_dir_instance.cleanup.assert_called_once()


@patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.get_settings')
@patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE', True)
@patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.Interpreter')
@patch('mlflow.set_tracking_uri')
@patch('mlflow.artifacts.download_artifacts')
@patch('tempfile.TemporaryDirectory')
def test_init_mlflow_success(
    mock_temp_dir_constructor: MagicMock,
    mock_download_artifacts: MagicMock,
    mock_set_tracking_uri: MagicMock,
    mock_interpreter_constructor: MagicMock,
    mock_tflite_available_patch: MagicMock, # Renamed to avoid conflict with actual TFLITE_AVAILABLE
    mock_get_settings_func: MagicMock,
    mock_metrics_collector: MagicMock,
    tmp_path: Path
):
    # 1. Setup Mocks
    mock_settings_obj = Settings(
        MLFLOW_TRACKING_URI="http://fake-mlflow:5000",
        ML_MODEL_NAME_IN_REGISTRY="test_model",
        ML_MODEL_VERSION_OR_STAGE="Production",
        ML_MODEL_PATH="fallback/model.tflite",
        ML_FEATURE_COUNT=FEATURE_COUNT_SETTING,
        ML_APPROVAL_THRESHOLD=0.7,
        ML_PREDICTION_CACHE_MAXSIZE=10,
        ML_PREDICTION_CACHE_TTL=60
    )
    mock_get_settings_func.return_value = mock_settings_obj

    mock_temp_dir_instance = MagicMock(spec=tempfile.TemporaryDirectory)
    # download_artifacts will create subdirectories, so use a sub-path for the model itself
    mock_temp_dir_instance.name = str(tmp_path / "mlflow_temp_root")
    (tmp_path / "mlflow_temp_root").mkdir(parents=True, exist_ok=True)
    mock_temp_dir_constructor.return_value = mock_temp_dir_instance

    # download_artifacts returns the path to the root of downloaded artifacts.
    # The .tflite file is expected to be found within this root.
    download_root_path = Path(mock_temp_dir_instance.name) / "test_model_prod_artifacts" # Simulate a subfolder created by download
    download_root_path.mkdir(parents=True, exist_ok=True)
    mock_download_artifacts.return_value = str(download_root_path)

    downloaded_model_path = download_root_path / "model.tflite"
    downloaded_model_path.touch()

    mock_interpreter_instance = MagicMock(spec=TFLiteInterpreter)
    mock_interpreter_instance.get_input_details.return_value = [{'shape': np.array([-1, mock_settings_obj.ML_FEATURE_COUNT])}]
    mock_interpreter_instance.get_output_details.return_value = [{'shape': np.array([-1, 2])}]
    mock_interpreter_constructor.return_value = mock_interpreter_instance

    # 2. Instantiate OptimizedPredictor
    predictor = OptimizedPredictor(metrics_collector=mock_metrics_collector)

    # 3. Assertions
    mock_set_tracking_uri.assert_called_once_with(mock_settings_obj.MLFLOW_TRACKING_URI)
    mock_download_artifacts.assert_called_once_with(
        artifact_uri=f"models:/{mock_settings_obj.ML_MODEL_NAME_IN_REGISTRY}/{mock_settings_obj.ML_MODEL_VERSION_OR_STAGE}",
        dst_path=mock_temp_dir_instance.name
    )
    mock_interpreter_constructor.assert_called_once_with(model_path=str(downloaded_model_path))
    assert predictor.model_path == str(downloaded_model_path)
    assert predictor.interpreter == mock_interpreter_instance
    assert predictor.temp_model_dir == mock_temp_dir_instance
    mock_temp_dir_instance.cleanup.assert_not_called()


@patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.get_settings')
@patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE', True)
@patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.Interpreter')
@patch('mlflow.set_tracking_uri')
@patch('mlflow.artifacts.download_artifacts')
def test_init_no_mlflow_fallback_success(
    mock_download_artifacts: MagicMock,
    mock_set_tracking_uri: MagicMock,
    mock_interpreter_constructor: MagicMock,
    mock_tflite_available: MagicMock, # To confirm it's True
    mock_get_settings_func: MagicMock,
    mock_metrics_collector: MagicMock,
    tmp_path: Path # Pytest fixture for temp path
):
    fallback_model_path = tmp_path / "fallback_model.tflite"
    fallback_model_path.touch() # Create dummy fallback model file

    mock_settings_obj = Settings(
        MLFLOW_TRACKING_URI=None, # MLflow not configured
        ML_MODEL_NAME_IN_REGISTRY=None,
        ML_MODEL_VERSION_OR_STAGE=None,
        ML_MODEL_PATH=str(fallback_model_path),
        ML_FEATURE_COUNT=FEATURE_COUNT_SETTING, # Using global for tests
        ML_APPROVAL_THRESHOLD=APPROVAL_THRESHOLD_SETTING, # Using global for tests
        ML_PREDICTION_CACHE_MAXSIZE=10,
        ML_PREDICTION_CACHE_TTL=60
    )
    mock_get_settings_func.return_value = mock_settings_obj

    mock_interpreter_instance = MagicMock(spec=TFLiteInterpreter)
    mock_interpreter_instance.get_input_details.return_value = [{'shape': np.array([-1, mock_settings_obj.ML_FEATURE_COUNT])}]
    mock_interpreter_instance.get_output_details.return_value = [{'shape': np.array([-1, 2])}]
    mock_interpreter_constructor.return_value = mock_interpreter_instance

    predictor = OptimizedPredictor(metrics_collector=mock_metrics_collector)

    mock_set_tracking_uri.assert_not_called()
    mock_download_artifacts.assert_not_called()
    mock_interpreter_constructor.assert_called_once_with(model_path=str(fallback_model_path))
    assert predictor.model_path == str(fallback_model_path)
    assert predictor.interpreter == mock_interpreter_instance
    assert predictor.temp_model_dir is None


# --- Prediction Tests (Refactored for Batch and Cache) ---

def generate_feature_sample(feature_count: int, value_offset: float = 0.0) -> np.ndarray:
    # Creates a 1D array
    return np.array([i + value_offset for i in range(feature_count)], dtype=np.float32)

@pytest.mark.asyncio
async def test_predict_batch_empty_input(optimized_predictor: OptimizedPredictor):
    predictions = await optimized_predictor.predict_batch([])
    assert predictions == []

@pytest.mark.asyncio
async def test_predict_batch_input_validation_errors(optimized_predictor: OptimizedPredictor, mock_metrics_collector: MagicMock):
    valid_sample = generate_feature_sample(optimized_predictor.feature_count)
    invalid_shape_sample = np.array([1,2,3], dtype=np.float32) # Wrong shape
    not_an_array = [1,2,3,4,5,6,7]

    # Batch with one invalid shape
    results = await optimized_predictor.predict_batch([valid_sample, invalid_shape_sample, valid_sample])
    assert len(results) == 3
    assert results[0]['ml_derived_decision'] != "ML_ERROR_FEATURE_SHAPE" # Assuming first is processed or cached if valid
    assert results[1]['ml_derived_decision'] == "ML_ERROR_FEATURE_SHAPE"
    assert results[2]['ml_derived_decision'] != "ML_ERROR_FEATURE_SHAPE"

    # Batch with non-ndarray
    results_non_array = await optimized_predictor.predict_batch([valid_sample, not_an_array]) # type: ignore
    assert len(results_non_array) == 2
    assert results_non_array[1]['ml_derived_decision'] == "ML_ERROR_FEATURE_FORMAT"


@pytest.mark.asyncio
async def test_prediction_cache_hit(optimized_predictor: OptimizedPredictor, mock_metrics_collector: MagicMock):
    features = generate_feature_sample(optimized_predictor.feature_count)
    cache_key = hashlib.sha256(features.tobytes()).hexdigest()
    cached_result = {'ml_score': 0.99, 'ml_derived_decision': 'ML_APPROVED_CACHED'}

    optimized_predictor.prediction_cache[cache_key] = cached_result

    predictions = await optimized_predictor.predict_batch([features])

    assert len(predictions) == 1
    assert predictions[0] == cached_result
    optimized_predictor.mock_interpreter_instance.invoke.assert_not_called()
    mock_metrics_collector.record_cache_operation.assert_called_with(cache_type='ml_prediction', operation_type='get', outcome='hit')

@pytest.mark.asyncio
async def test_prediction_cache_miss_then_hit(optimized_predictor: OptimizedPredictor, mock_metrics_collector: MagicMock):
    features = generate_feature_sample(optimized_predictor.feature_count)

    # Simulate interpreter output for the miss
    mock_output = np.array([[0.1, 0.9]], dtype=np.float32) # Batch of 1, 2 classes. Approve.
    optimized_predictor.mock_interpreter_instance.get_tensor.return_value = mock_output

    # First call (miss)
    predictions1 = await optimized_predictor.predict_batch([features])
    assert len(predictions1) == 1
    assert predictions1[0]['ml_derived_decision'] == 'ML_APPROVED'
    assert predictions1[0]['ml_score'] == pytest.approx(0.9)
    optimized_predictor.mock_interpreter_instance.invoke.assert_called_once()
    # Check cache miss recorded, then prediction recorded
    mock_metrics_collector.record_cache_operation.assert_called_with(cache_type='ml_prediction', operation_type='get', outcome='miss')
    mock_metrics_collector.record_ml_prediction.assert_called_with(outcome='ML_APPROVED', confidence_score=pytest.approx(0.9))

    # Reset invoke mock for second call check
    optimized_predictor.mock_interpreter_instance.invoke.reset_mock()
    mock_metrics_collector.record_cache_operation.reset_mock()

    # Second call (hit)
    predictions2 = await optimized_predictor.predict_batch([features])
    assert len(predictions2) == 1
    assert predictions2[0]['ml_derived_decision'] == 'ML_APPROVED'
    assert predictions2[0]['ml_score'] == pytest.approx(0.9)
    optimized_predictor.mock_interpreter_instance.invoke.assert_not_called()
    mock_metrics_collector.record_cache_operation.assert_called_with(cache_type='ml_prediction', operation_type='get', outcome='hit')

@pytest.mark.asyncio
async def test_prediction_cache_ttl_expiration(optimized_predictor: OptimizedPredictor, mock_metrics_collector: MagicMock, mock_settings: Settings):
    # This test needs to use the 'optimized_predictor' fixture which uses 'mock_settings'
    # We need to ensure 'mock_settings' has the short TTL for this specific test.
    # The fixture 'optimized_predictor' is already using 'mock_settings'.
    # We can't easily change mock_settings per test if it's a fixture used by another fixture.
    # So, let's re-initialize the cache on the existing predictor instance for this test.

    short_ttl = 0.1 # seconds
    optimized_predictor.prediction_cache = TTLCache(maxsize=128, ttl=short_ttl)

    features = generate_feature_sample(optimized_predictor.feature_count, value_offset=1.0) # Different features

    mock_output = np.array([[0.1, 0.9]], dtype=np.float32) # Approve
    optimized_predictor.mock_interpreter_instance.get_tensor.return_value = mock_output

    # First call (miss and cache)
    await optimized_predictor.predict_batch([features])
    optimized_predictor.mock_interpreter_instance.invoke.assert_called_once()
    mock_metrics_collector.record_cache_operation.assert_called_with(cache_type='ml_prediction', operation_type='get', outcome='miss')

    # Simulate time passing
    await asyncio.sleep(short_ttl + 0.05) # Sleep a bit longer than TTL

    # Second call (should be miss again due to TTL)
    optimized_predictor.mock_interpreter_instance.invoke.reset_mock()
    mock_metrics_collector.record_cache_operation.reset_mock()
    await optimized_predictor.predict_batch([features])

    optimized_predictor.mock_interpreter_instance.invoke.assert_called_once()
    mock_metrics_collector.record_cache_operation.assert_called_with(cache_type='ml_prediction', operation_type='get', outcome='miss')

@pytest.mark.asyncio
async def test_prediction_cache_maxsize(optimized_predictor: OptimizedPredictor, mock_metrics_collector: MagicMock):
    # Re-initialize cache with maxsize 1 for this test
    optimized_predictor.prediction_cache = TTLCache(maxsize=1, ttl=3600)

    features_a = generate_feature_sample(optimized_predictor.feature_count, value_offset=10.0)
    features_b = generate_feature_sample(optimized_predictor.feature_count, value_offset=20.0)

    # Mock interpreter output
    output_a = np.array([[0.1, 0.9]], dtype=np.float32) # Approve A
    output_b = np.array([[0.8, 0.2]], dtype=np.float32) # Reject B

    # Predict A (miss, cached)
    optimized_predictor.mock_interpreter_instance.get_tensor.return_value = output_a
    await optimized_predictor.predict_batch([features_a])
    optimized_predictor.mock_interpreter_instance.invoke.assert_called_once()
    assert len(optimized_predictor.prediction_cache) == 1
    mock_metrics_collector.record_cache_operation.assert_called_with(cache_type='ml_prediction', operation_type='get', outcome='miss')

    # Predict B (miss, cached, A should be evicted)
    optimized_predictor.mock_interpreter_instance.invoke.reset_mock()
    mock_metrics_collector.record_cache_operation.reset_mock()
    optimized_predictor.mock_interpreter_instance.get_tensor.return_value = output_b
    await optimized_predictor.predict_batch([features_b])
    optimized_predictor.mock_interpreter_instance.invoke.assert_called_once()
    assert len(optimized_predictor.prediction_cache) == 1
    mock_metrics_collector.record_cache_operation.assert_called_with(cache_type='ml_prediction', operation_type='get', outcome='miss')
    # Verify B is in cache (its key would be different from A's)
    cache_key_b = hashlib.sha256(features_b.tobytes()).hexdigest()
    assert cache_key_b in optimized_predictor.prediction_cache

    # Predict A again (should be a miss as it was evicted)
    optimized_predictor.mock_interpreter_instance.invoke.reset_mock()
    mock_metrics_collector.record_cache_operation.reset_mock()
    optimized_predictor.mock_interpreter_instance.get_tensor.return_value = output_a # Set output for A again
    await optimized_predictor.predict_batch([features_a])
    optimized_predictor.mock_interpreter_instance.invoke.assert_called_once()
    mock_metrics_collector.record_cache_operation.assert_called_with(cache_type='ml_prediction', operation_type='get', outcome='miss')

# Ensure existing health check tests (if any) are preserved or adapted
# For example:
@pytest.mark.asyncio
async def test_health_check_healthy(optimized_predictor: OptimizedPredictor, mock_settings: Settings):
    # This test assumes the predictor initialized correctly via the fixture (fallback path)
    # The fixture ensures interpreter is mocked and model_path is set from mock_settings.
    if not optimized_predictor.interpreter:
        pytest.skip("Skipping health check as interpreter was not initialized by fixture.")

    health = await optimized_predictor.health_check()
    assert health['status'] == "healthy"
    # optimized_predictor.model_path is set by the fixture to mock_settings.ML_MODEL_PATH
    # In the refactored __init__, self.model_path is updated.
    # The fixture sets up a scenario where fallback path is used.
    assert health['model_path'] == mock_settings.ML_MODEL_PATH # This should be correct as fixture uses fallback
```
