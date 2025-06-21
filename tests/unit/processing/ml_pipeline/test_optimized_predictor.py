import pytest
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
from typing import List, Dict, Any, Optional
import time # For mocking time for TTL tests
from cachetools import TTLCache # For type hinting

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

                predictor = OptimizedPredictor(
                    model_path=mock_settings.ML_MODEL_PATH,
                    metrics_collector=mock_metrics_collector,
                    feature_count=mock_settings.ML_FEATURE_COUNT
                )
                # Store the mocked interpreter instance on the predictor for easy access in tests
                predictor.mock_interpreter_instance = mock_interpreter_instance
                return predictor


# --- Initialization Tests (can be kept or simplified if covered by optimized_predictor fixture) ---
def test_init_tflite_not_available(mock_metrics_collector: MagicMock, mock_settings: Settings):
    with patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.get_settings', return_value=mock_settings):
        with patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE', False):
            predictor = OptimizedPredictor(
                model_path=mock_settings.ML_MODEL_PATH,
                feature_count=mock_settings.ML_FEATURE_COUNT,
                metrics_collector=mock_metrics_collector
            )
            assert predictor.interpreter is None
            assert isinstance(predictor.prediction_cache, TTLCache) # Cache should still init


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
    # This test assumes the predictor initialized correctly (mocked TFLite available and model file "exists")
    if not optimized_predictor.interpreter: # If init failed due to TFLITE_AVAILABLE=False in some other fixture state
        pytest.skip("Skipping health check as interpreter was not initialized, TFLITE_AVAILABLE might be False.")

    health = await optimized_predictor.health_check()
    assert health['status'] == "healthy"
    assert health['model_path'] == mock_settings.ML_MODEL_PATH
```
