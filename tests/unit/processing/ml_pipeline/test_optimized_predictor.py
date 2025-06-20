import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path
from typing import List, Dict, Any, Optional # Added Optional

# Conditional import for testing
TFLITE_INSTALLED_FOR_TEST = False
# Define a placeholder for tflite.Interpreter if the module isn't found
# This allows tests to run and mock this type.
try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
    # If the above import works, we can try to import the module itself for monkeypatching
    import tflite_runtime.interpreter as tflite_interpreter_module # For monkeypatching later
    TFLITE_INSTALLED_FOR_TEST = True
except ImportError:
    # Create a dummy class that mimics tflite.Interpreter for type hinting and mocking if tflite_runtime is not installed
    class TFLiteInterpreter: # type: ignore
        def __init__(self, model_path=None, model_content=None, experimental_delegates=None, num_threads=None):
            self.model_path = model_path # Store for verification
        def allocate_tensors(self): pass
        def get_input_details(self): return [{'shape': np.array([1, 7], dtype=np.int32), 'index': 0, 'dtype': np.float32}]
        def get_output_details(self): return [{'shape': np.array([1, 2], dtype=np.int32), 'index': 0, 'dtype': np.float32}]
        def set_tensor(self, tensor_index, value): pass
        def invoke(self): pass
        def get_tensor(self, tensor_index): return np.array([[0.25, 0.75]], dtype=np.float32) # Default mock output
    logger_for_dummy = structlog.get_logger(__name__)
    logger_for_dummy = structlog.get_logger(__name__) # Ensure structlog is imported if used here
    logger_for_dummy.warn("Dummy TFLiteInterpreter created as tflite_runtime is not installed.")


from claims_processor.src.processing.ml_pipeline.optimized_predictor import OptimizedPredictor, TFLITE_AVAILABLE as PRED_TFLITE_AVAILABLE
from claims_processor.src.core.config.settings import Settings, get_settings # For mocking settings
from claims_processor.src.core.monitoring.app_metrics import MetricsCollector # Added

# Use settings for model path and feature count consistently
BASE_SETTINGS = get_settings()
MODEL_PATH_SETTING = BASE_SETTINGS.ML_MODEL_PATH or "dummy_model.tflite"
FEATURE_COUNT_SETTING = BASE_SETTINGS.ML_FEATURE_COUNT
APPROVAL_THRESHOLD_SETTING = BASE_SETTINGS.ML_APPROVAL_THRESHOLD

@pytest.fixture
def mock_tflite_interpreter_instance() -> MagicMock:
    """Mocks a TFLite Interpreter instance with basic methods."""
    mock_instance = MagicMock(spec=TFLiteInterpreter)
    mock_instance.allocate_tensors = MagicMock()
    mock_instance.get_input_details = MagicMock(return_value=[{'shape': np.array([1, FEATURE_COUNT_SETTING], dtype=np.int32), 'index': 0, 'dtype': np.float32}])
    mock_instance.get_output_details = MagicMock(return_value=[{'shape': np.array([1, 2], dtype=np.int32), 'index': 0, 'dtype': np.float32}])
    mock_instance.set_tensor = MagicMock()
    mock_instance.invoke = MagicMock()
    # Default output for a single prediction (batch size 1, 2 classes)
    mock_instance.get_tensor = MagicMock(return_value=np.array([[0.25, 0.75]], dtype=np.float32))
    return mock_instance

@pytest.fixture
def mock_tflite_interpreter_constructor(monkeypatch, mock_tflite_interpreter_instance: MagicMock) -> Optional[MagicMock]:
    """Mocks the TFLite Interpreter constructor if TFLite is considered available."""
    if TFLITE_INSTALLED_FOR_TEST and PRED_TFLITE_AVAILABLE:
        # Path to the Interpreter class within the module where OptimizedPredictor imports it
        interpreter_path = "claims_processor.src.processing.ml_pipeline.optimized_predictor.Interpreter"
        mock_constructor = MagicMock(return_value=mock_tflite_interpreter_instance)
        monkeypatch.setattr(interpreter_path, mock_constructor)
        return mock_constructor
    return None

@pytest.fixture # Added
def mock_metrics_collector() -> MagicMock:
    return MagicMock(spec=MetricsCollector)

# --- Initialization Tests ---
def test_init_tflite_not_available(monkeypatch, mock_metrics_collector: MagicMock):
    monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE", False)
    predictor = OptimizedPredictor(
        model_path=MODEL_PATH_SETTING,
        feature_count=FEATURE_COUNT_SETTING,
        metrics_collector=mock_metrics_collector
    )
    assert predictor.interpreter is None

def test_init_model_path_not_configured(monkeypatch, mock_metrics_collector: MagicMock):
    monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE", True)
    if TFLITE_INSTALLED_FOR_TEST: # Ensure dummy or real Interpreter is available for import
         monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.Interpreter", TFLiteInterpreter)
    predictor = OptimizedPredictor(
        model_path="",
        feature_count=FEATURE_COUNT_SETTING,
        metrics_collector=mock_metrics_collector
    ) # Empty path
    assert predictor.interpreter is None

def test_init_model_file_not_found(monkeypatch, mock_metrics_collector: MagicMock):
    monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE", True)
    if TFLITE_INSTALLED_FOR_TEST:
         monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.Interpreter", TFLiteInterpreter)
    monkeypatch.setattr(Path, 'is_file', lambda self: False)
    predictor = OptimizedPredictor(
        model_path="non_existent.tflite",
        feature_count=FEATURE_COUNT_SETTING,
        metrics_collector=mock_metrics_collector
    )
    assert predictor.interpreter is None

@pytest.mark.skipif(not TFLITE_INSTALLED_FOR_TEST, reason="TFLite runtime not installed/found in test environment")
def test_init_model_loads_successfully(mock_tflite_interpreter_constructor: MagicMock, mock_tflite_interpreter_instance: MagicMock, monkeypatch, mock_metrics_collector: MagicMock):
    if not mock_tflite_interpreter_constructor: pytest.skip("Interpreter constructor not patched") # Should not happen if skipif works
    monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE", True)
    monkeypatch.setattr(Path, 'is_file', lambda self: True)

    predictor = OptimizedPredictor(
        model_path=MODEL_PATH_SETTING,
        feature_count=FEATURE_COUNT_SETTING,
        metrics_collector=mock_metrics_collector
    )

    mock_tflite_interpreter_constructor.assert_called_once_with(model_path=MODEL_PATH_SETTING)
    assert predictor.interpreter == mock_tflite_interpreter_instance
    mock_tflite_interpreter_instance.allocate_tensors.assert_called_once()

# --- Prediction Tests ---
@pytest.mark.asyncio
async def test_predict_batch_tflite_not_available(monkeypatch, mock_metrics_collector: MagicMock):
    monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE", False)
    predictor = OptimizedPredictor(
        model_path=MODEL_PATH_SETTING,
        feature_count=FEATURE_COUNT_SETTING,
        metrics_collector=mock_metrics_collector
    )
    features_batch = [np.random.rand(1, FEATURE_COUNT_SETTING).astype(np.float32)]
    predictions = await predictor.predict_batch(features_batch)
    assert len(predictions) == 1
    assert "error" in predictions[0]
    mock_metrics_collector.record_ml_prediction.assert_called_once_with(outcome="ML_ERROR_RUNTIME_UNAVAILABLE", confidence_score=None)
    mock_metrics_collector.record_ml_inference_duration.assert_not_called()


@pytest.mark.skipif(not TFLITE_INSTALLED_FOR_TEST, reason="TFLite runtime not installed/found in test environment")
@pytest.mark.asyncio
async def test_predict_batch_with_loaded_model(mock_tflite_interpreter_constructor: MagicMock, mock_tflite_interpreter_instance: MagicMock, monkeypatch, mock_metrics_collector: MagicMock):
    if not mock_tflite_interpreter_constructor: pytest.skip("Interpreter constructor not patched")
    monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE", True)
    monkeypatch.setattr(Path, 'is_file', lambda self: True) # Assume file exists for loading

    predictor = OptimizedPredictor(
        model_path=MODEL_PATH_SETTING,
        feature_count=FEATURE_COUNT_SETTING,
        metrics_collector=mock_metrics_collector
    )
    assert predictor.interpreter == mock_tflite_interpreter_instance # Model "loaded"

    # Test with a batch of 2 feature sets
    features1 = np.random.rand(1, FEATURE_COUNT_SETTING).astype(np.float32)
    features2 = np.random.rand(1, FEATURE_COUNT_SETTING).astype(np.float32)
    features_batch = [features1, features2]

    # Configure mock interpreter's get_tensor for two calls
    output_val1 = np.array([[0.1, 0.9]], dtype=np.float32) # Corresponds to ML_APPROVED with default threshold
    output_val2 = np.array([[0.8, 0.2]], dtype=np.float32) # Corresponds to ML_REJECTED
    mock_tflite_interpreter_instance.get_tensor.side_effect = [output_val1, output_val2]

    predictions = await predictor.predict_batch(features_batch)

    assert mock_tflite_interpreter_instance.set_tensor.call_count == 2
    mock_tflite_interpreter_instance.set_tensor.assert_any_call(predictor.input_details[0]['index'], features1)
    mock_tflite_interpreter_instance.set_tensor.assert_any_call(predictor.input_details[0]['index'], features2)
    assert mock_tflite_interpreter_instance.invoke.call_count == 2
    assert mock_tflite_interpreter_instance.get_tensor.call_count == 2

    assert len(predictions) == 2
    assert predictions[0]['ml_score'] == pytest.approx(0.9)
    assert predictions[0]['ml_derived_decision'] == "ML_APPROVED"
    assert predictions[1]['ml_score'] == pytest.approx(0.2)
    assert predictions[1]['ml_derived_decision'] == "ML_REJECTED"

    # Metrics assertions
    assert mock_metrics_collector.record_ml_inference_duration.call_count == 2
    mock_metrics_collector.record_ml_inference_duration.assert_any_call(pytest.approx(0, abs=1e-1)) # Duration is dynamic

    assert mock_metrics_collector.record_ml_prediction.call_count == 2
    mock_metrics_collector.record_ml_prediction.assert_any_call(outcome="ML_APPROVED", confidence_score=pytest.approx(0.9))
    mock_metrics_collector.record_ml_prediction.assert_any_call(outcome="ML_REJECTED", confidence_score=pytest.approx(0.2))


@pytest.mark.asyncio
async def test_predict_batch_input_validation(monkeypatch, mock_metrics_collector: MagicMock):
    # Simulate TFLite available and model loaded for this test, but focus on input validation
    monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE", True)
    if TFLITE_INSTALLED_FOR_TEST:
         monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.Interpreter", TFLiteInterpreter)
    else: # If TFLite not installed, mock Interpreter to allow OptimizedPredictor init to "succeed"
        monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.Interpreter", MagicMock(return_value=MagicMock(spec=TFLiteInterpreter)))

    monkeypatch.setattr(Path, 'is_file', lambda self: True)
    predictor = OptimizedPredictor(
        model_path=MODEL_PATH_SETTING,
        feature_count=FEATURE_COUNT_SETTING,
        metrics_collector=mock_metrics_collector
    )
    if predictor.interpreter is None and TFLITE_INSTALLED_FOR_TEST: # If mock setup failed, re-mock interpreter directly
        predictor.interpreter = MagicMock(spec=TFLiteInterpreter)
        predictor.input_details = [{'shape': np.array([1, FEATURE_COUNT_SETTING], dtype=np.int32), 'index': 0, 'dtype': np.float32}]
        predictor.output_details = [{'shape': np.array([1, 2], dtype=np.int32), 'index': 0, 'dtype': np.float32}]


    # Test wrong feature shape (not (1, N))
    predictions_wrong_shape = await predictor.predict_batch([np.random.rand(FEATURE_COUNT_SETTING).astype(np.float32)]) # 1D instead of 2D
    assert "error" in predictions_wrong_shape[0]
    assert predictions_wrong_shape[0]["error"] == "Feature shape mismatch"
    mock_metrics_collector.record_ml_prediction.assert_called_with(outcome="ML_ERROR_INVALID_INPUT", confidence_score=None)


    # Test wrong feature count
    predictions_wrong_count = await predictor.predict_batch([np.random.rand(1, FEATURE_COUNT_SETTING + 1).astype(np.float32)])
    assert "error" in predictions_wrong_count[0]
    # This will also be caught by the shape check if feature_count is part of shape detail
    # The error message might depend on how OptimizedPredictor checks this.
    # Based on current OptimizedPredictor, it's "Feature shape mismatch"
    assert predictions_wrong_count[0]["error"] == "Feature shape mismatch"
    # This will be the second call to record_ml_prediction with this outcome
    mock_metrics_collector.record_ml_prediction.assert_called_with(outcome="ML_ERROR_INVALID_INPUT", confidence_score=None)
    assert mock_metrics_collector.record_ml_prediction.call_count == 2


    # Test wrong dtype (if not converted and causes error)
    # Current OptimizedPredictor tries to convert to float32, so this might not directly cause an "ML_ERROR_INVALID_INPUT"
    # unless the conversion itself fails in a way that's caught.
    # If it proceeds, it might fail at set_tensor if the mock is strict, or pass if mock is loose.
    # Assuming conversion is successful, no new error metric here.
    # If we wanted to test a conversion failure, we'd need to mock the conversion part.
    # For now, no specific metric assertion here for dtype if it's handled by conversion.
    # mock_metrics_collector.reset_mock() # Reset for next specific check if needed

    # Ensure invoke and inference duration not called for these error cases
    mock_metrics_collector.record_ml_inference_duration.assert_not_called()
    if hasattr(predictor.interpreter, 'invoke'): # Check if interpreter was even set
        predictor.interpreter.invoke.assert_not_called()


# --- Health Check Tests ---
# ... (Existing health check tests can be kept, ensuring they use monkeypatch for TFLITE_AVAILABLE and Path.is_file) ...
@pytest.mark.asyncio
async def test_health_check_tflite_not_available(monkeypatch, mock_metrics_collector: MagicMock):
    monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE", False)
    predictor = OptimizedPredictor(
        model_path=MODEL_PATH_SETTING,
        feature_count=FEATURE_COUNT_SETTING,
        metrics_collector=mock_metrics_collector
    )
    health = await predictor.health_check()
    assert health['status'] == "unhealthy"; assert "runtime not available" in health['reason']

@pytest.mark.skipif(not TFLITE_INSTALLED_FOR_TEST, reason="TFLite runtime not installed/found in test environment")
@pytest.mark.asyncio
async def test_health_check_healthy(mock_tflite_interpreter_constructor: Optional[MagicMock], monkeypatch, mock_metrics_collector: MagicMock):
    if not mock_tflite_interpreter_constructor: pytest.skip("Interpreter constructor not patched")
    monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE", True)
    monkeypatch.setattr(Path, 'is_file', lambda self: True)
    predictor = OptimizedPredictor(
        model_path=MODEL_PATH_SETTING,
        feature_count=FEATURE_COUNT_SETTING,
        metrics_collector=mock_metrics_collector
    )
    health = await predictor.health_check()
    assert health['status'] == "healthy"
    assert health['model_path'] == MODEL_PATH_SETTING
    # Check if input_details and output_details exist before trying to stringify them
    if predictor.input_details:
        assert str(predictor.input_details) in health['input_details']
    else:
        assert "Input details not available" in health['input_details']

    if predictor.output_details:
        assert str(predictor.output_details) in health['output_details']
    else:
        assert "Output details not available" in health['output_details']
