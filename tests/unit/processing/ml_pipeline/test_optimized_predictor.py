import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path
from typing import List, Dict, Any # For type hints

# Conditional import for testing
TFLITE_INSTALLED_FOR_TEST = False
tflite_interpreter_module = None
try:
    # Attempt to import the specific Interpreter class
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
    # If successful, also make the module available if OptimizedPredictor uses tflite.Interpreter
    import tflite_runtime.interpreter as tflite_interpreter_module
    TFLITE_INSTALLED_FOR_TEST = True
except ImportError:
    # Define a dummy Interpreter if tflite_runtime is not available
    class TFLiteInterpreter: # type: ignore
        def __init__(self, model_path=None, model_content=None, experimental_delegates=None, num_threads=None): pass
        def allocate_tensors(self): pass
        def get_input_details(self): return [{'index': 0, 'shape': np.array([1, 7])}] # Dummy
        def get_output_details(self): return [{'index': 0}] # Dummy
        def set_tensor(self, tensor_index, value): pass
        def invoke(self): pass
        def get_tensor(self, tensor_index): return np.array([[0.5, 0.5]], dtype=np.float32) # Dummy output

from claims_processor.src.processing.ml_pipeline.optimized_predictor import OptimizedPredictor, TFLITE_AVAILABLE as PRED_TFLITE_AVAILABLE
from claims_processor.src.core.config.settings import Settings, get_settings


MODEL_PATH_SETTING = get_settings().ML_MODEL_PATH or "dummy_model.tflite" # Use from settings or a default for tests
FEATURE_COUNT_SETTING = get_settings().ML_FEATURE_COUNT

@pytest.fixture
def mock_tflite_interpreter_instance(monkeypatch):
    """Mocks a TFLite Interpreter instance."""
    mock_instance = MagicMock(spec=TFLiteInterpreter) # Use the (potentially dummy) Interpreter
    mock_instance.allocate_tensors = MagicMock()
    mock_instance.get_input_details = MagicMock(return_value=[{'shape': np.array([1, FEATURE_COUNT_SETTING], dtype=np.int32), 'index': 0}])
    mock_instance.get_output_details = MagicMock(return_value=[{'shape': np.array([1, 2], dtype=np.int32), 'index': 0}]) # Assuming 2 output classes
    mock_instance.set_tensor = MagicMock()
    mock_instance.invoke = MagicMock()
    # Default output for a single prediction (batch size 1, 2 classes)
    mock_instance.get_tensor = MagicMock(return_value=np.array([[0.25, 0.75]], dtype=np.float32))
    return mock_instance

@pytest.fixture
def mock_tflite_interpreter_constructor(monkeypatch, mock_tflite_interpreter_instance: MagicMock):
    """Mocks the TFLite Interpreter constructor."""
    if TFLITE_INSTALLED_FOR_TEST and PRED_TFLITE_AVAILABLE: # Only patch if the real one could be imported
        mock_constructor = MagicMock(return_value=mock_tflite_interpreter_instance)
        monkeypatch.setattr("tflite_runtime.interpreter.Interpreter", mock_constructor)
        return mock_constructor
    return None # Indicates constructor not patched as TFLite not available for predictor

# --- Initialization Tests ---
def test_init_tflite_not_available(monkeypatch):
    monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE", False)
    # Ensure the Interpreter symbol is None if the import failed
    monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.Interpreter", None)

    predictor = OptimizedPredictor(model_path=MODEL_PATH_SETTING, feature_count=FEATURE_COUNT_SETTING)
    assert predictor.interpreter is None
    assert predictor.model_path == MODEL_PATH_SETTING
    assert predictor.feature_count == FEATURE_COUNT_SETTING

def test_init_model_path_not_configured(monkeypatch):
    # Simulate TFLite being available but model path being None
    monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE", True)
    if TFLITE_INSTALLED_FOR_TEST: # Ensure Interpreter symbol exists if TFLITE_AVAILABLE is true
         monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.Interpreter", TFLiteInterpreter)

    predictor = OptimizedPredictor(model_path=None, feature_count=FEATURE_COUNT_SETTING)
    assert predictor.interpreter is None

def test_init_model_file_not_found(monkeypatch):
    monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE", True)
    if TFLITE_INSTALLED_FOR_TEST:
         monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.Interpreter", TFLiteInterpreter)

    monkeypatch.setattr(Path, 'is_file', lambda self: False) # Mock Path.is_file() to return False
    predictor = OptimizedPredictor(model_path="non_existent.tflite", feature_count=FEATURE_COUNT_SETTING)
    assert predictor.interpreter is None

@pytest.mark.skipif(not TFLITE_INSTALLED_FOR_TEST, reason="TFLite runtime not available in test environment")
def test_init_model_loads_successfully(mock_tflite_interpreter_constructor: Optional[MagicMock], mock_tflite_interpreter_instance: MagicMock, monkeypatch):
    if not mock_tflite_interpreter_constructor: pytest.skip("TFLite runtime not available to patch Interpreter")

    monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE", True)
    monkeypatch.setattr(Path, 'is_file', lambda self: True)

    predictor = OptimizedPredictor(model_path=MODEL_PATH_SETTING, feature_count=FEATURE_COUNT_SETTING)

    mock_tflite_interpreter_constructor.assert_called_once_with(model_path=MODEL_PATH_SETTING)
    assert predictor.interpreter == mock_tflite_interpreter_instance
    mock_tflite_interpreter_instance.allocate_tensors.assert_called_once()
    assert predictor.input_details is not None
    assert predictor.output_details is not None


# --- Prediction Tests ---
@pytest.mark.asyncio
async def test_predict_batch_tflite_not_available(monkeypatch):
    monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE", False)
    monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.Interpreter", None)
    predictor = OptimizedPredictor(model_path=MODEL_PATH_SETTING, feature_count=FEATURE_COUNT_SETTING)

    features_batch = [np.random.rand(1, FEATURE_COUNT_SETTING).astype(np.float32)]
    predictions = await predictor.predict_batch(features_batch)

    assert len(predictions) == 1
    assert "error" in predictions[0]
    assert predictions[0]["error"] == "Predictor not available or TFLite runtime missing"

@pytest.mark.asyncio
async def test_predict_batch_interpreter_not_loaded(monkeypatch):
    # Simulate TFLite available, but model loading failed (interpreter is None)
    monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE", True)
    if TFLITE_INSTALLED_FOR_TEST: # Ensure Interpreter symbol exists
         monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.Interpreter", TFLiteInterpreter)

    predictor = OptimizedPredictor(model_path=MODEL_PATH_SETTING, feature_count=FEATURE_COUNT_SETTING)
    predictor.interpreter = None # Force interpreter to None after init

    features_batch = [np.random.rand(1, FEATURE_COUNT_SETTING).astype(np.float32)]
    predictions = await predictor.predict_batch(features_batch)

    assert len(predictions) == 1
    assert "error" in predictions[0]
    assert predictions[0]["error"] == "Predictor not available or TFLite runtime missing"

@pytest.mark.skipif(not TFLITE_INSTALLED_FOR_TEST, reason="TFLite runtime not available in test environment")
@pytest.mark.asyncio
async def test_predict_batch_with_mocked_interpreter(mock_tflite_interpreter_constructor: Optional[MagicMock], mock_tflite_interpreter_instance: MagicMock, monkeypatch):
    if not mock_tflite_interpreter_constructor: pytest.skip("TFLite runtime not available to patch Interpreter")

    monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE", True)
    monkeypatch.setattr(Path, 'is_file', lambda self: True) # Assume model file exists

    predictor = OptimizedPredictor(model_path=MODEL_PATH_SETTING, feature_count=FEATURE_COUNT_SETTING)
    assert predictor.interpreter == mock_tflite_interpreter_instance # Ensure mock interpreter is used

    # FeatureExtractor returns list of (1,N), predictor's predict_batch iterates this.
    # Inside predict_batch, it handles the (1,N) by reshaping if necessary before set_tensor.
    # For this test, let's create features as list of 1D arrays as per predict_batch's internal expectation after its own reshaping logic.
    # Or, let's test with the (1,N) shape directly if the mock is simple.
    # The mock implementation of predict_batch in OptimizedPredictor handles (1,N) -> (N,)

    features1 = np.random.rand(1, FEATURE_COUNT_SETTING).astype(np.float32)
    features2 = np.random.rand(1, FEATURE_COUNT_SETTING).astype(np.float32)
    features_batch = [features1, features2]

    # Configure mock get_settings for ML_APPROVAL_THRESHOLD inside predict_batch's mock logic
    with patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.get_settings') as mock_get_settings_pred:
        mock_get_settings_pred.return_value = Settings(ML_APPROVAL_THRESHOLD=0.5) # Example threshold

        # If we were testing actual TFLite calls, we'd configure mock_tf_lite_interpreter_instance.get_tensor
        # For the current mock predict_batch logic, it doesn't use the interpreter.
        # This test is for the *stubbed* predict_batch logic.
        predictions = await predictor.predict_batch(features_batch)

    assert len(predictions) == 2
    for p in predictions:
        assert 'ml_score' in p
        assert 'ml_derived_decision' in p
        assert p['ml_derived_decision'] in ["ML_APPROVED", "ML_REJECTED"]

@pytest.mark.asyncio
async def test_predict_batch_feature_length_mismatch(monkeypatch):
    # Test with TFLite mocked away or predictor not fully initialized
    monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE", False)
    predictor = OptimizedPredictor(model_path=MODEL_PATH_SETTING, feature_count=FEATURE_COUNT_SETTING)

    features_wrong_length = [np.random.rand(1, FEATURE_COUNT_SETTING + 1).astype(np.float32)]

    # Mock settings for approval_threshold as it's used in the mock predict_batch
    with patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.get_settings') as mock_get_settings_pred:
        mock_get_settings_pred.return_value = Settings(ML_APPROVAL_THRESHOLD=0.5)
        predictions = await predictor.predict_batch(features_wrong_length)

    assert len(predictions) == 1
    assert predictions[0].get("error") == "Feature length mismatch"

# --- Health Check Tests ---
@pytest.mark.asyncio
async def test_health_check_tflite_not_available(monkeypatch):
    monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE", False)
    predictor = OptimizedPredictor(model_path=MODEL_PATH_SETTING, feature_count=FEATURE_COUNT_SETTING)
    health = await predictor.health_check()
    assert health['status'] == "unhealthy"
    assert "TensorFlow Lite runtime not available" in health['reason']

@pytest.mark.asyncio
async def test_health_check_model_file_not_found(monkeypatch):
    monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE", True)
    if TFLITE_INSTALLED_FOR_TEST: # Ensure Interpreter symbol exists if TFLITE_AVAILABLE is true
         monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.Interpreter", TFLiteInterpreter)
    monkeypatch.setattr(Path, 'is_file', lambda self: False)
    predictor = OptimizedPredictor(model_path="bad_path.tflite", feature_count=FEATURE_COUNT_SETTING)
    health = await predictor.health_check()
    assert health['status'] == "unhealthy"
    assert "Model file not found" in health['reason']

@pytest.mark.asyncio
async def test_health_check_interpreter_not_loaded(monkeypatch):
    monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE", True)
    if TFLITE_INSTALLED_FOR_TEST:
         monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.Interpreter", TFLiteInterpreter)
    monkeypatch.setattr(Path, 'is_file', lambda self: True) # File exists
    # Simulate Interpreter loading failure by patching constructor to raise error
    with patch("tflite_runtime.interpreter.Interpreter", side_effect=Exception("Failed to load interpreter")):
        predictor = OptimizedPredictor(model_path=MODEL_PATH_SETTING, feature_count=FEATURE_COUNT_SETTING)
    health = await predictor.health_check()
    assert health['status'] == "unhealthy"
    assert "TFLite interpreter not loaded" in health['reason']

@pytest.mark.skipif(not TFLITE_INSTALLED_FOR_TEST, reason="TFLite runtime not available in test environment")
@pytest.mark.asyncio
async def test_health_check_healthy(mock_tflite_interpreter_constructor: Optional[MagicMock], mock_tflite_interpreter_instance: MagicMock, monkeypatch):
    if not mock_tflite_interpreter_constructor: pytest.skip("TFLite runtime not available to patch Interpreter")
    monkeypatch.setattr("claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE", True)
    monkeypatch.setattr(Path, 'is_file', lambda self: True)

    predictor = OptimizedPredictor(model_path=MODEL_PATH_SETTING, feature_count=FEATURE_COUNT_SETTING)
    health = await predictor.health_check()
    assert health['status'] == "healthy"
    assert health['model_path'] == MODEL_PATH_SETTING
    assert str(predictor.input_details) in health['input_details'] # Check if details are stringified
    assert str(predictor.output_details) in health['output_details']
