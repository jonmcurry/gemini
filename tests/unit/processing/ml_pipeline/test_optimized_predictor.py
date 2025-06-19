import pytest
import numpy as np

from claims_processor.src.processing.ml_pipeline.optimized_predictor import OptimizedPredictor
from claims_processor.src.core.config.settings import get_settings

from unittest.mock import patch, MagicMock # Add MagicMock

# DUMMY_MODEL_PATH can be removed if using settings directly

@pytest.fixture
def mock_tf_lite_interpreter(): # A mock for tf.lite.Interpreter
    mock_interpreter = MagicMock()
    # Mock methods that would be called:
    mock_interpreter.allocate_tensors = MagicMock()
    # Make get_input_details/get_output_details return a list of dicts as TF Lite does
    mock_interpreter.get_input_details = MagicMock(return_value=[{'index': 0, 'shape': np.array([1, get_settings().ML_FEATURE_COUNT], dtype=np.int32)}])
    mock_interpreter.get_output_details = MagicMock(return_value=[{'index': 0, 'shape': np.array([1, 2], dtype=np.int32)}])
    mock_interpreter.set_tensor = MagicMock()
    mock_interpreter.invoke = MagicMock()
    mock_interpreter.get_tensor = MagicMock(return_value=np.random.rand(1, 2).astype(np.float32)) # Default return for get_tensor
    return mock_interpreter

@pytest.fixture
def predictor(monkeypatch) -> OptimizedPredictor: # Add monkeypatch for TENSORFLOW_AVAILABLE if needed by some tests explicitly
    settings = get_settings()
    model_path = settings.ML_MODEL_PATH if settings.ML_MODEL_PATH else "dummy/path/model.tflite"
    # For most tests here, we'll be patching Interpreter, so actual path doesn't always matter.
    # Some tests might want to test path handling, so using settings path is good.
    return OptimizedPredictor(model_path=model_path)

# Test _load_model behavior
def test_load_model_tensorflow_not_available(monkeypatch):
    monkeypatch.setattr('claims_processor.src.processing.ml_pipeline.optimized_predictor.TENSORFLOW_AVAILABLE', False)
    settings = get_settings()
    model_path = settings.ML_MODEL_PATH if settings.ML_MODEL_PATH else "dummy/path/model.tflite"

    # Patch 'tf' within the scope of OptimizedPredictor to ensure it's seen as None if used
    with patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.tf', None):
        predictor = OptimizedPredictor(model_path=model_path)
        assert predictor.model is None
        # Here, OptimizedPredictor's __init__ calls _load_model.
        # _load_model should log a warning and return early.

@patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.tf.lite.Interpreter')
def test_load_model_file_not_found(mock_interpreter_class_constructor: MagicMock, monkeypatch):
    # Ensure TensorFlow is "available" for this test's logic path
    monkeypatch.setattr('claims_processor.src.processing.ml_pipeline.optimized_predictor.TENSORFLOW_AVAILABLE', True)
    # Make sure 'tf' itself is not None for 'tf.lite.Interpreter' to be valid before patching
    monkeypatch.setattr('claims_processor.src.processing.ml_pipeline.optimized_predictor.tf', MagicMock())


    mock_interpreter_class_constructor.side_effect = FileNotFoundError("Mocked: Model file not found")

    predictor = OptimizedPredictor(model_path="non_existent_path.tflite")
    assert predictor.model is None
    mock_interpreter_class_constructor.assert_called_with(model_path="non_existent_path.tflite")

@patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.tf.lite.Interpreter')
def test_load_model_success(mock_interpreter_class_constructor: MagicMock, mock_tf_lite_interpreter: MagicMock, monkeypatch):
    monkeypatch.setattr('claims_processor.src.processing.ml_pipeline.optimized_predictor.TENSORFLOW_AVAILABLE', True)
    monkeypatch.setattr('claims_processor.src.processing.ml_pipeline.optimized_predictor.tf', MagicMock())

    mock_interpreter_class_constructor.return_value = mock_tf_lite_interpreter
    settings = get_settings()
    model_path = settings.ML_MODEL_PATH if settings.ML_MODEL_PATH else "dummy/path/model.tflite"

    predictor = OptimizedPredictor(model_path=model_path)
    assert predictor.model == mock_tf_lite_interpreter
    assert predictor.input_details is not None
    assert predictor.output_details is not None
    mock_interpreter_class_constructor.assert_called_with(model_path=model_path)
    mock_tf_lite_interpreter.allocate_tensors.assert_called_once()
    mock_tf_lite_interpreter.get_input_details.assert_called_once()
    mock_tf_lite_interpreter.get_output_details.assert_called_once()

@pytest.mark.asyncio
async def test_predict_async_with_mocked_successful_model_load(mock_tf_lite_interpreter: MagicMock, monkeypatch):
    settings = get_settings()
    model_path = settings.ML_MODEL_PATH if settings.ML_MODEL_PATH else "dummy/path/model.tflite"

    monkeypatch.setattr('claims_processor.src.processing.ml_pipeline.optimized_predictor.TENSORFLOW_AVAILABLE', True)
    monkeypatch.setattr('claims_processor.src.processing.ml_pipeline.optimized_predictor.tf', MagicMock())

    with patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.tf.lite.Interpreter', return_value=mock_tf_lite_interpreter):
        predictor = OptimizedPredictor(model_path=model_path)
        assert predictor.model == mock_tf_lite_interpreter

        num_samples = 3
        features = np.random.rand(num_samples, settings.ML_FEATURE_COUNT).astype(np.float32)

        # Adjust mock_tf_lite_interpreter's get_tensor for this specific call's expected output shape
        mock_tf_lite_interpreter.get_tensor.return_value = np.random.rand(num_samples, 2).astype(np.float32)

        predictions = await predictor.predict_async(features)

        mock_tf_lite_interpreter.set_tensor.assert_called_once()
        # Example: Assert arguments of set_tensor
        # called_args, _ = mock_tf_lite_interpreter.set_tensor.call_args
        # assert called_args[0] == predictor.input_details[0]['index']
        # np.testing.assert_array_equal(called_args[1], features)
        mock_tf_lite_interpreter.invoke.assert_called_once()
        mock_tf_lite_interpreter.get_tensor.assert_called_once_with(predictor.output_details[0]['index'])

        assert predictions.shape == (num_samples, 2)
        assert predictions.dtype == np.float32

# Test cases for when model is None (i.e., loading failed or TF not available)
@pytest.mark.asyncio
async def test_predict_async_returns_dummy_if_model_none(predictor: OptimizedPredictor, monkeypatch): # predictor fixture will try to load model
    # This test will cover cases where TF might be available but model loading failed for other reasons,
    # or if TF was initially available but we monkeypatch it to False here.
    monkeypatch.setattr(predictor, 'model', None) # Force model to be None

    settings = get_settings()
    dummy_features = np.random.rand(1, settings.ML_FEATURE_COUNT).astype(np.float32)
    predictions = await predictor.predict_async(dummy_features)

    assert isinstance(predictions, np.ndarray), "Predictions should be a NumPy array"
    assert predictions.shape == (1, 2), f"Expected shape (1, 2) for dummy prediction, but got {predictions.shape}"
    assert predictions.dtype == np.float32, "Dummy predictions should be float32 type"

@pytest.mark.asyncio
async def test_predict_async_returns_dummy_if_tf_not_available(monkeypatch):
    monkeypatch.setattr('claims_processor.src.processing.ml_pipeline.optimized_predictor.TENSORFLOW_AVAILABLE', False)
    monkeypatch.setattr('claims_processor.src.processing.ml_pipeline.optimized_predictor.tf', None)

    settings = get_settings()
    model_path = settings.ML_MODEL_PATH if settings.ML_MODEL_PATH else "dummy/path/model.tflite"
    predictor_no_tf = OptimizedPredictor(model_path=model_path) # _load_model will set model to None
    assert predictor_no_tf.model is None

    dummy_features = np.random.rand(1, settings.ML_FEATURE_COUNT).astype(np.float32)
    predictions = await predictor_no_tf.predict_async(dummy_features)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (1, 2)
    assert predictions.dtype == np.float32

# Refined initialization test
@patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.tf.lite.Interpreter')
def test_predictor_initialization_after_load_attempt(mock_interpreter_class: MagicMock, monkeypatch):
    settings = get_settings()
    model_path = settings.ML_MODEL_PATH if settings.ML_MODEL_PATH else "dummy/path/model.tflite"

    # Test when TF is not available
    monkeypatch.setattr('claims_processor.src.processing.ml_pipeline.optimized_predictor.TENSORFLOW_AVAILABLE', False)
    monkeypatch.setattr('claims_processor.src.processing.ml_pipeline.optimized_predictor.tf', None)
    predictor_no_tf = OptimizedPredictor(model_path=model_path)
    assert predictor_no_tf.model_path == model_path
    assert predictor_no_tf.model is None
    mock_interpreter_class.assert_not_called()

    # Test when TF is available but model file not found
    mock_interpreter_class.reset_mock() # Reset for next scenario
    monkeypatch.setattr('claims_processor.src.processing.ml_pipeline.optimized_predictor.TENSORFLOW_AVAILABLE', True)
    # Ensure 'tf' module mock itself is available for tf.lite.Interpreter attribute access
    tf_mock = MagicMock()
    tf_mock.lite.Interpreter.side_effect = FileNotFoundError("mocked file not found")
    monkeypatch.setattr('claims_processor.src.processing.ml_pipeline.optimized_predictor.tf', tf_mock)

    # The @patch decorator above still targets 'tf.lite.Interpreter' globally for the constructor call.
    # We need to ensure the side_effect is on the correct mock.
    # This is tricky because of the top-level 'tf' import and the patch.
    # A cleaner way: patch 'tf' itself if TENSORFLOW_AVAILABLE is True.
    # For this test, let's assume the @patch on the class works as intended for the constructor.
    # The patch should be on 'claims_processor.src.processing.ml_pipeline.optimized_predictor.tf.lite.Interpreter'
    # The mock_interpreter_class is this patched constructor. So its side_effect should be used.

    # Re-patching for this specific scenario inside the test if needed, or ensure fixture setup is right.
    # The fixture 'predictor' is not used here to control the patching context.
    with patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.tf.lite.Interpreter', side_effect=FileNotFoundError("mocked file not found")) as specific_mock_interpreter_ctor:
        predictor_file_not_found = OptimizedPredictor(model_path="badpath.tflite")
        assert predictor_file_not_found.model is None
        specific_mock_interpreter_ctor.assert_called_with(model_path="badpath.tflite")

# Original test_predict_async_returns_correct_shape_and_type is effectively covered by test_predict_async_returns_dummy_if_model_none
# if the default predictor fixture results in model = None (e.g. TF not available or dummy path not found).
# Let's keep it as a general fallback check.
@pytest.mark.asyncio
async def test_predict_async_fallback_dummy_prediction(predictor: OptimizedPredictor): # Uses the default predictor fixture
    settings = get_settings()
    # Input features: 1 sample, N features (matching FeatureExtractor's output)
    dummy_features = np.random.rand(1, settings.ML_FEATURE_COUNT).astype(np.float32)

    predictions = await predictor.predict_async(dummy_features)

    assert isinstance(predictions, np.ndarray), "Predictions should be a NumPy array"
    # Expecting 2 output classes (e.g., reject, approve probabilities)
    assert predictions.shape == (1, 2), \
        f"Expected shape (1, 2) for binary classification, but got {predictions.shape}"
    assert predictions.dtype == np.float32, "Predictions should be float32 type"

@pytest.mark.asyncio
async def test_predict_async_multiple_samples(predictor: OptimizedPredictor):
    settings = get_settings()
    num_samples = 5
    dummy_features_batch = np.random.rand(num_samples, settings.ML_FEATURE_COUNT).astype(np.float32)

    predictions = await predictor.predict_async(dummy_features_batch)

    assert predictions.shape == (num_samples, 2), \
        f"Expected shape ({num_samples}, 2), but got {predictions.shape}"

def test_predictor_initialization(predictor: OptimizedPredictor):
    settings = get_settings()
    expected_model_path = settings.ML_MODEL_PATH if settings.ML_MODEL_PATH else "dummy/path/model.tflite"
    assert predictor.model_path == expected_model_path
    # Check that the stub model object is set
    assert predictor.model == "dummy_model_object"

@pytest.mark.asyncio
async def test_predict_async_model_not_loaded(predictor: OptimizedPredictor):
    # Simulate model loading failure by setting model to None
    predictor.model = None

    settings = get_settings()
    dummy_features = np.random.rand(1, settings.ML_FEATURE_COUNT).astype(np.float32)
    predictions = await predictor.predict_async(dummy_features)

    # Check if it returns the default error prediction format
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (1, 2)
    assert np.all(predictions == np.full((1, 2), 0.0)), \
        "Expected default error prediction [[0.0, 0.0]] when model is not loaded"
