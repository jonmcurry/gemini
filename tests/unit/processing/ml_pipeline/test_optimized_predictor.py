import pytest
import numpy as np

from claims_processor.src.processing.ml_pipeline.optimized_predictor import OptimizedPredictor
from claims_processor.src.core.config.settings import get_settings

# DUMMY_MODEL_PATH = "models/test_dummy_model.tflite" # Not needed if using settings.ML_MODEL_PATH

@pytest.fixture
def predictor() -> OptimizedPredictor:
    # Uses the default model path from settings
    settings = get_settings()
    # Ensure ML_MODEL_PATH has a default in settings or is set in .env for tests if path is critical
    # For the stub, any string path is fine as it's not actually loaded.
    model_path = settings.ML_MODEL_PATH if settings.ML_MODEL_PATH else "dummy/path/model.tflite"
    return OptimizedPredictor(model_path=model_path)


@pytest.mark.asyncio
async def test_predict_async_returns_correct_shape_and_type(predictor: OptimizedPredictor):
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
