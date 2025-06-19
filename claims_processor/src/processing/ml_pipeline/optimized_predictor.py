import structlog
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

# Attempt to import TensorFlow Lite and set a flag.
TFLITE_AVAILABLE = False
Interpreter = None # Default to None
logger = structlog.get_logger(__name__) # Define logger globally

try:
    from tflite_runtime.interpreter import Interpreter
    TFLITE_AVAILABLE = True
    logger.info("TensorFlow Lite runtime found.")
except ImportError:
    logger.warn("TensorFlow Lite runtime not found. OptimizedPredictor will not be able to make real predictions if a model is loaded.")
except Exception as e: # Catch other potential errors during import
    logger.error("An unexpected error occurred during tflite_runtime import.", error=str(e), exc_info=True)


class OptimizedPredictor:
    """
    High-performance ML predictor using TensorFlow Lite, as per REQUIREMENTS.md.
    Handles model loading and batch predictions.
    """

    def __init__(self, model_path: str, feature_count: int = 7):
        """
        Initializes the OptimizedPredictor.

        Args:
            model_path: Path to the TensorFlow Lite model file.
            feature_count: Expected number of features for the model.
        """
        self.model_path = model_path
        self.feature_count = feature_count
        self.interpreter: Optional[Interpreter] = None # Type hint the interpreter
        self.input_details: Optional[List[Dict[str, Any]]] = None # More specific type hint
        self.output_details: Optional[List[Dict[str, Any]]] = None

        logger.info("OptimizedPredictor initializing...", model_path=self.model_path, tflite_available=TFLITE_AVAILABLE)

        if not TFLITE_AVAILABLE:
            logger.error("TensorFlow Lite runtime is not available. Cannot load model or make predictions.")
            return

        if not self.model_path: # Check if model_path is empty or None
            logger.error("Model path is not configured. Predictor will not work.")
            return

        model_file = Path(self.model_path)
        if not model_file.is_file():
            logger.error(f"ML model file not found at {self.model_path}. Predictor will not work.")
            return

        try:
            logger.info(f"Loading TFLite model from: {self.model_path}")
            self.interpreter = Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details() # Returns a list of dicts
            self.output_details = self.interpreter.get_output_details() # Returns a list of dicts

            if self.input_details:
                expected_shape = list(self.input_details[0]['shape'])
                if len(expected_shape) > 1 and expected_shape[-1] != self.feature_count:
                    logger.warn(
                        f"Model input shape {expected_shape} last dimension "
                        f"does not match expected feature_count {self.feature_count}. "
                        "Ensure model is compatible."
                    )

            logger.info("TFLite model loaded and tensors allocated successfully.",
                        input_details=self.input_details,
                        output_details=self.output_details)

        except Exception as e:
            logger.error(f"Failed to load TFLite model or allocate tensors: {e}", exc_info=True)
            self.interpreter = None


    async def predict_batch(self, features_batch: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Performs batch prediction on a list of feature sets.
        Each feature set in the list is a 1D NumPy array of shape (feature_count,).
        """
        if not self.interpreter or not TFLITE_AVAILABLE: # Check interpreter instance specifically
            logger.error("Predictor not initialized correctly or TFLite not available. Cannot predict.")
            return [{"error": "Predictor not available or TFLite runtime missing"}] * len(features_batch) if features_batch else []

        if not features_batch:
            return []

        logger.debug(f"Predicting batch of size {len(features_batch)}")
        predictions: List[Dict[str, Any]] = []

        # MOCK IMPLEMENTATION:
        # This part should be replaced by actual TFLite inference loop if a model is used.
        # For now, it provides mock predictions as per the prompt's structure.
        from claims_processor.src.core.config.settings import get_settings # For ML_APPROVAL_THRESHOLD
        settings = get_settings()
        approval_threshold = settings.ML_APPROVAL_THRESHOLD

        for i, features_sample_1d in enumerate(features_batch):
            # Ensure features_sample_1d is float32 and has the correct number of features.
            # The FeatureExtractor now returns (1, feature_count) for single,
            # so this might need adjustment if the input is a list of these 2D arrays.
            # Assuming features_sample_1d is already 1D of shape (feature_count,).
            if features_sample_1d.ndim == 2 and features_sample_1d.shape[0] == 1: # If (1,N) passed
                features_sample_1d = features_sample_1d.reshape(-1) # Convert to (N,)

            if features_sample_1d.shape[0] != self.feature_count:
                logger.warn(f"Sample {i} feature length {features_sample_1d.shape[0]} != model expected {self.feature_count}")
                predictions.append({"error": "Feature length mismatch", "ml_score": 0.0, "ml_derived_decision": "ERROR"})
                continue

            # Actual TFLite inference would look like:
            # input_data = np.expand_dims(features_sample_1d.astype(np.float32), axis=0)
            # self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            # self.interpreter.invoke()
            # output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            # score = float(output_data[0][1]) # Assuming index 1 is 'approve' probability
            # decision = "ML_APPROVED" if score >= approval_threshold else "ML_REJECTED"
            # predictions.append({'ml_score': round(score, 4), 'ml_derived_decision': decision})

            # Mock prediction logic:
            mock_score_sum = np.sum(features_sample_1d)
            mock_score = (mock_score_sum / self.feature_count) * 0.1 # Scale sum to be roughly in 0-1
            mock_score = float(min(max(0.0, mock_score + (0.05 * (i % 3))), 1.0)) # Vary it a bit, ensure float

            decision = "ML_APPROVED" if mock_score >= approval_threshold else "ML_REJECTED"
            predictions.append({'ml_score': round(mock_score, 4), 'ml_derived_decision': decision})
            logger.debug(f"Mock prediction for sample {i}: score={mock_score:.4f}, decision='{decision}'")

        return predictions

    async def health_check(self) -> Dict[str, Any]:
        """Performs a health check on the ML predictor."""
        if not TFLITE_AVAILABLE:
            return {"status": "unhealthy", "reason": "TensorFlow Lite runtime not available."}
        if not self.model_path or not Path(self.model_path).is_file(): # Check model_path attr
            return {"status": "unhealthy", "reason": f"Model file not found at {self.model_path or '[not configured]'}"}
        if self.interpreter is None:
            return {"status": "unhealthy", "reason": "TFLite interpreter not loaded (e.g., model load failed during init)."}

        # Basic check: interpreter object exists.
        # A more thorough check might try a dummy inference if it's fast enough.
        return {"status": "healthy", "model_path": self.model_path,
                "input_details": str(self.input_details), # Convert to str for JSON serializability
                "output_details": str(self.output_details)}
