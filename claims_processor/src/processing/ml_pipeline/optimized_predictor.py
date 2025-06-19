import structlog
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

# Attempt to import TensorFlow Lite and set a flag.
TFLITE_AVAILABLE = False
Interpreter = None # Default to None
logger = structlog.get_logger(__name__)

try:
    from tflite_runtime.interpreter import Interpreter
    TFLITE_AVAILABLE = True
    logger.info("TensorFlow Lite runtime found.")
except ImportError:
    logger.warn("TensorFlow Lite runtime not found. OptimizedPredictor will not be able to make real predictions if a model is loaded.")
except Exception as e:
    TFLITE_AVAILABLE = False # Ensure flag is false on any import error
    Interpreter = None
    logger.error("An unexpected error occurred during tflite_runtime import.", error=str(e), exc_info=True)

from ....core.config.settings import get_settings # For ML_APPROVAL_THRESHOLD

class OptimizedPredictor:
    """
    High-performance ML predictor using TensorFlow Lite, as per REQUIREMENTS.md.
    Handles model loading and batch predictions.
    """

    def __init__(self, model_path: str, feature_count: int = 7):
        self.model_path = model_path
        self.feature_count = feature_count
        self.interpreter: Optional[Interpreter] = None
        self.input_details: Optional[List[Dict[str, Any]]] = None
        self.output_details: Optional[List[Dict[str, Any]]] = None
        self.approval_threshold: float = 0.8 # Default

        logger.info("OptimizedPredictor initializing...", model_path=self.model_path, tflite_available=TFLITE_AVAILABLE)

        if not TFLITE_AVAILABLE:
            logger.error("TensorFlow Lite runtime is not available. Model loading skipped.")
            return

        if not self.model_path:
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
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            app_settings = get_settings() # Fetch settings to get threshold
            self.approval_threshold = app_settings.ML_APPROVAL_THRESHOLD
            logger.info(f"Approval threshold set to: {self.approval_threshold}")

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
        if not self.interpreter or not TFLITE_AVAILABLE:
            logger.error("Predictor not initialized correctly or TFLite not available. Cannot predict.")
            return [{"error": "Predictor not available", "ml_score": None, "ml_derived_decision": "ML_ERROR"}] * len(features_batch) if features_batch else []

        if not features_batch:
            return []

        logger.info(f"Performing TFLite inference for batch of size {len(features_batch)}")
        predictions: List[Dict[str, Any]] = []

        for i, features_sample_2d in enumerate(features_batch): # features_sample_2d is (1, feature_count)
            try:
                if not isinstance(features_sample_2d, np.ndarray):
                    logger.warn(f"Sample {i} is not a NumPy array. Skipping.")
                    predictions.append({"error": "Invalid feature format", "ml_score": None, "ml_derived_decision": "ML_ERROR"})
                    continue

                # FeatureExtractor returns (1, N). Model input_details[0]['shape'] is often also [1, N].
                if features_sample_2d.shape != tuple(self.input_details[0]['shape']):
                     # If model expects different batch size (e.g. dynamic batching with None in shape)
                     # or if feature count mismatches (already checked partly in FeatureExtractor)
                     # This check assumes model input batch size is 1 if features_sample_2d is (1,N)
                    if features_sample_2d.shape[1] != self.input_details[0]['shape'][-1]: # Check only last dim (features)
                        logger.warn(
                            f"Sample {i} feature count {features_sample_2d.shape[1]} "
                            f"does not match model expected {self.input_details[0]['shape'][-1]}. Skipping."
                        )
                        predictions.append({"error": "Feature count mismatch", "ml_score": None, "ml_derived_decision": "ML_ERROR"})
                        continue
                    # If only batch size differs, and model supports dynamic batch (e.g. shape [None, N]), it might be fine.
                    # For now, strict shape check for (1,N) vs model's (1,N)
                    if self.input_details[0]['shape'][0] == 1 and features_sample_2d.shape[0] != 1:
                         logger.warn(
                            f"Sample {i} feature batch dim {features_sample_2d.shape[0]} "
                            f"does not match model expected batch dim {self.input_details[0]['shape'][0]}. Skipping."
                        )
                         predictions.append({"error": "Feature batch dim mismatch", "ml_score": None, "ml_derived_decision": "ML_ERROR"})
                         continue


                if features_sample_2d.dtype != np.float32:
                    logger.warn(f"Sample {i} feature dtype {features_sample_2d.dtype} is not np.float32. Attempting conversion.")
                    try:
                        features_sample_2d = features_sample_2d.astype(np.float32)
                    except Exception as cast_e:
                        logger.error(f"Could not cast sample {i} features to np.float32: {cast_e}", exc_info=True)
                        predictions.append({"error": "Feature type conversion failed", "ml_score": None, "ml_derived_decision": "ML_ERROR"})
                        continue

                self.interpreter.set_tensor(self.input_details[0]['index'], features_sample_2d)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

                score = 0.0
                # Assuming output_data for a single sample is e.g., [[probability_class_0, probability_class_1]]
                # Or, if it's a regression model, it might be a single value.
                if output_data.shape == (1,1): # Single score output (e.g. sigmoid for approval)
                    score = float(output_data[0][0])
                elif output_data.shape == (1,2): # [prob_reject, prob_approve]
                    score = float(output_data[0][1]) # Probability of 'approve' is index 1
                else:
                    logger.warn(f"Unexpected TFLite model output shape: {output_data.shape}. Attempting to use first item.")
                    try:
                        score = float(output_data.item(0))
                    except Exception:
                        logger.error(f"Could not interpret TFLite model output shape: {output_data.shape}", exc_info=True)
                        predictions.append({"error": "Model output interpretation error", "ml_score": None, "ml_derived_decision": "ML_ERROR"})
                        continue

                decision = "ML_APPROVED" if score >= self.approval_threshold else "ML_REJECTED"
                predictions.append({'ml_score': round(score, 4), 'ml_derived_decision': decision})
                logger.debug(f"TFLite prediction for sample {i}: score={score:.4f}, decision='{decision}'")

            except Exception as e:
                logger.error(f"Error during TFLite inference for sample {i}: {e}", exc_info=True)
                predictions.append({"error": "TFLite inference exception", "ml_score": None, "ml_derived_decision": "ML_ERROR"})

        return predictions

    async def health_check(self) -> Dict[str, Any]:
        if not TFLITE_AVAILABLE:
            return {"status": "unhealthy", "reason": "TensorFlow Lite runtime not available."}
        if not self.model_path or not Path(self.model_path).is_file():
            return {"status": "unhealthy", "reason": f"Model file not found at {self.model_path or '[not configured]'}"}
        if self.interpreter is None:
            return {"status": "unhealthy", "reason": "TFLite interpreter not loaded (e.g., model load failed during init)."}

        return {"status": "healthy", "model_path": self.model_path,
                "input_details": str(self.input_details),
                "output_details": str(self.output_details)}
