import asyncio # Added for to_thread
import structlog
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import time # Import time

TFLITE_AVAILABLE = False
Interpreter = None
logger = structlog.get_logger(__name__)

try:
    from tflite_runtime.interpreter import Interpreter
    TFLITE_AVAILABLE = True
    logger.info("TensorFlow Lite runtime found.")
except ImportError:
    logger.warn("TensorFlow Lite runtime not found. OptimizedPredictor will not be able to make real predictions if a model is loaded.")
except Exception as e:
    TFLITE_AVAILABLE = False
    Interpreter = None
    logger.error("An unexpected error occurred during tflite_runtime import.", error=str(e), exc_info=True)

from ....core.config.settings import get_settings
from ....core.monitoring.app_metrics import MetricsCollector # Import MetricsCollector

class OptimizedPredictor:
    def __init__(self, model_path: str, metrics_collector: MetricsCollector, feature_count: int = 7):
        self.model_path = model_path
        self.feature_count = feature_count
        self.metrics_collector = metrics_collector # Store metrics_collector
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

            app_settings = get_settings()
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
                        input_details=self.input_details, output_details=self.output_details)
        except Exception as e:
            logger.error(f"Failed to load TFLite model or allocate tensors: {e}", exc_info=True)
            self.interpreter = None

    async def predict_batch(self, features_batch: List[np.ndarray]) -> List[Dict[str, Any]]:
        if not self.interpreter or not TFLITE_AVAILABLE:
            logger.error("Predictor not initialized or TFLite not available. Cannot predict.")
            error_result = {"error": "Predictor not available", "ml_score": None, "ml_derived_decision": "ML_ERROR"}
            if self.metrics_collector: # Check if metrics_collector is available
                for _ in features_batch: self.metrics_collector.record_ml_prediction(outcome="ML_ERROR_PREDICTOR_UNAVAILABLE", confidence_score=None)
            return [error_result] * len(features_batch) if features_batch else []

        if not features_batch: return []

        logger.info(f"Performing TFLite inference for batch of size {len(features_batch)}")
        predictions: List[Dict[str, Any]] = []

        for i, features_sample_2d in enumerate(features_batch):
            ml_decision_outcome_label = "ML_ERROR"
            score_for_metric: Optional[float] = None
            try:
                if not isinstance(features_sample_2d, np.ndarray):
                    logger.warn(f"Sample {i} is not a NumPy array. Skipping.")
                    predictions.append({"error": "Invalid feature format", "ml_score": None, "ml_derived_decision": "ML_ERROR"})
                    ml_decision_outcome_label="ML_ERROR_FEATURE_FORMAT"
                    if self.metrics_collector: self.metrics_collector.record_ml_prediction(outcome=ml_decision_outcome_label, confidence_score=None)
                    continue

                # features_sample_2d is actually a single sample's features.
                # It can be 1D (feature_count,) or 2D (1, feature_count).
                # The interpreter expects (1, feature_count).
                features_for_interpreter: Optional[np.ndarray] = None
                if features_sample_2d.ndim == 1 and features_sample_2d.shape[0] == self.feature_count:
                    features_for_interpreter = np.expand_dims(features_sample_2d, axis=0)
                elif features_sample_2d.shape == (1, self.feature_count): # Already in expected shape
                    features_for_interpreter = features_sample_2d
                else:
                    logger.warn(f"Sample {i} feature shape {features_sample_2d.shape} is not ({self.feature_count},) or (1, {self.feature_count}). Skipping.")
                    predictions.append({"error": "Feature shape mismatch", "ml_score": None, "ml_derived_decision": "ML_ERROR"})
                    ml_decision_outcome_label="ML_ERROR_FEATURE_SHAPE"
                    if self.metrics_collector: self.metrics_collector.record_ml_prediction(outcome=ml_decision_outcome_label, confidence_score=None)
                    continue

                final_features = features_for_interpreter # Use the (potentially) reshaped features
                if final_features.dtype != np.float32: # Check dtype of the final_features
                    logger.warn(f"Sample {i} feature dtype {final_features.dtype} is not np.float32. Attempting conversion.")
                    try:
                        final_features = final_features.astype(np.float32)
                    except Exception as cast_e:
                        logger.error(f"Could not cast sample {i} features to np.float32: {cast_e}", exc_info=True)
                        predictions.append({"error": "Feature type conversion failed", "ml_score": None, "ml_derived_decision": "ML_ERROR"})
                        ml_decision_outcome_label="ML_ERROR_FEATURE_TYPE"
                        if self.metrics_collector: self.metrics_collector.record_ml_prediction(outcome=ml_decision_outcome_label, confidence_score=None)
                        continue

                self.interpreter.set_tensor(self.input_details[0]['index'], final_features)

                invoke_start_time = time.perf_counter()
                await asyncio.to_thread(self.interpreter.invoke) # Asynchronous invocation
                invoke_duration_seconds = time.perf_counter() - invoke_start_time
                if self.metrics_collector: self.metrics_collector.record_ml_inference_duration(invoke_duration_seconds)

                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

                score: Optional[float] = None; decision = "ML_ERROR"
                if output_data.shape == (1,1): score = float(output_data[0][0])
                elif output_data.shape == (1,2): score = float(output_data[0][1])
                else:
                    logger.warn(f"Unexpected TFLite model output shape: {output_data.shape}. Cannot interpret score.")
                    ml_decision_outcome_label = "ML_ERROR_OUTPUT_SHAPE"

                if score is not None:
                    decision = "ML_APPROVED" if score >= self.approval_threshold else "ML_REJECTED"
                    score_for_metric = round(score, 4)
                    ml_decision_outcome_label = decision

                predictions.append({'ml_score': score_for_metric, 'ml_derived_decision': decision})
                if self.metrics_collector: self.metrics_collector.record_ml_prediction(outcome=ml_decision_outcome_label, confidence_score=score_for_metric)
                logger.debug(f"TFLite prediction for sample {i}: score={score_for_metric if score_for_metric is not None else 'N/A'}, decision='{decision}'")

            except Exception as e:
                logger.error(f"Error during TFLite inference for sample {i}: {e}", exc_info=True)
                predictions.append({"error": "TFLite inference exception", "ml_score": None, "ml_derived_decision": "ML_ERROR"})
                if self.metrics_collector: self.metrics_collector.record_ml_prediction(outcome="ML_ERROR_INFERENCE_EXCEPTION", confidence_score=None)

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
