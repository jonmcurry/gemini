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

        if not features_batch:
            return []

        logger.info(f"Performing TFLite inference for batch of size {len(features_batch)}")
        predictions: List[Dict[str, Any]] = []

        try:
            # Prepare batch input tensor
            # Ensure all feature samples are 1D and have the correct feature_count
            processed_feature_list = []
            for i, features_sample in enumerate(features_batch):
                if not isinstance(features_sample, np.ndarray):
                    logger.warn(f"Sample {i} is not a NumPy array. Marking as error for this sample.")
                    predictions.append({"error": "Invalid feature format", "ml_score": None, "ml_derived_decision": "ML_ERROR_FEATURE_FORMAT"})
                    if self.metrics_collector: self.metrics_collector.record_ml_prediction(outcome="ML_ERROR_FEATURE_FORMAT", confidence_score=None)
                    # Add a placeholder or skip; for batching, all inputs must be valid or skipped before batch formation.
                    # For now, this error will prevent batching. A more robust solution might involve returning error for this item and processing others.
                    # However, batch inference requires all items in the batch to be valid for np.array stacking.
                    # Alternative: pre-filter features_batch for valid items and then process.
                    # For this refactor, let's assume valid inputs or fail the whole batch if one is bad before np.array.
                    # A simpler approach: if a sample is bad, we can't form the batch tensor.
                    # Let's return individual errors for each sample if pre-batch validation fails.
                    # This loop is just for validation before creating the batch tensor.
                    if features_sample.ndim != 1 or features_sample.shape[0] != self.feature_count:
                        logger.error(f"Sample {i} has incorrect shape {features_sample.shape}. Expected ({self.feature_count},). Cannot proceed with batch.")
                        # If one sample is bad, the whole batch operation as a single tensor might be problematic.
                        # Fallback to individual error reporting for all items in batch for simplicity for this error type.
                        error_msg = f"Feature shape mismatch for sample {i}"
                        for _ in features_batch: # Mark all as error if one is bad for batching
                            predictions.append({"error": error_msg, "ml_score": None, "ml_derived_decision": "ML_ERROR_FEATURE_SHAPE"})
                            if self.metrics_collector: self.metrics_collector.record_ml_prediction(outcome="ML_ERROR_FEATURE_SHAPE", confidence_score=None)
                        return predictions
                processed_feature_list.append(features_sample.astype(np.float32)) # Ensure float32

            if not processed_feature_list: # Should be caught by "if not features_batch" but as a safeguard
                 return []

            batch_input_tensor = np.array(processed_feature_list)
            if batch_input_tensor.shape[1] != self.feature_count:
                logger.error(f"Batch input tensor has incorrect feature dimension: {batch_input_tensor.shape[1]}. Expected {self.feature_count}")
                # This indicates a fundamental issue if it happens after the loop above.
                error_msg = "Batch tensor feature dimension mismatch"
                for _ in features_batch:
                    predictions.append({"error": error_msg, "ml_score": None, "ml_derived_decision": "ML_ERROR_BATCH_TENSOR"})
                    if self.metrics_collector: self.metrics_collector.record_ml_prediction(outcome="ML_ERROR_BATCH_TENSOR", confidence_score=None)
                return predictions

            self.interpreter.set_tensor(self.input_details[0]['index'], batch_input_tensor)

            invoke_start_time = time.perf_counter()
            await asyncio.to_thread(self.interpreter.invoke)
            invoke_duration_seconds = time.perf_counter() - invoke_start_time
            if self.metrics_collector:
                # Record duration for the whole batch. If per-sample is needed, divide by batch size.
                self.metrics_collector.record_ml_inference_duration(invoke_duration_seconds)

            batch_output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

            # Process batch output
            for i in range(batch_output_data.shape[0]):
                sample_output = batch_output_data[i] # This is likely shape (1,) or (num_classes,)

                score: Optional[float] = None
                decision = "ML_ERROR" # Default for this sample
                ml_decision_outcome_label = "ML_ERROR" # Default for metrics

                # TFLite output for a single item in a batch might be (1,1) or (1,2) if that's how model is structured,
                # or just (1,) or (2,). Assuming output_details[0]['shape'] is like [Batch, Classes]
                # and sample_output is now effectively [Classes]
                if sample_output.ndim == 1: # e.g. shape (1,) or (2,)
                    if sample_output.shape[0] == 1: # Single score output
                        score = float(sample_output[0])
                    elif sample_output.shape[0] == 2: # Two-class output (prob_class0, prob_class1)
                        score = float(sample_output[1]) # Assuming positive class is at index 1
                    else:
                        logger.warn(f"Unexpected TFLite model output shape for sample {i}: {sample_output.shape}. Output tensor shape: {batch_output_data.shape}")
                        ml_decision_outcome_label = "ML_ERROR_OUTPUT_SHAPE"
                else: # Should not happen if batch_output_data is 2D and sample_output is batch_output_data[i]
                     logger.warn(f"Unexpected sample_output dimension for sample {i}: {sample_output.ndim}. Shape: {sample_output.shape}")
                     ml_decision_outcome_label = "ML_ERROR_OUTPUT_DIM"


                score_for_metric: Optional[float] = None
                if score is not None:
                    decision = "ML_APPROVED" if score >= self.approval_threshold else "ML_REJECTED"
                    score_for_metric = round(score, 4)
                    ml_decision_outcome_label = decision

                predictions.append({'ml_score': score_for_metric, 'ml_derived_decision': decision})
                if self.metrics_collector:
                    self.metrics_collector.record_ml_prediction(outcome=ml_decision_outcome_label, confidence_score=score_for_metric)
                # logger.debug for individual sample prediction can be very verbose for large batches, consider sampling or removing
                # logger.debug(f"TFLite prediction for sample {i} in batch: score={score_for_metric if score_for_metric is not None else 'N/A'}, decision='{decision}'")

        except Exception as e:
            logger.error(f"Error during TFLite batch inference: {e}", exc_info=True, batch_size=len(features_batch))
            # If a global batch error occurs, mark all predictions in this batch as errored
            # This overwrites any per-sample errors identified before the exception.
            predictions = [] # Clear any partial predictions
            for _ in features_batch:
                predictions.append({"error": "TFLite batch inference exception", "ml_score": None, "ml_derived_decision": "ML_ERROR_INFERENCE_EXCEPTION"})
                if self.metrics_collector:
                     self.metrics_collector.record_ml_prediction(outcome="ML_ERROR_INFERENCE_EXCEPTION", confidence_score=None)

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
