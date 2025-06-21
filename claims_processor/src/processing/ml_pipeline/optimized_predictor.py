import asyncio
import structlog
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import time
import hashlib # For cache key generation
from cachetools import TTLCache # For caching

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

import mlflow
import tempfile # For managing temporary download path

from ...core.config.settings import get_settings # Corrected relative import
from ...core.monitoring.app_metrics import MetricsCollector # Corrected relative import

class OptimizedPredictor:
    def __init__(self, metrics_collector: MetricsCollector): # Removed model_path and feature_count from args
        self.metrics_collector = metrics_collector
        self.interpreter: Optional[Interpreter] = None
        self.input_details: Optional[List[Dict[str, Any]]] = None
        self.output_details: Optional[List[Dict[str, Any]]] = None
        self.model_path: Optional[str] = None # Will be set by loading logic
        self.temp_model_dir: Optional[tempfile.TemporaryDirectory] = None # For MLflow downloaded models

        app_settings = get_settings()
        self.feature_count = app_settings.ML_FEATURE_COUNT # Get from settings
        self.approval_threshold = app_settings.ML_APPROVAL_THRESHOLD

        self.prediction_cache = TTLCache(
            maxsize=app_settings.ML_PREDICTION_CACHE_MAXSIZE,
            ttl=app_settings.ML_PREDICTION_CACHE_TTL
        )
        logger.info("OptimizedPredictor initializing...",
                    tflite_available=TFLITE_AVAILABLE,
                    cache_maxsize=app_settings.ML_PREDICTION_CACHE_MAXSIZE,
                    cache_ttl=app_settings.ML_PREDICTION_CACHE_TTL)

        model_uri_to_load = None
        using_mlflow = False

        if app_settings.MLFLOW_TRACKING_URI and \
           app_settings.ML_MODEL_NAME_IN_REGISTRY and \
           app_settings.ML_MODEL_VERSION_OR_STAGE:

            logger.info("MLflow configuration found. Attempting to load model from registry.",
                        tracking_uri=app_settings.MLFLOW_TRACKING_URI,
                        model_name=app_settings.ML_MODEL_NAME_IN_REGISTRY,
                        version_stage=app_settings.ML_MODEL_VERSION_OR_STAGE)
            try:
                mlflow.set_tracking_uri(app_settings.MLFLOW_TRACKING_URI)

                self.temp_model_dir = tempfile.TemporaryDirectory()
                # model_download_path = self.temp_model_dir.name # Not used directly, download_artifacts creates structure

                download_root = mlflow.artifacts.download_artifacts(
                    artifact_uri=f"models:/{app_settings.ML_MODEL_NAME_IN_REGISTRY}/{app_settings.ML_MODEL_VERSION_OR_STAGE}",
                    dst_path=self.temp_model_dir.name # Ensure artifacts are downloaded into the temp dir
                )

                tflite_files = list(Path(download_root).rglob("*.tflite"))
                if not tflite_files:
                    # Sometimes download_artifacts returns the path to a specific file if only one,
                    # or a directory. If download_root itself is the .tflite file:
                    if Path(download_root).is_file() and Path(download_root).name.endswith(".tflite"):
                        tflite_files = [Path(download_root)]
                    else:
                        raise FileNotFoundError(f"No .tflite file found in downloaded MLflow artifacts from {download_root}. Searched for *.tflite.")

                if len(tflite_files) > 1:
                    logger.warn(f"Multiple .tflite files found: {tflite_files}. Using the first one: {tflite_files[0]}")

                model_uri_to_load = str(tflite_files[0])
                self.model_path = model_uri_to_load # Update self.model_path to the downloaded one
                using_mlflow = True
                logger.info(f"Model artifact will be loaded from MLflow download: {self.model_path}")

            except Exception as e:
                logger.error("Failed to load model from MLflow Model Registry. Falling back to ML_MODEL_PATH if available.",
                             model_name=app_settings.ML_MODEL_NAME_IN_REGISTRY,
                             version_stage=app_settings.ML_MODEL_VERSION_OR_STAGE,
                             error=str(e), exc_info=True)
                if self.temp_model_dir: # Cleanup if temp_model_dir was created
                    self.temp_model_dir.cleanup()
                    self.temp_model_dir = None # Reset
                model_uri_to_load = app_settings.ML_MODEL_PATH # Fallback
                using_mlflow = False
        else:
            logger.info("MLflow configuration not fully provided. Using local ML_MODEL_PATH.")
            model_uri_to_load = app_settings.ML_MODEL_PATH

        # --- Start of TFLite Interpreter Loading ---
        if not TFLITE_AVAILABLE:
            logger.error("TensorFlow Lite runtime is not available. Model loading skipped.")
            return

        if not model_uri_to_load:
            logger.error("No model path configured (neither MLflow nor local ML_MODEL_PATH). Predictor will not work.")
            return

        self.model_path = str(model_uri_to_load) # Ensure it's a string for Path object
        model_file = Path(self.model_path)

        if not model_file.is_file():
            logger.error(f"ML model file not found at resolved path: {self.model_path}. Predictor will not work.")
            if using_mlflow and self.temp_model_dir:
                self.temp_model_dir.cleanup()
                self.temp_model_dir = None
            return
        try:
            logger.info(f"Loading TFLite model from final path: {self.model_path}")
            self.interpreter = Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            # app_settings already fetched, self.approval_threshold already set
            logger.info(f"Approval threshold set to: {self.approval_threshold}")

            if self.input_details:
                expected_shape = list(self.input_details[0]['shape'])
                # self.feature_count is now set from app_settings
                if len(expected_shape) > 1 and expected_shape[-1] != self.feature_count:
                    logger.warn(
                        f"Model input shape {expected_shape} last dimension "
                        f"does not match expected feature_count {self.feature_count} from settings. "
                        "Ensure model is compatible."
                    )
            logger.info("TFLite model loaded and tensors allocated successfully.",
                        input_details=self.input_details, output_details=self.output_details)
        except Exception as e:
            logger.error(f"Failed to load TFLite model or allocate tensors from path: {self.model_path}. Error: {e}", exc_info=True)
            self.interpreter = None
            if using_mlflow and self.temp_model_dir:
                self.temp_model_dir.cleanup()
                self.temp_model_dir = None
        # --- End of TFLite Interpreter Loading ---

    async def predict_batch(self, features_batch: List[np.ndarray]) -> List[Dict[str, Any]]:
        if not self.interpreter or not TFLITE_AVAILABLE:
            logger.error("Predictor not initialized or TFLite not available. Cannot predict.")
            error_result = {"error": "Predictor not available", "ml_score": None, "ml_derived_decision": "ML_ERROR"}
            if self.metrics_collector: # Check if metrics_collector is available
                for _ in features_batch: self.metrics_collector.record_ml_prediction(outcome="ML_ERROR_PREDICTOR_UNAVAILABLE", confidence_score=None)
            return [error_result] * len(features_batch) if features_batch else []

        if not features_batch:
            return []

        logger.info(f"Processing prediction for batch of size {len(features_batch)} with caching.")

        # Initialize results list with None placeholders
        predictions_results: List[Optional[Dict[str, Any]]] = [None] * len(features_batch)
        miss_indices: List[int] = []
        miss_features: List[np.ndarray] = []
        cache_keys_for_misses: List[str] = [] # To store cache keys for items that missed

        # 1. Input validation and Cache Lookup
        for i, features_sample_1d in enumerate(features_batch):
            if not isinstance(features_sample_1d, np.ndarray):
                logger.warn(f"Sample {i} is not a NumPy array. Marking as error.")
                predictions_results[i] = {"error": "Invalid feature format", "ml_score": None, "ml_derived_decision": "ML_ERROR_FEATURE_FORMAT"}
                if self.metrics_collector: self.metrics_collector.record_ml_prediction(outcome="ML_ERROR_FEATURE_FORMAT", confidence_score=None)
                continue # Skip this sample for caching/batching

            if features_sample_1d.ndim != 1 or features_sample_1d.shape[0] != self.feature_count:
                logger.warn(f"Sample {i} has incorrect shape {features_sample_1d.shape}. Expected ({self.feature_count},). Marking as error.")
                predictions_results[i] = {"error": "Feature shape mismatch", "ml_score": None, "ml_derived_decision": "ML_ERROR_FEATURE_SHAPE"}
                if self.metrics_collector: self.metrics_collector.record_ml_prediction(outcome="ML_ERROR_FEATURE_SHAPE", confidence_score=None)
                continue # Skip this sample

            # Ensure dtype is float32 for consistent hashing and for the model
            features_sample_1d = features_sample_1d.astype(np.float32)

            # Generate cache key
            cache_key = hashlib.sha256(features_sample_1d.tobytes()).hexdigest()

            if cache_key in self.prediction_cache:
                predictions_results[i] = self.prediction_cache[cache_key]
                if self.metrics_collector: self.metrics_collector.record_cache_operation(cache_type='ml_prediction', operation_type='get', outcome='hit')
                logger.debug(f"Cache hit for sample {i}", cache_key=cache_key)
            else:
                if self.metrics_collector: self.metrics_collector.record_cache_operation(cache_type='ml_prediction', operation_type='get', outcome='miss')
                miss_indices.append(i)
                miss_features.append(features_sample_1d) # Already float32
                cache_keys_for_misses.append(cache_key)
                logger.debug(f"Cache miss for sample {i}", cache_key=cache_key)

        # 2. Batch Inference for Cache Misses
        if miss_features:
            logger.info(f"Performing TFLite inference for {len(miss_features)} cache misses.")
            try:
                batch_input_tensor = np.array(miss_features) # Already list of float32 arrays
                # Shape check for batch_input_tensor (already done for individual samples)
                # if batch_input_tensor.shape[1] != self.feature_count: ... error handling ...

                self.interpreter.set_tensor(self.input_details[0]['index'], batch_input_tensor)

                invoke_start_time = time.perf_counter()
                await asyncio.to_thread(self.interpreter.invoke)
                invoke_duration_seconds = time.perf_counter() - invoke_start_time
                if self.metrics_collector:
                    self.metrics_collector.record_ml_inference_duration(invoke_duration_seconds)

                batch_output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

                for i, (original_index, cache_key_for_miss) in enumerate(zip(miss_indices, cache_keys_for_misses)):
                    sample_output = batch_output_data[i]
                    score: Optional[float] = None
                    decision = "ML_ERROR"
                    ml_decision_outcome_label = "ML_ERROR_PROCESSING_MISS" # Default for this stage

                    if sample_output.ndim == 1:
                        if sample_output.shape[0] == 1: score = float(sample_output[0])
                        elif sample_output.shape[0] == 2: score = float(sample_output[1])
                        else: ml_decision_outcome_label = "ML_ERROR_OUTPUT_SHAPE_MISS"
                    else: ml_decision_outcome_label = "ML_ERROR_OUTPUT_DIM_MISS"

                    score_for_metric: Optional[float] = None
                    if score is not None:
                        decision = "ML_APPROVED" if score >= self.approval_threshold else "ML_REJECTED"
                        score_for_metric = round(score, 4)
                        ml_decision_outcome_label = decision

                    current_prediction_result = {'ml_score': score_for_metric, 'ml_derived_decision': decision}
                    predictions_results[original_index] = current_prediction_result
                    self.prediction_cache[cache_key_for_miss] = current_prediction_result # Add to cache

                    if self.metrics_collector:
                        self.metrics_collector.record_ml_prediction(outcome=ml_decision_outcome_label, confidence_score=score_for_metric)

            except Exception as e:
                logger.error(f"Error during TFLite batch inference for cache misses: {e}", exc_info=True, num_misses=len(miss_features))
                # Mark all missed predictions as errored if batch inference fails
                for original_index in miss_indices:
                    predictions_results[original_index] = {"error": "TFLite batch inference exception on miss", "ml_score": None, "ml_derived_decision": "ML_ERROR_INFERENCE_EXCEPTION"}
                    if self.metrics_collector:
                         self.metrics_collector.record_ml_prediction(outcome="ML_ERROR_INFERENCE_EXCEPTION", confidence_score=None)

        # Ensure no None entries are left if some samples were skipped due to validation errors before caching logic
        # This should not happen if all paths correctly assign a dict to predictions_results[i]
        final_predictions = [res if res is not None else {"error": "Skipped due to pre-cache validation error", "ml_score": None, "ml_derived_decision": "ML_ERROR_SKIPPED"} for res in predictions_results]

        return final_predictions

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
