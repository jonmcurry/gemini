import numpy as np
import structlog
from typing import Any, Dict, Optional # Added Dict, Optional

logger = structlog.get_logger(__name__)

# Attempt to import tflite_runtime.interpreter
try:
    from tflite_runtime.interpreter import Interpreter
    TFLITE_RUNTIME_AVAILABLE = True
    logger.info("tflite_runtime library found.")
except ImportError:
    TFLITE_RUNTIME_AVAILABLE = False
    logger.warn("tflite_runtime library not found. OptimizedPredictor will use dummy predictions if model loading is attempted.")
    Interpreter = None # Assign to None so Interpreter() calls don't cause NameError later

class OptimizedPredictor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model: Any = None # Will store the tf.lite.Interpreter instance
        self.input_details: Optional[Any] = None # Changed from List[Dict] to Any for simplicity with TF details
        self.output_details: Optional[Any] = None # Changed from List[Dict] to Any

        logger.info("OptimizedPredictor initialized.", model_path=self.model_path)
        self._load_model() # Changed from _load_model_stub

    def _load_model(self):
        logger.info(f"Attempting to load model from {self.model_path}.")
        if not TFLITE_RUNTIME_AVAILABLE: # Use new flag
            logger.warn(f"tflite-runtime is not available. Cannot load TFLite model from {self.model_path}.")
            self.model = None
            return

        try:
            if not self.model_path:
                logger.warn("No model path configured in settings. Cannot load model.")
                self.model = None
                return

            interpreter_instance = Interpreter(model_path=self.model_path) # Use imported Interpreter
            interpreter_instance.allocate_tensors()

            self.input_details = interpreter_instance.get_input_details()
            self.output_details = interpreter_instance.get_output_details()
            self.model = interpreter_instance

            logger.info(
                "TFLite model loaded successfully using tflite_runtime.",
                model_path=self.model_path,
                input_details=self.input_details,
                output_details=self.output_details
            )
        except FileNotFoundError:
            logger.error(f"Model file not found at {self.model_path}. Ensure the model exists or the path is correct.")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load TFLite model using tflite_runtime from {self.model_path}", error_type=type(e).__name__, error=str(e), exc_info=False)
            self.model = None

        if self.model is None:
            logger.warn(f"Model loading failed or tflite-runtime not available. Predictor will use dummy predictions.", model_path=self.model_path)

    async def predict_async(self, features: np.ndarray) -> np.ndarray:
        logger.debug("Performing prediction", input_shape=features.shape, model_loaded=(self.model is not None))

        if self.model is None or not TFLITE_RUNTIME_AVAILABLE: # Use new flag
            if self.model is None and TFLITE_RUNTIME_AVAILABLE : # Log only if runtime is there but model failed to load
                 logger.warn("Model not loaded, returning dummy predictions.", model_path=self.model_path)
            elif not TFLITE_RUNTIME_AVAILABLE:
                 logger.warn("tflite-runtime not available, returning dummy predictions.")
            num_samples = features.shape[0]
            # Consistent dummy prediction (e.g., for binary classification)
            return np.random.rand(num_samples, 2).astype(np.float32)

        try:
            if self.input_details is None or self.output_details is None: # Should not happen if model loaded ok
                logger.error("Model input/output details not available despite model being loaded. Cannot perform prediction.")
                num_samples = features.shape[0]
                return np.random.rand(num_samples, 2).astype(np.float32)

            # Assuming features are correctly shaped (e.g., (num_samples, num_features))
            # And model input_details[0]['shape'] is compatible (e.g. [None, num_features] or [num_samples, num_features])
            self.model.set_tensor(self.input_details[0]['index'], features)
            self.model.invoke()
            output_data = self.model.get_tensor(self.output_details[0]['index'])

            logger.info("Prediction successful using loaded TFLite model.", output_shape=output_data.shape)
            return np.array(output_data).astype(np.float32)

        except Exception as e:
            logger.error("Error during TFLite model inference", error_type=type(e).__name__, error=str(e), exc_info=True)
            num_samples = features.shape[0]
            return np.random.rand(num_samples, 2).astype(np.float32) # Fallback to dummy
