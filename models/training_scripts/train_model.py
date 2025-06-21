import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import shutil # For cleaning up saved model directory
import os
import mlflow # Added
import mlflow.tensorflow # Added
from datetime import datetime # Added
import sys # Added
import argparse # Added
from typing import Optional, Dict # Added

# Determine paths relative to the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_PROJECT_DIR is one level up from SCRIPT_DIR (training_scripts), then another level up from 'models'
# No, ROOT_MODEL_DIR should be SCRIPT_DIR -> models -> saved_model
# SCRIPT_DIR = models/training_scripts/
# ..           = models/
# ../..        = project root
# So, models/saved_model should be os.path.join(SCRIPT_DIR, '..', 'saved_model')
ROOT_SAVED_MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'saved_model'))
MODEL_NAME = "claims_model"
# Final path for saving the versioned model, e.g., models/saved_model/claims_model
MODEL_BASE_PATH = os.path.join(ROOT_SAVED_MODEL_DIR, MODEL_NAME)

# Define model parameters
NUM_FEATURES = 7  # Should match FeatureExtractor output
NUM_CLASSES = 1   # Binary classification (Approve/Reject), outputting a single probability
MODEL_VERSION = "1" # Version subdirectory

def generate_mock_data(num_samples=1000, num_features=NUM_FEATURES):
    """Generates mock feature data and binary labels."""
    # Generate random features (e.g., simulating output from FeatureExtractor)
    # Values roughly between -1 and 5 (based on some normalized features like age, log_charges, encoded cats)
    features = np.random.rand(num_samples, num_features) * 6 - 1
    features = features.astype(np.float32)

    # Generate random binary labels (0 or 1)
    labels = np.random.randint(0, 2, size=num_samples).astype(np.float32)
    # Reshape labels to (num_samples, 1) if using binary_crossentropy with single output unit
    labels = labels.reshape(-1, 1)

    return features, labels

def define_model(num_features=NUM_FEATURES, num_classes=NUM_CLASSES) -> keras.Model:
    """Defines a simple Keras Sequential model."""
    model = keras.Sequential([
        layers.InputLayer(input_shape=(num_features,)), # Input layer for 7 features
        layers.Dense(16, activation='relu', name='dense_1'),
        layers.Dense(8, activation='relu', name='dense_2'),
        layers.Dense(num_classes, activation='sigmoid', name='output') # Sigmoid for binary probability
    ])
    return model

def train_and_save_model(
    model_base_save_path: str,
    model_version: str,
    mlflow_tracking_uri_arg: Optional[str] = None, # Renamed to avoid clash with local var
    mlflow_model_name_arg: Optional[str] = None, # Renamed
    mlflow_run_name_arg: Optional[str] = None, # Renamed
    mlflow_model_tags_arg: Optional[Dict[str, str]] = None # Renamed
):
    """Generates data, defines, trains, and saves the model to a versioned path, with MLflow logging."""
    print("Generating mock data...")
    features, labels = generate_mock_data()

    # Simple train/validation split
    split_ratio = 0.8
    split_index = int(len(features) * split_ratio)

    train_features, val_features = features[:split_index], features[split_index:]
    train_labels, val_labels = labels[:split_index], labels[split_index:]

    print(f"Training features shape: {train_features.shape}")
    print(f"Training labels shape: {train_labels.shape}")
    print(f"Validation features shape: {val_features.shape}")
    print(f"Validation labels shape: {val_labels.shape}")

    print("Defining model...")
    model = define_model()

    print("Compiling model...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    model.summary() # Print model summary

    print("Training model...")
    history = model.fit(
        train_features,
        train_labels,
        epochs=10, # Small number of epochs for a placeholder
        batch_size=32,
        validation_data=(val_features, val_labels),
        verbose=2 # Print less output during training
    )

    print("Training history:", history.history)

    print("Evaluating model on validation set...")
    val_loss, val_accuracy, val_precision, val_recall = model.evaluate(val_features, val_labels, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")

    # Save the model in TensorFlow SavedModel format
    # Create versioned directory: model_base_save_path/model_version
    versioned_model_path = os.path.join(model_base_save_path, model_version)

    # Clean up existing model directory if it exists to avoid errors during save
    if os.path.exists(versioned_model_path):
        print(f"Cleaning up existing model directory: {versioned_model_path}")
        shutil.rmtree(versioned_model_path)

    print(f"Saving model to: {versioned_model_path}")
    model.save(versioned_model_path) # Creates a directory with assets, variables, and .pb
    print(f"Model saved successfully to {versioned_model_path}")

    # --- MLflow Logging and Registration ---
    # Prioritize CLI args, then environment variables, then defaults for some
    effective_mlflow_tracking_uri = mlflow_tracking_uri_arg if mlflow_tracking_uri_arg is not None else os.environ.get("MLFLOW_TRACKING_URI")
    effective_mlflow_model_name = mlflow_model_name_arg if mlflow_model_name_arg is not None else os.environ.get("MLFLOW_MODEL_NAME", "claims_model_from_training_script")

    default_run_name = f"training_run_{model_version}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    effective_mlflow_run_name = mlflow_run_name_arg if mlflow_run_name_arg is not None else default_run_name

    if effective_mlflow_tracking_uri and effective_mlflow_model_name:
        print(f"MLflow: Configured to log to {effective_mlflow_tracking_uri} for model {effective_mlflow_model_name}")
        mlflow.set_tracking_uri(effective_mlflow_tracking_uri)

        try:
            with mlflow.start_run(run_name=effective_mlflow_run_name) as run:
                run_id = run.info.run_id
                print(f"MLflow: Started run {run_id} with name {effective_mlflow_run_name}")

                # Log parameters
                if mlflow_model_tags_arg:
                    mlflow.set_tags(mlflow_model_tags_arg)
                    print(f"MLflow: Set run tags: {mlflow_model_tags_arg}")

                mlflow.log_param("model_version_script", model_version)
                mlflow.log_param("num_features", NUM_FEATURES)
                mlflow.log_param("num_classes", NUM_CLASSES)
                mlflow.log_param("epochs", 10) # Hardcoded in script
                mlflow.log_param("batch_size", 32) # Hardcoded in script

                # Log metrics
                mlflow.log_metric("val_loss", val_loss)
                mlflow.log_metric("val_accuracy", val_accuracy)
                mlflow.log_metric("val_precision", val_precision)
                mlflow.log_metric("val_recall", val_recall)

                # Log the TensorFlow SavedModel
                print(f"MLflow: Logging SavedModel from {versioned_model_path} to MLflow.")
                mlflow.tensorflow.log_model(
                    tf_saved_model_dir=versioned_model_path,
                    artifact_path="saved_model",
                    registered_model_name=effective_mlflow_model_name # Use effective name
                )
                print(f"MLflow: Model logged and registered as '{effective_mlflow_model_name}'.")

        except Exception as mlflow_e:
            print(f"MLflow: Error during MLflow operations: {mlflow_e}", file=sys.stderr)
    else:
        print("MLflow: MLFLOW_TRACKING_URI or MLFLOW_MODEL_NAME not effectively configured. Skipping MLflow logging.")
    # --- End MLflow Logging and Registration ---

    # Placeholder comments for next MLOps steps
    print("\n--- MLOps Next Steps Placeholders ---")
    print(f"1. Convert the SavedModel to TensorFlow Lite (TFLite):")
    print(f"   - Use tf.lite.TFLiteConverter.from_saved_model('{versioned_model_path}')")
    print(f"   - Apply optimizations (e.g., tf.lite.Optimize.DEFAULT for size/latency).")
    print(f"   - Implement 8-bit integer quantization (representative dataset needed).")
    print(f"   - Save the .tflite model file (e.g., '{os.path.join(SCRIPT_DIR, '..', MODEL_NAME + '.tflite')}')")
    print(f"2. Version control the .tflite model (e.g., using Git LFS or a model registry).")
    print(f"3. Update application settings (ML_MODEL_PATH) to point to the new .tflite model.")
    print(f"4. Deploy the application with the updated model.")
    print(f"--- End MLOps Next Steps Placeholders ---")

if __name__ == '__main__':
    # Ensure the base model directory and the specific model's base path exist
    if not os.path.exists(ROOT_SAVED_MODEL_DIR):
        os.makedirs(ROOT_SAVED_MODEL_DIR)
        print(f"Created root saved model directory: {ROOT_SAVED_MODEL_DIR}")

    if not os.path.exists(MODEL_BASE_PATH):
        os.makedirs(MODEL_BASE_PATH)
        print(f"Created base model directory for '{MODEL_NAME}': {MODEL_BASE_PATH}")

    parser = argparse.ArgumentParser(description="Train and save a claims processing model, with MLflow integration.")
    parser.add_argument("--model-version", type=str, help="Version for the model being trained (e.g., '2', '3'). Overrides default.")
    parser.add_argument("--mlflow-tracking-uri", type=str, help="MLflow tracking server URI. Overrides MLFLOW_TRACKING_URI env var.")
    parser.add_argument("--mlflow-model-name", type=str, help="Name for the model in MLflow Model Registry. Overrides MLFLOW_MODEL_NAME env var.")
    parser.add_argument("--mlflow-run-name", type=str, help="Name for this MLflow run.")
    parser.add_argument("--mlflow-model-tags", type=str, help="Comma-separated key:value pairs for MLflow run tags (e.g., 'stage:dev,team:claims').")

    args = parser.parse_args()

    parsed_tags = None
    if args.mlflow_model_tags:
        try:
            parsed_tags = dict(tag.split(':', 1) for tag in args.mlflow_model_tags.split(',') if ':' in tag)
        except ValueError:
            print("Warning: --mlflow-model-tags format is invalid. Expected 'key1:value1,key2:value2'. Tags will not be set.", file=sys.stderr)

    effective_model_version = args.model_version if args.model_version else MODEL_VERSION

    train_and_save_model(
        MODEL_BASE_PATH,
        effective_model_version,
        mlflow_tracking_uri_arg=args.mlflow_tracking_uri,
        mlflow_model_name_arg=args.mlflow_model_name,
        mlflow_run_name_arg=args.mlflow_run_name,
        mlflow_model_tags_arg=parsed_tags
    )
```
