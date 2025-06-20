import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import shutil # For cleaning up saved model directory
import os

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

def train_and_save_model(model_base_save_path: str, model_version: str):
    """Generates data, defines, trains, and saves the model to a versioned path."""
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

    train_and_save_model(MODEL_BASE_PATH, MODEL_VERSION)
```
