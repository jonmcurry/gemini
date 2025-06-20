import pytest
import os
import shutil
import subprocess # To run the script as a separate process
import sys

# Determine paths relative to this test file's location
TEST_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_SCRIPT_DIR, '..', '..')) # Up two levels from tests/models/
TRAIN_SCRIPT_PATH = os.path.join(PROJECT_ROOT, 'models', 'training_scripts', 'train_model.py')

# Expected output model directory from train_model.py
# Based on train_model.py:
# SCRIPT_DIR = models/training_scripts/
# ROOT_SAVED_MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'saved_model')) -> models/saved_model
# MODEL_NAME = "claims_model"
# MODEL_BASE_PATH = os.path.join(ROOT_SAVED_MODEL_DIR, MODEL_NAME) -> models/saved_model/claims_model
# MODEL_VERSION = "1"
# versioned_model_path = os.path.join(MODEL_BASE_PATH, MODEL_VERSION) -> models/saved_model/claims_model/1
EXPECTED_MODEL_DIR_FROM_SCRIPT_ROOT = os.path.join('models', 'saved_model', 'claims_model', '1')
EXPECTED_MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, EXPECTED_MODEL_DIR_FROM_SCRIPT_ROOT)


# Fixture to clean up the created model directory after the test
@pytest.fixture(scope="function") # Run per test function
def cleanup_model_dir():
    # No setup needed before
    yield # Test runs here
    # Teardown: remove the directory if it was created
    # Check the versioned path first
    if os.path.exists(EXPECTED_MODEL_OUTPUT_DIR):
        try:
            shutil.rmtree(EXPECTED_MODEL_OUTPUT_DIR)
            print(f"Cleaned up test model directory: {EXPECTED_MODEL_OUTPUT_DIR}")
        except OSError as e:
            print(f"Error cleaning up test model directory {EXPECTED_MODEL_OUTPUT_DIR}: {e}")

    # Clean up parent directories if they were created by the script and are now empty
    # MODEL_BASE_PATH (models/saved_model/claims_model)
    model_base_path = os.path.dirname(EXPECTED_MODEL_OUTPUT_DIR)
    if os.path.exists(model_base_path) and not os.listdir(model_base_path):
        try:
            shutil.rmtree(model_base_path)
            print(f"Cleaned up empty base model directory: {model_base_path}")
        except OSError as e:
            print(f"Error cleaning up base model directory {model_base_path}: {e}")

    # ROOT_SAVED_MODEL_DIR (models/saved_model)
    root_saved_model_dir = os.path.dirname(model_base_path)
    if os.path.exists(root_saved_model_dir) and not os.listdir(root_saved_model_dir):
        try:
            shutil.rmtree(root_saved_model_dir)
            print(f"Cleaned up empty root saved model directory: {root_saved_model_dir}")
        except OSError as e:
            print(f"Error cleaning up root saved model directory {root_saved_model_dir}: {e}")


def test_train_model_script_runs_and_saves_model(cleanup_model_dir):
    """
    Tests that the train_model.py script runs without errors and creates
    the expected SavedModel directory.
    """
    # Ensure the script path is correct
    assert os.path.exists(TRAIN_SCRIPT_PATH), f"Training script not found at {TRAIN_SCRIPT_PATH}"

    # Run the script as a subprocess
    # This is a robust way to test a script's main execution block
    try:
        # Ensure PYTHONPATH includes the project root so imports in script work
        env = os.environ.copy()
        current_pythonpath = env.get('PYTHONPATH', '')
        # Add project root to the beginning of PYTHONPATH
        # This ensures that 'from claims_processor...' would work if the script used it.
        env['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + current_pythonpath

        process = subprocess.run(
            [sys.executable, TRAIN_SCRIPT_PATH],
            capture_output=True,
            text=True,
            check=True, # Will raise CalledProcessError if script exits with non-zero status
            cwd=PROJECT_ROOT, # Run script from project root for consistent relative paths
            env=env
        )
        print("Training script stdout:")
        print(process.stdout)
        if process.stderr: # Stderr might contain warnings from TensorFlow, so print if not empty
            print("Training script stderr:")
            print(process.stderr)

    except subprocess.CalledProcessError as e:
        print("Training script stdout (on error):")
        print(e.stdout)
        print("Training script stderr (on error):")
        print(e.stderr)
        pytest.fail(f"train_model.py script failed with exit code {e.returncode}: {e.stderr}")
    except FileNotFoundError:
        pytest.fail(f"Could not find Python interpreter '{sys.executable}' or script '{TRAIN_SCRIPT_PATH}'.")

    # Check if the SavedModel directory and key file were created
    assert os.path.isdir(EXPECTED_MODEL_OUTPUT_DIR), \
        f"SavedModel directory not found at {EXPECTED_MODEL_OUTPUT_DIR} after running script."

    expected_pb_file = os.path.join(EXPECTED_MODEL_OUTPUT_DIR, 'saved_model.pb')
    assert os.path.exists(expected_pb_file), \
        f"saved_model.pb not found in {EXPECTED_MODEL_OUTPUT_DIR}."

    # Optional: Check for variables directory as SavedModel format includes it
    expected_vars_dir = os.path.join(EXPECTED_MODEL_OUTPUT_DIR, 'variables')
    assert os.path.isdir(expected_vars_dir), \
        f"variables directory not found in {EXPECTED_MODEL_OUTPUT_DIR}."

    # Optional: Check for assets directory (may or may not exist depending on model)
    # expected_assets_dir = os.path.join(EXPECTED_MODEL_OUTPUT_DIR, 'assets')
    # assert os.path.isdir(expected_assets_dir), \
    #    f"assets directory not found in {EXPECTED_MODEL_OUTPUT_DIR}."
```
