import pytest
from unittest.mock import patch, MagicMock, call, ANY
import os
import sys
from datetime import datetime
import argparse # Won't directly mock argparse, but test its effects via function params
from typing import Dict, Optional

# Module to be tested
from models.training_scripts.train_model import (
    train_and_save_model,
    MODEL_BASE_PATH, # Base path for saving models, used in calls
    MODEL_NAME,      # Default model name component
    MODEL_VERSION as DEFAULT_MODEL_VERSION, # Default version if not overridden
    NUM_FEATURES,
    NUM_CLASSES
)

# --- Constants for Test ---
DEFAULT_EPOCHS = 10 # Matches hardcoded value in script
DEFAULT_BATCH_SIZE = 32 # Matches hardcoded value in script

# --- Pytest Fixtures (Optional, can also use direct patching in tests) ---

@pytest.fixture
def mock_tf_keras_model():
    """ Mocks the Keras model and its methods. """
    mock_model = MagicMock()
    # model.evaluate returns a list/tuple like [loss, acc, prec, recall]
    mock_model.evaluate.return_value = [0.5, 0.8, 0.7, 0.6]
    # model.fit returns a History object, mock its .history attribute
    mock_history = MagicMock()
    mock_history.history = {
        'loss': [1.0, 0.8], 'accuracy': [0.7, 0.75],
        'val_loss': [0.9, 0.85], 'val_accuracy': [0.65, 0.7]
    }
    mock_model.fit.return_value = mock_history
    return mock_model

@pytest.fixture
def mock_mlflow_start_run():
    """ Mocks mlflow.start_run context manager. """
    mock_run = MagicMock()
    mock_run.info.run_id = "test_run_id_123"

    mock_cm = MagicMock() # Mock for the context manager itself
    mock_cm.__enter__.return_value = mock_run # __enter__ should return the run object
    mock_cm.__exit__.return_value = None
    return mock_cm

# --- Test Cases ---

@patch('models.training_scripts.train_model.os.makedirs')
@patch('models.training_scripts.train_model.os.path.exists')
@patch('models.training_scripts.train_model.shutil.rmtree')
@patch('models.training_scripts.train_model.define_model') # Mocks the function that returns the Keras model
@patch('mlflow.set_tracking_uri')
@patch('mlflow.start_run')
@patch('mlflow.log_param')
@patch('mlflow.log_metric')
@patch('mlflow.tensorflow.log_model')
@patch('mlflow.set_tags')
def test_mlflow_disabled_no_args_no_env(
    mock_mlflow_set_tags: MagicMock,
    mock_mlflow_tf_log_model: MagicMock,
    mock_mlflow_log_metric: MagicMock,
    mock_mlflow_log_param: MagicMock,
    mock_mlflow_start_run_cm: MagicMock, # Patched mlflow.start_run
    mock_mlflow_set_uri: MagicMock,
    mock_define_model: MagicMock,
    mock_shutil_rmtree: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_makedirs: MagicMock,
    mock_tf_keras_model: MagicMock # Fixture providing mocked Keras model
):
    """
    Tests that MLflow functions are NOT called when no MLflow args or env vars are set.
    """
    mock_define_model.return_value = mock_tf_keras_model
    mock_os_path_exists.return_value = False # Simulate model dir doesn't exist initially

    # Simulate no MLflow env vars
    with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "", "MLFLOW_MODEL_NAME": ""}, clear=True):
        train_and_save_model(
            model_base_save_path=MODEL_BASE_PATH,
            model_version=DEFAULT_MODEL_VERSION
            # No MLflow args passed
        )

    # Assert Keras model methods were called (training happened)
    mock_define_model.assert_called_once()
    mock_tf_keras_model.compile.assert_called_once()
    mock_tf_keras_model.fit.assert_called_once()
    mock_tf_keras_model.evaluate.assert_called_once()
    mock_tf_keras_model.save.assert_called_once() # Check that model saving was attempted

    # Assert MLflow functions were NOT called
    mock_mlflow_set_uri.assert_not_called()
    mock_mlflow_start_run_cm.assert_not_called()
    mock_mlflow_log_param.assert_not_called()
    mock_mlflow_log_metric.assert_not_called()
    mock_mlflow_tf_log_model.assert_not_called()
    mock_mlflow_set_tags.assert_not_called()


@patch('models.training_scripts.train_model.os.makedirs')
@patch('models.training_scripts.train_model.os.path.exists')
@patch('models.training_scripts.train_model.shutil.rmtree')
@patch('models.training_scripts.train_model.define_model')
@patch('mlflow.set_tracking_uri')
@patch('mlflow.start_run') # This will be the mock_mlflow_start_run_fixture
@patch('mlflow.log_param')
@patch('mlflow.log_metric')
@patch('mlflow.tensorflow.log_model')
@patch('mlflow.set_tags')
def test_mlflow_enabled_via_cli_args(
    mock_mlflow_set_tags: MagicMock,
    mock_mlflow_tf_log_model: MagicMock,
    mock_mlflow_log_metric: MagicMock,
    mock_mlflow_log_param: MagicMock,
    mock_mlflow_start_run_fixture: MagicMock, # Patched mlflow.start_run
    mock_mlflow_set_uri: MagicMock,
    mock_define_model: MagicMock,
    mock_shutil_rmtree: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_makedirs: MagicMock,
    mock_tf_keras_model: MagicMock, # Fixture
    # mock_mlflow_start_run fixture is not directly passed if we are patching mlflow.start_run
    # The mock_mlflow_start_run_fixture *is* the patch of mlflow.start_run
    fixture_mock_mlflow_start_run_cm: MagicMock # This is the fixture we defined earlier
):
    """
    Tests that MLflow functions ARE called with CLI-provided values.
    """
    mock_define_model.return_value = mock_tf_keras_model
    mock_os_path_exists.return_value = False
    # Configure the mock for mlflow.start_run (which is mock_mlflow_start_run_fixture due to patching order)
    # The mock_mlflow_start_run_fixture (the patched version of mlflow.start_run)
    # should return the context manager provided by fixture_mock_mlflow_start_run_cm
    mock_mlflow_start_run_fixture.return_value = fixture_mock_mlflow_start_run_cm

    cli_tracking_uri = "http://cli-mlflow-server:1234"
    cli_model_name = "cli_claims_model"
    cli_run_name = "cli_test_run"
    cli_tags_dict = {"source": "cli", "test_type": "args"}
    cli_model_version = "77"

    # Simulate no MLflow env vars to ensure CLI args take precedence
    with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "", "MLFLOW_MODEL_NAME": ""}, clear=True):
        train_and_save_model(
            model_base_save_path=MODEL_BASE_PATH,
            model_version=cli_model_version,
            mlflow_tracking_uri_arg=cli_tracking_uri,
            mlflow_model_name_arg=cli_model_name,
            mlflow_run_name_arg=cli_run_name,
            mlflow_model_tags_arg=cli_tags_dict
        )

    mock_mlflow_set_uri.assert_called_once_with(cli_tracking_uri)
    mock_mlflow_start_run_fixture.assert_called_once_with(run_name=cli_run_name)

    # Check parameters logged
    expected_params = [
        call("model_version_script", cli_model_version),
        call("num_features", NUM_FEATURES),
        call("num_classes", NUM_CLASSES),
        call("epochs", DEFAULT_EPOCHS),
        call("batch_size", DEFAULT_BATCH_SIZE)
    ]
    mock_mlflow_log_param.assert_has_calls(expected_params, any_order=True)
    assert mock_mlflow_log_param.call_count == len(expected_params)

    # Check metrics logged (using evaluate return values from mock_tf_keras_model)
    # val_loss, val_accuracy, val_precision, val_recall = [0.5, 0.8, 0.7, 0.6]
    mock_mlflow_log_metric.assert_any_call("val_loss", 0.5)
    mock_mlflow_log_metric.assert_any_call("val_accuracy", 0.8)
    mock_mlflow_log_metric.assert_any_call("val_precision", 0.7)
    mock_mlflow_log_metric.assert_any_call("val_recall", 0.6)

    # Check model logging
    expected_model_save_path = os.path.join(MODEL_BASE_PATH, cli_model_version)
    mock_mlflow_tf_log_model.assert_called_once_with(
        tf_saved_model_dir=expected_model_save_path,
        artifact_path="saved_model",
        registered_model_name=cli_model_name
    )
    # Check tags
    mock_mlflow_set_tags.assert_called_once_with(cli_tags_dict)


@patch('models.training_scripts.train_model.os.makedirs')
@patch('models.training_scripts.train_model.os.path.exists')
@patch('models.training_scripts.train_model.shutil.rmtree')
@patch('models.training_scripts.train_model.define_model')
@patch('mlflow.set_tracking_uri')
@patch('mlflow.start_run')
@patch('mlflow.log_param')
@patch('mlflow.log_metric')
@patch('mlflow.tensorflow.log_model')
@patch('mlflow.set_tags')
@patch('os.environ.get') # To mock environment variables
def test_training_script_mlflow_enabled_via_env_vars(
    mock_os_environ_get: MagicMock,
    mock_mlflow_set_tags: MagicMock,
    mock_mlflow_tf_log_model: MagicMock,
    mock_mlflow_log_metric: MagicMock,
    mock_mlflow_log_param: MagicMock,
    mock_mlflow_start_run_patch: MagicMock, # This is the patched mlflow.start_run
    mock_mlflow_set_uri: MagicMock,
    mock_define_model: MagicMock,
    mock_shutil_rmtree: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_makedirs: MagicMock,
    mock_tf_keras_model: MagicMock,
    mock_mlflow_start_run # Fixture for context manager
):
    """
    Tests that MLflow functions ARE called with values from environment variables.
    """
    mock_define_model.return_value = mock_tf_keras_model
    mock_os_path_exists.return_value = False
    mock_mlflow_start_run_patch.return_value = mock_mlflow_start_run # Configure patched start_run

    env_tracking_uri = "http://env-mlflow-server:5678"
    env_model_name = "env_claims_model"

    # Simulate os.environ.get returning values for MLflow vars
    def environ_get_side_effect(key, default=None):
        if key == "MLFLOW_TRACKING_URI":
            return env_tracking_uri
        if key == "MLFLOW_MODEL_NAME":
            return env_model_name
        return default
    mock_os_environ_get.side_effect = environ_get_side_effect

    # Call with no CLI args for MLflow, relying on environment variables
    train_and_save_model(
        model_base_save_path=MODEL_BASE_PATH,
        model_version=DEFAULT_MODEL_VERSION
        # mlflow_tracking_uri_arg, mlflow_model_name_arg, etc., are None
    )

    mock_mlflow_set_uri.assert_called_once_with(env_tracking_uri)
    # Run name will be the default generated one as it's not set by env var or CLI
    mock_mlflow_start_run_patch.assert_called_once_with(run_name=ANY)

    # Check parameters logged (model_version will be default)
    mock_mlflow_log_param.assert_any_call("model_version_script", DEFAULT_MODEL_VERSION)
    # ... (other param checks similar to CLI test, but with default model version)

    # Check model logging (name from env var)
    expected_model_save_path = os.path.join(MODEL_BASE_PATH, DEFAULT_MODEL_VERSION)
    mock_mlflow_tf_log_model.assert_called_once_with(
        tf_saved_model_dir=expected_model_save_path,
        artifact_path="saved_model",
        registered_model_name=env_model_name
    )
    mock_mlflow_set_tags.assert_not_called() # No tags from env vars in this setup


@patch('models.training_scripts.train_model.os.makedirs')
@patch('models.training_scripts.train_model.os.path.exists')
@patch('models.training_scripts.train_model.shutil.rmtree')
@patch('models.training_scripts.train_model.define_model')
@patch('mlflow.set_tracking_uri')
@patch('mlflow.start_run')
@patch('mlflow.log_param')
@patch('mlflow.log_metric')
@patch('mlflow.tensorflow.log_model')
@patch('mlflow.set_tags')
@patch('os.environ.get')
def test_training_script_cli_overrides_env_vars_for_mlflow(
    mock_os_environ_get: MagicMock,
    mock_mlflow_set_tags: MagicMock,
    mock_mlflow_tf_log_model: MagicMock,
    mock_mlflow_log_metric: MagicMock,
    mock_mlflow_log_param: MagicMock,
    mock_mlflow_start_run_patch: MagicMock,
    mock_mlflow_set_uri: MagicMock,
    mock_define_model: MagicMock,
    mock_shutil_rmtree: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_makedirs: MagicMock,
    mock_tf_keras_model: MagicMock,
    mock_mlflow_start_run # Fixture
):
    """
    Tests that CLI arguments for MLflow override environment variables.
    """
    mock_define_model.return_value = mock_tf_keras_model
    mock_os_path_exists.return_value = False
    mock_mlflow_start_run_patch.return_value = mock_mlflow_start_run

    env_tracking_uri = "http://should_be_overridden_by_cli_env_uri"
    env_model_name = "should_be_overridden_by_cli_env_model_name"

    cli_tracking_uri = "http://actual_cli_uri"
    cli_model_name = "actual_cli_model_name"
    cli_run_name = "run_name_from_cli"

    # Simulate os.environ.get returning ENV values
    def environ_get_side_effect(key, default=None):
        if key == "MLFLOW_TRACKING_URI":
            return env_tracking_uri
        if key == "MLFLOW_MODEL_NAME":
            return env_model_name
        return default
    mock_os_environ_get.side_effect = environ_get_side_effect

    train_and_save_model(
        model_base_save_path=MODEL_BASE_PATH,
        model_version=DEFAULT_MODEL_VERSION,
        mlflow_tracking_uri_arg=cli_tracking_uri, # CLI arg provided
        mlflow_model_name_arg=cli_model_name,   # CLI arg provided
        mlflow_run_name_arg=cli_run_name        # CLI arg provided
    )

    mock_mlflow_set_uri.assert_called_once_with(cli_tracking_uri) # Should use CLI arg
    mock_mlflow_start_run_patch.assert_called_once_with(run_name=cli_run_name)

    mock_mlflow_tf_log_model.assert_called_once_with(
        tf_saved_model_dir=ANY, # Path check not focus here
        artifact_path="saved_model",
        registered_model_name=cli_model_name # Should use CLI arg
    )


@patch('models.training_scripts.train_model.os.makedirs')
@patch('models.training_scripts.train_model.os.path.exists')
@patch('models.training_scripts.train_model.shutil.rmtree')
@patch('models.training_scripts.train_model.define_model')
# Minimal MLflow mocks as versioning is primary focus, but some MLflow calls will occur
@patch('mlflow.set_tracking_uri')
@patch('mlflow.start_run')
@patch('mlflow.log_param')
@patch('mlflow.tensorflow.log_model') # To check model name if MLflow is active
def test_training_script_model_version_override(
    mock_mlflow_tf_log_model: MagicMock,
    mock_mlflow_log_param: MagicMock,
    mock_mlflow_start_run_patch: MagicMock,
    mock_mlflow_set_uri: MagicMock,
    mock_define_model: MagicMock,
    mock_shutil_rmtree: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_makedirs: MagicMock,
    mock_tf_keras_model: MagicMock,
    mock_mlflow_start_run # Fixture
):
    """
    Tests that the --model-version CLI argument overrides the default model version.
    """
    mock_define_model.return_value = mock_tf_keras_model
    mock_os_path_exists.return_value = False # Assume dir needs creation
    mock_mlflow_start_run_patch.return_value = mock_mlflow_start_run # Configure context manager

    cli_model_version = "99"

    # Simulate MLflow being minimally configured to ensure that part of code runs
    # if model_version is logged as a param there.
    cli_tracking_uri = "http://temp-mlflow.com"
    cli_model_name = "version_test_model"

    with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": cli_tracking_uri, "MLFLOW_MODEL_NAME": cli_model_name}, clear=True):
        train_and_save_model(
            model_base_save_path=MODEL_BASE_PATH,
            model_version=cli_model_version, # This is the key CLI arg being tested
            mlflow_tracking_uri_arg=cli_tracking_uri, # Provide to activate MLflow path
            mlflow_model_name_arg=cli_model_name    # Provide to activate MLflow path
        )

    # Check that model.save was called with the correct versioned path
    expected_model_save_path = os.path.join(MODEL_BASE_PATH, cli_model_version)
    mock_tf_keras_model.save.assert_called_once_with(expected_model_save_path)

    # Check that MLflow logged the correct model version
    mock_mlflow_log_param.assert_any_call("model_version_script", cli_model_version)


@patch('models.training_scripts.train_model.os.makedirs')
@patch('models.training_scripts.train_model.os.path.exists')
@patch('models.training_scripts.train_model.shutil.rmtree')
@patch('models.training_scripts.train_model.define_model')
@patch('mlflow.set_tracking_uri')
@patch('mlflow.start_run')
@patch('mlflow.set_tags') # Focus of this test
# Minimal other MLflow mocks
@patch('mlflow.log_param')
@patch('mlflow.log_metric')
@patch('mlflow.tensorflow.log_model')
def test_mlflow_tags_usage_in_function(
    mock_mlflow_tf_log_model: MagicMock,
    mock_mlflow_log_metric: MagicMock,
    mock_mlflow_log_param: MagicMock,
    mock_mlflow_set_tags: MagicMock, # Patched mlflow.set_tags
    mock_mlflow_start_run_patch: MagicMock,
    mock_mlflow_set_uri: MagicMock,
    mock_define_model: MagicMock,
    mock_shutil_rmtree: MagicMock,
    mock_os_path_exists: MagicMock,
    mock_os_makedirs: MagicMock,
    mock_tf_keras_model: MagicMock,
    mock_mlflow_start_run # Fixture
):
    """
    Tests that mlflow.set_tags is called correctly by train_and_save_model
    if a tags dictionary is provided.
    """
    mock_define_model.return_value = mock_tf_keras_model
    mock_os_path_exists.return_value = False
    mock_mlflow_start_run_patch.return_value = mock_mlflow_start_run

    tags_to_pass = {"test_tag_key": "test_tag_value", "framework": "tensorflow"}

    # Simulate MLflow being configured to ensure set_tags is reachable
    tracking_uri = "http://tags-test-mlflow.com"
    model_name = "tags_test_model"
    with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": tracking_uri, "MLFLOW_MODEL_NAME": model_name}, clear=True):
        train_and_save_model(
            model_base_save_path=MODEL_BASE_PATH,
            model_version=DEFAULT_MODEL_VERSION,
            mlflow_model_tags_arg=tags_to_pass, # Pass the pre-parsed dict
            mlflow_tracking_uri_arg=tracking_uri,
            mlflow_model_name_arg=model_name
        )

    mock_mlflow_set_tags.assert_called_once_with(tags_to_pass)

    # Test with no tags
    mock_mlflow_set_tags.reset_mock()
    with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": tracking_uri, "MLFLOW_MODEL_NAME": model_name}, clear=True):
        train_and_save_model(
            model_base_save_path=MODEL_BASE_PATH,
            model_version=DEFAULT_MODEL_VERSION,
            mlflow_model_tags_arg=None, # No tags
            mlflow_tracking_uri_arg=tracking_uri,
            mlflow_model_name_arg=model_name
        )
    mock_mlflow_set_tags.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__])
