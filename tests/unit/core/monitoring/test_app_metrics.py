import pytest
from unittest.mock import MagicMock, patch
import time

# Assuming app_metrics.py is in claims_processor.src.core.monitoring
from claims_processor.src.core.monitoring.app_metrics import (
    MetricsCollector,
    CLAIMS_PROCESSED_TOTAL,
    CLAIMS_PROCESSING_DURATION_SECONDS,
    CLAIMS_THROUGHPUT_CLAIMS_PER_SECOND,
    DATABASE_QUERY_DURATION_SECONDS,
    ML_PREDICTIONS_TOTAL,
    ML_PREDICTION_CONFIDENCE,
    CACHE_OPERATIONS_TOTAL,
    ML_INFERENCE_DURATION_SECONDS
)
# Import the module to allow checking attributes like ML_INFERENCE_DURATION_SECONDS
import claims_processor.src.core.monitoring.app_metrics as metrics_collector_module


@pytest.fixture
def metrics_collector() -> MetricsCollector:
    return MetricsCollector()

# Patch all global metric objects for isolation in tests
@pytest.fixture(autouse=True)
def mock_global_metrics(monkeypatch):
    monkeypatch.setattr("claims_processor.src.core.monitoring.app_metrics.CLAIMS_PROCESSED_TOTAL", MagicMock(spec=CLAIMS_PROCESSED_TOTAL))
    monkeypatch.setattr("claims_processor.src.core.monitoring.app_metrics.CLAIMS_PROCESSING_DURATION_SECONDS", MagicMock(spec=CLAIMS_PROCESSING_DURATION_SECONDS))
    monkeypatch.setattr("claims_processor.src.core.monitoring.app_metrics.CLAIMS_THROUGHPUT_CLAIMS_PER_SECOND", MagicMock(spec=CLAIMS_THROUGHPUT_CLAIMS_PER_SECOND))
    monkeypatch.setattr("claims_processor.src.core.monitoring.app_metrics.DATABASE_QUERY_DURATION_SECONDS", MagicMock(spec=DATABASE_QUERY_DURATION_SECONDS))
    monkeypatch.setattr("claims_processor.src.core.monitoring.app_metrics.ML_PREDICTIONS_TOTAL", MagicMock(spec=ML_PREDICTIONS_TOTAL))
    monkeypatch.setattr("claims_processor.src.core.monitoring.app_metrics.ML_PREDICTION_CONFIDENCE", MagicMock(spec=ML_PREDICTION_CONFIDENCE))
    monkeypatch.setattr("claims_processor.src.core.monitoring.app_metrics.CACHE_OPERATIONS_TOTAL", MagicMock(spec=CACHE_OPERATIONS_TOTAL))
    # Ensure ML_INFERENCE_DURATION_SECONDS is mocked
    monkeypatch.setattr("claims_processor.src.core.monitoring.app_metrics.ML_INFERENCE_DURATION_SECONDS", MagicMock(spec=ML_INFERENCE_DURATION_SECONDS))


def test_record_batch_processed(metrics_collector: MetricsCollector):
    # Adjusted to reflect the actual logic in record_batch_processed for throughput
    claims_by_status = {'completed_transferred': 10, 'validation_failed': 2}
    # Assuming 'completed_transferred' is a success state for throughput calculation
    metrics_collector.record_batch_processed(batch_size=12, duration_seconds=1.5, claims_by_final_status=claims_by_status)

    CLAIMS_PROCESSING_DURATION_SECONDS.observe.assert_called_once_with(1.5)
    # Throughput is calculated based on 'successfully_processed' which includes 'completed_transferred'
    expected_throughput = claims_by_status.get('completed_transferred', 0) / 1.5
    CLAIMS_THROUGHPUT_CLAIMS_PER_SECOND.set.assert_called_once_with(expected_throughput)
    CLAIMS_PROCESSED_TOTAL.labels(final_status='completed_transferred').inc.assert_called_once_with(10)
    CLAIMS_PROCESSED_TOTAL.labels(final_status='validation_failed').inc.assert_called_once_with(2)

def test_record_batch_processed_zero_duration(metrics_collector: MetricsCollector):
    claims_by_status = {'completed_transferred': 10}
    metrics_collector.record_batch_processed(batch_size=10, duration_seconds=0, claims_by_final_status=claims_by_status)
    CLAIMS_PROCESSING_DURATION_SECONDS.observe.assert_not_called() # Duration is not observed if 0
    CLAIMS_THROUGHPUT_CLAIMS_PER_SECOND.set.assert_called_once_with(0) # Throughput is 0 if duration is 0
    CLAIMS_PROCESSED_TOTAL.labels(final_status='completed_transferred').inc.assert_called_once_with(10)

def test_record_batch_processed_zero_success(metrics_collector: MetricsCollector):
    claims_by_status = {'validation_failed': 10} # No 'completed_transferred' or 'processing_complete'
    metrics_collector.record_batch_processed(batch_size=10, duration_seconds=1.5, claims_by_final_status=claims_by_status)
    CLAIMS_PROCESSING_DURATION_SECONDS.observe.assert_called_once_with(1.5)
    CLAIMS_THROUGHPUT_CLAIMS_PER_SECOND.set.assert_called_once_with(0) # Throughput is 0 if no successful claims
    CLAIMS_PROCESSED_TOTAL.labels(final_status='validation_failed').inc.assert_called_once_with(10)


def test_record_ml_prediction(metrics_collector: MetricsCollector):
    metrics_collector.record_ml_prediction(outcome="ML_APPROVED", confidence_score=0.95)
    ML_PREDICTIONS_TOTAL.labels(outcome="ML_APPROVED").inc.assert_called_once_with(1)
    ML_PREDICTION_CONFIDENCE.observe.assert_called_once_with(0.95)

def test_record_ml_prediction_no_score(metrics_collector: MetricsCollector):
    metrics_collector.record_ml_prediction(outcome="ML_SKIPPED", confidence_score=None)
    ML_PREDICTIONS_TOTAL.labels(outcome="ML_SKIPPED").inc.assert_called_once_with(1)
    ML_PREDICTION_CONFIDENCE.observe.assert_not_called()


def test_record_cache_operation(metrics_collector: MetricsCollector):
    metrics_collector.record_cache_operation(cache_type='rvu', operation_type='get', outcome='hit')
    CACHE_OPERATIONS_TOTAL.labels(cache_type='rvu', operation_type='get', outcome='hit').inc.assert_called_once_with(1)

def test_record_database_query_duration(metrics_collector: MetricsCollector):
    metrics_collector.record_database_query_duration(query_name='test_query', duration_seconds=0.05)
    DATABASE_QUERY_DURATION_SECONDS.labels(query_name='test_query').observe.assert_called_once_with(0.05)

def test_timer_context_manager(metrics_collector: MetricsCollector):
    with metrics_collector.time_db_query('timed_op'):
        time.sleep(0.001) # Simulate work
    DATABASE_QUERY_DURATION_SECONDS.labels(query_name='timed_op').observe.assert_called_once()
    # Check that observed duration is positive
    args, _ = DATABASE_QUERY_DURATION_SECONDS.labels(query_name='timed_op').observe.call_args
    assert args[0] > 0.0009 # allow for slight timing inaccuracies

def test_record_ml_inference_duration(metrics_collector: MetricsCollector):
    metrics_collector.record_ml_inference_duration(duration_seconds=0.005)
    ML_INFERENCE_DURATION_SECONDS.observe.assert_called_once_with(0.005)

# Add a test for the case where a status in claims_by_final_status has a count of 0
def test_record_batch_processed_zero_count_status(metrics_collector: MetricsCollector):
    claims_by_status = {'completed_transferred': 10, 'validation_failed': 0}
    metrics_collector.record_batch_processed(batch_size=10, duration_seconds=1.5, claims_by_final_status=claims_by_status)

    CLAIMS_PROCESSING_DURATION_SECONDS.observe.assert_called_once_with(1.5)
    expected_throughput = 10 / 1.5
    CLAIMS_THROUGHPUT_CLAIMS_PER_SECOND.set.assert_called_once_with(expected_throughput)
    CLAIMS_PROCESSED_TOTAL.labels(final_status='completed_transferred').inc.assert_called_once_with(10)
    # Ensure .inc is not called for statuses with 0 count
    CLAIMS_PROCESSED_TOTAL.labels(final_status='validation_failed').inc.assert_not_called()

# Test that other success states are also counted for throughput
def test_record_batch_processed_other_success_state_for_throughput(metrics_collector: MetricsCollector):
    claims_by_status = {'processing_complete': 5, 'validation_failed': 2}
    # 'processing_complete' is another success state in the implementation
    metrics_collector.record_batch_processed(batch_size=7, duration_seconds=1.0, claims_by_final_status=claims_by_status)

    CLAIMS_PROCESSING_DURATION_SECONDS.observe.assert_called_once_with(1.0)
    expected_throughput = 5 / 1.0 # Only 'processing_complete' contributes here
    CLAIMS_THROUGHPUT_CLAIMS_PER_SECOND.set.assert_called_once_with(expected_throughput)
    CLAIMS_PROCESSED_TOTAL.labels(final_status='processing_complete').inc.assert_called_once_with(5)
    CLAIMS_PROCESSED_TOTAL.labels(final_status='validation_failed').inc.assert_called_once_with(2)

# Test that if both success states are present, they are summed for throughput
def test_record_batch_processed_combined_success_states_for_throughput(metrics_collector: MetricsCollector):
    claims_by_status = {'processing_complete': 5, 'completed_transferred': 3, 'ml_rejected': 2}
    metrics_collector.record_batch_processed(batch_size=10, duration_seconds=2.0, claims_by_final_status=claims_by_status)

    CLAIMS_PROCESSING_DURATION_SECONDS.observe.assert_called_once_with(2.0)
    # Both 'processing_complete' and 'completed_transferred' contribute to successfully_processed
    expected_throughput = (5 + 3) / 2.0
    CLAIMS_THROUGHPUT_CLAIMS_PER_SECOND.set.assert_called_once_with(expected_throughput)
    CLAIMS_PROCESSED_TOTAL.labels(final_status='processing_complete').inc.assert_called_once_with(5)
    CLAIMS_PROCESSED_TOTAL.labels(final_status='completed_transferred').inc.assert_called_once_with(3)
    CLAIMS_PROCESSED_TOTAL.labels(final_status='ml_rejected').inc.assert_called_once_with(2)

# Test the monkeypatching of ML_INFERENCE_DURATION_SECONDS specifically
def test_ml_inference_duration_is_mocked(metrics_collector: MetricsCollector):
    # This test primarily ensures the mock_global_metrics fixture correctly mocks ML_INFERENCE_DURATION_SECONDS
    assert isinstance(ML_INFERENCE_DURATION_SECONDS, MagicMock)
    metrics_collector.record_ml_inference_duration(duration_seconds=0.001)
    ML_INFERENCE_DURATION_SECONDS.observe.assert_called_once_with(0.001)

# Test that the global metrics are indeed MagicMock objects
def test_global_metrics_are_mocked():
    assert isinstance(CLAIMS_PROCESSED_TOTAL, MagicMock)
    assert isinstance(CLAIMS_PROCESSING_DURATION_SECONDS, MagicMock)
    assert isinstance(CLAIMS_THROUGHPUT_CLAIMS_PER_SECOND, MagicMock)
    assert isinstance(DATABASE_QUERY_DURATION_SECONDS, MagicMock)
    assert isinstance(ML_PREDICTIONS_TOTAL, MagicMock)
    assert isinstance(ML_PREDICTION_CONFIDENCE, MagicMock)
    assert isinstance(CACHE_OPERATIONS_TOTAL, MagicMock)
    assert isinstance(ML_INFERENCE_DURATION_SECONDS, MagicMock)

# Ensure __init__.py exists for the test directory
# This is not a test function but a placeholder for thought process
# if __name__ == "__main__":
#   # Create __init__.py if it doesn't exist
#   pass
# Actually, better to use the tool to create __init__.py if needed.
# For now, assume it exists or will be created as part of test running infrastructure.
# The plan did not include creating __init__.py here, so I will stick to the plan.
# I will create __init__.py for tests/unit/core/monitoring if it's missing in a later step or if tests fail.

# Test the Timer class indirectly via time_db_query
# The timer's __exit__ calls record_database_query_duration on the collector instance
@patch.object(MetricsCollector, 'record_database_query_duration')
def test_timer_calls_record_database_query_duration(mock_record_db_query, metrics_collector: MetricsCollector):
    query_name = "my_test_query"
    with metrics_collector.time_db_query(query_name):
        time.sleep(0.0001)  # Simulate some work

    mock_record_db_query.assert_called_once()
    args, _ = mock_record_db_query.call_args
    assert args[0] == query_name
    assert isinstance(args[1], float) and args[1] > 0

# Test that CLAIMS_PROCESSED_TOTAL is not called if count is not > 0
def test_claims_processed_total_not_called_for_zero_count(metrics_collector: MetricsCollector):
    claims_by_status = {'completed_transferred': 0, 'validation_failed': 5}
    metrics_collector.record_batch_processed(batch_size=5, duration_seconds=1.0, claims_by_final_status=claims_by_status)

    CLAIMS_PROCESSED_TOTAL.labels(final_status='completed_transferred').inc.assert_not_called()
    CLAIMS_PROCESSED_TOTAL.labels(final_status='validation_failed').inc.assert_called_once_with(5)

# Test that CLAIMS_THROUGHPUT_CLAIMS_PER_SECOND is set to 0 if duration is positive but no claims are successfully processed.
def test_throughput_zero_if_no_successful_claims_positive_duration(metrics_collector: MetricsCollector):
    claims_by_status = {'validation_failed': 5} # No success states
    metrics_collector.record_batch_processed(batch_size=5, duration_seconds=1.0, claims_by_final_status=claims_by_status)

    CLAIMS_THROUGHPUT_CLAIMS_PER_SECOND.set.assert_called_once_with(0)

# Check if `tests/unit/core/monitoring/__init__.py` exists, create if not.
# This is important for pytest to discover the tests correctly.
# I will do this as a separate step.
# For now, assuming the test file itself is created correctly.

```
