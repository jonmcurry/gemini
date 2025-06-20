from prometheus_client import Counter, Histogram, Gauge, REGISTRY
import time
import structlog # Added for logger definition
from typing import Dict, Optional # Added for type hints

logger = structlog.get_logger(__name__)

# --- Prometheus Metric Definitions ---
# These are defined globally so they are registered with the default REGISTRY

# Processing metrics
CLAIMS_PROCESSED_TOTAL = Counter(
    'claims_processed_total',
    'Total claims processed, labeled by their final status.',
    ['final_status']  # e.g., 'completed_transferred', 'validation_failed', 'ml_rejected', 'processing_error'
)

CLAIMS_PROCESSING_DURATION_SECONDS = Histogram(
    'claims_processing_duration_seconds',
    'Time spent processing a batch of claims, in seconds.',
    # Buckets can be customized e.g. [.005, .01, .025, .05, ..., 15] for 100k claims in <15s
    # Using buckets from previous metrics.py for CLAIMS_BATCH_PROCESSING_DURATION_SECONDS
    buckets=(1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 300.0, float('inf'))
)

CLAIMS_THROUGHPUT_CLAIMS_PER_SECOND = Gauge(
    'claims_throughput_claims_per_second',
    'Current claims processing throughput for a completed batch.'
)

# Database metrics (basic examples)
DATABASE_QUERY_DURATION_SECONDS = Histogram(
    'database_query_duration_seconds',
    'Duration of key database queries, in seconds.',
    ['query_name'] # e.g., 'fetch_pending_claims', 'transfer_to_production', 'update_staging_status'
)

# ML metrics
ML_PREDICTIONS_TOTAL = Counter(
    'ml_predictions_total',
    'Total ML predictions made, labeled by outcome.',
    ['outcome'] # e.g., 'ML_APPROVED', 'ML_REJECTED', 'ML_ERROR', 'ML_SKIPPED'
)

ML_PREDICTION_CONFIDENCE = Histogram(
    'ml_prediction_confidence',
    'Distribution of ML prediction confidence scores (ml_score).',
    buckets=tuple(x / 10.0 for x in range(11)) # 0.0, 0.1, ..., 1.0
)

# Cache metrics
# This definition will replace/override any previous CACHE_OPERATIONS_TOTAL if this module is imported after.
# Ensure consistency or consolidate metric definitions.
CACHE_OPERATIONS_TOTAL = Counter(
    'cache_operations_total',
    'Total cache operations, labeled by type and outcome.',
    ['cache_type', 'operation_type', 'outcome']
)

ML_INFERENCE_DURATION_SECONDS = Histogram(
    'ml_inference_duration_seconds',
    'Duration of individual ML model inference (invoke) calls, in seconds.',
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, float('inf')) # Fine-grained buckets
)

logger.info("Application Prometheus metrics defined in app_metrics.py.")


class MetricsCollector:
    """
    Collects and exposes application metrics using Prometheus client.
    """

    def __init__(self):
        logger.info("MetricsCollector initialized (stateless, uses global metrics).")
        pass

    def record_batch_processed(self, batch_size: int, duration_seconds: float, claims_by_final_status: Dict[str, int]):
        """
        Record metrics for a processed batch.
        """
        if duration_seconds > 0:
            CLAIMS_PROCESSING_DURATION_SECONDS.observe(duration_seconds)
            # Ensure successfully_processed_count is available for accurate throughput
            # For now, using batch_size as a proxy for attempted, actual throughput might be based on successfully_processed_count
            # Let's assume claims_by_final_status contains a count for 'successfully_processed_count' or similar.
            successfully_processed = claims_by_final_status.get('processing_complete', 0) + \
                                     claims_by_final_status.get('completed_transferred', 0) # Example of success states

            if successfully_processed > 0:
                throughput = successfully_processed / duration_seconds
                CLAIMS_THROUGHPUT_CLAIMS_PER_SECOND.set(throughput)
            else: # If no claims were successfully processed in this batch
                CLAIMS_THROUGHPUT_CLAIMS_PER_SECOND.set(0)
        else: # Batch duration was zero or negative (unlikely but handle)
            CLAIMS_THROUGHPUT_CLAIMS_PER_SECOND.set(0)

        for status, count in claims_by_final_status.items():
            if count > 0:
                CLAIMS_PROCESSED_TOTAL.labels(final_status=status).inc(count)

        logger.debug("Batch processed metrics recorded", batch_size=batch_size, duration_s=duration_seconds, status_counts=claims_by_final_status)

    def record_ml_prediction(self, outcome: str, confidence_score: Optional[float]):
        ML_PREDICTIONS_TOTAL.labels(outcome=outcome).inc()
        if confidence_score is not None: # Only observe if a score is available
            ML_PREDICTION_CONFIDENCE.observe(confidence_score)

    def record_cache_operation(self, cache_type: str, operation_type: str, outcome: str):
        CACHE_OPERATIONS_TOTAL.labels(cache_type=cache_type, operation_type=operation_type, outcome=outcome).inc()

    def record_database_query_duration(self, query_name: str, duration_seconds: float):
        DATABASE_QUERY_DURATION_SECONDS.labels(query_name=query_name).observe(duration_seconds)

    def record_ml_inference_duration(self, duration_seconds: float): # New method
        ML_INFERENCE_DURATION_SECONDS.observe(duration_seconds)

    # Helper for timing code blocks (can be used with 'with' statement)
    class Timer:
        def __init__(self, collector_instance: 'MetricsCollector', metric_observer_func, *metric_labels):
            self.collector = collector_instance # Not used if observer_func is bound or static
            self.metric_observer_func = metric_observer_func
            self.metric_labels = metric_labels
            self.start_time: Optional[float] = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.start_time is not None:
                duration_seconds = time.perf_counter() - self.start_time
                if self.metric_labels:
                    self.metric_observer_func(*self.metric_labels, duration_seconds) # For labeled Histograms like DB query
                else:
                    self.metric_observer_func(duration_seconds) # For non-labeled Histograms like ML inference

    def time_db_query(self, query_name: str) -> Timer:
        """Returns a Timer context manager for a database query."""
        # Pass the specific method of the collector instance if methods are not static
        return self.Timer(self, self.record_database_query_duration, query_name)
        # If record_database_query_duration were static or global func:
        # return self.Timer(MetricsCollector, MetricsCollector.record_database_query_duration, query_name)
        # Or if it's a bound method:
        # return self.Timer(self, self.record_database_query_duration, query_name)
        # The current Timer uses self.collector.record_database_query_duration(self.query_name, duration_seconds)
        # which implies record_database_query_duration is an instance method.
        # The original Timer was fine, let's revert Timer to be specific for DB queries for now,
        # and use direct timing for ML inference as per plan.

    # Reverting Timer to its simpler form for time_db_query
    class Timer:
        def __init__(self, collector_instance: 'MetricsCollector', query_name: str):
            self.collector = collector_instance
            self.query_name = query_name
            self.start_time: Optional[float] = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.start_time is not None:
                duration_seconds = time.perf_counter() - self.start_time
                self.collector.record_database_query_duration(self.query_name, duration_seconds)

    def time_db_query(self, query_name: str) -> Timer:
        return self.Timer(self, query_name)


# Optional: Create a global instance if this collector is to be used as a singleton directly
# metrics_collector = MetricsCollector()
# This is often not necessary if methods are called directly on the class,
# or if an instance is managed by a DI framework or explicitly passed around.
# Since methods are instance methods, an instance would be needed.
# For now, services will create an instance of MetricsCollector or have one injected.
# If metrics objects are global, the class mostly serves as a namespace for methods.
# Let's make methods static if they only operate on global metrics and don't need 'self'.
# However, the Timer class needs 'self.collector' if collector_instance is 'self'.
# Keeping as instance methods, assuming an instance of MetricsCollector will be created and used.
