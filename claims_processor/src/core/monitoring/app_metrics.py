from prometheus_client import Counter, Histogram, Gauge, REGISTRY
import time
import structlog # Added for logger definition
from typing import Dict, Optional # Added for type hints

logger = structlog.get_logger(__name__)

# --- Prometheus Metric Definitions ---
# These are defined globally so they are registered with the default REGISTRY

# --- Prometheus Metric Definitions (Consolidated) ---
# These are defined globally so they are registered with the default REGISTRY

# 1. Claims Processing Metrics
CLAIMS_PROCESSED_TOTAL = Counter(
    'claims_processed_total',
    'Total claims processed, labeled by their final status.',
    ['final_status']  # e.g., 'completed_transferred', 'validation_failed', 'ml_rejected', 'processing_error'
)

# Renamed from CLAIMS_PROCESSING_DURATION_SECONDS
CLAIMS_BATCH_PROCESSING_DURATION_SECONDS = Histogram(
    'claims_batch_processing_duration_seconds', # Renamed
    'Time spent processing a batch of claims, in seconds.',
    buckets=(1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, float('inf')) # Adjusted buckets
)

# Added new metric for individual claims
CLAIM_INDIVIDUAL_PROCESSING_DURATION_SECONDS = Histogram(
    'claim_individual_processing_duration_seconds',
    'Time spent processing an individual claim end-to-end (can be high level).',
    # Buckets suitable for individual claim processing time (likely shorter than batches)
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf'))
)

# Default buckets for stage durations
STAGE_DURATION_BUCKETS = (
    0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0,
    2.5, 5.0, float('inf')
)

VALIDATION_STAGE_DURATION_SECONDS = Histogram(
    'validation_stage_duration_seconds',
    'Time spent in the validation stage for a single claim.',
    buckets=STAGE_DURATION_BUCKETS
)

ML_STAGE_DURATION_SECONDS = Histogram(
    'ml_stage_duration_seconds',
    'Time spent in the ML feature extraction and prediction stage for a single claim.',
    buckets=STAGE_DURATION_BUCKETS
)

RVU_STAGE_DURATION_SECONDS = Histogram(
    'rvu_stage_duration_seconds',
    'Time spent in the RVU calculation stage for a single claim.',
    buckets=STAGE_DURATION_BUCKETS
)

# Renamed from CLAIMS_THROUGHPUT_CLAIMS_PER_SECOND
CLAIMS_THROUGHPUT_GAUGE = Gauge(
    'claims_throughput_gauge', # Renamed
    'Current claims processing throughput for a completed batch (claims per second).'
)

# 2. Database Metrics
DATABASE_QUERY_DURATION_SECONDS = Histogram(
    'database_query_duration_seconds',
    'Duration of key database queries, in seconds.',
    ['query_name'] # e.g., 'fetch_pending_claims', 'transfer_to_production', 'update_staging_status'
)

# Added new metric
DATABASE_CONNECTIONS_ACTIVE_GAUGE = Gauge(
    'database_connections_active_gauge',
    'Current number of active database connections.',
    ['database_name'] # e.g., 'staging_db_pool', 'production_db_pool'
)

# 3. ML Metrics
ML_PREDICTIONS_TOTAL = Counter(
    'ml_predictions_total',
    'Total ML predictions made, labeled by outcome.',
    ['outcome'] # e.g., 'ML_APPROVED', 'ML_REJECTED', 'ML_ERROR', 'ML_SKIPPED'
)

# Renamed from ML_PREDICTION_CONFIDENCE
ML_PREDICTION_CONFIDENCE_HISTOGRAM = Histogram(
    'ml_prediction_confidence_histogram', # Renamed
    'Distribution of ML prediction confidence scores (ml_score).',
    buckets=tuple(x / 10.0 for x in range(11)) # 0.0, 0.1, ..., 1.0
)

ML_INFERENCE_DURATION_SECONDS = Histogram(
    'ml_inference_duration_seconds',
    'Duration of individual ML model inference (invoke) calls, in seconds.',
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, float('inf'))
)

# 4. Cache Metrics
CACHE_OPERATIONS_TOTAL = Counter(
    'cache_operations_total',
    'Total cache operations, labeled by type and outcome.',
    ['cache_type', 'operation_type', 'outcome'] # e.g., rvu_cache, get, hit/miss/error
)

logger.info("Application Prometheus metrics consolidated and defined in app_metrics.py.")


class MetricsCollector:
    """
    Collects and exposes application metrics using Prometheus client.
    """

    def __init__(self):
        logger.info("MetricsCollector initialized (stateless, uses global metrics).")
        pass

    def record_batch_processed(self, batch_size: int, duration_seconds: float, claims_by_final_status: Dict[str, int]):
        """
        Record metrics for a processed batch. Uses CLAIMS_BATCH_PROCESSING_DURATION_SECONDS.
        """
        if duration_seconds > 0:
            CLAIMS_BATCH_PROCESSING_DURATION_SECONDS.observe(duration_seconds) # Use renamed metric

            # Calculate successfully processed claims for throughput calculation
            # This definition of "successful" might need adjustment based on actual final_status values
            successfully_processed = sum(
                count for status, count in claims_by_final_status.items()
                if status in ['completed_transferred', 'processing_complete'] # Define what constitutes success
            )

            if successfully_processed > 0:
                throughput = successfully_processed / duration_seconds
                CLAIMS_THROUGHPUT_GAUGE.set(throughput) # Use renamed metric
            else:
                CLAIMS_THROUGHPUT_GAUGE.set(0)
        else:
            CLAIMS_THROUGHPUT_GAUGE.set(0)

        for status, count in claims_by_final_status.items():
            if count > 0:
                CLAIMS_PROCESSED_TOTAL.labels(final_status=status).inc(count)

        logger.debug("Batch processed metrics recorded", batch_size=batch_size, duration_s=duration_seconds, status_counts=claims_by_final_status)

    def record_individual_claim_duration(self, duration_seconds: float):
        """Records the processing duration for an individual claim."""
        CLAIM_INDIVIDUAL_PROCESSING_DURATION_SECONDS.observe(duration_seconds)

    def record_ml_prediction(self, outcome: str, confidence_score: Optional[float]):
        ML_PREDICTIONS_TOTAL.labels(outcome=outcome).inc()
        if confidence_score is not None:
            ML_PREDICTION_CONFIDENCE_HISTOGRAM.observe(confidence_score) # Use renamed metric

    def record_cache_operation(self, cache_type: str, operation_type: str, outcome: str):
        CACHE_OPERATIONS_TOTAL.labels(cache_type=cache_type, operation_type=operation_type, outcome=outcome).inc()

    def record_database_query_duration(self, query_name: str, duration_seconds: float):
        DATABASE_QUERY_DURATION_SECONDS.labels(query_name=query_name).observe(duration_seconds)

    def set_database_connections_active(self, db_name: str, count: int):
        """Sets the current number of active database connections."""
        DATABASE_CONNECTIONS_ACTIVE_GAUGE.labels(database_name=db_name).set(count)

    def record_ml_inference_duration(self, duration_seconds: float):
        ML_INFERENCE_DURATION_SECONDS.observe(duration_seconds)

    def record_validation_stage_duration(self, duration_seconds: float):
        """Records the duration of the validation stage for a single claim."""
        VALIDATION_STAGE_DURATION_SECONDS.observe(duration_seconds)

    def record_ml_stage_duration(self, duration_seconds: float):
        """Records the duration of the ML stage for a single claim."""
        ML_STAGE_DURATION_SECONDS.observe(duration_seconds)

    def record_rvu_stage_duration(self, duration_seconds: float):
        """Records the duration of the RVU calculation stage for a single claim."""
        RVU_STAGE_DURATION_SECONDS.observe(duration_seconds)

    # Timer class for timing database queries
    class _DatabaseTimer: # Renamed to avoid potential confusion if Timer is used elsewhere
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

    def time_db_query(self, query_name: str) -> _DatabaseTimer: # Return type updated
        """Returns a Timer context manager for a database query."""
        return self._DatabaseTimer(self, query_name)


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
