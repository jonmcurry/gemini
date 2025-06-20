from prometheus_client import Counter, Histogram, Gauge
import structlog

logger = structlog.get_logger(__name__)

# --- Claim Processing Metrics ---

CLAIMS_PROCESSED_TOTAL = Counter(
    'claims_processed_total',
    'Total claims processed by the system.',
    ['final_status'] # Labels: e.g., "processing_complete", "ml_rejected", "validation_failed", "conversion_error", "rvu_calculation_failed"
)

# Histogram for individual claim processing duration (from start of _process_single_claim_concurrently to its end)
CLAIM_PROCESSING_DURATION_SECONDS = Histogram(
    'claim_processing_duration_seconds',
    'Time spent processing individual claims within a batch.',
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0, float('inf'))
)

# Histogram for overall batch processing duration
CLAIMS_BATCH_PROCESSING_DURATION_SECONDS = Histogram(
    'claims_batch_processing_duration_seconds',
    'Time spent processing a complete batch of claims.',
    buckets=(1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 300.0, float('inf'))
)

CLAIMS_THROUGHPUT_LAST_BATCH = Gauge(
    'claims_throughput_last_batch',
    'Number of claims successfully processed per second in the last completed batch.'
)

# --- Machine Learning Metrics ---

ML_PREDICTIONS_TOTAL = Counter(
    'ml_predictions_total',
    'Total ML predictions made or attempted.',
    ['ml_decision_outcome'] # Labels: e.g., "ML_APPROVED", "ML_REJECTED", "ML_PROCESSING_ERROR", "ML_ERROR_UNEXPECTED_FORMAT"
)

ML_PREDICTION_CONFIDENCE_SCORE = Histogram(
    'ml_prediction_confidence_score',
    'Distribution of ML prediction confidence scores (for approval probability).',
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, float('inf'))
)

# --- Cache Metrics ---

CACHE_OPERATIONS_TOTAL = Counter(
    'cache_operations_total',
    'Total cache operations performed.',
    ['cache_type', 'operation_type'] # cache_type: "rvu_cache", operation_type: "hit", "miss", "set_error", "get_error"
)

logger.info("Prometheus metrics defined.")
