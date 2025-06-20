# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a high-performance healthcare claims processing system designed to process 6,667+ claims per second. It validates healthcare claims, calculates RVU (Relative Value Unit) values, applies ML predictions, and transfers data from staging to production databases.

## Key Technologies

- **Python 3.11+** with FastAPI for the web framework
- **PostgreSQL 14+** with asyncpg for high-performance async database operations
- **TensorFlow** for ML-based claim classification
- **Memcached** for caching RVU data
- **SQLAlchemy 2.0+** with async support
- **Alembic** for database migrations
- **Prometheus** for metrics collection
- **structlog** for structured logging

## Common Development Commands

### Running the Application
```bash
# Start the FastAPI application
uvicorn claims_processor.src.main:app --reload --host 0.0.0.0 --port 8000

# With specific environment
export DATABASE_URL="postgresql+asyncpg://user:password@localhost:5432/claims_staging_db"
uvicorn claims_processor.src.main:app --reload
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=claims_processor --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/ --benchmark-only

# Run a single test file
pytest tests/unit/processing/pipeline/test_parallel_claims_processor.py -v

# Run tests with specific markers
pytest -m asyncio
```

### Database Operations
```bash
# Run all migrations
alembic upgrade head

# Create a new migration
alembic revision --autogenerate -m "Description of changes"

# Rollback one migration
alembic downgrade -1

# View migration history
alembic history

# Check current revision
alembic current
```

### Code Quality
```bash
# Format code with black (if installed)
black claims_processor/

# Lint with flake8 (if installed)
flake8 claims_processor/

# Type checking with mypy (if installed)
mypy claims_processor/
```

## Architecture Overview

### Core Components

1. **Parallel Processing Pipeline** (`claims_processor/src/processing/pipeline/parallel_claims_processor.py`)
   - Main entry point for claim processing
   - Handles batch processing with configurable concurrency
   - Stages: Fetch → Validate & ML Process → Calculate RVU → Transfer → Update Status

2. **Database Layer**
   - **Staging DB**: Initial claim storage and processing queue
   - **Production DB**: Final processed claims with analytics data
   - Models in `claims_processor/src/core/database/models/`
   - Connection pooling via `db_session.py`

3. **ML Pipeline**
   - Feature extraction: `ml_pipeline/feature_extractor.py`
   - Prediction: `ml_pipeline/optimized_predictor.py`
   - Uses TensorFlow Lite for optimized inference
   - Configurable approval threshold (default: 0.8)

4. **Security & Compliance**
   - HIPAA-compliant audit logging via `audit_logger_service.py`
   - Field-level encryption for sensitive data via `encryption_service.py`
   - All patient data is encrypted at rest

5. **Caching Layer**
   - Memcached integration for RVU data caching
   - TTL-based cache management
   - Cache service in `cache_manager.py`

### API Endpoints

- `GET /health` - Basic health check
- `GET /ready` - Readiness probe with dependency checks
- `GET /metrics` - Prometheus metrics endpoint
- `POST /api/v1/claims/process` - Process claims batch
- `POST /api/v1/data-transfer/transfer` - Transfer claims to production

### Configuration

Environment variables are defined in `.env` (copy from `.env.template`):
- `DATABASE_URL` - PostgreSQL connection string for staging DB
- `TEST_DATABASE_URL` - Test database connection
- `MEMCACHED_HOST/PORT` - Cache server configuration
- `MAX_CONCURRENT_CLAIM_PROCESSING` - Concurrency limit
- `ML_MODEL_PATH` - Path to TensorFlow Lite model
- `ML_APPROVAL_THRESHOLD` - ML confidence threshold

### Performance Considerations

1. **Batch Processing**
   - Optimal batch sizes: 5,000-50,000 claims
   - Concurrent operations limited by semaphores
   - Bulk database operations using multi-row inserts

2. **Database Optimization**
   - Monthly partitioned tables for claim line items
   - Comprehensive indexes on frequently queried fields
   - Connection pooling with NullPool for async operations

3. **Concurrency Settings**
   - `VALIDATION_CONCURRENCY = 100`
   - `RVU_CALCULATION_CONCURRENCY = 80`
   - `TRANSFER_CONCURRENCY = 16`
   - `MAX_CONCURRENT_BATCHES = 2`

### Error Handling

- Failed claims are stored in `failed_claims` table
- Automatic retry logic with exponential backoff
- Comprehensive error logging with structlog
- Validation errors include detailed field-level information

### Security Notes

- All patient-identifiable information (PII) is encrypted
- Audit logs track all data access and modifications
- No secrets or API keys should be committed to the repository
- Use environment variables for all sensitive configuration

### Development Tips

1. Always run tests before committing code
2. Use the readiness endpoint to verify all services are healthy
3. Monitor the `/metrics` endpoint for performance data
4. Check logs for detailed error information (JSON formatted)
5. When modifying database schema, always create a migration
6. The ML model file must exist at the path specified in `ML_MODEL_PATH`
7. RVU data is loaded from `data/rvu_data.csv` on startup

### Claude Rules
Rule 1: NEVER disable or remove a feature to fix a bug or error.
Rule 2: NEVER fix an error or bug by hiding it.
Rule 3: NO silent fallbacks or silent failures, all problems should be loud and proud.
Rule 4: Always check online documentation of every package used and do everything the officially recommended way.
Rule 5: Clean up your mess. Remove any temporary and/or outdated files or scripts that were only meant to be used once and no longer serve a purpose.