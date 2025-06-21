import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession # For type hint
from decimal import Decimal
from datetime import date, datetime, timezone

from claims_processor.src.processing.claims_processing_service import ClaimProcessingService
from claims_processor.src.core.database.models.claims_db import ClaimModel, ClaimLineItemModel
from claims_processor.src.api.models.claim_models import ProcessableClaim, ProcessableClaimLineItem
from claims_processor.src.core.monitoring.app_metrics import MetricsCollector
from claims_processor.src.core.config.settings import Settings

# Using a fixed UTC for tests
try:
    from zoneinfo import ZoneInfo
    UTC = ZoneInfo("UTC")
except ImportError:
    UTC = timezone.utc

@pytest.fixture
def mock_db_session():
    session = AsyncMock(spec=AsyncSession)
    async_cm = AsyncMock()
    session.begin = MagicMock(return_value=async_cm)
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.refresh = AsyncMock()
    return session

@pytest.fixture
def mock_metrics_collector():
    collector = MagicMock(spec=MetricsCollector)
    collector.record_individual_claim_duration = MagicMock()
    collector.record_batch_processed = MagicMock()
    # Mock other methods of MetricsCollector if they are directly called by ClaimProcessingService
    return collector

@pytest.fixture
def mock_settings_fixture(tmp_path: Path) -> Settings:
    dummy_model_file = tmp_path / "dummy_model.tflite"
    dummy_model_file.touch()
    settings = Settings(
        ML_MODEL_PATH=str(dummy_model_file),
        ML_FEATURE_COUNT=7,
        ML_APPROVAL_THRESHOLD=0.8,
        FETCH_BATCH_SIZE=50, # Default for ClaimProcessingService's batch fetching
        VALIDATION_CONCURRENCY=5, # Test specific values
        RVU_CALCULATION_CONCURRENCY=3, # Test specific values
        MAX_CONCURRENT_CLAIM_PROCESSING=10, # Old value, still in settings model
        ML_PREDICTION_CACHE_MAXSIZE=10,
        ML_PREDICTION_CACHE_TTL=60
    )
    return settings

@pytest.fixture
def claim_processing_service_instance(mock_db_session, mock_metrics_collector, mock_settings_fixture):
    with patch('claims_processor.src.processing.claims_processing_service.get_settings', return_value=mock_settings_fixture), \
         patch('claims_processor.src.processing.rvu_service.get_settings', return_value=mock_settings_fixture), \
         patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.get_settings', return_value=mock_settings_fixture), \
         patch('claims_processor.src.core.cache.cache_manager.get_settings', return_value=mock_settings_fixture):
        with patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE', True):
            service = ClaimProcessingService(db_session=mock_db_session, metrics_collector=mock_metrics_collector)
            # Mock sub-services directly on the instance for finer control in tests
            service.validator = MagicMock()
            service.feature_extractor = MagicMock()
            # OptimizedPredictor instance already created, mock its predict_batch if needed
            service.predictor.predict_batch = AsyncMock()
            # RVUService instance already created, mock its calculate_rvu_for_claim
            service.rvu_service.calculate_rvu_for_claim = AsyncMock()
            service._store_failed_claim = AsyncMock() # Mock this helper
            service._update_claim_and_lines_in_db = AsyncMock() # Mock this helper
            return service

# Helper to create a basic ClaimModel for tests
def create_mock_db_claim(claim_id_str="test_claim_001", status="pending", priority=1, id_val=None) -> ClaimModel:
    if id_val is None:
        try: id_val = int(claim_id_str.split('_')[-1])
        except ValueError: id_val = 1
    return ClaimModel(
        id=id_val, claim_id=claim_id_str, facility_id="FAC001", patient_account_number="PATACC001",
        patient_first_name="Testy", patient_last_name="McTestFace", patient_date_of_birth="1990-01-01",
        medical_record_number="MRN001", financial_class="COM", insurance_type="PPO", insurance_plan_id="PLAN01",
        service_from_date=date(2023, 1, 1), service_to_date=date(2023, 1, 2), total_charges=Decimal("100.00"),
        processing_status=status, priority=priority, created_at=datetime.now(UTC), updated_at=datetime.now(UTC)
    )

def create_mock_processable_claim_from_db(db_claim: ClaimModel) -> ProcessableClaim:
    # Simulates ProcessableClaim.model_validate(db_claim)
    return ProcessableClaim(
        id=db_claim.id, claim_id=db_claim.claim_id, facility_id=db_claim.facility_id,
        patient_account_number=db_claim.patient_account_number,
        medical_record_number=db_claim.medical_record_number,
        patient_first_name=db_claim.patient_first_name, patient_last_name=db_claim.patient_last_name,
        # Assuming patient_date_of_birth in DB is string, convert to date for ProcessableClaim if model expects date
        patient_date_of_birth=date.fromisoformat(db_claim.patient_date_of_birth) if isinstance(db_claim.patient_date_of_birth, str) else db_claim.patient_date_of_birth,
        insurance_type=db_claim.insurance_type, insurance_plan_id=db_claim.insurance_plan_id,
        financial_class=db_claim.financial_class, service_from_date=db_claim.service_from_date,
        service_to_date=db_claim.service_to_date, total_charges=db_claim.total_charges,
        processing_status=db_claim.processing_status, batch_id=db_claim.batch_id,
        created_at=db_claim.created_at, updated_at=db_claim.updated_at, processed_at=db_claim.processed_at,
        line_items=[], # Add line items if needed for specific tests
        ml_score=db_claim.ml_score, ml_derived_decision=db_claim.ml_derived_decision,
        processing_duration_ms=db_claim.processing_duration_ms
    )


# --- Test __init__ ---
def test_init_semaphores(claim_processing_service_instance: ClaimProcessingService, mock_settings_fixture: Settings):
    assert claim_processing_service_instance.validation_ml_semaphore._value == mock_settings_fixture.VALIDATION_CONCURRENCY
    assert claim_processing_service_instance.rvu_semaphore._value == mock_settings_fixture.RVU_CALCULATION_CONCURRENCY

# --- Tests for _perform_validation_and_ml ---
@pytest.mark.asyncio
async def test_val_ml_success(claim_processing_service_instance: ClaimProcessingService):
    mock_db_claim = create_mock_db_claim()
    processable_claim = create_mock_processable_claim_from_db(mock_db_claim)
    processable_claim.processing_status = "converted_to_pydantic" # Initial status

    claim_processing_service_instance.validator.validate_claim.return_value = [] # No validation errors
    claim_processing_service_instance.feature_extractor.extract_features.return_value = np.array([1.0, 2.0]) # Dummy features
    claim_processing_service_instance.predictor.predict_batch.return_value = [{"ml_score": 0.9, "ml_derived_decision": "ML_APPROVED"}]

    result_claim = await claim_processing_service_instance._perform_validation_and_ml(processable_claim, mock_db_claim)

    assert result_claim.processing_status == "ml_complete"
    assert result_claim.ml_derived_decision == "ML_APPROVED"
    claim_processing_service_instance._store_failed_claim.assert_not_called()

@pytest.mark.asyncio
async def test_val_ml_validation_fails(claim_processing_service_instance: ClaimProcessingService):
    mock_db_claim = create_mock_db_claim()
    processable_claim = create_mock_processable_claim_from_db(mock_db_claim)
    validation_errors = ["Test validation error"]
    claim_processing_service_instance.validator.validate_claim.return_value = validation_errors

    result_claim = await claim_processing_service_instance._perform_validation_and_ml(processable_claim, mock_db_claim)

    assert result_claim.processing_status == "validation_failed"
    claim_processing_service_instance._store_failed_claim.assert_called_once_with(
        db_claim_to_update=mock_db_claim,
        processable_claim_instance=processable_claim,
        failed_stage="VALIDATION",
        reason=f"Validation errors: {'; '.join(validation_errors)}"
    )
    claim_processing_service_instance.predictor.predict_batch.assert_not_called()

@pytest.mark.asyncio
async def test_val_ml_ml_rejects(claim_processing_service_instance: ClaimProcessingService):
    mock_db_claim = create_mock_db_claim()
    processable_claim = create_mock_processable_claim_from_db(mock_db_claim)
    claim_processing_service_instance.validator.validate_claim.return_value = []
    claim_processing_service_instance.feature_extractor.extract_features.return_value = np.array([1.0, 2.0])
    claim_processing_service_instance.predictor.predict_batch.return_value = [{"ml_score": 0.1, "ml_derived_decision": "ML_REJECTED"}]

    result_claim = await claim_processing_service_instance._perform_validation_and_ml(processable_claim, mock_db_claim)

    assert result_claim.processing_status == "ml_rejected"
    claim_processing_service_instance._store_failed_claim.assert_called_once()
    assert claim_processing_service_instance._store_failed_claim.call_args[1]['failed_stage'] == "ML_REJECTION"

# --- Tests for _perform_rvu_calculation ---
@pytest.mark.asyncio
async def test_rvu_success(claim_processing_service_instance: ClaimProcessingService):
    mock_db_claim = create_mock_db_claim()
    processable_claim = create_mock_processable_claim_from_db(mock_db_claim)
    processable_claim.processing_status = "ml_complete" # Prerequisite status

    result_claim = await claim_processing_service_instance._perform_rvu_calculation(processable_claim, mock_db_claim)

    assert result_claim.processing_status == "processing_complete"
    claim_processing_service_instance.rvu_service.calculate_rvu_for_claim.assert_called_once_with(processable_claim, mock_db_session)
    claim_processing_service_instance._store_failed_claim.assert_not_called()

@pytest.mark.asyncio
async def test_rvu_fails(claim_processing_service_instance: ClaimProcessingService):
    mock_db_claim = create_mock_db_claim()
    processable_claim = create_mock_processable_claim_from_db(mock_db_claim)
    processable_claim.processing_status = "ml_complete"
    claim_processing_service_instance.rvu_service.calculate_rvu_for_claim.side_effect = Exception("RVU Boom")

    result_claim = await claim_processing_service_instance._perform_rvu_calculation(processable_claim, mock_db_claim)

    assert result_claim.processing_status == "rvu_calculation_failed"
    claim_processing_service_instance._store_failed_claim.assert_called_once()
    assert claim_processing_service_instance._store_failed_claim.call_args[1]['failed_stage'] == "RVU_CALCULATION_FAILED"

# --- Tests for _process_single_claim_concurrently (Orchestrator) ---
@pytest.mark.asyncio
async def test_orchestrator_full_success_path(claim_processing_service_instance: ClaimProcessingService, mock_metrics_collector):
    mock_db_claim = create_mock_db_claim()

    # Mock stage methods to indicate success
    claim_processing_service_instance._perform_validation_and_ml = AsyncMock(
        return_value=create_mock_processable_claim_from_db(mock_db_claim) # Return modified claim
    )
    claim_processing_service_instance._perform_validation_and_ml.return_value.processing_status = "ml_complete"

    claim_processing_service_instance._perform_rvu_calculation = AsyncMock(
        return_value=create_mock_processable_claim_from_db(mock_db_claim)
    )
    claim_processing_service_instance._perform_rvu_calculation.return_value.processing_status = "processing_complete"


    result_dict = await claim_processing_service_instance._process_single_claim_concurrently(mock_db_claim)

    assert result_dict["status"] == "processing_complete"
    claim_processing_service_instance._perform_validation_and_ml.assert_called_once()
    claim_processing_service_instance._perform_rvu_calculation.assert_called_once()
    claim_processing_service_instance._update_claim_and_lines_in_db.assert_called_once()
    # Check that the final status passed to _update_claim_and_lines_in_db is "processing_complete"
    assert claim_processing_service_instance._update_claim_and_lines_in_db.call_args[0][2] == "processing_complete"
    mock_metrics_collector.record_individual_claim_duration.assert_called_once()


@pytest.mark.asyncio
async def test_orchestrator_pydantic_conversion_fails(claim_processing_service_instance: ClaimProcessingService, mock_metrics_collector):
    mock_db_claim = create_mock_db_claim()

    with patch('claims_processor.src.api.models.claim_models.ProcessableClaim.model_validate', side_effect=ValueError("Pydantic error")):
        result_dict = await claim_processing_service_instance._process_single_claim_concurrently(mock_db_claim)

    assert result_dict["status"] == "conversion_error"
    claim_processing_service_instance._store_failed_claim.assert_called_once_with(
        db_claim_to_update=mock_db_claim,
        processable_claim_instance=None,
        failed_stage="CONVERSION_ERROR",
        reason="Pydantic conversion error: Pydantic error"
    )
    claim_processing_service_instance._update_claim_and_lines_in_db.assert_called_once()
    assert claim_processing_service_instance._update_claim_and_lines_in_db.call_args[0][2] == "conversion_error"
    claim_processing_service_instance._perform_validation_and_ml.assert_not_called()
    claim_processing_service_instance._perform_rvu_calculation.assert_not_called()
    mock_metrics_collector.record_individual_claim_duration.assert_called_once()


# Keep existing tests for _update_claim_and_lines_in_db and _fetch_pending_claims
# ... (tests from previous state) ...
# (The content below is from the previous version of the file, with minor adaptations)

# --- Tests for _update_claim_and_lines_in_db ---
@pytest.mark.asyncio
async def test_update_db_success_first_attempt(claim_processing_service_instance: ClaimProcessingService, mock_db_session):
    mock_claim = create_mock_db_claim()
    # Reset the service's _update_claim_and_lines_in_db to use the real one for this test
    claim_processing_service_instance._update_claim_and_lines_in_db = ClaimProcessingService._update_claim_and_lines_in_db.__get__(claim_processing_service_instance, ClaimProcessingService)

    await claim_processing_service_instance._update_claim_and_lines_in_db(
        db_claim_to_update=mock_claim,
        processed_pydantic_claim=None,
        new_status="processing_complete"
    )

    mock_db_session.add.assert_called_with(mock_claim)
    mock_db_session.commit.assert_called_once()
    mock_db_session.rollback.assert_not_called()
    mock_db_session.refresh.assert_called_once_with(mock_claim)

@pytest.mark.asyncio
@patch('asyncio.sleep', new_callable=AsyncMock)
async def test_update_db_retry_then_success(mock_sleep: AsyncMock, claim_processing_service_instance: ClaimProcessingService, mock_db_session):
    mock_claim = create_mock_db_claim()
    claim_processing_service_instance._update_claim_and_lines_in_db = ClaimProcessingService._update_claim_and_lines_in_db.__get__(claim_processing_service_instance, ClaimProcessingService)

    mock_db_session.commit.side_effect = [OperationalError("DB error", {}, None), AsyncMock()]

    await claim_processing_service_instance._update_claim_and_lines_in_db(
        db_claim_to_update=mock_claim,
        processed_pydantic_claim=None,
        new_status="processing_complete"
    )

    assert mock_db_session.commit.call_count == 2
    mock_db_session.rollback.assert_called_once()
    mock_sleep.assert_called_once()
    assert mock_db_session.add.call_count == 2
    mock_db_session.refresh.assert_called_once_with(mock_claim)

@pytest.mark.asyncio
@patch('asyncio.sleep', new_callable=AsyncMock)
async def test_update_db_all_retries_fail(mock_sleep: AsyncMock, claim_processing_service_instance: ClaimProcessingService, mock_db_session):
    mock_claim = create_mock_db_claim()
    claim_processing_service_instance._update_claim_and_lines_in_db = ClaimProcessingService._update_claim_and_lines_in_db.__get__(claim_processing_service_instance, ClaimProcessingService)
    max_retries = 3

    mock_db_session.commit.side_effect = OperationalError("DB error", {}, None)

    with pytest.raises(OperationalError):
        await claim_processing_service_instance._update_claim_and_lines_in_db(
            db_claim_to_update=mock_claim,
            processed_pydantic_claim=None,
            new_status="processing_complete"
        )

    assert mock_db_session.commit.call_count == max_retries
    assert mock_db_session.rollback.call_count == max_retries
    assert mock_sleep.call_count == max_retries - 1

@pytest.mark.asyncio
async def test_update_db_non_retryable_exception(claim_processing_service_instance: ClaimProcessingService, mock_db_session):
    mock_claim = create_mock_db_claim()
    claim_processing_service_instance._update_claim_and_lines_in_db = ClaimProcessingService._update_claim_and_lines_in_db.__get__(claim_processing_service_instance, ClaimProcessingService)

    mock_db_session.commit.side_effect = ValueError("Non-retryable error")

    with pytest.raises(ValueError):
        await claim_processing_service_instance._update_claim_and_lines_in_db(
            db_claim_to_update=mock_claim,
            processed_pydantic_claim=None,
            new_status="processing_complete"
        )

    mock_db_session.commit.assert_called_once()
    mock_db_session.rollback.assert_called_once()

# --- Tests for _fetch_pending_claims ordering ---
@pytest.mark.asyncio
async def test_fetch_pending_claims_ordering(claim_processing_service_instance: ClaimProcessingService, mock_db_session):
    mock_result_proxy = AsyncMock()
    mock_result_proxy.scalars.return_value.all.return_value = [create_mock_db_claim()]
    mock_db_session.execute.return_value = mock_result_proxy

    await claim_processing_service_instance._fetch_pending_claims(effective_batch_size=10)

    mock_db_session.execute.assert_called_once()
    executed_statement = mock_db_session.execute.call_args[0][0]

    from sqlalchemy.dialects import postgresql
    compiled_query = str(executed_statement.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))

    assert "ORDER BY claims.priority DESC, claims.created_at ASC" in compiled_query
    assert "LIMIT 10" in compiled_query

# --- Tests for process_pending_claims_batch error aggregation ---
@pytest.mark.asyncio
async def test_process_batch_error_aggregation(claim_processing_service_instance: ClaimProcessingService, mock_metrics_collector):
    num_claims = 5
    num_exceptions = 2
    # Use create_mock_db_claim for consistency
    mock_claims = [create_mock_db_claim(claim_id_str=f"claim_{i}", id_val=i) for i in range(num_claims)]

    claim_processing_service_instance._fetch_pending_claims = AsyncMock(return_value=mock_claims)

    # _process_single_claim_concurrently is now an orchestrator.
    # To test error aggregation in process_pending_claims_batch, we make _process_single_claim_concurrently return error dicts.
    async def mock_orchestrator_side_effect(db_claim: ClaimModel):
        if db_claim.id < (num_claims - num_exceptions):
            return {"status": "processing_complete", "claim_id": db_claim.claim_id, "ml_decision": "ML_APPROVED"}
        else:
            # Simulate an error case being returned by the orchestrator
            return {"status": "unhandled_orchestration_error", "claim_id": db_claim.claim_id, "ml_decision": "N/A", "error_detail_str": "Simulated error"}
            # Alternatively, to test asyncio.gather(return_exceptions=True) catching direct exceptions:
            # raise ValueError(f"Simulated processing error for {db_claim.claim_id}")


    claim_processing_service_instance._process_single_claim_concurrently = AsyncMock(side_effect=mock_orchestrator_side_effect)

    summary = await claim_processing_service_instance.process_pending_claims_batch(batch_size_override=num_claims)

    assert summary["attempted_claims"] == num_claims
    # The "by_status" field now directly contains the counts from claims_by_final_status_for_metric
    assert summary["by_status"].get("processing_complete") == (num_claims - num_exceptions)
    assert summary["by_status"].get("unhandled_orchestration_error") == num_exceptions # if _process_single_claim_concurrently returns such status
                                                                                  # or if it raises an exception caught by gather

    expected_statuses_for_metrics = {
        'processing_complete': num_claims - num_exceptions,
        'unhandled_orchestration_error': num_exceptions
    }

    mock_metrics_collector.record_batch_processed.assert_called_once()
    args, _ = mock_metrics_collector.record_batch_processed.call_args
    assert args[0] == num_claims
    assert args[2] == expected_statuses_for_metrics

```
