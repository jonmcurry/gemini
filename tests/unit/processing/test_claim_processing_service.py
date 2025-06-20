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
from claims_processor.src.core.config.settings import Settings # For type hinting mock

@pytest.fixture
def mock_db_session():
    session = AsyncMock(spec=AsyncSession)
    async_cm = AsyncMock() # For 'async with session.begin()'
    session.begin = MagicMock(return_value=async_cm)
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.refresh = AsyncMock() # Mock refresh as it's used in _update_claim_and_lines_in_db
    return session

@pytest.fixture
def mock_metrics_collector():
    collector = MagicMock(spec=MetricsCollector)
    # Configure specific methods if their return values or side effects are important
    # For now, a general MagicMock is often sufficient.
    return collector

@pytest.fixture
def mock_settings_fixture(): # Renamed to avoid conflict with 'settings' module
    settings = MagicMock(spec=Settings)
    # Default settings relevant to ClaimProcessingService
    settings.MAX_CONCURRENT_CLAIM_PROCESSING = 10
    settings.ML_MODEL_PATH = "dummy/path/model.tflite" # Or None if model loading is skipped
    settings.ML_FEATURE_COUNT = 7
    settings.ML_APPROVAL_THRESHOLD = 0.8
    settings.FETCH_BATCH_SIZE = 100 # For process_pending_claims_batch default
    # Add other settings as they become relevant for tests
    return settings

@pytest.fixture
def claim_processing_service(mock_db_session, mock_metrics_collector, mock_settings_fixture):
    # Patch get_settings used by ClaimProcessingService and its components like RVUService, OptimizedPredictor
    with patch('claims_processor.src.processing.claims_processing_service.get_settings', return_value=mock_settings_fixture), \
         patch('claims_processor.src.processing.rvu_service.get_settings', return_value=mock_settings_fixture), \
         patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.get_settings', return_value=mock_settings_fixture), \
         patch('claims_processor.src.core.cache.cache_manager.get_settings', return_value=mock_settings_fixture): # If cache_manager uses get_settings directly
        # Mock OptimizedPredictor's TFLITE_AVAILABLE to True to allow initialization
        with patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE', True):
            service = ClaimProcessingService(db_session=mock_db_session, metrics_collector=mock_metrics_collector)
    return service

# Helper to create a basic ClaimModel for tests
def create_mock_claim_model(claim_id_str="test_claim_001", status="pending", priority=1, id_val=None) -> ClaimModel:
    # Ensure id_val is deterministic if not provided
    if id_val is None:
        try:
            id_val = int(claim_id_str.split('_')[-1])
        except ValueError:
            id_val = 1 # Default if parsing fails

    return ClaimModel(
        id=id_val,
        claim_id=claim_id_str,
        facility_id="FAC001",
        patient_account_number="PATACC001",
        patient_first_name="Testy",
        patient_last_name="McTestFace",
        patient_date_of_birth="1990-01-01", # Encrypted string
        medical_record_number="MRN001",
        financial_class="COM",
        insurance_type="PPO",
        insurance_plan_id="PLAN01",
        service_from_date=date(2023, 1, 1),
        service_to_date=date(2023, 1, 2),
        total_charges=Decimal("1000.00"),
        processing_status=status,
        priority=priority,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        line_items=[]
    )

# Helper to create a ProcessableClaim (Pydantic)
def create_mock_processable_claim(claim_id_str="test_claim_001", id_val=None) -> ProcessableClaim:
    if id_val is None:
        try:
            id_val = int(claim_id_str.split('_')[-1])
        except ValueError:
            id_val = 1
    return ProcessableClaim(
        id=id_val,
        claim_id=claim_id_str,
        facility_id="FAC001",
        patient_account_number="PATACC001",
        patient_first_name="Testy",
        patient_last_name="McTestFace",
        patient_date_of_birth="1990-01-01",
        medical_record_number="MRN001",
        financial_class="COM",
        insurance_type="PPO",
        insurance_plan_id="PLAN01",
        service_from_date=date(2023, 1, 1),
        service_to_date=date(2023, 1, 2),
        total_charges=Decimal("1000.00"),
        line_items=[]
    )

# --- Tests for _update_claim_and_lines_in_db ---
@pytest.mark.asyncio
async def test_update_db_success_first_attempt(claim_processing_service: ClaimProcessingService, mock_db_session):
    mock_claim = create_mock_claim_model()

    await claim_processing_service._update_claim_and_lines_in_db(
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
async def test_update_db_retry_then_success(mock_sleep: AsyncMock, claim_processing_service: ClaimProcessingService, mock_db_session):
    mock_claim = create_mock_claim_model()

    mock_db_session.commit.side_effect = [OperationalError("DB error", {}, None), AsyncMock()]

    await claim_processing_service._update_claim_and_lines_in_db(
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
async def test_update_db_all_retries_fail(mock_sleep: AsyncMock, claim_processing_service: ClaimProcessingService, mock_db_session):
    mock_claim = create_mock_claim_model()
    max_retries = 3

    mock_db_session.commit.side_effect = OperationalError("DB error", {}, None)

    with pytest.raises(OperationalError):
        await claim_processing_service._update_claim_and_lines_in_db(
            db_claim_to_update=mock_claim,
            processed_pydantic_claim=None,
            new_status="processing_complete"
        )

    assert mock_db_session.commit.call_count == max_retries
    assert mock_db_session.rollback.call_count == max_retries
    assert mock_sleep.call_count == max_retries - 1

@pytest.mark.asyncio
async def test_update_db_non_retryable_exception(claim_processing_service: ClaimProcessingService, mock_db_session):
    mock_claim = create_mock_claim_model()

    mock_db_session.commit.side_effect = ValueError("Non-retryable error")

    with pytest.raises(ValueError):
        await claim_processing_service._update_claim_and_lines_in_db(
            db_claim_to_update=mock_claim,
            processed_pydantic_claim=None,
            new_status="processing_complete"
        )

    mock_db_session.commit.assert_called_once()
    mock_db_session.rollback.assert_called_once()

# --- Tests for _fetch_pending_claims ordering ---
@pytest.mark.asyncio
async def test_fetch_pending_claims_ordering(claim_processing_service: ClaimProcessingService, mock_db_session):
    mock_result_proxy = AsyncMock()
    mock_result_proxy.scalars.return_value.all.return_value = [create_mock_claim_model()]
    mock_db_session.execute.return_value = mock_result_proxy

    await claim_processing_service._fetch_pending_claims(effective_batch_size=10)

    mock_db_session.execute.assert_called_once()
    executed_statement = mock_db_session.execute.call_args[0][0]

    from sqlalchemy.dialects import postgresql
    compiled_query = str(executed_statement.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))

    assert "ORDER BY claims.priority DESC, claims.created_at ASC" in compiled_query
    assert "LIMIT 10" in compiled_query

# --- Tests for process_pending_claims_batch error aggregation ---
@pytest.mark.asyncio
async def test_process_batch_error_aggregation(claim_processing_service: ClaimProcessingService, mock_metrics_collector):
    num_claims = 5
    num_exceptions = 2
    mock_claims = [create_mock_claim_model(claim_id_str=f"claim_{i}", id_val=i) for i in range(num_claims)]

    claim_processing_service._fetch_pending_claims = AsyncMock(return_value=mock_claims)

    async def mock_process_single(claim: ClaimModel):
        if claim.id < (num_claims - num_exceptions): # Use .id which is now deterministic
            return {"status": "processing_complete", "claim_id": claim.claim_id, "ml_decision": "ML_APPROVED"}
        else:
            raise ValueError(f"Simulated processing error for {claim.claim_id}")

    claim_processing_service._process_single_claim_concurrently = AsyncMock(side_effect=mock_process_single)

    summary = await claim_processing_service.process_pending_claims_batch(batch_size_override=num_claims)

    assert summary["attempted_claims"] == num_claims
    assert summary["successfully_processed_count"] == (num_claims - num_exceptions)
    assert summary["other_exceptions"] == num_exceptions

    expected_statuses = {
        'processing_complete': num_claims - num_exceptions,
        'unhandled_exception_in_gather': num_exceptions
    }

    mock_metrics_collector.record_batch_processed.assert_called_once()
    args, _ = mock_metrics_collector.record_batch_processed.call_args
    assert args[0] == num_claims
    assert args[2] == expected_statuses
