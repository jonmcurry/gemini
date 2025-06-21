import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession # For type hint
from decimal import Decimal
from datetime import date, datetime, timezone
from pathlib import Path
import hashlib # For A/B test hashing

from claims_processor.src.processing.claims_processing_service import ClaimProcessingService
from claims_processor.src.core.database.models.claims_db import ClaimModel, ClaimLineItemModel
from claims_processor.src.api.models.claim_models import ProcessableClaim, ProcessableClaimLineItem
from claims_processor.src.core.monitoring.app_metrics import MetricsCollector
from claims_processor.src.core.config.settings import Settings
from claims_processor.src.processing.ml_pipeline.optimized_predictor import OptimizedPredictor # For patching

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
    return collector

@pytest.fixture
def mock_settings(tmp_path: Path) -> Settings: # Renamed from mock_settings_fixture for consistency
    # Base settings used by default in claim_processing_service fixture
    # Specific tests can override these by re-patching get_settings or modifying this fixture's return
    dummy_model_file = tmp_path / "dummy_model.tflite"
    dummy_model_file.touch()

    # For A/B testing path, create another dummy file
    dummy_challenger_model_file = tmp_path / "challenger_model.tflite"
    dummy_challenger_model_file.touch()

    return Settings(
        ML_MODEL_PATH=str(dummy_model_file),
        ML_FEATURE_COUNT=7,
        ML_APPROVAL_THRESHOLD=0.8,
        FETCH_BATCH_SIZE=50,
        VALIDATION_CONCURRENCY=5,
        RVU_CALCULATION_CONCURRENCY=3,
        ML_PREDICTION_CACHE_MAXSIZE=10,
        ML_PREDICTION_CACHE_TTL=60,
        # A/B Test defaults (can be overridden in specific tests)
        ML_CHALLENGER_MODEL_PATH=str(dummy_challenger_model_file), # Default to having a path
        ML_AB_TEST_TRAFFIC_PERCENTAGE_TO_CHALLENGER=0.0, # Default to A/B off
        ML_AB_TEST_CLAIM_ID_SALT="test_salt"
    )

@pytest.fixture
def claim_processing_service(mock_db_session, mock_metrics_collector, mock_settings):
    # This fixture provides a service instance where get_settings is patched globally for all modules
    # that might import it during the service's __init__ chain.
    with patch('claims_processor.src.processing.claims_processing_service.get_settings', return_value=mock_settings), \
         patch('claims_processor.src.processing.rvu_service.get_settings', return_value=mock_settings), \
         patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.get_settings', return_value=mock_settings), \
         patch('claims_processor.src.core.cache.cache_manager.get_settings', return_value=mock_settings):
        # Ensure TFLITE_AVAILABLE is True for OptimizedPredictor initialization
        with patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE', True):
            # We need to mock OptimizedPredictor constructor *before* ClaimProcessingService is instantiated
            with patch('claims_processor.src.processing.claims_processing_service.OptimizedPredictor') as MockOptPredictor:
                # Configure the mock OptimizedPredictor instances that will be created
                mock_primary_predictor_inst = MagicMock(spec=OptimizedPredictor)
                mock_primary_predictor_inst.predict_batch = AsyncMock()

                mock_challenger_predictor_inst = MagicMock(spec=OptimizedPredictor)
                mock_challenger_predictor_inst.predict_batch = AsyncMock()

                # Side effect for constructor: return primary, then challenger if called again
                MockOptPredictor.side_effect = [mock_primary_predictor_inst, mock_challenger_predictor_inst]

                service = ClaimProcessingService(db_session=mock_db_session, metrics_collector=mock_metrics_collector)

                # Attach mocks to service instance for easy access in tests
                service.primary_predictor_mock = mock_primary_predictor_inst # type: ignore
                if service.challenger_predictor: # If challenger was created
                    service.challenger_predictor_mock = mock_challenger_predictor_inst # type: ignore

                # Mock other sub-services/helpers
                service.validator = MagicMock()
                service.feature_extractor = MagicMock()
                service.rvu_service.calculate_rvu_for_claim = AsyncMock()
                service._store_failed_claim = AsyncMock()
                service._update_claim_and_lines_in_db = AsyncMock()
                return service

# Helper functions (create_mock_db_claim, create_mock_processable_claim_from_db) remain the same

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
        patient_date_of_birth=date.fromisoformat(db_claim.patient_date_of_birth) if isinstance(db_claim.patient_date_of_birth, str) else db_claim.patient_date_of_birth,
        insurance_type=db_claim.insurance_type, insurance_plan_id=db_claim.insurance_plan_id,
        financial_class=db_claim.financial_class, service_from_date=db_claim.service_from_date,
        service_to_date=db_claim.service_to_date, total_charges=db_claim.total_charges,
        processing_status=db_claim.processing_status, batch_id=db_claim.batch_id,
        created_at=db_claim.created_at, updated_at=db_claim.updated_at, processed_at=db_claim.processed_at,
        line_items=[],
        ml_score=db_claim.ml_score, ml_derived_decision=db_claim.ml_derived_decision,
        processing_duration_ms=db_claim.processing_duration_ms
    )

# --- Test __init__ for A/B Predictor Setup ---
def test_init_ab_testing_no_challenger_path(mock_db_session, mock_metrics_collector, mock_settings):
    mock_settings.ML_CHALLENGER_MODEL_PATH = None # Ensure no challenger path
    with patch('claims_processor.src.processing.claims_processing_service.get_settings', return_value=mock_settings), \
         patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE', True):
        service = ClaimProcessingService(db_session=mock_db_session, metrics_collector=mock_metrics_collector)
        assert service.challenger_predictor is None

def test_init_ab_testing_challenger_path_zero_traffic(mock_db_session, mock_metrics_collector, mock_settings):
    mock_settings.ML_AB_TEST_TRAFFIC_PERCENTAGE_TO_CHALLENGER = 0.0 # Zero traffic
    # ML_CHALLENGER_MODEL_PATH is set by default in mock_settings
    with patch('claims_processor.src.processing.claims_processing_service.get_settings', return_value=mock_settings), \
         patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE', True):
        # The current logic in __init__ instantiates challenger if path is set AND traffic > 0.
        # So, if traffic is 0, challenger_predictor should be None (or not instantiated).
        # Let's refine the OptimizedPredictor patch to check calls.
        with patch('claims_processor.src.processing.claims_processing_service.OptimizedPredictor') as MockOptPredictor:
            service = ClaimProcessingService(db_session=mock_db_session, metrics_collector=mock_metrics_collector)
            # Primary is always called. Challenger should not be if traffic is 0.
            # This depends on the exact logic in __init__ (if path set but traffic 0, is it instantiated?)
            # Based on current ClaimProcessingService.__init__, it is NOT instantiated if traffic is 0.
            assert service.challenger_predictor is None
            MockOptPredictor.assert_called_once() # Only primary

@patch('claims_processor.src.processing.claims_processing_service.OptimizedPredictor')
def test_init_ab_testing_with_active_challenger(MockOptPredictor: MagicMock, mock_db_session, mock_metrics_collector, mock_settings: Settings, tmp_path: Path):
    mock_settings.ML_CHALLENGER_MODEL_PATH = str(tmp_path / "challenger.tflite")
    (tmp_path / "challenger.tflite").touch()
    mock_settings.ML_AB_TEST_TRAFFIC_PERCENTAGE_TO_CHALLENGER = 0.5

    with patch('claims_processor.src.processing.claims_processing_service.get_settings', return_value=mock_settings), \
         patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE', True):
        service = ClaimProcessingService(db_session=mock_db_session, metrics_collector=mock_metrics_collector)

        assert MockOptPredictor.call_count == 2
        # Check primary model path
        assert MockOptPredictor.call_args_list[0][1]['model_path'] == mock_settings.ML_MODEL_PATH
        # Check challenger model path
        assert MockOptPredictor.call_args_list[1][1]['model_path'] == mock_settings.ML_CHALLENGER_MODEL_PATH
        assert service.primary_predictor is not None
        assert service.challenger_predictor is not None

@patch('claims_processor.src.processing.claims_processing_service.OptimizedPredictor')
@patch('claims_processor.src.processing.claims_processing_service.logger') # Mock logger
def test_init_ab_testing_challenger_instantiation_fails(mock_logger: MagicMock, MockOptPredictor: MagicMock, mock_db_session, mock_metrics_collector, mock_settings: Settings, tmp_path: Path):
    mock_settings.ML_CHALLENGER_MODEL_PATH = str(tmp_path / "challenger_fail.tflite")
    (tmp_path / "challenger_fail.tflite").touch()
    mock_settings.ML_AB_TEST_TRAFFIC_PERCENTAGE_TO_CHALLENGER = 0.5

    # First call (primary) is fine, second call (challenger) raises error
    MockOptPredictor.side_effect = [MagicMock(spec=OptimizedPredictor), Exception("Challenger load error")]

    with patch('claims_processor.src.processing.claims_processing_service.get_settings', return_value=mock_settings), \
         patch('claims_processor.src.processing.ml_pipeline.optimized_predictor.TFLITE_AVAILABLE', True):
        service = ClaimProcessingService(db_session=mock_db_session, metrics_collector=mock_metrics_collector)
        assert service.challenger_predictor is None
        mock_logger.error.assert_called_with(
            "Failed to instantiate Challenger OptimizedPredictor. A/B test will be inactive.",
            path=mock_settings.ML_CHALLENGER_MODEL_PATH, error="Challenger load error", exc_info=True
        )

# --- Tests for A/B Routing in _perform_validation_and_ml ---
@pytest.mark.asyncio
@patch('hashlib.sha256')
async def test_ab_routing_to_primary_model(mock_sha256: MagicMock, claim_processing_service_instance: ClaimProcessingService, mock_settings: Settings):
    mock_settings.ML_CHALLENGER_MODEL_PATH = "challenger.tflite" # Ensure challenger path is set
    mock_settings.ML_AB_TEST_TRAFFIC_PERCENTAGE_TO_CHALLENGER = 0.5 # 50% traffic
    claim_processing_service_instance.settings = mock_settings # Update service's settings view

    # Ensure challenger predictor mock exists on the service instance for this test
    claim_processing_service_instance.challenger_predictor = MagicMock(spec=OptimizedPredictor)
    claim_processing_service_instance.challenger_predictor.predict_batch = AsyncMock()


    mock_hash_obj = MagicMock()
    mock_hash_obj.hexdigest.return_value = "9abc" # int("9abc", 16) % 100 = 39964 % 100 = 64 (>= 50, so primary)
    mock_sha256.return_value = mock_hash_obj

    mock_db_claim = create_mock_db_claim(claim_id_str="claim_route_primary")
    processable_claim = create_mock_processable_claim_from_db(mock_db_claim)

    claim_processing_service_instance.validator.validate_claim.return_value = []
    claim_processing_service_instance.feature_extractor.extract_features.return_value = np.array([1.0]*7)
    claim_processing_service_instance.primary_predictor_mock.predict_batch.return_value = [{"ml_score": 0.8, "ml_derived_decision": "ML_APPROVED"}]


    await claim_processing_service_instance._perform_validation_and_ml(processable_claim, mock_db_claim)

    claim_processing_service_instance.primary_predictor_mock.predict_batch.assert_called_once()
    claim_processing_service_instance.challenger_predictor.predict_batch.assert_not_called()
    assert "control" in processable_claim.ml_model_version_used

@pytest.mark.asyncio
@patch('hashlib.sha256')
async def test_ab_routing_to_challenger_model(mock_sha256: MagicMock, claim_processing_service_instance: ClaimProcessingService, mock_settings: Settings):
    mock_settings.ML_CHALLENGER_MODEL_PATH = "challenger.tflite"
    mock_settings.ML_AB_TEST_TRAFFIC_PERCENTAGE_TO_CHALLENGER = 0.5 # 50% traffic
    claim_processing_service_instance.settings = mock_settings

    claim_processing_service_instance.challenger_predictor = MagicMock(spec=OptimizedPredictor)
    claim_processing_service_instance.challenger_predictor.predict_batch = AsyncMock(return_value=[{"ml_score": 0.85, "ml_derived_decision": "ML_APPROVED"}])

    mock_hash_obj = MagicMock()
    mock_hash_obj.hexdigest.return_value = "1abc" # int("1abc", 16) % 100 = 6844 % 100 = 44 (< 50, so challenger)
    mock_sha256.return_value = mock_hash_obj

    mock_db_claim = create_mock_db_claim(claim_id_str="claim_route_challenger")
    processable_claim = create_mock_processable_claim_from_db(mock_db_claim)

    claim_processing_service_instance.validator.validate_claim.return_value = []
    claim_processing_service_instance.feature_extractor.extract_features.return_value = np.array([1.0]*7)

    await claim_processing_service_instance._perform_validation_and_ml(processable_claim, mock_db_claim)

    claim_processing_service_instance.challenger_predictor.predict_batch.assert_called_once()
    claim_processing_service_instance.primary_predictor_mock.predict_batch.assert_not_called()
    assert "challenger" in processable_claim.ml_model_version_used

# --- Test for persistence of ml_model_version_used ---
@pytest.mark.asyncio
async def test_orchestrator_persists_ml_model_version(claim_processing_service_instance: ClaimProcessingService, mock_settings: Settings):
    mock_db_claim = create_mock_db_claim()

    # Ensure A/B testing is active for this test to make ml_model_version_used more interesting
    mock_settings.ML_CHALLENGER_MODEL_PATH = "challenger.tflite"
    mock_settings.ML_AB_TEST_TRAFFIC_PERCENTAGE_TO_CHALLENGER = 1.0 # Force challenger
    claim_processing_service_instance.settings = mock_settings
    # Re-initialize challenger predictor if it was None due to default fixture settings
    if not claim_processing_service_instance.challenger_predictor:
        claim_processing_service_instance.challenger_predictor = MagicMock(spec=OptimizedPredictor)
        claim_processing_service_instance.challenger_predictor.predict_batch = AsyncMock()

    # Mock _perform_validation_and_ml to set ml_model_version_used
    async def mock_val_ml_side_effect(claim, db_claim):
        claim.ml_model_version_used = "challenger:test"
        claim.processing_status = "ml_complete"
        claim.ml_derived_decision = "ML_APPROVED" # Ensure it passes this stage
        return claim
    claim_processing_service_instance._perform_validation_and_ml = AsyncMock(side_effect=mock_val_ml_side_effect)

    # Mock _perform_rvu_calculation to set final status
    async def mock_rvu_side_effect(claim, db_claim):
        claim.processing_status = "processing_complete"
        return claim
    claim_processing_service_instance._perform_rvu_calculation = AsyncMock(side_effect=mock_rvu_side_effect)

    # Unmock _update_claim_and_lines_in_db to test its actual logic for this field
    real_update_method = ClaimProcessingService._update_claim_and_lines_in_db.__get__(claim_processing_service_instance, ClaimProcessingService)
    with patch.object(claim_processing_service_instance, '_update_claim_and_lines_in_db', new_callable=AsyncMock, side_effect=real_update_method) as mock_update_method_spy:
        await claim_processing_service_instance._process_single_claim_concurrently(mock_db_claim)

        mock_update_method_spy.assert_called_once()
        # Check the ClaimModel instance passed to the real _update_claim_and_lines_in_db
        # The first positional argument to _update_claim_and_lines_in_db is db_claim_to_update (ClaimModel)
        # The second is processed_pydantic_claim (ProcessableClaim)
        call_args_instance = mock_update_method_spy.call_args[0][0] # This is db_claim_to_update
        processed_pydantic_claim_arg = mock_update_method_spy.call_args[0][1]

        assert processed_pydantic_claim_arg.ml_model_version_used == "challenger:test"
        # The actual assertion should be on db_claim_to_update *inside* the _update_claim_and_lines_in_db,
        # which is now part of the real method call due to side_effect.
        # So we check the state of mock_db_claim *after* the call, assuming commit was mocked.
        # The mock_db_session.add will capture the state.
        final_db_claim_state = claim_processing_service_instance.mock_db_session.add.call_args[0][0]
        assert final_db_claim_state.ml_model_version_used == "challenger:test"


# Preserving existing tests below and adapting them if necessary.
# (The content of existing tests like _update_db_*, _fetch_pending_claims_ordering, _process_batch_error_aggregation are here)
# ... (tests from previous state, ensure they use claim_processing_service_instance fixture) ...

# --- Tests for _update_claim_and_lines_in_db ---
@pytest.mark.asyncio
async def test_update_db_success_first_attempt(claim_processing_service_instance: ClaimProcessingService, mock_db_session):
    mock_claim = create_mock_db_claim()
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
    mock_claims = [create_mock_db_claim(claim_id_str=f"claim_{i}", id_val=i) for i in range(num_claims)]

    claim_processing_service_instance._fetch_pending_claims = AsyncMock(return_value=mock_claims)

    async def mock_orchestrator_side_effect(db_claim: ClaimModel):
        if db_claim.id < (num_claims - num_exceptions):
            return {"status": "processing_complete", "claim_id": db_claim.claim_id, "ml_decision": "ML_APPROVED", "ml_model_version_used": "control:test"}
        else:
            return {"status": "unhandled_orchestration_error", "claim_id": db_claim.claim_id, "ml_decision": "N/A", "error_detail_str": "Simulated error", "ml_model_version_used": "control:test"}

    claim_processing_service_instance._process_single_claim_concurrently = AsyncMock(side_effect=mock_orchestrator_side_effect)

    summary = await claim_processing_service_instance.process_pending_claims_batch(batch_size_override=num_claims)

    assert summary["attempted_claims"] == num_claims
    assert summary["by_status"].get("processing_complete") == (num_claims - num_exceptions)
    assert summary["by_status"].get("unhandled_orchestration_error") == num_exceptions

    expected_statuses_for_metrics = {
        'processing_complete': num_claims - num_exceptions,
        'unhandled_orchestration_error': num_exceptions
    }

    mock_metrics_collector.record_batch_processed.assert_called_once()
    args, _ = mock_metrics_collector.record_batch_processed.call_args
    assert args[0] == num_claims
    assert args[2] == expected_statuses_for_metrics
```
