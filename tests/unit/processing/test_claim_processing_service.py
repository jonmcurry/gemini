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
    session.bulk_update_mappings = AsyncMock() # Added for testing batch updates
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
                # service._update_claim_and_lines_in_db = AsyncMock() # Removed as method will be deleted
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
    if not claim_processing_service_instance.challenger_predictor: # Ensure it's mocked if settings enable it
        claim_processing_service_instance.challenger_predictor = MagicMock(spec=OptimizedPredictor)
        claim_processing_service_instance.challenger_predictor.predict_batch = AsyncMock()
    elif hasattr(claim_processing_service_instance, 'challenger_predictor_mock'): # If fixture created it
         claim_processing_service_instance.challenger_predictor = claim_processing_service_instance.challenger_predictor_mock


    # Mock _perform_validation_and_ml to set ml_model_version_used and other relevant fields
    async def mock_val_ml_side_effect(claim_arg, db_claim_arg): # Renamed args to avoid clash
        claim_arg.ml_model_version_used = "challenger:test_version_from_val_ml"
        claim_arg.processing_status = "ml_complete" # Assume validation and ML were successful
        claim_arg.ml_derived_decision = "ML_APPROVED"
        claim_arg.ml_score = Decimal("0.95")
        return claim_arg # Return the modified ProcessableClaim
    claim_processing_service_instance._perform_validation_and_ml = AsyncMock(side_effect=mock_val_ml_side_effect)

    # Mock _perform_rvu_calculation to set final status
    async def mock_rvu_side_effect(claim_arg, db_claim_arg): # Renamed args
        claim_arg.processing_status = "processing_complete" # Assume RVU calculation successful
        return claim_arg # Return the modified ProcessableClaim
    claim_processing_service_instance._perform_rvu_calculation = AsyncMock(side_effect=mock_rvu_side_effect)

    # Call the orchestrator
    result = await claim_processing_service_instance._process_single_claim_concurrently(mock_db_claim)

    # Assert that the returned ProcessableClaim instance has the ml_model_version_used set
    assert isinstance(result, ProcessableClaim)
    assert result.ml_model_version_used == "challenger:test_version_from_val_ml"
    assert result.processing_status == "processing_complete" # Check final status from RVU mock

    # Further checks on what _process_single_claim_concurrently returns can be added here.
    # The main point is that _update_claim_and_lines_in_db is no longer called from it.
    # The state is carried on the ProcessableClaim instance.

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
# Renamed and refactored from test_process_batch_error_aggregation
@pytest.mark.asyncio
async def test_batch_processing_handles_mixed_results_and_commits_successfully(
    claim_processing_service_instance: ClaimProcessingService,
    mock_metrics_collector: MagicMock,
    mock_db_session: MagicMock # Get the session mock directly to check calls on its context manager
):
    num_total_claims = 5
    num_processing_complete = 2
    num_ml_rejected = 1
    num_conversion_error = 1
    num_unhandled_exception = 1 # from _process_single_claim_concurrently

    mock_db_claims = [create_mock_db_claim(claim_id_str=f"claim_{i}", id_val=i+1) for i in range(num_total_claims)]

    claim_processing_service_instance._fetch_pending_claims = AsyncMock(return_value=mock_db_claims)

    # --- Setup side effects for _process_single_claim_concurrently ---
    mock_results_from_single_processing = []

    # Successful processing
    for i in range(num_processing_complete):
        pc = create_mock_processable_claim_from_db(mock_db_claims[i])
        pc.processing_status = "processing_complete"
        pc.ml_derived_decision = "ML_APPROVED"
        pc.ml_model_version_used = "control:test_v1"
        pc.ml_score = Decimal("0.85")
        pc.processing_duration_ms = 120.0
        # Simulate RVU calculation results
        pc.line_items = [ProcessableClaimLineItem(id=(i*10)+j+1, claim_db_id=pc.id, line_number=j+1, service_date=date(2023,1,1), procedure_code="P01", units=1, charge_amount=Decimal(10), rvu_total=Decimal("2.5")) for j in range(2)]
        mock_results_from_single_processing.append(pc)

    # ML Rejected
    idx = num_processing_complete
    pc_rejected = create_mock_processable_claim_from_db(mock_db_claims[idx])
    pc_rejected.processing_status = "ml_rejected"
    pc_rejected.ml_derived_decision = "ML_REJECTED"
    pc_rejected.ml_model_version_used = "control:test_v1"
    pc_rejected.ml_score = Decimal("0.5")
    pc_rejected.processing_duration_ms = 80.0
    mock_results_from_single_processing.append(pc_rejected)

    # Conversion Error
    idx = num_processing_complete + num_ml_rejected
    db_claim_for_conversion_error = mock_db_claims[idx]
    mock_results_from_single_processing.append({
        "error": "conversion_error", "original_db_id": db_claim_for_conversion_error.id,
        "claim_id": db_claim_for_conversion_error.claim_id, "reason": "Simulated Pydantic conversion error",
        "processing_duration_ms": 10.0
    })

    # Unhandled Exception from _process_single_claim_concurrently
    idx = num_processing_complete + num_ml_rejected + num_conversion_error
    mock_results_from_single_processing.append(RuntimeError(f"Simulated unhandled error for claim {mock_db_claims[idx].claim_id}"))

    claim_processing_service_instance._process_single_claim_concurrently = AsyncMock(side_effect=mock_results_from_single_processing)

    # _store_failed_claim is already an AsyncMock from the fixture.
    # db.begin().__aexit__ (commit) should be successful by default from fixture.
    mock_db_session.begin.return_value.__aexit__ = AsyncMock(return_value=None) # Explicitly ensure success

    # --- Call the batch processing method ---
    summary = await claim_processing_service_instance.process_pending_claims_batch(batch_size_override=num_total_claims)

    # --- Assertions ---
    assert summary["attempted_claims"] == num_total_claims
    assert summary["by_status"].get("processing_complete") == num_processing_complete
    assert summary["by_status"].get("ml_rejected") == num_ml_rejected
    assert summary["by_status"].get("conversion_error") == num_conversion_error
    assert summary["by_status"].get("unhandled_orchestration_error") == num_unhandled_exception

    # Assert _store_failed_claim calls
    # It's called by _perform_validation_and_ml for 'ml_rejected'
    # It's called by process_pending_claims_batch for 'conversion_error'
    # It's called by process_pending_claims_batch for unhandled exceptions from _process_single_claim_concurrently
    assert claim_processing_service_instance._store_failed_claim.call_count == (num_ml_rejected + num_conversion_error + num_unhandled_exception)

    # Assert bulk update calls
    # One call for successful_claim_updates (processing_complete + ml_rejected)
    # One call for failed_claim_updates_for_conversion_error (conversion_error + unhandled_orchestration_error)
    # One call for successful_line_item_updates

    # Check ClaimModel updates
    assert mock_db_session.bulk_update_mappings.call_count >= 2 # At least one for successful, one for failed. Could be 3 if line items.

    successful_updates_call = None
    failed_updates_call = None
    line_item_updates_call = None

    for call_args in mock_db_session.bulk_update_mappings.call_args_list:
        model_type, mappings = call_args[0]
        if model_type == ClaimModel:
            # Check content of mappings to differentiate
            is_successful_batch = any(d.get('processing_status') in ["processing_complete", "ml_rejected"] for d in mappings)
            is_failed_batch = any(d.get('processing_status') in ["conversion_error", "unhandled_orchestration_error"] for d in mappings)
            if is_successful_batch:
                successful_updates_call = mappings
            elif is_failed_batch:
                failed_updates_call = mappings
        elif model_type == ClaimLineItemModel:
            line_item_updates_call = mappings

    assert successful_updates_call is not None and len(successful_updates_call) == (num_processing_complete + num_ml_rejected)
    assert failed_updates_call is not None and len(failed_updates_call) == (num_conversion_error + num_unhandled_exception)

    expected_line_item_updates = num_processing_complete * 2 # 2 line items per 'processing_complete' claim
    if expected_line_item_updates > 0:
        assert line_item_updates_call is not None and len(line_item_updates_call) == expected_line_item_updates
        assert line_item_updates_call[0]['rvu_total'] == Decimal("2.5")
    else:
        assert line_item_updates_call is None


    # Assert successful commit
    mock_db_session.begin.return_value.__aexit__.assert_called_once_with(None, None, None)

    # Assert metrics
    mock_metrics_collector.record_batch_processed.assert_called_once()
    metrics_call_kwargs = mock_metrics_collector.record_batch_processed.call_args.kwargs
    assert metrics_call_kwargs['batch_size'] == num_total_claims
    assert metrics_call_kwargs['claims_by_final_status'] == summary["by_status"]


@pytest.mark.asyncio
@patch('asyncio.sleep', new_callable=AsyncMock)
async def test_batch_commit_fails_retries_then_succeeds(
    mock_async_sleep: AsyncMock, # Patched asyncio.sleep
    claim_processing_service_instance: ClaimProcessingService,
    mock_db_session: MagicMock,
    mock_metrics_collector: MagicMock
):
    num_claims = 2
    mock_db_claims = [create_mock_db_claim(f"claim_retry_{i}", id_val=i) for i in range(num_claims)]
    claim_processing_service_instance._fetch_pending_claims = AsyncMock(return_value=mock_db_claims)

    processable_results = []
    for db_claim in mock_db_claims:
        pc = create_mock_processable_claim_from_db(db_claim)
        pc.processing_status = "processing_complete"
        pc.ml_derived_decision = "ML_APPROVED"
        pc.processing_duration_ms = 50.0
        processable_results.append(pc)
    claim_processing_service_instance._process_single_claim_concurrently = AsyncMock(return_value=processable_results[0] if num_claims == 1 else AsyncMock(side_effect=processable_results))

    # Mock commit to fail once, then succeed
    mock_db_session.begin.return_value.__aexit__ = AsyncMock(side_effect=[
        OperationalError("DB commit failed", {}, None), None
    ])

    summary = await claim_processing_service_instance.process_pending_claims_batch(num_claims)

    assert mock_db_session.begin.return_value.__aexit__.call_count == 2 # Called twice (1 fail, 1 success)
    mock_async_sleep.assert_called_once() # Sleep between retries
    # bulk_update_mappings would be called twice because the operation is retried
    assert mock_db_session.bulk_update_mappings.call_count == 2 * (1 if num_claims > 0 else 0) # 1 for ClaimModel, potentially 0 if no line items

    assert summary["attempted_claims"] == num_claims
    assert summary["by_status"].get("processing_complete") == num_claims
    assert summary["db_commit_errors"] == 0 # Should ultimately succeed
    mock_metrics_collector.record_batch_processed.assert_called_once()


@pytest.mark.asyncio
@patch('asyncio.sleep', new_callable=AsyncMock)
async def test_batch_commit_all_retries_fail(
    mock_async_sleep: AsyncMock,
    claim_processing_service_instance: ClaimProcessingService,
    mock_db_session: MagicMock,
    mock_metrics_collector: MagicMock
):
    num_claims = 2
    max_retries = 3 # Matches default in service
    mock_db_claims = [create_mock_db_claim(f"claim_all_fail_{i}", id_val=i) for i in range(num_claims)]
    claim_processing_service_instance._fetch_pending_claims = AsyncMock(return_value=mock_db_claims)

    processable_results = []
    for db_claim in mock_db_claims:
        pc = create_mock_processable_claim_from_db(db_claim)
        pc.processing_status = "processing_complete" # Initially processed fine
        pc.ml_derived_decision = "ML_APPROVED"
        pc.processing_duration_ms = 50.0
        processable_results.append(pc)
    claim_processing_service_instance._process_single_claim_concurrently = AsyncMock(return_value=processable_results[0] if num_claims == 1 else AsyncMock(side_effect=processable_results))

    # Mock commit to always fail
    mock_db_session.begin.return_value.__aexit__ = AsyncMock(side_effect=OperationalError("DB commit always fails", {}, None))

    # No pytest.raises here because the method should catch OperationalError and log it, then report in summary
    summary = await claim_processing_service_instance.process_pending_claims_batch(num_claims)

    assert mock_db_session.begin.return_value.__aexit__.call_count == max_retries
    assert mock_async_sleep.call_count == max_retries - 1

    assert summary["attempted_claims"] == num_claims
    assert summary["by_status"].get("db_commit_error") == num_claims # All claims marked as db_commit_error
    assert summary["db_commit_errors"] == num_claims # Specific count for this type of error
    mock_metrics_collector.record_batch_processed.assert_called_once()
    metrics_call_kwargs = mock_metrics_collector.record_batch_processed.call_args.kwargs
    assert metrics_call_kwargs['claims_by_final_status'].get("db_commit_error") == num_claims


@pytest.mark.asyncio
async def test_batch_commit_non_retryable_error(
    claim_processing_service_instance: ClaimProcessingService,
    mock_db_session: MagicMock,
    mock_metrics_collector: MagicMock
):
    num_claims = 1
    mock_db_claims = [create_mock_db_claim("claim_non_retry_fail")]
    claim_processing_service_instance._fetch_pending_claims = AsyncMock(return_value=mock_db_claims)

    pc = create_mock_processable_claim_from_db(mock_db_claims[0])
    pc.processing_status = "processing_complete"
    claim_processing_service_instance._process_single_claim_concurrently = AsyncMock(return_value=pc)

    mock_db_session.begin.return_value.__aexit__ = AsyncMock(side_effect=ValueError("Non-retryable DB error"))

    # The service method should catch this, log, and then the claims are marked as db_commit_error
    summary = await claim_processing_service_instance.process_pending_claims_batch(num_claims)

    mock_db_session.begin.return_value.__aexit__.assert_called_once() # Not retried
    assert summary["attempted_claims"] == num_claims
    assert summary["by_status"].get("db_commit_error") == num_claims
    assert summary["db_commit_errors"] == num_claims
    mock_metrics_collector.record_batch_processed.assert_called_once()

# Test for empty batch processing (already existed, ensure it's still valid)
@pytest.mark.asyncio
async def test_process_pending_claims_batch_no_claims(
    claim_processing_service_instance: ClaimProcessingService,
    mock_metrics_collector: MagicMock
):
    claim_processing_service_instance._fetch_pending_claims = AsyncMock(return_value=[]) # No claims fetched

    summary = await claim_processing_service_instance.process_pending_claims_batch()

    assert summary["attempted_claims"] == 0
    assert summary["successfully_processed_count"] == 0
    assert summary["message"] == "No pending claims to process."
    mock_metrics_collector.record_batch_processed.assert_called_once_with(
        batch_size=0,
        duration_seconds=pytest.approx(0, abs=0.1), # Duration should be very small
        claims_by_final_status={}
    )
```
