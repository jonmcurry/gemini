import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import List, Any, Optional, Tuple # Added Tuple
from decimal import Decimal
from datetime import date, datetime, timezone
import numpy as np

# Models and Services
from claims_processor.src.core.database.models.claims_db import ClaimModel, ClaimLineItemModel
from claims_processor.src.api.models.claim_models import ProcessableClaim, ProcessableClaimLineItem
from claims_processor.src.core.cache.cache_manager import CacheManager
from claims_processor.src.processing.validation.claim_validator import ClaimValidator
from claims_processor.src.processing.rvu_service import RVUService
from claims_processor.src.processing.ml_pipeline.feature_extractor import FeatureExtractor
from claims_processor.src.processing.ml_pipeline.optimized_predictor import OptimizedPredictor
from claims_processor.src.processing.pipeline.parallel_claims_processor import ParallelClaimsProcessor
from claims_processor.src.core.database.models.claims_production_db import ClaimsProductionModel
from claims_processor.src.core.database.models.failed_claims_db import FailedClaimModel # Import new model
from claims_processor.src.core.monitoring.app_metrics import MetricsCollector # Added
from sqlalchemy.sql import insert, update # For checking statements in mocks

# --- Fixtures --- (existing fixtures are fine)
@pytest.fixture
def mock_db_session_factory_and_session():
    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.execute = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()
    async_cm = AsyncMock()
    async_cm.__aenter__.return_value = mock_session
    async_cm.__aexit__.return_value = None
    mock_session_factory = MagicMock()
    mock_session_factory.return_value = async_cm
    return mock_session_factory, mock_session

@pytest.fixture
def mock_cache_manager() -> MagicMock:
    return MagicMock(spec=CacheManager)

@pytest.fixture
def mock_metrics_collector() -> MagicMock: # Added
    return MagicMock(spec=MetricsCollector)

@pytest.fixture
def mock_claim_validator() -> MagicMock:
    validator = MagicMock(spec=ClaimValidator)
    validator.validate_claim = MagicMock(return_value=[])
    return validator

@pytest.fixture
def mock_rvu_service() -> MagicMock:
    service = MagicMock(spec=RVUService)
    service.calculate_rvu_for_claim = AsyncMock(return_value=None)
    return service

@pytest.fixture
def mock_feature_extractor() -> MagicMock:
    extractor = MagicMock(spec=FeatureExtractor)
    extractor.extract_features.return_value = np.random.rand(1, 7).astype(np.float32)
    return extractor

@pytest.fixture
def mock_optimized_predictor() -> MagicMock:
    predictor = MagicMock(spec=OptimizedPredictor)
    predictor.predict_batch = AsyncMock(return_value=[{'ml_score': 0.7, 'ml_derived_decision': 'ML_APPROVED'}])
    return predictor

@pytest.fixture
def processor(
    mock_db_session_factory_and_session,
    mock_claim_validator: MagicMock,
    mock_rvu_service: MagicMock,
    mock_feature_extractor: MagicMock,
    mock_optimized_predictor: MagicMock,
    mock_metrics_collector: MagicMock # Added
) -> ParallelClaimsProcessor:
    mock_session_factory, _ = mock_db_session_factory_and_session
    service = ParallelClaimsProcessor(
        db_session_factory=mock_session_factory,
        claim_validator=mock_claim_validator,
        rvu_service=mock_rvu_service,
        feature_extractor=mock_feature_extractor,
        optimized_predictor=mock_optimized_predictor,
        metrics_collector=mock_metrics_collector # Added
    )
    return service

# Helper functions (create_mock_db_claim, create_processable_claim) remain mostly as before
# Ensure create_processable_claim initializes all fields of ProcessableClaim for tests
def create_mock_db_claim(claim_id_val: str, db_id: int, status: str = 'pending', num_line_items: int = 1) -> MagicMock:
    # ... (implementation from previous step, ensure all fields are set) ...
    claim = MagicMock(spec=ClaimModel); claim.id = db_id; claim.claim_id = claim_id_val
    claim.facility_id = f"fac_{db_id}"; claim.patient_account_number = f"pac_{db_id}"
    claim.service_from_date = date(2023,1,1); claim.service_to_date = date(2023,1,5)
    claim.total_charges = Decimal("100.00"); claim.processing_status = status; claim.batch_id = None
    claim.created_at = datetime.now(timezone.utc); claim.updated_at = datetime.now(timezone.utc)
    claim.processed_at = None; claim.ml_score = None; claim.ml_derived_decision = None
    claim.processing_duration_ms = None; claim.transferred_to_prod_at = None
    claim.line_items = []
    for i in range(num_line_items):
        line = MagicMock(spec=ClaimLineItemModel)
        line.id = (db_id * 100) + i; line.claim_db_id = db_id; line.line_number = i + 1
        line.service_date = date(2023, 1, 1 + i); line.procedure_code = f"proc_{i+1}"; line.units = 1
        line.charge_amount = Decimal("100.00"); line.rvu_total = None
        line.created_at = datetime.now(timezone.utc); line.updated_at = datetime.now(timezone.utc)
        claim.line_items.append(line)
    return claim

def create_processable_claim(
    claim_id_val: str, db_id: int, status: str = 'processing',
    num_line_items: int = 1, batch_id: Optional[str]=None,
    ml_score: Optional[float] = None, ml_decision: Optional[str] = None,
    processing_duration: Optional[float] = None
) -> ProcessableClaim:
    # ... (implementation from previous step, ensure all fields are set) ...
    lines = []
    for i in range(num_line_items):
        lines.append(ProcessableClaimLineItem(
            id=(db_id*100)+i, claim_db_id=db_id, line_number=i+1, service_date=date(2023,1,1+i),
            procedure_code=f"P{i+1}", units=1, charge_amount=Decimal("100.00") / (num_line_items if num_line_items >0 else 1),
            created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc)
        ))
    return ProcessableClaim(
        id=db_id, claim_id=claim_id_val, facility_id=f"fac_{db_id}", patient_account_number=f"pac_{db_id}",
        patient_date_of_birth=date(1990,1,1), service_from_date=date(2023,1,1), service_to_date=date(2023,1,3),
        total_charges=Decimal("100.00"), processing_status=status, created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc), line_items=lines, batch_id=batch_id,
        ml_score=ml_score, ml_derived_decision=ml_decision, processing_duration_ms=processing_duration
    )


# --- Existing Tests for _fetch, _validate, _calculate_rvus, _apply_ml_predictions (condensed for brevity) ---
@pytest.mark.asyncio
async def test_fetch_claims_no_pending(processor: ParallelClaimsProcessor, mock_db_session_factory_and_session, mock_metrics_collector: MagicMock): # Minimal + metrics
    _, mock_session = mock_db_session_factory_and_session
    mock_ids_result = AsyncMock()
    mock_ids_result.fetchall.return_value = [] # No IDs with 'pending' status

    # Simulate the two execute calls in _fetch_claims_parallel
    # 1. Update 'pending' to 'processing' (returns 0 rows)
    # 2. Fetch claims by batch_id (returns 0 rows)
    mock_session.execute.side_effect = [AsyncMock(rowcount=0), AsyncMock(scalars=AsyncMock(return_value=[]))]

    assert await processor._fetch_claims_parallel(mock_session, batch_id="test_batch", limit=10) == []

    # Check metrics calls for DB operations
    assert mock_metrics_collector.record_database_query_duration.call_count == 2
    mock_metrics_collector.record_database_query_duration.assert_any_call("fetch_batch_for_processing_with_lock", pytest.approx(0, abs=1e-1)) # Duration is dynamic
    mock_metrics_collector.record_database_query_duration.assert_any_call("fetch_claim_details_for_batch", pytest.approx(0, abs=1e-1))


@pytest.mark.asyncio
async def test_validate_claims_all_valid(processor, mock_claim_validator): # Minimal
    claims = [create_processable_claim("c1",1)]
    mock_claim_validator.validate_claim.return_value = []
    valid, invalid = await processor._validate_claims_parallel(claims)
    assert len(valid) == 1
    assert len(invalid) == 0

@pytest.mark.asyncio
async def test_calculate_rvus_for_claims_success(processor, mock_db_session_factory_and_session, mock_rvu_service): # Minimal
    _, mock_session = mock_db_session_factory_and_session
    claims = [create_processable_claim("r1",1)]
    await processor._calculate_rvus_for_claims(mock_session, claims)
    mock_rvu_service.calculate_rvu_for_claim.assert_called_once()

@pytest.mark.asyncio
async def test_apply_ml_predictions_success(processor: ParallelClaimsProcessor, mock_feature_extractor: MagicMock, mock_optimized_predictor: MagicMock): # Minimal
    claim1 = create_processable_claim("ml_c1", 1)
    claims_to_process = [claim1]
    features_c1 = np.array([[0.1]*7], dtype=np.float32)
    mock_feature_extractor.extract_features.return_value = features_c1
    mock_optimized_predictor.predict_batch.return_value = [{'ml_score': 0.9, 'ml_derived_decision': 'ML_APPROVED'}]

    await processor._apply_ml_predictions(claims_to_process)

    mock_feature_extractor.extract_features.assert_called_once_with(claim1)
    # Check that predict_batch was called with a list containing the features
    mock_optimized_predictor.predict_batch.assert_called_once()
    call_args_list = mock_optimized_predictor.predict_batch.call_args[0][0]
    assert len(call_args_list) == 1
    assert np.array_equal(call_args_list[0], features_c1)

    assert claim1.ml_score == 0.9
    assert claim1.ml_derived_decision == "ML_APPROVED"


# --- New Tests for _transfer_claims_to_production ---
@pytest.mark.asyncio
async def test_transfer_claims_to_production_success(processor: ParallelClaimsProcessor, mock_db_session_factory_and_session, mock_metrics_collector: MagicMock):
    _, mock_session = mock_db_session_factory_and_session
    claims_to_transfer = [
        create_processable_claim("c1_prod", 101, ml_score=0.85, processing_duration_ms=120.0, ml_derived_decision="ML_APPROVED"),
        create_processable_claim("c2_prod", 102, ml_score=0.92, processing_duration_ms=150.0, ml_derived_decision="ML_APPROVED")
    ]
    mock_session.execute.return_value = AsyncMock() # For insert

    count = await processor._transfer_claims_to_production(mock_session, claims_to_transfer, {'throughput_achieved': Decimal("10.0")})

    assert count == 2
    mock_session.execute.assert_called_once()
    args, _ = mock_session.execute.call_args
    assert isinstance(args[0], type(insert(ClaimsProductionModel)))
    assert len(args[1]) == 2
    assert args[1][0]['claim_id'] == "c1_prod"
    assert args[1][0]['throughput_achieved'] == Decimal("10.0")
    mock_metrics_collector.record_database_query_duration.assert_called_once_with("bulk_insert_production_claims", pytest.approx(0, abs=1e-1))


# --- New Tests for _set_final_status_for_staging_claims ---
@pytest.mark.asyncio
async def test_set_final_status_for_staging_claims_success(processor: ParallelClaimsProcessor, mock_db_session_factory_and_session, mock_metrics_collector: MagicMock):
    _, mock_session = mock_db_session_factory_and_session
    claim1 = create_processable_claim("sfs1", 401)
    claim2 = create_processable_claim("sfs2", 402)
    claims_and_statuses = [(claim1, "VALIDATION_FAILED"), (claim2, "ML_REJECTED")]

    mock_execute_result_vf = MagicMock()
    mock_execute_result_vf.rowcount = 1
    mock_execute_result_ml = MagicMock()
    mock_execute_result_ml.rowcount = 1
    mock_session.execute.side_effect = [mock_execute_result_vf, mock_execute_result_ml]

    count = await processor._set_final_status_for_staging_claims(mock_session, claims_and_statuses)

    assert count == 2
    assert mock_session.execute.call_count == 2
    mock_metrics_collector.record_database_query_duration.assert_called_once_with("update_staging_failed_or_rejected", pytest.approx(0, abs=1e-1))


# --- Updated Orchestration Test for process_claims_parallel ---
@pytest.mark.asyncio
async def test_process_claims_parallel_e2e_with_failures_and_routing(
    processor: ParallelClaimsProcessor,
    mock_db_session_factory_and_session,
    mock_claim_validator: MagicMock,
    mock_rvu_service: MagicMock,
    mock_feature_extractor: MagicMock,
    mock_optimized_predictor: MagicMock,
    mock_metrics_collector: MagicMock # Added
):
    mock_session_factory, mock_session = mock_db_session_factory_and_session

    # Setup claims
    c_valid_pass = create_processable_claim("c_pass", 1, batch_id="b_e2e")
    c_fail_validation = create_processable_claim("c_val_fail", 2, batch_id="b_e2e")
    c_ml_reject = create_processable_claim("c_ml_reject", 3, batch_id="b_e2e")

    all_fetched_claims = [c_valid_pass, c_fail_validation, c_ml_reject]

    # Mock _fetch_claims_parallel
    with patch.object(processor, '_fetch_claims_parallel', new_callable=AsyncMock, return_value=all_fetched_claims) as mock_fetch:
        # Mock _validate_claims_parallel behavior
        mock_claim_validator.validate_claim.side_effect = lambda claim: ["Error"] if claim.claim_id == "c_val_fail" else []

        # Mock ML behavior (_apply_ml_predictions uses these)
        # Ensure correct features are passed to predict_batch based on the claims that reach ML stage
        features_pass = np.array([[0.1]*7], dtype=np.float32)
        features_reject = np.array([[0.2]*7], dtype=np.float32)

        def feature_extractor_side_effect(claim):
            if claim.claim_id == "c_pass": return features_pass
            if claim.claim_id == "c_ml_reject": return features_reject
            return None # Should not happen for these claims
        mock_feature_extractor.extract_features.side_effect = feature_extractor_side_effect

        # Predict_batch will receive a list of features for claims that passed validation
        mock_optimized_predictor.predict_batch.return_value = [
            {'ml_score': 0.9, 'ml_derived_decision': 'ML_APPROVED'}, # For c_pass's features
            {'ml_score': 0.3, 'ml_derived_decision': 'ML_REJECTED'}  # For c_ml_reject's features
        ]

        # Mock RVU service (only c_pass should reach here)
        mock_rvu_service.calculate_rvu_for_claim.return_value = None

        # Mock Transfer service (only c_pass should reach here)
        # Add batch_metrics to the expected call for _transfer_claims_to_production
        with patch.object(processor, '_transfer_claims_to_production', new_callable=AsyncMock, return_value=1) as mock_transfer, \
             patch.object(processor, '_update_staging_claims_status', new_callable=AsyncMock, return_value=1) as mock_update_transferred, \
             patch.object(processor, '_route_failed_claims', new_callable=AsyncMock, return_value=2) as mock_route_failed, \
             patch.object(processor, '_set_final_status_for_staging_claims', new_callable=AsyncMock, return_value=2) as mock_set_final_status:

            summary = await processor.process_claims_parallel(batch_id="b_e2e", limit=3)

    # Assertions
    mock_fetch.assert_called_once_with(mock_session, "b_e2e", 3)
    assert mock_claim_validator.validate_claim.call_count == 3 # Called for all fetched

    # ML assertions
    assert mock_feature_extractor.extract_features.call_count == 2 # For c_pass, c_ml_reject
    mock_optimized_predictor.predict_batch.assert_called_once_with([features_pass, features_reject])


    # RVU assertions
    mock_rvu_service.calculate_rvu_for_claim.assert_called_once_with(c_valid_pass, mock_session) # Only for c_pass

    # Transfer assertions - check batch_metrics argument
    mock_transfer.assert_called_once()
    transfer_args = mock_transfer.call_args[0]
    assert transfer_args[0] == mock_session
    assert transfer_args[1] == [c_valid_pass] # Only c_pass is transferred
    assert 'throughput_achieved' in transfer_args[2] # Check batch_metrics dict
    assert isinstance(transfer_args[2]['throughput_achieved'], Decimal)


    # Staging update assertions
    mock_update_transferred.assert_called_once_with(mock_session, [c_valid_pass.id])

    # Failed claims routing and status update assertions
    mock_route_failed.assert_called_once()
    failed_routed_args = mock_route_failed.call_args[0][1]
    assert len(failed_routed_args) == 2
    assert any(item[0].claim_id == "c_val_fail" for item in failed_routed_args)
    assert any(item[0].claim_id == "c_ml_reject" for item in failed_routed_args)

    mock_set_final_status.assert_called_once()
    set_final_status_args = mock_set_final_status.call_args[0][1]
    assert len(set_final_status_args) == 2
    assert any(item[0].claim_id == "c_val_fail" and item[1] == "VALIDATION_FAILED" for item in set_final_status_args)
    assert any(item[0].claim_id == "c_ml_reject" and item[1] == "ML_REJECTED_OR_ERROR" for item in set_final_status_args)


    assert summary["fetched_count"] == 3
    assert summary["validation_passed_count"] == 2
    assert summary["validation_failed_count"] == 1
    assert summary["ml_prediction_attempted_count"] == 2
    assert summary["ml_rejected_count"] == 1
    assert summary["rvu_calculation_completed_count"] == 1
    assert summary["transferred_to_prod_count"] == 1
    assert summary["staging_updated_transferred_count"] == 1
    assert summary["staging_updated_failed_count"] == 2
    assert summary["failed_claims_routed_count"] == 2
    assert mock_session.commit.call_count >= 1 # Commit can be called multiple times in loops

    # Metrics Collector Assertions
    mock_metrics_collector.record_batch_processed.assert_called_once()
    batch_processed_args = mock_metrics_collector.record_batch_processed.call_args[1]
    assert batch_processed_args['batch_size'] == 3 # limit
    assert isinstance(batch_processed_args['duration_seconds'], float)
    assert batch_processed_args['claims_by_final_status'] == {
        'validation_failed': 1,
        'ml_rejected_or_error': 1, # Based on how status is set for c_ml_reject
        'completed_transferred': 1 # For c_pass
    }

    # Check DB query timer calls (count may vary based on actual execution paths of mocks)
    # We expect at least one call for each distinct timer used in the flow.
    # _fetch_claims_parallel (mocked out, so its internal timers won't be called on the processor's collector)
    # _transfer_claims_to_production (called)
    # _update_staging_claims_status (called)
    # _route_failed_claims (called)
    # _set_final_status_for_staging_claims (called)
    db_timer_calls = mock_metrics_collector.record_database_query_duration.call_args_list

    # Check that the names of the timers are among the calls
    expected_timer_names = [
        "bulk_insert_production_claims", # from _transfer_claims_to_production
        "update_staging_transferred",    # from _update_staging_claims_status
        "bulk_insert_failed_claims",     # from _route_failed_claims
        "update_staging_failed_or_rejected" # from _set_final_status_for_staging_claims
    ]
    actual_timer_names_called = [call[0][0] for call in db_timer_calls]
    for name in expected_timer_names:
        assert name in actual_timer_names_called

    # Ensure processing_duration_ms is set for the claim that was transferred
    assert c_valid_pass.processing_duration_ms is not None
    assert isinstance(c_valid_pass.processing_duration_ms, float)
    assert c_valid_pass.processing_duration_ms > 0

    # Ensure batch_metrics (throughput) was passed to _transfer_claims_to_production
    # This was already checked in mock_transfer.assert_called_once_with
    # transfer_args[2] was {'throughput_achieved': Decimal(...)}
    assert isinstance(transfer_args[2]['throughput_achieved'], Decimal)
    # The actual throughput value depends on the duration, which is dynamic.
    # We can check it's non-negative.
    assert transfer_args[2]['throughput_achieved'] >= Decimal("0")
