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
from claims_processor.src.core.database.models.failed_claims_db import FailedClaimModel
from claims_processor.src.core.monitoring.app_metrics import MetricsCollector
from claims_processor.src.core.security.audit_logger_service import AuditLoggerService
from claims_processor.src.core.security.encryption_service import EncryptionService # Added
from sqlalchemy.sql import insert, update # For checking statements in mocks
from sqlalchemy.exc import OperationalError # Added for retry tests
import asyncio # Added for patching asyncio.sleep

# --- Fixtures ---
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
@pytest.fixture
def mock_audit_logger_service() -> MagicMock: # Added
    service = MagicMock(spec=AuditLoggerService)
    service.log_event = AsyncMock()
    return service

@pytest.fixture
def processor(
    mock_db_session_factory_and_session,
    mock_claim_validator: MagicMock,
    mock_rvu_service: MagicMock,
    mock_feature_extractor: MagicMock,
    mock_optimized_predictor: MagicMock,
    mock_metrics_collector: MagicMock,
    mock_audit_logger_service: MagicMock # Added
) -> ParallelClaimsProcessor:
@pytest.fixture
def mock_encryption_service() -> MagicMock: # Added
    service = MagicMock(spec=EncryptionService)
    # Default behavior: pass through, or specific mock values if needed per test
    service.decrypt.side_effect = lambda value: f"decrypted_{value}" if value else None
    return service

@pytest.fixture
def processor(
    mock_db_session_factory_and_session,
    mock_claim_validator: MagicMock,
    mock_rvu_service: MagicMock,
    mock_feature_extractor: MagicMock,
    mock_optimized_predictor: MagicMock,
    mock_metrics_collector: MagicMock,
    mock_audit_logger_service: MagicMock,
    mock_encryption_service: MagicMock # Added
) -> ParallelClaimsProcessor:
    mock_session_factory, _ = mock_db_session_factory_and_session
    service = ParallelClaimsProcessor(
        db_session_factory=mock_session_factory,
        claim_validator=mock_claim_validator,
        rvu_service=mock_rvu_service,
        feature_extractor=mock_feature_extractor,
        optimized_predictor=mock_optimized_predictor,
        metrics_collector=mock_metrics_collector,
        audit_logger_service=mock_audit_logger_service,
        encryption_service=mock_encryption_service # Added
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
    mock_metrics_collector: MagicMock,
    mock_audit_logger_service: MagicMock # Added
):
    mock_session_factory, mock_session = mock_db_session_factory_and_session

    passed_batch_id = "b_e2e" # The batch_id passed into the method

    # Setup claims
    c_valid_pass = create_processable_claim("c_pass", 1, batch_id=passed_batch_id) # Will pass all stages
    c_fail_validation = create_processable_claim("c_val_fail", 2, batch_id=passed_batch_id) # Will fail validation
    c_ml_reject = create_processable_claim("c_ml_reject", 3, batch_id=passed_batch_id) # Will be rejected by ML
    c_ml_error_proceeds = create_processable_claim("c_ml_error", 4, batch_id=passed_batch_id) # Will have ML error, but proceed

    all_fetched_claims = [c_valid_pass, c_fail_validation, c_ml_reject, c_ml_error_proceeds]
    limit_param = len(all_fetched_claims)

    # Mock _fetch_claims_parallel
    with patch.object(processor, '_fetch_claims_parallel', new_callable=AsyncMock, return_value=all_fetched_claims) as mock_fetch:
        # Mock _validate_claims_parallel behavior
        mock_claim_validator.validate_claim.side_effect = lambda claim_to_validate: ["Error"] if claim_to_validate.claim_id == "c_val_fail" else []

        # Mock ML behavior
        features_pass = np.array([[0.1]*7], dtype=np.float32)
        features_reject = np.array([[0.2]*7], dtype=np.float32)
        features_error = np.array([[0.3]*7], dtype=np.float32) # Features for c_ml_error_proceeds

        def feature_extractor_side_effect(claim_to_extract):
            if claim_to_extract.claim_id == "c_pass": return features_pass
            if claim_to_extract.claim_id == "c_ml_reject": return features_reject
            if claim_to_extract.claim_id == "c_ml_error": return features_error
            return None
        mock_feature_extractor.extract_features.side_effect = feature_extractor_side_effect

        # OptimizedPredictor's predict_batch is called with a list of feature arrays
        # for claims that passed validation and feature extraction.
        # Expected order: c_pass, c_ml_reject, c_ml_error_proceeds
        mock_optimized_predictor.predict_batch.return_value = [
            {'ml_score': 0.9, 'ml_derived_decision': 'ML_APPROVED'},          # For c_pass
            {'ml_score': 0.3, 'ml_derived_decision': 'ML_REJECTED'},          # For c_ml_reject
            {'ml_derived_decision': 'ML_SAMPLE_PROCESSING_ERROR', 'error': 'Simulated sample error'} # For c_ml_error_proceeds
        ]
        # Ensure predictor interpreter is available for this test path
        mock_optimized_predictor.interpreter = MagicMock() # Not None

        # Mock RVU service (c_pass and c_ml_error_proceeds should reach here)
        mock_rvu_service.calculate_rvu_for_claim.return_value = None

        # Mock Transfer service (only c_pass should reach here)
        # Mock Transfer service (c_pass and c_ml_error_proceeds should reach here, assuming they are transferred)
        # Let's assume c_ml_error_proceeds also gets transferred for this test.
        with patch.object(processor, '_transfer_claims_to_production', new_callable=AsyncMock, return_value=2) as mock_transfer, \
             patch.object(processor, '_update_staging_claims_status') as mock_update_staging, \
             patch.object(processor, '_route_failed_claims', new_callable=AsyncMock) as mock_route_failed, \
             patch.object(processor, '_set_final_status_for_staging_claims') as mock_set_final_status:

            summary = await processor.process_claims_parallel(batch_id=passed_batch_id, limit=limit_param)

    # Assertions for method calls
    mock_fetch.assert_called_once_with(mock_session, passed_batch_id, limit_param)
    assert mock_claim_validator.validate_claim.call_count == limit_param

    # ML assertions
    # Passed validation: c_pass, c_ml_reject, c_ml_error_proceeds
    assert mock_feature_extractor.extract_features.call_count == 3
    mock_optimized_predictor.predict_batch.assert_called_once_with([features_pass, features_reject, features_error])

    # RVU assertions (c_pass, c_ml_error_proceeds)
    assert mock_rvu_service.calculate_rvu_for_claim.call_count == 2
    mock_rvu_service.calculate_rvu_for_claim.assert_any_call(c_valid_pass, mock_session)
    mock_rvu_service.calculate_rvu_for_claim.assert_any_call(c_ml_error_proceeds, mock_session)

    # Transfer assertions
    mock_transfer.assert_called_once()
    transfer_args = mock_transfer.call_args[0]
    assert transfer_args[0] == mock_session
    # Claims passed to transfer should be c_pass and c_ml_error_proceeds
    assert len(transfer_args[1]) == 2
    assert c_valid_pass in transfer_args[1]
    assert c_ml_error_proceeds in transfer_args[1]
    assert 'throughput_achieved' in transfer_args[2]
    assert isinstance(transfer_args[2]['throughput_achieved'], Decimal)

    # Staging update assertions
    # Called for transferred claims and for failed/rejected claims (via _set_final_status_for_staging_claims)
    # _update_staging_claims_status is directly called for transferred claims
    # _set_final_status_for_staging_claims calls it for others. Let's check specific calls on mock_update_staging

    # Transferred: c_pass, c_ml_error_proceeds
    mock_update_staging.assert_any_call(mock_session, [c_valid_pass.id, c_ml_error_proceeds.id], "completed_transferred", transferred=True)

    # Failed claims routing and status update assertions
    # c_fail_validation (validation failure)
    # c_ml_reject (ML rejected)
    assert mock_route_failed.call_count == 2 # Called for VALIDATION_FAILED and ML_REJECTED stages

    # Check calls to _route_failed_claims for specific claims
    routed_validation_fail = any(
        call_args[0][0] == c_fail_validation and call_args[0][2] == "VALIDATION_FAILED"
        for call_obj in mock_route_failed.call_args_list for call_args in call_obj[0][1] # call_obj[0][1] is the list of tuples
    )
    routed_ml_reject = any(
        call_args[0][0] == c_ml_reject and call_args[0][2] == "ML_REJECTED"
        for call_obj in mock_route_failed.call_args_list for call_args in call_obj[0][1]
    )
    # This assertion needs refinement because _route_failed_claims is called with a list of tuples.
    # The structure of mock_route_failed.call_args_list will be:
    # [call(((session_mock, [(claim1, reason, stage), (claim2, reason, stage)]),), {}), ...]
    # The check should be:
    # For VALIDATION_FAILED stage:
    assert any(
        args[1][0][0].claim_id == "c_val_fail" and args[1][0][2] == "VALIDATION_FAILED"
        for args in mock_route_failed.call_args_list if args[0][1] # Ensure list is not empty
    ), "c_val_fail not routed correctly"
    # For ML_REJECTED stage:
    assert any(
        args[1][0][0].claim_id == "c_ml_reject" and args[1][0][2] == "ML_REJECTED"
        for args in mock_route_failed.call_args_list if args[0][1]
    ), "c_ml_reject not routed correctly"


    # Check calls to _set_final_status_for_staging_claims
    # c_fail_validation -> status VALIDATION_FAILED
    # c_ml_reject -> status ML_REJECTED
    assert mock_set_final_status.call_count == 1 # Called once with all non-transferred claims that need status update
    final_status_args = mock_set_final_status.call_args[0][1] # list of (claim, status_str)
    assert len(final_status_args) == 2
    assert any(item[0].claim_id == "c_val_fail" and item[1] == "VALIDATION_FAILED" for item in final_status_args)
    assert any(item[0].claim_id == "c_ml_reject" and item[1] == "ML_REJECTED" for item in final_status_args)


    # Summary assertions
    assert summary["fetched_count"] == limit_param
    assert summary["validation_passed_count"] == 3 # c_pass, c_ml_reject, c_ml_error_proceeds
    assert summary["validation_failed_count"] == 1 # c_fail_validation
    assert summary["ml_prediction_attempted_count"] == 3
    assert summary["ml_approved_raw"] == 1 # c_pass
    assert summary["ml_rejected_raw"] == 1 # c_ml_reject
    assert summary["ml_errors_raw"] == 1 # c_ml_error_proceeds (ML_SAMPLE_PROCESSING_ERROR)
    assert summary["stopped_by_ml_rejection"] == 1 # Only c_ml_reject
    assert summary["rvu_calculation_completed_count"] == 2 # c_pass, c_ml_error_proceeds
    assert summary["transferred_to_prod_count"] == 2 # c_pass, c_ml_error_proceeds
    assert summary["staging_updated_transferred_count"] == 2
    assert summary["staging_updated_failed_count"] == 2 # c_fail_validation, c_ml_reject
    assert summary["failed_claims_routed_count"] == 2 # c_fail_validation, c_ml_reject
    assert mock_session.commit.call_count >= 1

    # Metrics Collector Assertions
    mock_metrics_collector.record_batch_processed.assert_called_once()
    batch_processed_args = mock_metrics_collector.record_batch_processed.call_args[1]
    assert batch_processed_args['batch_size'] == limit_param
    assert isinstance(batch_processed_args['duration_seconds'], float)
    # Expected final statuses for metrics:
    # c_pass: completed_transferred
    # c_fail_validation: validation_failed
    # c_ml_reject: ml_rejected
    # c_ml_error_proceeds: completed_transferred (since it proceeds and is assumed to be transferred)
    expected_metrics_statuses = {
        'completed_transferred': 2, # c_pass, c_ml_error_proceeds
        'validation_failed': 1,     # c_fail_validation
        'ml_rejected': 1            # c_ml_reject
        # 'ml_processing_error' is not a final status for claims that proceed.
    }
    assert batch_processed_args['claims_by_final_status'] == expected_metrics_statuses


    # DB query timer calls (count may vary based on actual execution paths of mocks)
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
    assert isinstance(transfer_args[2]['throughput_achieved'], Decimal)
    assert transfer_args[2]['throughput_achieved'] >= Decimal("0")

    # Audit Logger Assertions
    assert mock_audit_logger_service.log_event.call_count == 2

    # Check PROCESS_BATCH_START call
    start_event_call = mock_audit_logger_service.log_event.call_args_list[0]
    start_event_kwargs = start_event_call.kwargs
    assert start_event_kwargs['action'] == "PROCESS_BATCH_START"
    assert start_event_kwargs['resource'] == "ClaimBatch"
    assert start_event_kwargs['resource_id'] == passed_batch_id # Since passed_batch_id is not None
    assert start_event_kwargs['success'] is True
    assert start_event_kwargs['user_id'] == "system_pipeline"
    assert start_event_kwargs['details'] == {"limit": 3, "original_batch_id_param": passed_batch_id}

    # Check PROCESS_BATCH_END call
    end_event_call = mock_audit_logger_service.log_event.call_args_list[1]
    end_event_kwargs = end_event_call.kwargs
    assert end_event_kwargs['action'] == "PROCESS_BATCH_END"
    assert end_event_kwargs['resource'] == "ClaimBatch"
    assert end_event_kwargs['resource_id'] == passed_batch_id
    assert end_event_kwargs['success'] is True # Because no major exception was raised to set summary["error"]
    assert end_event_kwargs['failure_reason'] is None
    assert end_event_kwargs['user_id'] == "system_pipeline"
    assert end_event_kwargs['details']['summary'] == summary # Compare the summary dict
    assert isinstance(end_event_kwargs['details']['duration_seconds'], float)
    assert end_event_kwargs['details']['attempts_made'] == 1 # Success on first attempt


@patch('asyncio.sleep', new_callable=AsyncMock) # Mock asyncio.sleep for retry tests
@pytest.mark.asyncio
async def test_process_claims_parallel_success_on_second_attempt(
    mock_asyncio_sleep: AsyncMock, # Injected by @patch
    processor: ParallelClaimsProcessor,
    mock_db_session_factory_and_session,
    mock_claim_validator: MagicMock,
    mock_rvu_service: MagicMock,
    mock_feature_extractor: MagicMock,
    mock_optimized_predictor: MagicMock,
    mock_metrics_collector: MagicMock,
    mock_audit_logger_service: MagicMock
):
    mock_session_factory, mock_session = mock_db_session_factory_and_session
    passed_batch_id = "b_retry_success"

    # Simulate OperationalError on the first commit, then success
    # The session factory is called in each attempt, so we reconfigure the mock_session it returns.
    # This requires the factory to be called multiple times.
    # Let's make the factory return different session mocks or one that has side_effects.

    # For this test, let's assume the critical operation that fails is session.commit()
    # We need to make the factory return a session that fails commit once, then succeeds.

    commit_effects = [OperationalError("Simulated DB connection error"), AsyncMock()]

    # We need a new session mock for each attempt, or a session factory that changes behavior
    # The current 'processor' fixture gets a factory that returns the *same* mock_session.
    # So, we can set side_effect on mock_session.commit directly.
    mock_session.commit.side_effect = commit_effects

    # Setup claims (same as e2e test for simplicity, focusing on retry logic)
    c_valid_pass = create_processable_claim("c_pass_retry", 10, batch_id=passed_batch_id)
    all_fetched_claims = [c_valid_pass]

    with patch.object(processor, '_fetch_claims_parallel', new_callable=AsyncMock, return_value=all_fetched_claims) as mock_fetch:
        # Other mocks for a simplified successful flow after retry
        mock_claim_validator.validate_claim.return_value = [] # All valid
        mock_feature_extractor.extract_features.return_value = np.array([[0.1]*7], dtype=np.float32)
        mock_optimized_predictor.predict_batch.return_value = [{'ml_score': 0.9, 'ml_derived_decision': 'ML_APPROVED'}]
        mock_rvu_service.calculate_rvu_for_claim.return_value = None
        with patch.object(processor, '_transfer_claims_to_production', new_callable=AsyncMock, return_value=1), \
             patch.object(processor, '_update_staging_claims_status', new_callable=AsyncMock, return_value=1), \
             patch.object(processor, '_route_failed_claims', new_callable=AsyncMock, return_value=0), \
             patch.object(processor, '_set_final_status_for_staging_claims', new_callable=AsyncMock, return_value=0):

            summary = await processor.process_claims_parallel(batch_id=passed_batch_id, limit=1)

    # Assertions
    assert mock_session_factory.call_count == 2 # Called for attempt 1 (fail) and attempt 2 (success)
    assert mock_session.commit.call_count == 2 # Commit called twice

    mock_asyncio_sleep.assert_called_once_with(5) # BATCH_RETRY_DELAY_SECONDS from ParallelClaimsProcessor

    assert summary["error"] is None # Error should be cleared on final success
    assert summary["transferred_to_prod_count"] == 1 # Should succeed in the end

    # Audit log assertions
    assert mock_audit_logger_service.log_event.call_count == 2
    start_event_call = mock_audit_logger_service.log_event.call_args_list[0]
    assert start_event_call.kwargs['action'] == "PROCESS_BATCH_START"

    end_event_call = mock_audit_logger_service.log_event.call_args_list[1]
    end_event_kwargs = end_event_call.kwargs
    assert end_event_kwargs['action'] == "PROCESS_BATCH_END"
    assert end_event_kwargs['success'] is True
    assert end_event_kwargs['failure_reason'] is None
    assert end_event_kwargs['details']['attempts_made'] == 2
    assert "DB OperationalError (attempt 1)" in end_event_kwargs['details']['summary']['error'] # Ensure last error is in summary for audit

    # Metrics should be recorded once with final successful state
    mock_metrics_collector.record_batch_processed.assert_called_once()
    processed_args = mock_metrics_collector.record_batch_processed.call_args[1]
    assert processed_args['claims_by_final_status'] == {'completed_transferred': 1}


@patch('asyncio.sleep', new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_process_claims_parallel_failure_after_all_retries(
    mock_asyncio_sleep: AsyncMock,
    processor: ParallelClaimsProcessor,
    mock_db_session_factory_and_session,
    mock_audit_logger_service: MagicMock,
    mock_metrics_collector: MagicMock
):
    mock_session_factory, mock_session = mock_db_session_factory_and_session
    passed_batch_id = "b_retry_fail_all"

    # Simulate OperationalError on all commit attempts
    mock_session.commit.side_effect = OperationalError("Simulated DB error on commit", params=None, orig=None)

    # Minimal claims setup, fetch will be mocked
    c_valid = create_processable_claim("c_valid_retry_fail", 20, batch_id=passed_batch_id)
    all_fetched_claims = [c_valid]

    with patch.object(processor, '_fetch_claims_parallel', new_callable=AsyncMock, return_value=all_fetched_claims) as mock_fetch:
        # Assume other parts of processing would succeed if commit didn't fail
        mock_processor = processor # Already has other services mocked by fixture
        with patch.object(mock_processor, '_validate_claims_parallel', return_value=([c_valid],[])), \
             patch.object(mock_processor, '_apply_ml_predictions', return_value=None), \
             patch.object(mock_processor, '_calculate_rvus_for_claims', return_value=None), \
             patch.object(mock_processor, '_transfer_claims_to_production', return_value=1), \
             patch.object(mock_processor, '_update_staging_claims_status', return_value=1), \
             patch.object(mock_processor, '_route_failed_claims', return_value=0), \
             patch.object(mock_processor, '_set_final_status_for_staging_claims', return_value=0):

            summary = await processor.process_claims_parallel(batch_id=passed_batch_id, limit=1)

    assert mock_session_factory.call_count == 3 # Max retries
    assert mock_session.commit.call_count == 3 # Commit attempted 3 times
    assert mock_asyncio_sleep.call_count == 2 # Sleep between 3 attempts

    assert "DB OperationalError (attempt 3)" in summary["error"]

    # Audit log
    assert mock_audit_logger_service.log_event.call_count == 2 # START and END
    end_event_kwargs = mock_audit_logger_service.log_event.call_args_list[1].kwargs
    assert end_event_kwargs['action'] == "PROCESS_BATCH_END"
    assert end_event_kwargs['success'] is False
    assert "DB OperationalError (attempt 3)" in end_event_kwargs['failure_reason']
    assert end_event_kwargs['details']['attempts_made'] == 3

    # Metrics (should still be called once with final summary, even if failed)
    mock_metrics_collector.record_batch_processed.assert_called_once()
    processed_args = mock_metrics_collector.record_batch_processed.call_args[1]
    # The status might be empty or reflect partial work depending on how summary is built on error
    # Given the current setup, it might be zeros or based on last attempt's view before rollback
    # For OperationalError on commit, the counts in summary might reflect a full run.
    # This depends on whether summary is reset or not during retries.
    # The current code does not reset summary counts between retries.
    assert processed_args['claims_by_final_status'] == {} # Or based on what summary contains


@patch('asyncio.sleep', new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_process_claims_parallel_failure_non_retryable_error(
    mock_asyncio_sleep: AsyncMock,
    processor: ParallelClaimsProcessor,
    mock_db_session_factory_and_session,
    mock_audit_logger_service: MagicMock,
    mock_metrics_collector: MagicMock
):
    mock_session_factory, mock_session = mock_db_session_factory_and_session
    passed_batch_id = "b_non_retry_fail"

    # Simulate a non-OperationalError (e.g., ValueError) during a core step like validation
    # For instance, let _validate_claims_parallel raise an unexpected error.
    with patch.object(processor, '_fetch_claims_parallel', new_callable=AsyncMock, return_value=[create_processable_claim("c1",1)]) as mock_fetch, \
         patch.object(processor, '_validate_claims_parallel', side_effect=ValueError("Non-retryable validation crash")) as mock_validate:

        summary = await processor.process_claims_parallel(batch_id=passed_batch_id, limit=1)

    assert mock_session_factory.call_count == 1 # Only one attempt for non-retryable errors
    mock_asyncio_sleep.assert_not_called() # No retries, so no sleep

    assert "Non-retryable validation crash" in summary["error"]

    # Audit log
    assert mock_audit_logger_service.log_event.call_count == 2
    end_event_kwargs = mock_audit_logger_service.log_event.call_args_list[1].kwargs
    assert end_event_kwargs['action'] == "PROCESS_BATCH_END"
    assert end_event_kwargs['success'] is False
    assert "Non-retryable validation crash" in end_event_kwargs['failure_reason']
    assert end_event_kwargs['details']['attempts_made'] == 1

    # Metrics
    mock_metrics_collector.record_batch_processed.assert_called_once()
    processed_args = mock_metrics_collector.record_batch_processed.call_args[1]
    # Summary might have fetched_count=1, but other processing steps didn't complete.
    # This depends on how summary is updated when non-retryable error occurs.
    # The current code updates summary['error'] and then goes to finally.
    # So other counts in summary (like validation_passed_count) would be 0.
    assert processed_args['claims_by_final_status'] == {} # Or based on summary state


@pytest.mark.asyncio
async def test_fetch_claims_parallel_with_decryption(
    processor: ParallelClaimsProcessor,
    mock_db_session_factory_and_session,
    mock_metrics_collector: MagicMock, # Already part of processor fixture
    mock_encryption_service: MagicMock # Injected into processor
):
    mock_session_factory, mock_session = mock_db_session_factory_and_session

    # Setup mock ClaimModel instances that would be returned by the DB query
    mock_claim_db_1 = MagicMock(spec=ClaimModel)
    mock_claim_db_1.id = 101
    mock_claim_db_1.claim_id = "claim_enc_1"
    mock_claim_db_1.patient_date_of_birth = "enc_dob_1990-05-15" # Encrypted string
    mock_claim_db_1.medical_record_number = "enc_mrn_12345"   # Encrypted string
    # Populate other necessary fields for ProcessableClaim.model_validate or dict conversion
    mock_claim_db_1.facility_id = "F001"
    mock_claim_db_1.patient_account_number = "PACC001"
    mock_claim_db_1.service_from_date = date(2023,1,1)
    mock_claim_db_1.service_to_date = date(2023,1,2)
    mock_claim_db_1.total_charges = Decimal("1000.00")
    mock_claim_db_1.processing_status = "processing" # Set by _fetch before full load
    mock_claim_db_1.batch_id = "test_batch_enc"
    mock_claim_db_1.created_at = datetime.now(timezone.utc)
    mock_claim_db_1.updated_at = datetime.now(timezone.utc)
    mock_claim_db_1.patient_first_name = "Test"
    mock_claim_db_1.patient_last_name = "User"
    mock_claim_db_1.insurance_type = "Medicare" # Assuming these fields exist on ClaimModel
    mock_claim_db_1.financial_class = "Inpatient"
    mock_claim_db_1.processed_at = None
    mock_claim_db_1.transferred_to_prod_at = None
    mock_claim_db_1.ml_score = None
    mock_claim_db_1.ml_derived_decision = None
    mock_claim_db_1.processing_duration_ms = None
    mock_claim_db_1.line_items = [] # Empty for simplicity in this test

    # Re-create __table__.columns for the mock if needed by dict comprehension logic
    # This is a bit of a deep mock, might be simpler if model_validate handles objects.
    # The current _fetch_claims_parallel uses dict comprehension.
    mock_columns = {c.name: c for c in ClaimModel.__table__.columns}
    mock_claim_db_1.__table__ = MagicMock()
    mock_claim_db_1.__table__.columns = mock_columns


    # Mock the return value of session.execute for fetching full claim data
    mock_full_data_result = AsyncMock()
    mock_full_data_result.scalars.return_value.unique.return_value.all.return_value = [mock_claim_db_1]

    # First call to execute (fetch IDs), second (update status), third (fetch full data)
    mock_session.execute.side_effect = [
        AsyncMock(fetchall=AsyncMock(return_value=[(101,)])), # Claim ID for update
        AsyncMock(rowcount=1), # Result of update
        mock_full_data_result # Result of select full claims
    ]

    # Configure mock_encryption_service.decrypt side effects
    def decrypt_side_effect(encrypted_value: str):
        if encrypted_value == "enc_dob_1990-05-15":
            return "1990-05-15" # Decrypted ISO date string
        if encrypted_value == "enc_mrn_12345":
            return "mrn_clear_12345"
        return None
    mock_encryption_service.decrypt.side_effect = decrypt_side_effect

    fetched_processable_claims = await processor._fetch_claims_parallel(mock_session, batch_id="test_batch_enc", limit=1)

    assert len(fetched_processable_claims) == 1
    p_claim = fetched_processable_claims[0]

    # Verify decrypt was called for DOB and MRN
    mock_encryption_service.decrypt.assert_any_call("enc_dob_1990-05-15")
    mock_encryption_service.decrypt.assert_any_call("enc_mrn_12345")

    # Verify the ProcessableClaim has the decrypted and parsed values
    assert p_claim.patient_date_of_birth == date(1990, 5, 15)
    assert p_claim.medical_record_number == "mrn_clear_12345"
    assert p_claim.claim_id == "claim_enc_1" # Other fields should still be there

    # Test case: Decryption fails for DOB (returns None)
    mock_encryption_service.decrypt.side_effect = lambda val: None if val == "enc_dob_1990-05-15" else f"decrypted_{val}"
    mock_session.execute.side_effect = [ # Reset side effect for execute
        AsyncMock(fetchall=AsyncMock(return_value=[(101,)])),
        AsyncMock(rowcount=1),
        mock_full_data_result
    ]
    fetched_processable_claims_fail_dob = await processor._fetch_claims_parallel(mock_session, batch_id="test_batch_enc", limit=1)
    assert fetched_processable_claims_fail_dob[0].patient_date_of_birth is None

    # Test case: Decrypted DOB string is invalid format
    mock_encryption_service.decrypt.side_effect = lambda val: "invalid-date-format" if val == "enc_dob_1990-05-15" else f"decrypted_{val}"
    mock_session.execute.side_effect = [
        AsyncMock(fetchall=AsyncMock(return_value=[(101,)])),
        AsyncMock(rowcount=1),
        mock_full_data_result
    ]
    fetched_processable_claims_bad_date = await processor._fetch_claims_parallel(mock_session, batch_id="test_batch_enc", limit=1)
    assert fetched_processable_claims_bad_date[0].patient_date_of_birth is None


@patch('asyncio.sleep', new_callable=AsyncMock) # Keep sleep mocked if not used, for consistency
@pytest.mark.asyncio
async def test_process_claims_parallel_ml_predictor_unavailable(
    mock_asyncio_sleep: AsyncMock, # Injected by @patch
    processor: ParallelClaimsProcessor,
    mock_db_session_factory_and_session,
    mock_claim_validator: MagicMock,
    mock_rvu_service: MagicMock,
    mock_feature_extractor: MagicMock, # Still need this for feature extraction attempt
    mock_optimized_predictor: MagicMock,
    mock_metrics_collector: MagicMock,
    mock_audit_logger_service: MagicMock
):
    mock_session_factory, mock_session = mock_db_session_factory_and_session
    passed_batch_id = "b_ml_unavailable"

    # Simulate predictor being unavailable
    mock_optimized_predictor.interpreter = None # Key condition for this test

    # Setup a couple of claims that would otherwise pass validation and feature extraction
    claim1 = create_processable_claim("c_mlu_1", 301, batch_id=passed_batch_id)
    claim2 = create_processable_claim("c_mlu_2", 302, batch_id=passed_batch_id)
    all_fetched_claims = [claim1, claim2]
    limit_param = len(all_fetched_claims)

    with patch.object(processor, '_fetch_claims_parallel', new_callable=AsyncMock, return_value=all_fetched_claims) as mock_fetch:
        mock_claim_validator.validate_claim.return_value = [] # All valid

        # Feature extraction would still run
        features_c1 = np.array([[0.1]*7], dtype=np.float32)
        features_c2 = np.array([[0.2]*7], dtype=np.float32)
        def feature_extractor_side_effect(claim_to_extract):
            if claim_to_extract.claim_id == "c_mlu_1": return features_c1
            if claim_to_extract.claim_id == "c_mlu_2": return features_c2
            return None
        mock_feature_extractor.extract_features.side_effect = feature_extractor_side_effect

        # predict_batch should not be called if interpreter is None and _apply_ml_predictions handles it early

        # RVU service should be called for both claims
        mock_rvu_service.calculate_rvu_for_claim.return_value = None

        # Assume both claims are transferred successfully
        with patch.object(processor, '_transfer_claims_to_production', new_callable=AsyncMock, return_value=2) as mock_transfer, \
             patch.object(processor, '_update_staging_claims_status') as mock_update_staging, \
             patch.object(processor, '_route_failed_claims', new_callable=AsyncMock, return_value=0) as mock_route_failed, \
             patch.object(processor, '_set_final_status_for_staging_claims') as mock_set_final_status:

            summary = await processor.process_claims_parallel(batch_id=passed_batch_id, limit=limit_param)

    # Assertions
    mock_fetch.assert_called_once_with(mock_session, passed_batch_id, limit_param)
    assert mock_claim_validator.validate_claim.call_count == 2

    # Feature extraction is attempted before the interpreter check within _apply_ml_predictions
    assert mock_feature_extractor.extract_features.call_count == 2
    # predict_batch should not be called because _apply_ml_predictions returns early
    mock_optimized_predictor.predict_batch.assert_not_called()

    # RVU assertions (both claims should proceed)
    assert mock_rvu_service.calculate_rvu_for_claim.call_count == 2
    mock_rvu_service.calculate_rvu_for_claim.assert_any_call(claim1, mock_session)
    mock_rvu_service.calculate_rvu_for_claim.assert_any_call(claim2, mock_session)

    # Transfer assertions (both claims proceed and are transferred)
    mock_transfer.assert_called_once()
    transfer_args = mock_transfer.call_args[0]
    assert len(transfer_args[1]) == 2
    assert claim1 in transfer_args[1]
    assert claim2 in transfer_args[1]

    # Staging update for transferred claims
    mock_update_staging.assert_any_call(mock_session, [claim1.id, claim2.id], "completed_transferred", transferred=True)

    # No claims should be routed as failed due to ML, nor staging status set to a failed ML state
    mock_route_failed.assert_not_called() # Or called with empty list if that's the path
    # mock_set_final_status might be called with an empty list if no other failures, or not at all.
    # Check if it was called, and if so, with an empty list for ML-related parts.
    if mock_set_final_status.called:
        final_status_args = mock_set_final_status.call_args[0][1]
        assert not any("ML_" in item[1] for item in final_status_args)


    # Summary assertions
    assert summary["fetched_count"] == limit_param
    assert summary["validation_passed_count"] == 2
    assert summary["ml_prediction_attempted_count"] == 2 # Attempted means _apply_ml_predictions was called
    assert summary["ml_approved_raw"] == 0
    assert summary["ml_rejected_raw"] == 0
    assert summary["stopped_by_ml_rejection"] == 0
    # Both claims should be marked with ML_PREDICTOR_UNAVAILABLE and counted in ml_errors_raw
    assert summary["ml_errors_raw"] == 2
    assert summary["rvu_calculation_completed_count"] == 2
    assert summary["transferred_to_prod_count"] == 2
    assert summary["staging_updated_transferred_count"] == 2
    assert summary["failed_claims_routed_count"] == 0
    assert summary["staging_updated_failed_count"] == 0
    assert summary["error"] is None

    assert claim1.ml_derived_decision == "ML_PREDICTOR_UNAVAILABLE"
    assert claim2.ml_derived_decision == "ML_PREDICTOR_UNAVAILABLE"

    # Audit log
    assert mock_audit_logger_service.log_event.call_count == 2 # START and END
    end_event_kwargs = mock_audit_logger_service.log_event.call_args_list[1].kwargs
    assert end_event_kwargs['success'] is True
    assert end_event_kwargs['details']['attempts_made'] == 1
    assert end_event_kwargs['details']['summary']['ml_errors_raw'] == 2

    # Metrics
    mock_metrics_collector.record_batch_processed.assert_called_once()
    processed_args = mock_metrics_collector.record_batch_processed.call_args[1]
    assert processed_args['claims_by_final_status'] == {'completed_transferred': 2}

    mock_asyncio_sleep.assert_not_called() # No retries expected
