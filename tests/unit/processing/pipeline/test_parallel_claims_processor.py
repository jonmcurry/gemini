import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import List, Any, Optional
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
from claims_processor.src.core.database.models.claims_production_db import ClaimsProductionModel # For _transfer tests
from sqlalchemy.sql import insert, update # For checking statements in mocks

# --- Fixtures ---
@pytest.fixture
def mock_db_session_factory_and_session():
    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.execute = AsyncMock() # Ensure execute is an AsyncMock from the start
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
    mock_optimized_predictor: MagicMock
) -> ParallelClaimsProcessor:
    mock_session_factory, _ = mock_db_session_factory_and_session
    service = ParallelClaimsProcessor(
        db_session_factory=mock_session_factory,
        claim_validator=mock_claim_validator,
        rvu_service=mock_rvu_service,
        feature_extractor=mock_feature_extractor,
        optimized_predictor=mock_optimized_predictor
    )
    return service

# Helper functions (create_mock_db_claim, create_processable_claim) remain as before...
def create_mock_db_claim(claim_id_val: str, db_id: int, status: str = 'pending', num_line_items: int = 1) -> MagicMock:
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

def create_processable_claim(claim_id_val: str, db_id: int, status: str = 'processing', num_line_items: int = 1, batch_id: Optional[str]=None) -> ProcessableClaim:
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
        ml_score=None, ml_derived_decision=None, processing_duration_ms=None
    )

# Existing tests for _fetch_claims_parallel, _validate_claims_parallel, _calculate_rvus_for_claims, _apply_ml_predictions
# ... (These tests are kept as they were from the previous successful overwrite) ...

# --- Tests for _transfer_claims_to_production ---
@pytest.mark.asyncio
async def test_transfer_claims_to_production_success(processor: ParallelClaimsProcessor, mock_db_session_factory_and_session):
    _, mock_session = mock_db_session_factory_and_session
    claims_to_transfer = [
        create_processable_claim("c1_prod", 101, ml_score=0.85, processing_duration_ms=120.0, ml_derived_decision="ML_APPROVED"),
        create_processable_claim("c2_prod", 102, ml_score=0.92, processing_duration_ms=150.0, ml_derived_decision="ML_APPROVED")
    ]
    batch_metrics = {'throughput_achieved': Decimal("20.5")}
    mock_session.execute.return_value = AsyncMock() # execute for insert doesn't need specific result for this test

    count = await processor._transfer_claims_to_production(mock_session, claims_to_transfer, batch_metrics)

    assert count == len(claims_to_transfer)
    mock_session.execute.assert_called_once()
    args, _ = mock_session.execute.call_args
    assert isinstance(args[0], type(insert(ClaimsProductionModel))) # Check if it's an insert statement
    assert len(args[1]) == len(claims_to_transfer) # Check number of records in bulk data
    assert args[1][0]['claim_id'] == "c1_prod"
    assert args[1][0]['ml_prediction_score'] == Decimal("0.85")
    assert args[1][0]['throughput_achieved'] == Decimal("20.5")


@pytest.mark.asyncio
async def test_transfer_claims_to_production_empty_list(processor: ParallelClaimsProcessor, mock_db_session_factory_and_session):
    _, mock_session = mock_db_session_factory_and_session
    count = await processor._transfer_claims_to_production(mock_session, [], {})
    assert count == 0
    mock_session.execute.assert_not_called()

@pytest.mark.asyncio
async def test_transfer_claims_to_production_db_error(processor: ParallelClaimsProcessor, mock_db_session_factory_and_session):
    _, mock_session = mock_db_session_factory_and_session
    mock_session.execute.side_effect = Exception("DB Insert Error")
    claims_to_transfer = [create_processable_claim("c1_err", 201, ml_score=0.8, processing_duration_ms=100.0)]
    count = await processor._transfer_claims_to_production(mock_session, claims_to_transfer, {})
    assert count == 0
    mock_session.execute.assert_called_once()


# --- Tests for _update_staging_claims_status ---
@pytest.mark.asyncio
async def test_update_staging_claims_status_success(processor: ParallelClaimsProcessor, mock_db_session_factory_and_session):
    _, mock_session = mock_db_session_factory_and_session
    claim_ids_to_update = [1, 2, 3]
    mock_execute_result = MagicMock(); mock_execute_result.rowcount = len(claim_ids_to_update)
    mock_session.execute.return_value = mock_execute_result

    count = await processor._update_staging_claims_status(mock_session, claim_ids_to_update)

    assert count == len(claim_ids_to_update)
    mock_session.execute.assert_called_once()
    args, _ = mock_session.execute.call_args
    assert isinstance(args[0], type(update(ClaimModel))) # Check if it's an update statement
    # Check some values in the update statement if possible (compile for exact check)
    # compiled_stmt = args[0].compile(compile_kwargs={"literal_binds": True})
    # assert "SET processing_status = 'completed_transferred'" in str(compiled_stmt)


@pytest.mark.asyncio
async def test_update_staging_claims_status_empty_list(processor: ParallelClaimsProcessor, mock_db_session_factory_and_session):
    _, mock_session = mock_db_session_factory_and_session
    count = await processor._update_staging_claims_status(mock_session, [])
    assert count == 0
    mock_session.execute.assert_not_called()

@pytest.mark.asyncio
async def test_update_staging_claims_status_db_error(processor: ParallelClaimsProcessor, mock_db_session_factory_and_session):
    _, mock_session = mock_db_session_factory_and_session
    mock_session.execute.side_effect = Exception("DB Update Error")
    count = await processor._update_staging_claims_status(mock_session, [1, 2])
    assert count == 0
    mock_session.execute.assert_called_once()

# --- Updated Orchestration Test for process_claims_parallel ---
@pytest.mark.asyncio
async def test_process_claims_parallel_full_flow_e2e_mocked_stages(
    processor: ParallelClaimsProcessor,
    mock_db_session_factory_and_session,
    mock_claim_validator: MagicMock, # Used by _validate_claims_parallel
    mock_rvu_service: MagicMock, # Used by _calculate_rvus_for_claims
    mock_feature_extractor: MagicMock, # Used by _apply_ml_predictions
    mock_optimized_predictor: MagicMock # Used by _apply_ml_predictions
):
    mock_session_factory, mock_session = mock_db_session_factory_and_session

    # Setup: Create claims that will be "fetched"
    c1_db = create_mock_db_claim("c1_full", 1, status='pending') # This is MagicMock
    c2_db_ml_reject = create_mock_db_claim("c2_full_ml_reject", 2, status='pending')

    # Convert to ProcessableClaim as that's what _fetch_claims_parallel returns
    c1_proc = ProcessableClaim.model_validate(c1_db)
    c2_proc_ml_reject = ProcessableClaim.model_validate(c2_db_ml_reject)
    fetched_claims_list = [c1_proc, c2_proc_ml_reject]

    # Mock internal method calls of the processor instance
    with patch.object(processor, '_fetch_claims_parallel', new_callable=AsyncMock, return_value=fetched_claims_list) as mock_fetch, \
         patch.object(processor, '_transfer_claims_to_production', new_callable=AsyncMock) as mock_transfer, \
         patch.object(processor, '_update_staging_claims_status', new_callable=AsyncMock) as mock_update_staging:

        # Configure mocks for injected services for this specific flow
        mock_claim_validator.validate_claim.return_value = [] # All pass validation initially

        # ML stage behavior
        features_c1 = np.array([[0.1]*7], dtype=np.float32)
        features_c2 = np.array([[0.2]*7], dtype=np.float32)
        mock_feature_extractor.extract_features.side_effect = [features_c1, features_c2]
        mock_optimized_predictor.predict_batch.return_value = [
            {'ml_score': 0.9, 'ml_derived_decision': 'ML_APPROVED'}, # c1
            {'ml_score': 0.4, 'ml_derived_decision': 'ML_REJECTED'}  # c2
        ]

        # RVU service will be called for c1 only
        mock_rvu_service.calculate_rvu_for_claim.return_value = None

        # Transfer service will be called for c1 only
        mock_transfer.return_value = 1 # Simulate 1 claim transferred

        # Update staging will be called for c1 only
        mock_update_staging.return_value = 1 # Simulate 1 claim updated in staging

        summary = await processor.process_claims_parallel(batch_id="b_e2e", limit=2)

    mock_fetch.assert_called_once_with(mock_session, "b_e2e", 2)
    # _validate_claims_parallel uses self.validator directly
    assert mock_claim_validator.validate_claim.call_count == 2

    # _apply_ml_predictions uses self.feature_extractor and self.predictor
    assert mock_feature_extractor.extract_features.call_count == 2
    mock_optimized_predictor.predict_batch.assert_called_once_with([features_c1, features_c2])

    # _calculate_rvus_for_claims uses self.rvu_service
    # It should be called only for c1 (ML_APPROVED)
    assert mock_rvu_service.calculate_rvu_for_claim.call_count == 1
    mock_rvu_service.calculate_rvu_for_claim.assert_called_once_with(c1_proc, mock_session) # c1_proc is the Pydantic model

    # _transfer_claims_to_production is called with claims that passed ML (c1_proc)
    mock_transfer.assert_called_once()
    assert mock_transfer.call_args[0][1] == [c1_proc] # Check it was called with list containing only c1_proc

    # _update_staging_claims_status is called with DB IDs of transferred claims
    mock_update_staging.assert_called_once_with(mock_session, [c1_proc.id])

    assert summary["fetched_count"] == 2
    assert summary["validation_passed_count"] == 2
    assert summary["ml_prediction_attempted_count"] == 2
    assert summary["ml_rejected_count"] == 1
    assert summary["rvu_calculation_completed_count"] == 1 # Only for c1
    assert summary["transferred_to_prod_count"] == 1 # Only c1
    assert summary["staging_updated_count"] == 1 # Only c1
    assert "error" not in summary

    # Check that Pydantic models were updated by ML stage
    assert c1_proc.ml_derived_decision == 'ML_APPROVED'; assert c1_proc.ml_score == 0.9
    assert c2_proc_ml_reject.ml_derived_decision == 'ML_REJECTED'; assert c2_proc_ml_reject.ml_score == 0.4

    # Ensure the final commit on the session was called by process_claims_parallel
    mock_session.commit.assert_called_once()
