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

# --- Fixtures ---
@pytest.fixture
def mock_db_session_factory_and_session():
    mock_session = AsyncMock(spec=AsyncSession)
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

# Helper to create mock DB ClaimModel instances
def create_mock_db_claim(claim_id_val: str, db_id: int, status: str = 'pending', num_line_items: int = 1) -> MagicMock:
    claim = MagicMock(spec=ClaimModel)
    claim.id = db_id; claim.claim_id = claim_id_val; claim.facility_id = f"fac_{db_id}"
    claim.patient_account_number = f"pac_{db_id}"; claim.service_from_date = date(2023, 1, 1)
    claim.service_to_date = date(2023, 1, 5); claim.total_charges = Decimal("100.00")
    claim.processing_status = status; claim.batch_id = None
    claim.created_at = datetime.now(timezone.utc); claim.updated_at = datetime.now(timezone.utc)
    claim.processed_at = None; claim.ml_score = None; claim.ml_derived_decision = None; claim.processing_duration_ms = None
    claim.transferred_to_prod_at = None
    claim.line_items = []
    for i in range(num_line_items):
        line = MagicMock(spec=ClaimLineItemModel)
        line.id = (db_id * 100) + i; line.claim_db_id = db_id; line.line_number = i + 1
        line.service_date = date(2023, 1, 1 + i); line.procedure_code = f"proc_{i+1}"; line.units = 1
        line.charge_amount = Decimal("50.00") if num_line_items == 2 else Decimal("100.00")
        line.rvu_total = None; line.created_at = datetime.now(timezone.utc); line.updated_at = datetime.now(timezone.utc)
        claim.line_items.append(line)
    return claim

def create_processable_claim(claim_id_val: str, db_id: int, status: str = 'processing', num_line_items: int = 1, batch_id: Optional[str]=None) -> ProcessableClaim:
    line_items_data = []
    for i in range(num_line_items):
        line_items_data.append(ProcessableClaimLineItem(
            id=(db_id * 100) + i, claim_db_id=db_id, line_number=i + 1,
            service_date=date(2023, 1, 1 + i), procedure_code=f"proc_{i+1}", units=1,
            charge_amount=Decimal("50.00") if num_line_items == 2 else Decimal("100.00"),
            created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc), rvu_total=None
        ))
    return ProcessableClaim(
        id=db_id, claim_id=claim_id_val, facility_id=f"fac_{db_id}",
        patient_account_number=f"pac_{db_id}", service_from_date=date(2023, 1, 1),
        service_to_date=date(2023, 1, 5), total_charges=Decimal("100.00"),
        processing_status=status, created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc), line_items=line_items_data, batch_id=batch_id,
        ml_score=None, ml_derived_decision=None, processing_duration_ms=None # Initialize new fields
    )

# --- Tests for _fetch_claims_parallel (largely unchanged) ---
@pytest.mark.asyncio
async def test_fetch_claims_no_pending(processor: ParallelClaimsProcessor, mock_db_session_factory_and_session):
    _, mock_session = mock_db_session_factory_and_session
    mock_ids_result = AsyncMock(); mock_ids_result.fetchall.return_value = []
    mock_session.execute.return_value = mock_ids_result
    fetched_claims = await processor._fetch_claims_parallel(mock_session, batch_id="b1", limit=10)
    assert fetched_claims == []

# --- Tests for _validate_claims_parallel (largely unchanged) ---
@pytest.mark.asyncio
async def test_validate_claims_all_valid(processor: ParallelClaimsProcessor, mock_claim_validator: MagicMock):
    claims_to_validate = [create_processable_claim("c1",1,batch_id="b1"), create_processable_claim("c2",2,batch_id="b1")]
    mock_claim_validator.validate_claim.return_value = []
    valid, invalid = await processor._validate_claims_parallel(claims_to_validate)
    assert len(valid) == 2; assert len(invalid) == 0

# --- Tests for _calculate_rvus_for_claims (largely unchanged) ---
@pytest.mark.asyncio
async def test_calculate_rvus_for_claims_success(processor: ParallelClaimsProcessor, mock_db_session_factory_and_session, mock_rvu_service: MagicMock):
    _, mock_session = mock_db_session_factory_and_session
    claims = [create_processable_claim("r1",101), create_processable_claim("r2",102)]
    await processor._calculate_rvus_for_claims(mock_session, claims)
    assert mock_rvu_service.calculate_rvu_for_claim.call_count == 2

# --- Tests for _apply_ml_predictions (new method being tested) ---
@pytest.mark.asyncio
async def test_apply_ml_predictions_success(processor: ParallelClaimsProcessor, mock_feature_extractor: MagicMock, mock_optimized_predictor: MagicMock):
    claim1 = create_processable_claim("ml_c1", 1); claim2 = create_processable_claim("ml_c2", 2)
    claims_to_process = [claim1, claim2]
    features_c1 = np.array([[0.1]*7], dtype=np.float32); features_c2 = np.array([[0.2]*7], dtype=np.float32)
    mock_feature_extractor.extract_features.side_effect = [features_c1, features_c2]
    mock_optimized_predictor.predict_batch.return_value = [
        {'ml_score': 0.9, 'ml_derived_decision': 'ML_APPROVED'},
        {'ml_score': 0.4, 'ml_derived_decision': 'ML_REJECTED'}
    ]
    await processor._apply_ml_predictions(claims_to_process)
    assert mock_feature_extractor.extract_features.call_count == 2
    mock_optimized_predictor.predict_batch.assert_called_once_with([features_c1, features_c2])
    assert claim1.ml_score == 0.9; assert claim1.ml_derived_decision == "ML_APPROVED"
    assert claim2.ml_score == 0.4; assert claim2.ml_derived_decision == "ML_REJECTED"

@pytest.mark.asyncio
async def test_apply_ml_predictions_feature_extraction_fails(processor: ParallelClaimsProcessor, mock_feature_extractor: MagicMock, mock_optimized_predictor: MagicMock):
    claim1 = create_processable_claim("ml_fe_ok", 1); claim2_fe_fail = create_processable_claim("ml_fe_fail", 2)
    claims_to_process = [claim1, claim2_fe_fail]
    features_c1 = np.array([[0.1]*7], dtype=np.float32)
    mock_feature_extractor.extract_features.side_effect = [features_c1, None] # claim2 fails extraction
    mock_optimized_predictor.predict_batch.return_value = [{'ml_score': 0.8, 'ml_derived_decision': 'ML_APPROVED'}]
    await processor._apply_ml_predictions(claims_to_process)
    assert mock_feature_extractor.extract_features.call_count == 2
    mock_optimized_predictor.predict_batch.assert_called_once_with([features_c1]) # Only claim1 features sent
    assert claim1.ml_score == 0.8; assert claim1.ml_derived_decision == "ML_APPROVED"
    assert claim2_fe_fail.ml_derived_decision == "ML_SKIPPED_NO_FEATURES"; assert claim2_fe_fail.ml_score is None

@pytest.mark.asyncio
async def test_apply_ml_predictions_batch_prediction_fails(processor: ParallelClaimsProcessor, mock_feature_extractor: MagicMock, mock_optimized_predictor: MagicMock):
    claim1 = create_processable_claim("ml_pred_fail1", 1); claim2 = create_processable_claim("ml_pred_fail2", 2)
    claims_to_process = [claim1, claim2]
    features_c1 = np.array([[0.1]*7], dtype=np.float32); features_c2 = np.array([[0.2]*7], dtype=np.float32)
    mock_feature_extractor.extract_features.side_effect = [features_c1, features_c2]
    mock_optimized_predictor.predict_batch.side_effect = Exception("Batch Prediction Error")
    await processor._apply_ml_predictions(claims_to_process)
    assert claim1.ml_derived_decision == "ML_PREDICTION_BATCH_ERROR"; assert claim1.ml_score is None
    assert claim2.ml_derived_decision == "ML_PREDICTION_BATCH_ERROR"; assert claim2.ml_score is None

# --- Updated Test for process_claims_parallel (Orchestration) ---
@pytest.mark.asyncio
async def test_process_claims_parallel_full_flow_mocked_ml(
    processor: ParallelClaimsProcessor, mock_db_session_factory_and_session,
    mock_rvu_service: MagicMock, mock_feature_extractor: MagicMock, mock_optimized_predictor: MagicMock
):
    mock_session_factory, mock_session = mock_db_session_factory_and_session
    c1 = create_processable_claim("c1_ml_full", 1, batch_id="b_ml_full")
    c2 = create_processable_claim("c2_ml_full", 2, batch_id="b_ml_full")
    fetched_claims_list = [c1, c2]

    with patch.object(processor, '_fetch_claims_parallel', new_callable=AsyncMock, return_value=fetched_claims_list) as mock_fetch, \
         patch.object(processor, '_validate_claims_parallel', new_callable=AsyncMock, return_value=(fetched_claims_list, [])) as mock_validate, \
         patch.object(processor, '_calculate_rvus_for_claims', new_callable=AsyncMock) as mock_rvu_stage: # Keep RVU stage mocked for focus

        features_c1 = np.array([[0.1]*7], dtype=np.float32); features_c2 = np.array([[0.2]*7], dtype=np.float32)
        mock_feature_extractor.extract_features.side_effect = [features_c1, features_c2]
        mock_optimized_predictor.predict_batch.return_value = [
            {'ml_score': 0.9, 'ml_derived_decision': 'ML_APPROVED'},
            {'ml_score': 0.4, 'ml_derived_decision': 'ML_REJECTED'}
        ]

        summary = await processor.process_claims_parallel(batch_id="b_ml_full", limit=2)

    mock_fetch.assert_called_once_with(mock_session, "b_ml_full", 2)
    mock_validate.assert_called_once_with(fetched_claims_list)
    assert mock_feature_extractor.extract_features.call_count == 2
    mock_optimized_predictor.predict_batch.assert_called_once_with([features_c1, features_c2])
    mock_rvu_stage.assert_called_once_with(mock_session, fetched_claims_list) # RVU still called for all valid (for now)

    assert summary["fetched_count"] == 2; assert summary["validation_passed_count"] == 2
    assert summary["ml_prediction_attempted_count"] == 2
    assert summary["rvu_calculation_completed_count"] == 2
    assert "error" not in summary
    assert c1.ml_derived_decision == 'ML_APPROVED'; assert c1.ml_score == 0.9
    assert c2.ml_derived_decision == 'ML_REJECTED'; assert c2.ml_score == 0.4
