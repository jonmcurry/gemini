import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import List, Any, Optional # Added Optional for batch_id in create_processable_claim
from decimal import Decimal
from datetime import date, datetime, timezone

# Models and Services
from claims_processor.src.core.database.models.claims_db import ClaimModel, ClaimLineItemModel
from claims_processor.src.api.models.claim_models import ProcessableClaim, ProcessableClaimLineItem
from claims_processor.src.core.cache.cache_manager import CacheManager
from claims_processor.src.processing.validation.claim_validator import ClaimValidator
from claims_processor.src.processing.rvu_service import RVUService # Import RVUService
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
def processor(
    mock_db_session_factory_and_session,
    mock_claim_validator: MagicMock,
    mock_rvu_service: MagicMock
) -> ParallelClaimsProcessor:
    mock_session_factory, _ = mock_db_session_factory_and_session
    # ParallelClaimsProcessor now takes validator and rvu_service instances
    service = ParallelClaimsProcessor(
        db_session_factory=mock_session_factory,
        claim_validator=mock_claim_validator,
        rvu_service=mock_rvu_service
    )
    return service

# Helper to create mock DB ClaimModel instances
def create_mock_db_claim(claim_id_val: str, db_id: int, status: str = 'pending', num_line_items: int = 1) -> MagicMock:
    claim = MagicMock(spec=ClaimModel)
    claim.id = db_id
    claim.claim_id = claim_id_val
    claim.facility_id = f"fac_{db_id}"
    claim.patient_account_number = f"pac_{db_id}"
    claim.service_from_date = date(2023, 1, 1)
    claim.service_to_date = date(2023, 1, 5)
    claim.total_charges = Decimal("100.00")
    claim.processing_status = status
    claim.batch_id = None
    claim.created_at = datetime.now(timezone.utc)
    claim.updated_at = datetime.now(timezone.utc)
    claim.processed_at = None
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
        updated_at=datetime.now(timezone.utc), line_items=line_items_data, batch_id=batch_id
    )

# --- Tests for _fetch_claims_parallel ---
@pytest.mark.asyncio
async def test_fetch_claims_no_pending(processor: ParallelClaimsProcessor, mock_db_session_factory_and_session):
    _, mock_session = mock_db_session_factory_and_session
    mock_ids_result = AsyncMock(); mock_ids_result.fetchall.return_value = []
    mock_session.execute.return_value = mock_ids_result
    fetched_claims = await processor._fetch_claims_parallel(mock_session, batch_id="b1", limit=10)
    assert fetched_claims == []
    mock_session.execute.assert_called_once()
    assert "update" not in str(mock_session.execute.call_args_list[0][0][0]).lower()

@pytest.mark.asyncio
async def test_fetch_claims_success(processor: ParallelClaimsProcessor, mock_db_session_factory_and_session):
    _, mock_session = mock_db_session_factory_and_session
    mock_db_claim1 = create_mock_db_claim("c1", 1); mock_db_claim2 = create_mock_db_claim("c2", 2)
    mock_ids_result = AsyncMock(); mock_ids_result.fetchall.return_value = [(1,), (2,)]
    mock_update_result = AsyncMock()
    mock_full_claims_result = AsyncMock()
    mock_full_claims_result.scalars.return_value.unique.return_value.all.return_value = [mock_db_claim1, mock_db_claim2]
    mock_session.execute.side_effect = [mock_ids_result, mock_update_result, mock_full_claims_result]
    fetched_claims = await processor._fetch_claims_parallel(mock_session, batch_id="b1", limit=10)
    assert len(fetched_claims) == 2; assert fetched_claims[0].claim_id == "c1"
    assert mock_session.execute.call_count == 3
    update_call = mock_session.execute.call_args_list[1]
    update_stmt_args = update_call[0][0]
    assert "UPDATE claims" in str(update_stmt_args)
    assert "processing_status = :processing_status" in str(update_stmt_args)
    assert "batch_id = :batch_id" in str(update_stmt_args)
    assert "WHERE claims.id IN (__[POSTCOMPILE_id_1])" in str(update_stmt_args)

@pytest.mark.asyncio
async def test_fetch_claims_db_error_during_select_ids(processor: ParallelClaimsProcessor, mock_db_session_factory_and_session):
    _, mock_session = mock_db_session_factory_and_session
    mock_session.execute.side_effect = Exception("DB Read Error")
    with pytest.raises(Exception, match="DB Read Error"):
        await processor._fetch_claims_parallel(mock_session, batch_id="b1", limit=10)

@pytest.mark.asyncio
async def test_fetch_claims_pydantic_conversion_error(processor: ParallelClaimsProcessor, mock_db_session_factory_and_session):
    _, mock_session = mock_db_session_factory_and_session
    mock_db_claim_valid_sqla = create_mock_db_claim("valid1", 1)
    mock_db_claim_invalid_structure = MagicMock(spec=ClaimModel); mock_db_claim_invalid_structure.claim_id = "invalid_structure"
    mock_ids_result = AsyncMock(); mock_ids_result.fetchall.return_value = [(1,), (2,)]
    mock_update_result = AsyncMock()
    mock_full_claims_result = AsyncMock()
    mock_full_claims_result.scalars.return_value.unique.return_value.all.return_value = [mock_db_claim_valid_sqla, mock_db_claim_invalid_structure]
    mock_session.execute.side_effect = [mock_ids_result, mock_update_result, mock_full_claims_result]
    original_model_validate = ProcessableClaim.model_validate
    def mock_model_validate_custom(obj):
        if obj == mock_db_claim_invalid_structure: raise ValueError("Pydantic conversion failed")
        return original_model_validate(obj)
    with patch('claims_processor.src.processing.pipeline.parallel_claims_processor.ProcessableClaim.model_validate', side_effect=mock_model_validate_custom) as mock_val_call:
        fetched_claims = await processor._fetch_claims_parallel(mock_session, "b1", 10)
    assert len(fetched_claims) == 1; assert fetched_claims[0].claim_id == "valid1"
    assert mock_val_call.call_count == 2


# --- Tests for _validate_claims_parallel ---
@pytest.mark.asyncio
async def test_validate_claims_all_valid(processor: ParallelClaimsProcessor, mock_claim_validator: MagicMock):
    claims_to_validate = [create_processable_claim("c1",1,batch_id="b1"), create_processable_claim("c2",2,batch_id="b1")]
    mock_claim_validator.validate_claim.return_value = []
    valid, invalid = await processor._validate_claims_parallel(claims_to_validate)
    assert len(valid) == 2; assert len(invalid) == 0; assert mock_claim_validator.validate_claim.call_count == 2

@pytest.mark.asyncio
async def test_validate_claims_some_invalid(processor: ParallelClaimsProcessor, mock_claim_validator: MagicMock):
    v = create_processable_claim("c1",1,batch_id="b1"); inv = create_processable_claim("c2",2,batch_id="b1")
    mock_claim_validator.validate_claim.side_effect = lambda c: ["Err"] if c.claim_id == "c2" else []
    valid, invalid = await processor._validate_claims_parallel([v, inv])
    assert len(valid) == 1; assert valid[0].claim_id == "c1"; assert len(invalid) == 1; assert invalid[0].claim_id == "c2"

# --- Tests for _calculate_rvus_for_claims ---
@pytest.mark.asyncio
async def test_calculate_rvus_for_claims_success(processor: ParallelClaimsProcessor, mock_db_session_factory_and_session, mock_rvu_service: MagicMock):
    _, mock_session = mock_db_session_factory_and_session
    claims = [create_processable_claim("r1",101), create_processable_claim("r2",102)]
    await processor._calculate_rvus_for_claims(mock_session, claims)
    assert mock_rvu_service.calculate_rvu_for_claim.call_count == 2
    mock_rvu_service.calculate_rvu_for_claim.assert_any_call(claims[0], mock_session)
    mock_rvu_service.calculate_rvu_for_claim.assert_any_call(claims[1], mock_session)

@pytest.mark.asyncio
async def test_calculate_rvus_for_claims_one_fails(processor: ParallelClaimsProcessor, mock_db_session_factory_and_session, mock_rvu_service: MagicMock):
    _, mock_session = mock_db_session_factory_and_session
    c1 = create_processable_claim("r_ok",201); c2_fail = create_processable_claim("r_fail",202)
    async def rvu_side_effect(claim, session):
        if claim.claim_id == "r_fail": raise Exception("RVU Error")
    mock_rvu_service.calculate_rvu_for_claim.side_effect = rvu_side_effect
    await processor._calculate_rvus_for_claims(mock_session, [c1, c2_fail])
    assert mock_rvu_service.calculate_rvu_for_claim.call_count == 2

@pytest.mark.asyncio
async def test_calculate_rvus_for_claims_no_claims(processor: ParallelClaimsProcessor, mock_db_session_factory_and_session, mock_rvu_service: MagicMock):
    _, mock_session = mock_db_session_factory_and_session
    await processor._calculate_rvus_for_claims(mock_session, [])
    mock_rvu_service.calculate_rvu_for_claim.assert_not_called()

# --- Test for process_claims_parallel (Orchestration) ---
@pytest.mark.asyncio
async def test_process_claims_parallel_success_with_rvu(
    processor: ParallelClaimsProcessor, mock_db_session_factory_and_session, mock_rvu_service: MagicMock
): # mock_claim_validator is used by processor fixture
    mock_session_factory, mock_session = mock_db_session_factory_and_session
    c1 = create_processable_claim("orc1", 1, batch_id="b_orc"); c2 = create_processable_claim("orc2", 2, batch_id="b_orc")
    fetched_list = [c1, c2]

    with patch.object(processor, '_fetch_claims_parallel', new_callable=AsyncMock, return_value=fetched_list) as mock_fetch, \
         patch.object(processor, '_validate_claims_parallel', new_callable=AsyncMock, return_value=(fetched_list, [])) as mock_validate:

        summary = await processor.process_claims_parallel(batch_id="b_orc", limit=2)

    mock_fetch.assert_called_once_with(mock_session, "b_orc", 2)
    mock_validate.assert_called_once_with(fetched_list)
    assert mock_rvu_service.calculate_rvu_for_claim.call_count == 2
    mock_rvu_service.calculate_rvu_for_claim.assert_any_call(c1, mock_session)
    mock_rvu_service.calculate_rvu_for_claim.assert_any_call(c2, mock_session)

    assert summary["fetched_count"] == 2
    assert summary["validation_passed_count"] == 2
    assert summary["validation_failed_count"] == 0
    assert summary["rvu_calculation_completed_count"] == 2
    assert "error" not in summary
