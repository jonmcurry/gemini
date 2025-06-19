import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import List, Any
from decimal import Decimal
from datetime import date, datetime, timezone

# Models to be mocked or instantiated
from claims_processor.src.core.database.models.claims_db import ClaimModel, ClaimLineItemModel
from claims_processor.src.api.models.claim_models import ProcessableClaim, ProcessableClaimLineItem
from claims_processor.src.core.cache.cache_manager import CacheManager
from claims_processor.src.processing.validation.claim_validator import ClaimValidator
from claims_processor.src.processing.pipeline.parallel_claims_processor import ParallelClaimsProcessor

# --- Fixtures ---
@pytest.fixture
def mock_db_session_factory_and_session(): # Combined for easier use
    mock_session = AsyncMock(spec=AsyncSession) # Mock the session object that will be yielded

    # Mock the context manager part of the session factory
    async_cm = AsyncMock()
    async_cm.__aenter__.return_value = mock_session
    async_cm.__aexit__.return_value = None # To handle 'async with ... as session:'

    mock_session_factory = MagicMock() # This is the factory function (e.g., AsyncSessionLocal)
    mock_session_factory.return_value = async_cm # Factory returns the async context manager

    return mock_session_factory, mock_session

@pytest.fixture
def mock_cache_manager():
    return MagicMock(spec=CacheManager)

@pytest.fixture
def mock_claim_validator_fixture(): # Renamed to avoid conflict with imported class
    validator = MagicMock(spec=ClaimValidator)
    validator.validate_claim = MagicMock(return_value=[]) # Default to valid (empty error list)
    return validator

@pytest.fixture
def processor(mock_db_session_factory_and_session, mock_cache_manager, mock_claim_validator_fixture) -> ParallelClaimsProcessor:
    mock_session_factory, _ = mock_db_session_factory_and_session # Unpack
    # Patch the ClaimValidator instantiation within ParallelClaimsProcessor's __init__
    with patch('claims_processor.src.processing.pipeline.parallel_claims_processor.ClaimValidator', return_value=mock_claim_validator_fixture):
        service = ParallelClaimsProcessor(
            db_session_factory=mock_session_factory,
            cache_manager=mock_cache_manager
        )
    return service


# Helper to create mock DB ClaimModel instances
def create_mock_db_claim(claim_id_val: str, db_id: int, status: str = 'pending', num_line_items: int = 1) -> ClaimModel:
    # Using MagicMock to simulate SQLAlchemy models for more control over attributes if needed.
    # Actual model instances can also be used if they don't trigger DB calls on init.
    claim = MagicMock(spec=ClaimModel)
    claim.id = db_id
    claim.claim_id = claim_id_val
    claim.facility_id = f"fac_{db_id}"
    claim.patient_account_number = f"pac_{db_id}"
    claim.service_from_date = date(2023, 1, 1)
    claim.service_to_date = date(2023, 1, 5)
    claim.total_charges = Decimal("100.00")
    claim.processing_status = status
    claim.batch_id = None # Important for tests
    claim.created_at = datetime.now(timezone.utc)
    claim.updated_at = datetime.now(timezone.utc)
    claim.processed_at = None
    claim.line_items = []
    for i in range(num_line_items):
        line = MagicMock(spec=ClaimLineItemModel)
        line.id = (db_id * 100) + i
        line.claim_db_id = db_id
        line.line_number = i + 1
        line.service_date = date(2023, 1, 1 + i)
        line.procedure_code = f"proc_{i+1}"
        line.units = 1
        line.charge_amount = Decimal("50.00") if num_line_items == 2 else Decimal("100.00")
        line.rvu_total = None
        line.created_at = datetime.now(timezone.utc)
        line.updated_at = datetime.now(timezone.utc)
        claim.line_items.append(line)
    return claim

# Helper to create ProcessableClaim Pydantic models
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
        updated_at=datetime.now(timezone.utc), line_items=line_items_data,
        batch_id=batch_id # Make sure ProcessableClaim can accept batch_id
    )

# --- Tests for _fetch_claims_parallel ---
@pytest.mark.asyncio
async def test_fetch_claims_no_pending(processor: ParallelClaimsProcessor, mock_db_session_factory_and_session):
    _, mock_session = mock_db_session_factory_and_session
    mock_ids_result = AsyncMock()
    mock_ids_result.fetchall.return_value = []
    mock_session.execute.return_value = mock_ids_result

    fetched_claims = await processor._fetch_claims_parallel(mock_session, batch_id="b1", limit=10)

    assert fetched_claims == []
    mock_session.execute.assert_called_once()
    assert "update" not in str(mock_session.execute.call_args_list[0][0][0]).lower() # Check the SQL query string

@pytest.mark.asyncio
async def test_fetch_claims_success(processor: ParallelClaimsProcessor, mock_db_session_factory_and_session):
    _, mock_session = mock_db_session_factory_and_session

    mock_db_claim1 = create_mock_db_claim("c1", 1)
    mock_db_claim2 = create_mock_db_claim("c2", 2)

    mock_ids_result = AsyncMock(); mock_ids_result.fetchall.return_value = [(1,), (2,)]
    mock_update_result = AsyncMock() # For the update statement
    mock_full_claims_result = AsyncMock()
    mock_full_claims_result.scalars.return_value.unique.return_value.all.return_value = [mock_db_claim1, mock_db_claim2]

    mock_session.execute.side_effect = [mock_ids_result, mock_update_result, mock_full_claims_result]

    fetched_claims = await processor._fetch_claims_parallel(mock_session, batch_id="b1", limit=10)

    assert len(fetched_claims) == 2
    assert fetched_claims[0].claim_id == "c1"
    assert fetched_claims[1].claim_id == "c2"
    assert len(fetched_claims[0].line_items) == 1 # Based on create_mock_db_claim default

    assert mock_session.execute.call_count == 3

    # Check the update call more carefully
    update_call = mock_session.execute.call_args_list[1] # Second call to execute
    update_stmt_args = update_call[0][0] # The first positional argument to execute() is the statement
    assert "UPDATE claims" in str(update_stmt_args) # Check for UPDATE keyword
    assert "processing_status = :processing_status" in str(update_stmt_args) # Check for parameterization
    assert "batch_id = :batch_id" in str(update_stmt_args)
    assert "WHERE claims.id IN (__[POSTCOMPILE_id_1])" in str(update_stmt_args) # Check for ID list (SQLAlchemy internal representation)

@pytest.mark.asyncio
async def test_fetch_claims_db_error_during_select_ids(processor: ParallelClaimsProcessor, mock_db_session_factory_and_session):
    _, mock_session = mock_db_session_factory_and_session
    mock_session.execute.side_effect = Exception("DB Read Error")

    with pytest.raises(Exception, match="DB Read Error"):
        await processor._fetch_claims_parallel(mock_session, batch_id="b1", limit=10)
    # Rollback is handled by the main orchestrator's session context, not asserted here directly on mock_session typically

@pytest.mark.asyncio
async def test_fetch_claims_pydantic_conversion_error(processor: ParallelClaimsProcessor, mock_db_session_factory_and_session):
    _, mock_session = mock_db_session_factory_and_session

    mock_db_claim_valid_sqla = create_mock_db_claim("valid1", 1) # This is a MagicMock
    # To make ProcessableClaim.model_validate(mock_db_claim_valid_sqla) work,
    # mock_db_claim_valid_sqla needs to act like a dict or have attributes. MagicMock does.

    mock_db_claim_invalid_structure = MagicMock(spec=ClaimModel)
    mock_db_claim_invalid_structure.claim_id = "invalid_structure"
    # Add other necessary fields that ProcessableClaim expects or it will fail validation.
    # Forcing a more direct Pydantic error is harder if ClaimModel itself is already well-defined.
    # The test here is more about how the loop handles an error *during* model_validate.

    mock_ids_result = AsyncMock(); mock_ids_result.fetchall.return_value = [(1,), (2,)]
    mock_update_result = AsyncMock()
    mock_full_claims_result = AsyncMock()
    # Simulate one valid model and one that will cause Pydantic error
    mock_full_claims_result.scalars.return_value.unique.return_value.all.return_value = [
        mock_db_claim_valid_sqla,
        mock_db_claim_invalid_structure # This one will cause the error
    ]
    mock_session.execute.side_effect = [mock_ids_result, mock_update_result, mock_full_claims_result]

    def mock_model_validate_custom(obj):
        if obj == mock_db_claim_invalid_structure:
            raise ValueError("Pydantic conversion failed due to bad structure")
        return ProcessableClaim.model_validate(obj) # Original for others

    with patch('claims_processor.src.processing.pipeline.parallel_claims_processor.ProcessableClaim.model_validate', side_effect=mock_model_validate_custom):
        fetched_claims = await processor._fetch_claims_parallel(mock_session, "b1", 10)

    assert len(fetched_claims) == 1
    assert fetched_claims[0].claim_id == "valid1"

# --- Tests for _validate_claims_parallel ---
@pytest.mark.asyncio
async def test_validate_claims_all_valid(processor: ParallelClaimsProcessor, mock_claim_validator_fixture):
    claims_to_validate = [create_processable_claim("c1", 1, batch_id="b1"), create_processable_claim("c2", 2, batch_id="b1")]
    mock_claim_validator_fixture.validate_claim.return_value = []

    valid_claims, invalid_claims = await processor._validate_claims_parallel(claims_to_validate)

    assert len(valid_claims) == 2
    assert len(invalid_claims) == 0
    assert mock_claim_validator_fixture.validate_claim.call_count == 2

@pytest.mark.asyncio
async def test_validate_claims_some_invalid(processor: ParallelClaimsProcessor, mock_claim_validator_fixture):
    claim1_valid = create_processable_claim("c1", 1, batch_id="b1")
    claim2_invalid = create_processable_claim("c2", 2, batch_id="b1")
    claims_to_validate = [claim1_valid, claim2_invalid]

    def side_effect_validate(claim):
        if claim.claim_id == "c2":
            return ["Error: Invalid field"]
        return []
    mock_claim_validator_fixture.validate_claim.side_effect = side_effect_validate

    valid_claims, invalid_claims = await processor._validate_claims_parallel(claims_to_validate)

    assert len(valid_claims) == 1
    assert valid_claims[0].claim_id == "c1"
    assert len(invalid_claims) == 1
    assert invalid_claims[0].claim_id == "c2"
    assert mock_claim_validator_fixture.validate_claim.call_count == 2

@pytest.mark.asyncio
async def test_validate_claims_all_invalid(processor: ParallelClaimsProcessor, mock_claim_validator_fixture):
    claims_to_validate = [create_processable_claim("c1", 1, batch_id="b1"), create_processable_claim("c2", 2, batch_id="b1")]
    mock_claim_validator_fixture.validate_claim.return_value = ["Error: General error"]

    valid_claims, invalid_claims = await processor._validate_claims_parallel(claims_to_validate)

    assert len(valid_claims) == 0
    assert len(invalid_claims) == 2
    assert mock_claim_validator_fixture.validate_claim.call_count == 2
