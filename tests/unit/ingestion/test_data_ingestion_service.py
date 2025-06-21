import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Callable, Any, Tuple # Added Tuple for fixture return type
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID, uuid4 # For checking batch_id format
from datetime import date

from claims_processor.src.ingestion.data_ingestion_service import DataIngestionService
from claims_processor.src.api.models.ingestion_models import IngestionClaim, IngestionClaimLineItem
from claims_processor.src.core.database.models.claims_db import ClaimModel # For asserting type passed to add_all
from claims_processor.src.core.security.encryption_service import EncryptionService


@pytest.fixture
def mock_db_session_factory_and_session() -> Tuple[MagicMock, AsyncMock]:
    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.add_all = MagicMock()

    # Mock the begin() context manager on the session
    mock_session_begin_cm = AsyncMock() # Context manager for session.begin()
    mock_session_begin_cm.__aenter__.return_value = mock_session # Simulate entering 'async with session.begin()'
    mock_session_begin_cm.__aexit__.return_value = None # Simulate successful exit
    mock_session.begin = MagicMock(return_value=mock_session_begin_cm)

    # Mock the factory that returns an async context manager for the session
    async_cm_factory_yields = AsyncMock() # Context manager for db_session_factory()
    async_cm_factory_yields.__aenter__.return_value = mock_session
    async_cm_factory_yields.__aexit__.return_value = None

    mock_session_factory_instance = MagicMock(spec=Callable[[], Any])
    mock_session_factory_instance.return_value = async_cm_factory_yields

    return mock_session_factory_instance, mock_session

@pytest.fixture
def mock_encryption_service() -> MagicMock:
    service = MagicMock(spec=EncryptionService)
    # Simple pass-through encryption for testing, or specific encrypted values
    service.encrypt.side_effect = lambda x: f"encrypted_{x}" if x else None
    return service

@pytest.fixture
def data_ingestion_service(
    mock_db_session_factory_and_session: Tuple[MagicMock, AsyncMock],
    mock_encryption_service: MagicMock
) -> DataIngestionService:
    factory_mock, _ = mock_db_session_factory_and_session
    return DataIngestionService(
        db_session_factory=factory_mock,
        encryption_service=mock_encryption_service
    )

# Helper to create IngestionClaim objects
def create_ingestion_claim_data(claim_id: str, num_lines: int = 1, patient_dob_str: Optional[str] = "2000-01-01", mrn: Optional[str] = "mrn123") -> IngestionClaim:
    line_items = []
    for i in range(num_lines):
        line_items.append(IngestionClaimLineItem(
            line_number=i + 1,
            service_date=date(2023, 1, 1 + i),
            procedure_code=f"P{i+1}",
            units=1,
            charge_amount=Decimal(f"{100.00 + i}")
        ))
    return IngestionClaim(
        claim_id=claim_id,
        facility_id="FAC001",
        patient_account_number=f"PATACC_{claim_id}",
        medical_record_number=mrn,
        patient_date_of_birth=date.fromisoformat(patient_dob_str) if patient_dob_str else None,
        service_from_date=date(2023, 1, 1),
        service_to_date=date(2023, 1, num_lines if num_lines > 0 else 1),
        total_charges=Decimal(f"{100.00 * num_lines if num_lines > 0 else 100.00}"),
        line_items=line_items,
        ingestion_batch_id="source_batch_1"
    )

@pytest.mark.asyncio
async def test_ingest_claims_batch_success(
    data_ingestion_service: DataIngestionService,
    mock_db_session_factory_and_session: Tuple[MagicMock, AsyncMock],
    mock_encryption_service: MagicMock
):
    _, mock_session = mock_db_session_factory_and_session
    raw_claims = [
        create_ingestion_claim_data("claim1", num_lines=2, patient_dob_str="1990-10-10", mrn="MRN001"),
        create_ingestion_claim_data("claim2", num_lines=1, patient_dob_str="1985-05-05", mrn="MRN002")
    ]

    summary = await data_ingestion_service.ingest_claims_batch(raw_claims, "test_batch_001")

    assert summary["received_claims"] == 2
    assert summary["successfully_staged_claims"] == 2
    assert summary["failed_ingestion_claims"] == 0
    assert not summary["errors"]
    assert UUID(summary["ingestion_batch_id"]) # If generated, or matches provided

    mock_session.add_all.assert_called_once()
    added_models = mock_session.add_all.call_args[0][0]
    assert len(added_models) == 2

    # Check first claim
    db_claim1 = added_models[0]
    assert isinstance(db_claim1, ClaimModel)
    assert db_claim1.claim_id == "claim1"
    assert db_claim1.patient_date_of_birth == "encrypted_1990-10-10"
    assert db_claim1.medical_record_number == "encrypted_MRN001"
    assert db_claim1.processing_status == "pending"
    assert db_claim1.batch_id == "source_batch_1" # From IngestionClaim.ingestion_batch_id
    assert len(db_claim1.line_items) == 2
    assert db_claim1.line_items[0].procedure_code == "P1"

    # Check encryption calls
    mock_encryption_service.encrypt.assert_any_call("1990-10-10")
    mock_encryption_service.encrypt.assert_any_call("MRN001")
    mock_encryption_service.encrypt.assert_any_call("1985-05-05")
    mock_encryption_service.encrypt.assert_any_call("MRN002")
    assert mock_encryption_service.encrypt.call_count == 4


@pytest.mark.asyncio
async def test_ingest_claims_batch_empty_input(data_ingestion_service: DataIngestionService):
    summary = await data_ingestion_service.ingest_claims_batch([], "test_batch_empty")
    assert summary["received_claims"] == 0
    assert summary["successfully_staged_claims"] == 0
    assert summary["failed_ingestion_claims"] == 0
    assert summary["ingestion_batch_id"] == "test_batch_empty"
    assert len(summary["errors"]) == 1
    assert summary["errors"][0]["error_message"] == "No data provided"

@pytest.mark.asyncio
async def test_ingest_claims_batch_encryption_failure(
    data_ingestion_service: DataIngestionService,
    mock_encryption_service: MagicMock
):
    raw_claims = [create_ingestion_claim_data("claim_enc_fail", mrn="mrn_to_fail_enc")]
    mock_encryption_service.encrypt.side_effect = lambda x: None if x == "mrn_to_fail_enc" else f"encrypted_{x}"

    summary = await data_ingestion_service.ingest_claims_batch(raw_claims, "batch_enc_fail")

    assert summary["received_claims"] == 1
    assert summary["successfully_staged_claims"] == 0
    assert summary["failed_ingestion_claims"] == 1
    assert len(summary["errors"]) == 1
    assert summary["errors"][0]["claim_id"] == "claim_enc_fail"
    assert "Medical record number encryption failed" in summary["errors"][0]["error"]

@pytest.mark.asyncio
async def test_ingest_claims_batch_db_error(
    data_ingestion_service: DataIngestionService,
    mock_db_session_factory_and_session: Tuple[MagicMock, AsyncMock]
):
    factory_mock, mock_session = mock_db_session_factory_and_session
    raw_claims = [create_ingestion_claim_data("claim_db_fail")]

    # Simulate DB error on session.begin().__aexit__ (commit)
    mock_session.begin.return_value.__aexit__ = AsyncMock(side_effect=Exception("DB Commit Error"))

    summary = await data_ingestion_service.ingest_claims_batch(raw_claims, "batch_db_fail")

    assert summary["received_claims"] == 1
    assert summary["successfully_staged_claims"] == 0
    assert summary["failed_ingestion_claims"] == 1 # The one claim failed due to DB error
    assert len(summary["errors"]) == 1
    assert "batch_db_error" in summary["errors"][0]
    assert summary["errors"][0]["batch_db_error"] == "DB Commit Error"

@pytest.mark.asyncio
async def test_ingest_claims_batch_uses_provided_batch_id_if_ingestion_batch_id_is_none(
    data_ingestion_service: DataIngestionService,
    mock_db_session_factory_and_session: Tuple[MagicMock, AsyncMock]
):
    _, mock_session = mock_db_session_factory_and_session
    claim_data = create_ingestion_claim_data("claim_no_source_batch", num_lines=1)
    claim_data.ingestion_batch_id = None # Explicitly set to None

    raw_claims = [claim_data]
    provided_id = "override_batch_002"

    summary = await data_ingestion_service.ingest_claims_batch(raw_claims, provided_id)

    assert summary["successfully_staged_claims"] == 1
    mock_session.add_all.assert_called_once()
    added_models = mock_session.add_all.call_args[0][0]
    assert added_models[0].batch_id == provided_id

@pytest.mark.asyncio
async def test_ingest_claims_batch_generates_batch_id_if_none_provided(
    data_ingestion_service: DataIngestionService,
    mock_db_session_factory_and_session: Tuple[MagicMock, AsyncMock]
):
    _, mock_session = mock_db_session_factory_and_session
    claim_data = create_ingestion_claim_data("claim_gen_batch_id", num_lines=1)
    claim_data.ingestion_batch_id = None # Source system did not provide one

    raw_claims = [claim_data]

    summary = await data_ingestion_service.ingest_claims_batch(raw_claims, None) # Service should generate one

    assert summary["successfully_staged_claims"] == 1
    generated_batch_id = summary["ingestion_batch_id"]
    assert generated_batch_id is not None
    try:
        UUID(generated_batch_id) # Check if it's a valid UUID string
    except ValueError:
        pytest.fail(f"Generated batch_id '{generated_batch_id}' is not a valid UUID.")

    mock_session.add_all.assert_called_once()
    added_models = mock_session.add_all.call_args[0][0]
    assert added_models[0].batch_id == generated_batch_id

@pytest.mark.asyncio
async def test_ingest_claims_partial_encryption_failure(
    data_ingestion_service: DataIngestionService,
    mock_db_session_factory_and_session: Tuple[MagicMock, AsyncMock],
    mock_encryption_service: MagicMock
):
    _, mock_session = mock_db_session_factory_and_session

    claim_ok = create_ingestion_claim_data("claim_ok", mrn="MRN_OK", patient_dob_str="2000-01-01")
    claim_fail_mrn = create_ingestion_claim_data("claim_fail_mrn", mrn="MRN_TO_FAIL", patient_dob_str="2001-01-01")
    claim_fail_dob = create_ingestion_claim_data("claim_fail_dob", mrn="MRN_OK_2", patient_dob_str="DOB_TO_FAIL") # DOB format will cause date.fromisoformat error before encryption

    raw_claims = [claim_ok, claim_fail_mrn, claim_fail_dob]

    def encryption_side_effect(value_to_encrypt: str):
        if value_to_encrypt == "MRN_TO_FAIL":
            return None # Simulate encryption failure for this MRN
        if value_to_encrypt == "DOB_TO_FAIL": # This case won't be hit due to earlier Pydantic/conversion error
            return None
        if value_to_encrypt is None:
            return None
        return f"encrypted_{value_to_encrypt}"

    mock_encryption_service.encrypt.side_effect = encryption_side_effect

    summary = await data_ingestion_service.ingest_claims_batch(raw_claims, "batch_partial_fail")

    assert summary["received_claims"] == 3
    assert summary["successfully_staged_claims"] == 1
    assert summary["failed_ingestion_claims"] == 2
    assert len(summary["errors"]) == 2

    # Check successful claim was added
    mock_session.add_all.assert_called_once()
    added_models = mock_session.add_all.call_args[0][0]
    assert len(added_models) == 1
    assert added_models[0].claim_id == "claim_ok"
    assert added_models[0].medical_record_number == "encrypted_MRN_OK"
    assert added_models[0].patient_date_of_birth == "encrypted_2000-01-01"

    # Check error details
    error_claim_ids = {err["claim_id"] for err in summary["errors"]}
    assert "claim_fail_mrn" in error_claim_ids
    assert "claim_fail_dob" in error_claim_ids

    for err in summary["errors"]:
        if err["claim_id"] == "claim_fail_mrn":
            assert "Medical record number encryption failed" in err["error"]
        if err["claim_id"] == "claim_fail_dob":
            # This error will be caught by Pydantic during IngestionClaim model creation if dob_val is not a valid date string
            # or during the _map_to_db_model if date.fromisoformat fails.
            # DataIngestionService's _map_to_db_model catches general Exception.
            assert "Error mapping IngestionClaim to ClaimModel" in err["error"] or "Invalid isoformat string" in err["error"]
```
