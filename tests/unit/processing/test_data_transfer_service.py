import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone, date
from decimal import Decimal

from claims_processor.src.processing.data_transfer_service import DataTransferService
from claims_processor.src.core.database.models.claims_db import ClaimModel
from claims_processor.src.core.database.models.claims_production_db import ClaimsProductionModel


@pytest.fixture
def mock_db_session() -> MagicMock:
    session = MagicMock(spec=AsyncSession)
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.add = MagicMock() # Added from my plan
    session.add_all = MagicMock() # Added from my plan
    # For 'async with session.begin()'
    async_cm = AsyncMock()
    session.begin = MagicMock(return_value=async_cm)
    return session

@pytest.fixture
def mock_settings_fixture(): # Renamed to avoid conflict with a potential 'settings' variable
    settings = MagicMock()
    settings.TRANSFER_BATCH_SIZE = 100
    # Add other settings if DataTransferService uses them directly
    return settings

@pytest.fixture
def data_transfer_service(mock_db_session: MagicMock, mock_settings_fixture: MagicMock) -> DataTransferService:
    # DataTransferService now initializes self.settings = get_settings()
    with patch('claims_processor.src.processing.data_transfer_service.get_settings', return_value=mock_settings_fixture):
        service = DataTransferService(db_session=mock_db_session)
    return service

# --- Test _select_claims_from_staging ---
@pytest.mark.asyncio
async def test_select_claims_from_staging_fetches_correct_claims(
    data_transfer_service: DataTransferService,
    mock_db_session: MagicMock
):
    mock_claim_1 = ClaimModel(id=1, claim_id="C1", processing_status="processing_complete", transferred_to_prod_at=None, created_at=datetime.now(timezone.utc))

    mock_result_proxy = MagicMock() # Mock for the result of execute
    mock_result_proxy.scalars.return_value.all.return_value = [mock_claim_1]
    mock_db_session.execute.return_value = mock_result_proxy

    selected_claims = await data_transfer_service._select_claims_from_staging(limit=10)

    mock_db_session.execute.assert_called_once()
    assert len(selected_claims) == 1
    assert selected_claims[0].claim_id == "C1"

@pytest.mark.asyncio
async def test_select_claims_from_staging_no_claims_ready(
    data_transfer_service: DataTransferService,
    mock_db_session: MagicMock
):
    mock_result_proxy = MagicMock()
    mock_result_proxy.scalars.return_value.all.return_value = []
    mock_db_session.execute.return_value = mock_result_proxy

    selected_claims = await data_transfer_service._select_claims_from_staging(limit=10)
    assert len(selected_claims) == 0

# --- Test _map_staging_to_production_records ---
def test_map_staging_to_production_records_maps_correctly(
    data_transfer_service: DataTransferService
):
    staging_claim = ClaimModel(
        id=1, claim_id="C1_MAP", facility_id="F001", patient_account_number="P001",
        patient_first_name="Test", patient_last_name="User", patient_date_of_birth=date(1990,1,1),
        service_from_date=date(2023,1,1), service_to_date=date(2023,1,5),
        total_charges=Decimal("120.50"),
        ml_score=Decimal("0.95"), ml_derived_decision="ML_APPROVED",
        processing_duration_ms=550,
        created_at=datetime.now(timezone.utc), # Required by model
        updated_at=datetime.now(timezone.utc)  # Required by model
    )
    mapped_records = data_transfer_service._map_staging_to_production_records([staging_claim])

    assert len(mapped_records) == 1
    prod_rec = mapped_records[0]

    assert prod_rec["id"] == staging_claim.id
    assert prod_rec["claim_id"] == staging_claim.claim_id
    assert prod_rec["ml_prediction_score"] == staging_claim.ml_score
    assert prod_rec["risk_category"] == "LOW" # Based on 0.95
    assert prod_rec["processing_duration_ms"] == staging_claim.processing_duration_ms
    assert prod_rec["throughput_achieved"] is None
    # Additional assertions from my planned test
    assert prod_rec["patient_first_name"] == "Test"
    assert prod_rec["patient_last_name"] == "User"
    # Assuming patient_date_of_birth is mapped directly as string if it was string in ClaimModel
    # If ClaimModel.patient_date_of_birth is date, and mapping keeps it as date, this is fine.
    # The current ClaimModel in data_transfer_service uses it as date, so mapping should be date.
    assert prod_rec["patient_date_of_birth"] == date(1990,1,1)
    assert prod_rec["service_from_date"] == date(2023,1,1)
    assert prod_rec["service_to_date"] == date(2023,1,5)

def test_map_staging_to_production_empty_list(data_transfer_service: DataTransferService):
    assert data_transfer_service._map_staging_to_production_records([]) == []

def test_map_staging_to_production_records_risk_categories( # Name kept from existing
    data_transfer_service: DataTransferService
):
    claims_data = [
        {"id": 1, "ml_score": Decimal("0.9"), "expected_risk": "LOW"},
        {"id": 2, "ml_score": Decimal("0.7"), "expected_risk": "MEDIUM"},
        {"id": 3, "ml_score": Decimal("0.4"), "expected_risk": "HIGH"},
        {"id": 4, "ml_score": None, "expected_risk": "UNKNOWN"},
    ]
    now = datetime.now(timezone.utc)
    staging_claims = [
        ClaimModel(id=d["id"], claim_id=f"C{d['id']}", ml_score=d["ml_score"],
                   facility_id="F", patient_account_number="P",
                   service_from_date=date.today(), service_to_date=date.today(),
                   total_charges=Decimal(1), processing_status="processing_complete",
                   created_at=now, updated_at=now)
        for d in claims_data
    ]
    mapped_records = data_transfer_service._map_staging_to_production_records(staging_claims)
    for i, rec in enumerate(mapped_records):
        assert rec["risk_category"] == claims_data[i]["expected_risk"]

# --- Test _bulk_insert_to_production ---
@pytest.mark.asyncio
async def test_bulk_insert_to_production_success(
    data_transfer_service: DataTransferService,
    mock_db_session: MagicMock
):
    test_records = [{"claim_id": "B1"}, {"claim_id": "B2"}]

    count = await data_transfer_service._bulk_insert_to_production(test_records)

    assert count == 2
    mock_db_session.execute.assert_called_once()
    # To check arguments of execute:
    args, _ = mock_db_session.execute.call_args
    # args[0] is the insert statement, args[1] is the list of dicts
    assert str(args[0]).startswith("INSERT INTO claims_production") # Basic check
    assert args[1] == test_records
    mock_db_session.commit.assert_called_once()
    mock_db_session.rollback.assert_not_called()

@pytest.mark.asyncio
async def test_bulk_insert_to_production_failure(
    data_transfer_service: DataTransferService,
    mock_db_session: MagicMock
):
    test_records = [{"claim_id": "B_FAIL"}]
    mock_db_session.execute.side_effect = Exception("DB Insert Error")

    count = await data_transfer_service._bulk_insert_to_production(test_records)

    assert count == 0
    mock_db_session.execute.assert_called_once()
    mock_db_session.rollback.assert_called_once()
    mock_db_session.commit.assert_not_called()

@pytest.mark.asyncio
async def test_bulk_insert_no_records(data_transfer_service: DataTransferService, mock_db_session: MagicMock):
    count = await data_transfer_service._bulk_insert_to_production([])
    assert count == 0
    mock_db_session.execute.assert_not_called()

# --- Test _update_staging_claims_after_transfer ---
@pytest.mark.asyncio
async def test_update_staging_claims_success(
    data_transfer_service: DataTransferService,
    mock_db_session: MagicMock
):
    claims_to_update = [ClaimModel(id=1), ClaimModel(id=2)]
    mock_execute_result = MagicMock()
    mock_execute_result.rowcount = 2
    mock_db_session.execute.return_value = mock_execute_result

    await data_transfer_service._update_staging_claims_after_transfer(claims_to_update)

    mock_db_session.execute.assert_called_once()
    mock_db_session.commit.assert_called_once()
    mock_db_session.rollback.assert_not_called()

@pytest.mark.asyncio
async def test_update_staging_claims_failure(
    data_transfer_service: DataTransferService,
    mock_db_session: MagicMock
):
    claims_to_update = [ClaimModel(id=1)]
    mock_db_session.execute.side_effect = Exception("DB Update Error")

    await data_transfer_service._update_staging_claims_after_transfer(claims_to_update)

    mock_db_session.execute.assert_called_once()
    mock_db_session.rollback.assert_called_once()
    mock_db_session.commit.assert_not_called()

@pytest.mark.asyncio
async def test_update_staging_no_claims(data_transfer_service: DataTransferService, mock_db_session: MagicMock):
    await data_transfer_service._update_staging_claims_after_transfer([])
    mock_db_session.execute.assert_not_called()

# --- Test transfer_claims_to_production (Orchestration) ---
@pytest.mark.asyncio
@patch.object(DataTransferService, '_select_claims_from_staging', new_callable=AsyncMock)
@patch.object(DataTransferService, '_map_staging_to_production_records')
@patch.object(DataTransferService, '_bulk_insert_to_production', new_callable=AsyncMock)
@patch.object(DataTransferService, '_update_staging_claims_after_transfer', new_callable=AsyncMock)
async def test_transfer_claims_orchestration_success(
    mock_update_staging: MagicMock,
    mock_bulk_insert: MagicMock,
    mock_map_records: MagicMock,
    mock_select_claims: MagicMock,
    data_transfer_service: DataTransferService
):
    mock_staging_claims = [ClaimModel(id=1, claim_id="C1_FULL")]
    mock_production_dicts = [{"claim_id": "C1_FULL_PROD"}]

    mock_select_claims.return_value = mock_staging_claims
    mock_map_records.return_value = mock_production_dicts
    mock_bulk_insert.return_value = len(mock_production_dicts)

    result = await data_transfer_service.transfer_claims_to_production(limit=5)

    mock_select_claims.assert_called_once_with(5)
    mock_map_records.assert_called_once_with(mock_staging_claims)
    mock_bulk_insert.assert_called_once_with(mock_production_dicts)
    mock_update_staging.assert_called_once_with(mock_staging_claims)

    assert result["successfully_transferred"] == len(mock_production_dicts)
    assert result["selected_from_staging"] == len(mock_staging_claims)
    assert result["mapped_to_production_format"] == len(mock_production_dicts)

@pytest.mark.asyncio
@patch.object(DataTransferService, '_select_claims_from_staging', new_callable=AsyncMock)
async def test_transfer_claims_orchestration_no_claims_selected(
    mock_select_claims: MagicMock,
    data_transfer_service: DataTransferService
):
    mock_select_claims.return_value = []

    result = await data_transfer_service.transfer_claims_to_production(limit=5)

    mock_select_claims.assert_called_once_with(5)
    assert result["message"] == "No claims to transfer."
    assert result["successfully_transferred"] == 0
    assert result["selected_from_staging"] == 0

@pytest.mark.asyncio
@patch.object(DataTransferService, '_select_claims_from_staging', new_callable=AsyncMock)
@patch.object(DataTransferService, '_map_staging_to_production_records')
@patch.object(DataTransferService, '_bulk_insert_to_production', new_callable=AsyncMock)
@patch.object(DataTransferService, '_update_staging_claims_after_transfer', new_callable=AsyncMock)
async def test_transfer_claims_orchestration_insert_fails(
    mock_update_staging: MagicMock,
    mock_bulk_insert: MagicMock,
    mock_map_records: MagicMock,
    mock_select_claims: MagicMock,
    data_transfer_service: DataTransferService
):
    mock_staging_claims = [ClaimModel(id=1)]
    mock_production_dicts = [{"id":1}]

    mock_select_claims.return_value = mock_staging_claims
    mock_map_records.return_value = mock_production_dicts
    mock_bulk_insert.return_value = 0 # Simulate insert failure

    result = await data_transfer_service.transfer_claims_to_production(limit=5)

    mock_select_claims.assert_called_once()
    mock_map_records.assert_called_once()
    mock_bulk_insert.assert_called_once()
    mock_update_staging.assert_not_called()

    assert result["successfully_transferred"] == 0
