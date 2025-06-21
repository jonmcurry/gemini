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
    session.add = MagicMock()
    session.add_all = MagicMock()

    # For 'async with session.begin()'
    mock_transaction_cm = AsyncMock() # Context manager for session.begin()
    # __aenter__ on this CM should return the session itself if structure is `async with session.begin() as tx_session:`
    # but DataTransferService uses `async with self.db.begin():` without `as...`, so __aenter__ doesn't need specific return.
    # However, the operations are then on `self.db` (our `session` mock).
    mock_transaction_cm.__aexit__ = AsyncMock(return_value=None) # Default success for commit
    session.begin.return_value = mock_transaction_cm

    # Setup for asyncpg raw connection mocking: session.get_bind().get_raw_connection()
    mock_raw_pg_conn = AsyncMock()
    mock_raw_pg_conn.execute = AsyncMock(return_value="INSERT 0 1") # Default success
    mock_raw_pg_conn.copy_records_to_table = AsyncMock()
    mock_raw_pg_conn.is_closed = MagicMock(return_value=False)
    mock_raw_pg_conn.close = AsyncMock()

    mock_sqla_connection = AsyncMock() # Mock for SQLAlchemy Connection object
    mock_sqla_connection.get_raw_connection = AsyncMock(return_value=mock_raw_pg_conn)

    # To mock `await self.db.get_bind()` returning an object that has `get_raw_connection`
    # If `self.db` is directly the session, then `self.db.connection()` is the call.
    # Let's assume `self.db` is the session, so `self.db.get_bind()` is not standard.
    # `self.db.connection()` would return an `AsyncConnection` which then has `get_raw_connection()`
    # For testing, we can mock `get_bind()` if the code was `await self.db.get_bind().get_raw_connection()`
    # The actual code in _upsert_to_production_with_copy is `await self.db.get_bind()`
    # which implies self.db is an Engine-like object or the Session has get_bind().
    # Let's assume self.db (the session mock) should have get_bind() mocked.

    mock_engine_or_conn_facade = AsyncMock() # This is what get_bind() might return
    mock_engine_or_conn_facade.get_raw_connection = AsyncMock(return_value=mock_raw_pg_conn)
    session.get_bind = AsyncMock(return_value=mock_engine_or_conn_facade)

    session.mock_raw_pg_conn = mock_raw_pg_conn # Attach for easy access in tests
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
        ml_model_version_used="control:model_v1.2", # Added
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
    assert prod_rec["ml_model_version_used"] == staging_claim.ml_model_version_used # Added
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
                   ml_model_version_used=f"v_test_{d['id']}", # Added
                   facility_id="F", patient_account_number="P",
                   service_from_date=date.today(), service_to_date=date.today(),
                   total_charges=Decimal(1), processing_status="processing_complete",
                   created_at=now, updated_at=now)
        for d in claims_data
    ]
    mapped_records = data_transfer_service._map_staging_to_production_records(staging_claims)
    for i, rec in enumerate(mapped_records):
        assert rec["risk_category"] == claims_data[i]["expected_risk"]
        assert rec["ml_model_version_used"] == f"v_test_{claims_data[i]['id']}" # Added

# --- Tests for _format_data_for_copy ---
def test_format_data_for_copy_empty(data_transfer_service: DataTransferService):
    assert data_transfer_service._format_data_for_copy([], ["col1", "col2"]) == []

def test_format_data_for_copy_basic(data_transfer_service: DataTransferService):
    records = [
        {"id": 1, "name": "Alice", "value": 100},
        {"id": 2, "name": "Bob", "value": 200}
    ]
    column_names = ["id", "name", "value"]
    expected = [
        (1, "Alice", 100),
        (2, "Bob", 200)
    ]
    assert data_transfer_service._format_data_for_copy(records, column_names) == expected

def test_format_data_for_copy_missing_key(data_transfer_service: DataTransferService):
    records = [
        {"id": 1, "name": "Alice"}, # "value" is missing
        {"id": 2, "value": 200}   # "name" is missing
    ]
    column_names = ["id", "name", "value"]
    expected = [
        (1, "Alice", None),
        (2, None, 200)
    ]
    assert data_transfer_service._format_data_for_copy(records, column_names) == expected

def test_format_data_for_copy_different_order(data_transfer_service: DataTransferService):
    records = [{"id": 1, "name": "Alice", "value": 100}]
    column_names = ["name", "value", "id"] # Different order
    expected = [("Alice", 100, 1)]
    assert data_transfer_service._format_data_for_copy(records, column_names) == expected


# --- Tests for _upsert_to_production_with_copy ---
@pytest.mark.asyncio
async def test_upsert_with_copy_success(
    data_transfer_service: DataTransferService,
    mock_db_session: MagicMock # mock_db_session is the SQLAlchemy AsyncSession mock
):
    test_records = [
        {"id": 1, "claim_id": "B1", "ml_model_version_used": "control_v1"},
        {"id": 2, "claim_id": "B2", "ml_model_version_used": "challenger_v1"}
    ]
    # mock_db_session fixture is already enhanced with mock_raw_pg_conn
    mock_raw_pg_conn = mock_db_session.mock_raw_pg_conn
    mock_raw_pg_conn.execute.side_effect = [
        AsyncMock(return_value=None), # For CREATE TEMP TABLE
        AsyncMock(return_value="INSERT 0 2")  # For INSERT...SELECT, 2 rows affected
    ]
    mock_raw_pg_conn.copy_records_to_table = AsyncMock() # Reset for this test

    # Patch _format_data_for_copy to return a known value
    formatted_tuples = [(r['id'], r['claim_id'], Ellipsis) for r in test_records] # Simplified for example
    data_transfer_service._format_data_for_copy = MagicMock(return_value=formatted_tuples)

    count = await data_transfer_service._upsert_to_production_with_copy(test_records)

    assert count == 2
    data_transfer_service._format_data_for_copy.assert_called_once()
    # Check calls to asyncpg connection
    assert mock_raw_pg_conn.execute.call_count == 2

    # Call 1: CREATE TEMP TABLE
    create_temp_table_call = mock_raw_pg_conn.execute.call_args_list[0]
    assert "CREATE TEMP TABLE" in create_temp_table_call[0][0]
    assert "ON COMMIT DROP" in create_temp_table_call[0][0]

    # Call 2: INSERT ... SELECT
    insert_select_call = mock_raw_pg_conn.execute.call_args_list[1]
    assert "INSERT INTO claims_production" in insert_select_call[0][0]
    assert "ON CONFLICT (id) DO UPDATE" in insert_select_call[0][0]
    assert '"updated_at" = NOW()' in insert_select_call[0][0] # Check updated_at is set

    mock_raw_pg_conn.copy_records_to_table.assert_called_once_with(
        unittest.mock.ANY,
        records=formatted_tuples,
        columns=unittest.mock.ANY,
        timeout=60
    )

@pytest.mark.asyncio
async def test_upsert_with_copy_formatting_returns_empty(
    data_transfer_service: DataTransferService,
    mock_db_session: MagicMock
):
    test_records = [{"id": 1, "claim_id": "B1"}]
    data_transfer_service._format_data_for_copy = MagicMock(return_value=[]) # Formatter returns empty

    mock_raw_pg_conn = mock_db_session.mock_raw_pg_conn # Get from enhanced fixture

    count = await data_transfer_service._upsert_to_production_with_copy(test_records)

    assert count == 0
    # CREATE TEMP TABLE might still be called before format check, depending on implementation
    # Based on current code, get_bind is called first, then format.
    # If format returns empty, copy_records_to_table and subsequent execute are skipped.
    mock_raw_pg_conn.execute.assert_called_once() # Only for CREATE TEMP TABLE
    assert "CREATE TEMP TABLE" in mock_raw_pg_conn.execute.call_args[0][0]
    mock_raw_pg_conn.copy_records_to_table.assert_not_called()


@pytest.mark.asyncio
async def test_upsert_with_copy_copy_records_fails(
    data_transfer_service: DataTransferService,
    mock_db_session: MagicMock
):
    test_records = [{"id": 1, "claim_id": "B_FAIL_COPY"}]
    mock_raw_pg_conn = mock_db_session.mock_raw_pg_conn

    data_transfer_service._format_data_for_copy = MagicMock(return_value=[(1, "B_FAIL_COPY")])
    mock_raw_pg_conn.execute = AsyncMock() # First execute for CREATE TEMP is fine
    mock_raw_pg_conn.copy_records_to_table = AsyncMock(side_effect=RuntimeError("Asyncpg COPY failed"))

    with pytest.raises(RuntimeError, match="Asyncpg COPY failed"):
        await data_transfer_service._upsert_to_production_with_copy(test_records)

    mock_raw_pg_conn.execute.assert_called_once() # CREATE TEMP
    # close() might be called in the except block of the SUT
    # mock_raw_pg_conn.close.assert_called_once() # This depends on exact SUT error handling for raw conn

@pytest.mark.asyncio
async def test_upsert_with_copy_insert_select_fails(
    data_transfer_service: DataTransferService,
    mock_db_session: MagicMock
):
    test_records = [{"id": 1, "claim_id": "B_FAIL_INSERT"}]
    mock_raw_pg_conn = mock_db_session.mock_raw_pg_conn

    data_transfer_service._format_data_for_copy = MagicMock(return_value=[(1, "B_FAIL_INSERT")])
    mock_raw_pg_conn.copy_records_to_table = AsyncMock() # COPY is fine
    # First execute for CREATE TEMP, second for INSERT...SELECT
    mock_raw_pg_conn.execute.side_effect = [
        AsyncMock(return_value=None), # CREATE TEMP
        RuntimeError("Asyncpg INSERT...SELECT failed")
    ]

    with pytest.raises(RuntimeError, match="Asyncpg INSERT...SELECT failed"):
        await data_transfer_service._upsert_to_production_with_copy(test_records)

    assert mock_raw_pg_conn.execute.call_count == 2
    # mock_raw_pg_conn.close.assert_called_once() # This depends on SUT error handling

@pytest.mark.asyncio
async def test_upsert_with_copy_no_records(data_transfer_service: DataTransferService, mock_db_session: MagicMock): # Test name kept
    count = await data_transfer_service._upsert_to_production_with_copy([])
    assert count == 0
    mock_db_session.get_bind.assert_not_called()


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
@patch.object(DataTransferService, '_upsert_to_production_with_copy', new_callable=AsyncMock) # Changed target
@patch.object(DataTransferService, '_update_staging_claims_after_transfer', new_callable=AsyncMock)
async def test_transfer_claims_orchestration_success(
    mock_update_staging: MagicMock,
    mock_upsert_with_copy: MagicMock, # Changed name
    mock_map_records: MagicMock,
    mock_select_claims: MagicMock,
    data_transfer_service: DataTransferService
):
    mock_staging_claims = [
        ClaimModel(id=1, claim_id="C1_FULL", ml_model_version_used="control:model_v1.0")
    ]
    mock_production_dicts = [
        {"claim_id": "C1_FULL_PROD", "id":1, "ml_model_version_used": "control:model_v1.0"}
    ]

    mock_select_claims.return_value = mock_staging_claims
    mock_map_records.return_value = mock_production_dicts
    mock_upsert_with_copy.return_value = len(mock_production_dicts) # Changed name

    # Mock the db.begin() context manager for the main transaction
    mock_begin_cm = AsyncMock()
    data_transfer_service.db.begin.return_value = mock_begin_cm

    result = await data_transfer_service.transfer_claims_to_production(limit=5)

    mock_select_claims.assert_called_once_with(5)
    mock_map_records.assert_called_once_with(mock_staging_claims)
    mock_upsert_with_copy.assert_called_once_with(mock_production_dicts) # Changed name
    mock_update_staging.assert_called_once_with(mock_staging_claims)
    mock_begin_cm.__aenter__.assert_called_once() # Check transaction was started
    mock_begin_cm.__aexit__.assert_called_once()  # Check transaction was finalized

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
@patch.object(DataTransferService, '_upsert_to_production_with_copy', new_callable=AsyncMock) # Changed target
@patch.object(DataTransferService, '_update_staging_claims_after_transfer', new_callable=AsyncMock)
async def test_transfer_claims_orchestration_insert_fails(
    mock_update_staging: MagicMock,
    mock_upsert_with_copy: MagicMock, # Changed name
    mock_map_records: MagicMock,
    mock_select_claims: MagicMock,
    data_transfer_service: DataTransferService
):
    mock_staging_claims = [
        ClaimModel(id=1, claim_id="C1_FAIL", ml_model_version_used="control:model_v1.1")
    ]
    mock_production_dicts = [
        {"id":1, "claim_id": "C1_FAIL", "ml_model_version_used": "control:model_v1.1"}
    ]

    mock_select_claims.return_value = mock_staging_claims
    mock_map_records.return_value = mock_production_dicts
    mock_upsert_with_copy.return_value = 0 # Simulate insert failure # Changed name

    # Mock the db.begin() context manager
    mock_begin_cm = AsyncMock()
    data_transfer_service.db.begin.return_value = mock_begin_cm

    result = await data_transfer_service.transfer_claims_to_production(limit=5)

    mock_select_claims.assert_called_once()
    mock_map_records.assert_called_once()
    mock_upsert_with_copy.assert_called_once() # Changed name
    mock_update_staging.assert_not_called()
    mock_begin_cm.__aenter__.assert_called_once() # Transaction started
    # __aexit__ is still called even if we log warnings and don't update staging.
    # The transaction should still complete if the upsert itself didn't raise an error.
    mock_begin_cm.__aexit__.assert_called_once()


    assert result["successfully_transferred"] == 0
