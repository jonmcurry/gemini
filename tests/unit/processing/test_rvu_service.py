import pytest
from decimal import Decimal
from datetime import date, datetime
from unittest.mock import MagicMock, AsyncMock

from claims_processor.src.processing.rvu_service import RVUService, DEFAULT_RVU_VALUE
from claims_processor.src.api.models.claim_models import ProcessableClaim, ProcessableClaimLineItem
from claims_processor.src.core.cache.cache_manager import CacheManager
from claims_processor.src.core.monitoring.app_metrics import MetricsCollector
from sqlalchemy.ext.asyncio import AsyncSession


# Test RVU data that would be returned by DB mock
MOCK_DB_RVU_DATA = {
    "99213": Decimal("2.11"),
    "99214": Decimal("3.28"),
    "80053": Decimal("0.85"),
    # UNKNOWN_PROC will not be in this map, should use DEFAULT_RVU_VALUE
}


@pytest.fixture
def mock_cache_manager() -> MagicMock:
    mcm = MagicMock(spec=CacheManager)
    mcm.get = AsyncMock(return_value=None) # Default: cache miss
    mcm.set = AsyncMock()
    return mcm

@pytest.fixture
def mock_metrics_collector() -> MagicMock:
    mmc = MagicMock(spec=MetricsCollector)
    # Mock the time_db_query context manager
    mock_timer = MagicMock()
    mock_timer.__enter__ = MagicMock(return_value=None) # __enter__ should return the context
    mock_timer.__exit__ = MagicMock(return_value=False) # __exit__ takes exc_type, exc_val, exc_tb
    mmc.time_db_query.return_value = mock_timer
    return mmc

@pytest.fixture
def mock_db_session() -> MagicMock:
    session = MagicMock(spec=AsyncSession)
    # Mock the execute method to return an object that has a scalar_one_or_none async method
    mock_execute_result = AsyncMock()

    async def scalar_one_or_none_side_effect(*args, **kwargs):
        # This side effect needs to know which procedure code is being queried
        # The actual select statement is not available here directly,
        # so we rely on the order of calls or more complex mocking if needed.
        # For now, let's assume test setup will make this clear.
        # This will be set by individual tests.
        return getattr(mock_execute_result, "current_rvu_value", None)

    mock_execute_result.scalar_one_or_none = AsyncMock(side_effect=scalar_one_or_none_side_effect)
    session.execute = AsyncMock(return_value=mock_execute_result)
    return session

@pytest.fixture
def rvu_service(mock_cache_manager: MagicMock, mock_metrics_collector: MagicMock) -> RVUService:
    # RVUService no longer loads from CSV, so no patching of _load_rvu_data_from_csv
    service = RVUService(cache_manager=mock_cache_manager, metrics_collector=mock_metrics_collector)
    return service

@pytest.fixture
def sample_line_items_for_rvu_tests() -> list[ProcessableClaimLineItem]:
    now = datetime.now()
    return [
        ProcessableClaimLineItem(id=10, claim_db_id=1, line_number=1, service_date=date.today(), procedure_code="99213", units=1, charge_amount=Decimal("100"), created_at=now, updated_at=now, rvu_total=None),
        ProcessableClaimLineItem(id=11, claim_db_id=1, line_number=2, service_date=date.today(), procedure_code="80053", units=2, charge_amount=Decimal("50"), created_at=now, updated_at=now, rvu_total=None),
        ProcessableClaimLineItem(id=12, claim_db_id=1, line_number=3, service_date=date.today(), procedure_code="UNKNOWN_PROC", units=1, charge_amount=Decimal("75"), created_at=now, updated_at=now, rvu_total=None),
    ]

@pytest.fixture
def sample_claim_for_rvu_tests(sample_line_items_for_rvu_tests: list[ProcessableClaimLineItem]) -> ProcessableClaim:
    return ProcessableClaim(
        id=1, claim_id="RVUTEST001", facility_id="F001", patient_account_number="P001",
        service_from_date=date.today(), service_to_date=date.today(),
        total_charges=sum(li.charge_amount for li in sample_line_items_for_rvu_tests),
        processing_status="validation_passed", created_at=datetime.now(), updated_at=datetime.now(),
        line_items=sample_line_items_for_rvu_tests
    )

@pytest.mark.asyncio
async def test_calculate_rvu_cache_miss_db_hit(
    rvu_service: RVUService,
    mock_cache_manager: MagicMock,
    mock_db_session: MagicMock,
    mock_metrics_collector: MagicMock,
    sample_claim_for_rvu_tests: ProcessableClaim
):
    mock_cache_manager.get.return_value = None # Cache miss for all

    # Configure DB mock responses
    def db_side_effect(stmt): # stmt is the SQLAlchemy select object
        # This is a simplified way to check which procedure code is being queried.
        # A more robust mock would parse the statement.
        # For this test, we assume codes are queried in order of line items.
        proc_code_being_queried = None
        if "99213" in str(stmt.compile(compile_kwargs={"literal_binds": True})):
            proc_code_being_queried = "99213"
        elif "80053" in str(stmt.compile(compile_kwargs={"literal_binds": True})):
            proc_code_being_queried = "80053"
        elif "UNKNOWN_PROC" in str(stmt.compile(compile_kwargs={"literal_binds": True})):
            proc_code_being_queried = "UNKNOWN_PROC" # Will return None from DB

        mock_execute_result = AsyncMock()
        mock_execute_result.scalar_one_or_none = AsyncMock(return_value=MOCK_DB_RVU_DATA.get(proc_code_being_queried))
        return mock_execute_result

    mock_db_session.execute.side_effect = db_side_effect

    await rvu_service.calculate_rvu_for_claim(sample_claim_for_rvu_tests, db_session=mock_db_session)

    line1, line2, line3 = sample_claim_for_rvu_tests.line_items

    assert line1.rvu_total == MOCK_DB_RVU_DATA["99213"] * Decimal(line1.units)
    assert line2.rvu_total == MOCK_DB_RVU_DATA["80053"] * Decimal(line2.units)
    assert line3.rvu_total == DEFAULT_RVU_VALUE * Decimal(line3.units) # Fell back to default

    # Cache assertions
    assert mock_cache_manager.get.call_count == 3 # One for each line item
    mock_cache_manager.set.assert_any_call(f"rvu:{line1.procedure_code}", str(MOCK_DB_RVU_DATA[line1.procedure_code]), ttl=3600)
    mock_cache_manager.set.assert_any_call(f"rvu:{line2.procedure_code}", str(MOCK_DB_RVU_DATA[line2.procedure_code]), ttl=3600)
    mock_cache_manager.set.assert_any_call(f"rvu:{line3.procedure_code}", str(DEFAULT_RVU_VALUE), ttl=3600) # Default was cached

    # Metrics assertions
    # Cache operations: 3 misses (initial get), then potentially sets (not directly measured by record_cache_operation here)
    # The service logic calls record_cache_operation for 'get' with 'miss' outcome.
    mock_metrics_collector.record_cache_operation.assert_any_call(cache_type='rvu', operation_type='get', outcome='miss')
    assert mock_metrics_collector.record_cache_operation.call_count == 3 # 3 misses for 'get'

    # DB operations: 3 calls to _get_rvu_from_db (one for each line item due to cache misses)
    assert mock_db_session.execute.call_count == 3
    assert mock_metrics_collector.time_db_query.call_count == 3
    mock_metrics_collector.time_db_query.assert_called_with("fetch_rvu_data_by_procedure_code")


@pytest.mark.asyncio
async def test_calculate_rvu_cache_hit(
    rvu_service: RVUService,
    mock_cache_manager: MagicMock,
    mock_db_session: MagicMock,
    mock_metrics_collector: MagicMock,
    sample_claim_for_rvu_tests: ProcessableClaim
):
    # Configure cache to return hits for all items
    async def cache_get_side_effect(key_bytes):
        key = key_bytes.decode('utf-8')
        proc_code = key.split(":")[1]
        if proc_code == "99213": return str(MOCK_DB_RVU_DATA["99213"]).encode('utf-8')
        if proc_code == "80053": return str(MOCK_DB_RVU_DATA["80053"]).encode('utf-8')
        if proc_code == "UNKNOWN_PROC": return str(DEFAULT_RVU_VALUE).encode('utf-8') # Default was cached
        return None
    mock_cache_manager.get.side_effect = cache_get_side_effect

    await rvu_service.calculate_rvu_for_claim(sample_claim_for_rvu_tests, db_session=mock_db_session)

    line1, line2, line3 = sample_claim_for_rvu_tests.line_items
    assert line1.rvu_total == MOCK_DB_RVU_DATA["99213"] * Decimal(line1.units)
    assert line2.rvu_total == MOCK_DB_RVU_DATA["80053"] * Decimal(line2.units)
    assert line3.rvu_total == DEFAULT_RVU_VALUE * Decimal(line3.units)

    # DB should not be called
    mock_db_session.execute.assert_not_called()

    # Cache set should not be called
    mock_cache_manager.set.assert_not_called()

    # Metrics
    mock_metrics_collector.record_cache_operation.assert_any_call(cache_type='rvu', operation_type='get', outcome='hit')
    assert mock_metrics_collector.record_cache_operation.call_count == 3 # 3 hits for 'get'
    mock_metrics_collector.time_db_query.assert_not_called()


@pytest.mark.asyncio
async def test_calculate_rvu_cache_parse_error(
    rvu_service: RVUService,
    mock_cache_manager: MagicMock,
    mock_db_session: MagicMock,
    mock_metrics_collector: MagicMock,
    sample_claim_for_rvu_tests: ProcessableClaim
):
    # Line 1 will have a cache parse error, Line 2 normal DB hit, Line 3 normal cache hit (for default)
    async def cache_get_side_effect(key_bytes):
        key = key_bytes.decode('utf-8')
        if key.endswith("99213"): return b"invalid-decimal" # Parse error
        if key.endswith("UNKNOWN_PROC"): return str(DEFAULT_RVU_VALUE).encode('utf-8') # Cache hit for default
        return None # Miss for 80053
    mock_cache_manager.get.side_effect = cache_get_side_effect

    # DB responses for cache misses/errors
    def db_side_effect(stmt):
        mock_execute_result = AsyncMock()
        if "99213" in str(stmt.compile(compile_kwargs={"literal_binds": True})): # Queried after parse error
            mock_execute_result.scalar_one_or_none = AsyncMock(return_value=MOCK_DB_RVU_DATA["99213"])
        elif "80053" in str(stmt.compile(compile_kwargs={"literal_binds": True})): # Queried after cache miss
            mock_execute_result.scalar_one_or_none = AsyncMock(return_value=MOCK_DB_RVU_DATA["80053"])
        else: # UNKNOWN_PROC should be a cache hit
            mock_execute_result.scalar_one_or_none = AsyncMock(return_value=None)
        return mock_execute_result
    mock_db_session.execute.side_effect = db_side_effect

    await rvu_service.calculate_rvu_for_claim(sample_claim_for_rvu_tests, db_session=mock_db_session)

    line1, line2, line3 = sample_claim_for_rvu_tests.line_items
    assert line1.rvu_total == MOCK_DB_RVU_DATA["99213"] * Decimal(line1.units) # Fetched from DB
    assert line2.rvu_total == MOCK_DB_RVU_DATA["80053"] * Decimal(line2.units) # Fetched from DB
    assert line3.rvu_total == DEFAULT_RVU_VALUE * Decimal(line3.units)       # From cache

    # Metrics
    # 99213: get -> parse_error, then another 'get' -> 'miss' (internal to logic before DB call)
    # 80053: get -> miss
    # UNKNOWN_PROC: get -> hit
    mock_metrics_collector.record_cache_operation.assert_any_call(cache_type='rvu', operation_type='get', outcome='parse_error') # For 99213
    mock_metrics_collector.record_cache_operation.assert_any_call(cache_type='rvu', operation_type='get', outcome='miss') # For 99213 (after parse error) AND 80053
    mock_metrics_collector.record_cache_operation.assert_any_call(cache_type='rvu', operation_type='get', outcome='hit')    # For UNKNOWN_PROC
    # Total calls: 1 parse_error, 2 misses, 1 hit
    assert mock_metrics_collector.record_cache_operation.call_count == (1 + 2 + 1)

    # DB calls for 99213 (after parse error) and 80053 (after miss)
    assert mock_db_session.execute.call_count == 2
    assert mock_metrics_collector.time_db_query.call_count == 2
    mock_metrics_collector.time_db_query.assert_called_with("fetch_rvu_data_by_procedure_code")


@pytest.mark.asyncio
async def test_calculate_rvu_no_line_items(
    rvu_service: RVUService,
    mock_db_session: MagicMock,
    mock_metrics_collector: MagicMock,
    sample_claim_for_rvu_tests: ProcessableClaim
):
    claim_copy = sample_claim_for_rvu_tests.model_copy(deep=True)
    claim_copy.line_items = []
    await rvu_service.calculate_rvu_for_claim(claim_copy, db_session=mock_db_session)
    assert not claim_copy.line_items
    mock_metrics_collector.record_cache_operation.assert_not_called()
    mock_metrics_collector.time_db_query.assert_not_called()
