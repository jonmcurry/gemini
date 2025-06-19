import pytest
from decimal import Decimal
from datetime import date, datetime
from unittest.mock import patch, MagicMock # Added MagicMock

from claims_processor.src.processing.rvu_service import RVUService # MEDICARE_CONVERSION_FACTOR can be imported if needed
from claims_processor.src.api.models.claim_models import ProcessableClaim, ProcessableClaimLineItem
from claims_processor.src.core.cache.cache_manager import CacheManager # For mock cache manager

# Test RVU data similar to what CSV would provide
TEST_RVU_DATA_MAP = {
    "99213": Decimal("2.11"),
    "99214": Decimal("3.28"),
    "80053": Decimal("0.85"),
    "DEFAULT_RVU": Decimal("1.00")
}

@pytest.fixture
def mock_cache_manager():
    mcm = MagicMock(spec=CacheManager)
    # Configure the async methods of the mock.
    # For simple cases, MagicMock might handle awaitables, but explicit async mocks are better.
    # However, 'get' and 'set' in CacheManager are async. MagicMock needs configuring for this.
    # A simple way is to make the mock methods themselves async def functions or assign Coroutine Mocks.
    # For now, let's assume MagicMock's default awaitable behavior is sufficient or adjust if tests fail.
    # A common pattern:
    async def async_magic_mock(*args, **kwargs):
        # This function can be used as a side_effect or directly assigned to mock async methods
        if mcm.get is args[0].obj: # Check if this is the 'get' method
             return mcm.get.return_value # Return the configured return_value for get
        return MagicMock() # Default for other async calls if any

    mcm.get = MagicMock(return_value=None) # Default: cache miss
    mcm.set = MagicMock()
    # If methods were async, they'd need to be AsyncMock or similar in Python 3.8+
    # For now, let's test and see. If 'await mcm.get()' fails, this needs refinement.
    # It's likely fine as MagicMock tries to fake awaitables.
    return mcm

@pytest.fixture
@patch('claims_processor.src.processing.rvu_service.RVUService._load_rvu_data_from_csv', return_value=TEST_RVU_DATA_MAP)
def rvu_service(_mock_load_data: MagicMock, mock_cache_manager: MagicMock) -> RVUService:
    service = RVUService(cache_manager=mock_cache_manager)
    return service

@pytest.fixture
def sample_line_items_for_rvu() -> list[ProcessableClaimLineItem]:
    # Using datetime.now() for created_at/updated_at for simplicity in test data
    now = datetime.now()
    return [
        ProcessableClaimLineItem(id=10, claim_db_id=1, line_number=1, service_date=date.today(), procedure_code="99213", units=1, charge_amount=Decimal("100"), created_at=now, updated_at=now, rvu_total=None),
        ProcessableClaimLineItem(id=11, claim_db_id=1, line_number=2, service_date=date.today(), procedure_code="80053", units=2, charge_amount=Decimal("50"), created_at=now, updated_at=now, rvu_total=None),
        ProcessableClaimLineItem(id=12, claim_db_id=1, line_number=3, service_date=date.today(), procedure_code="UNKNOWN_PROC", units=1, charge_amount=Decimal("75"), created_at=now, updated_at=now, rvu_total=None),
    ]

# Renamed fixtures for clarity to avoid collision if old ones were somehow still referenced
@pytest.fixture
def sample_line_items_for_rvu_tests() -> list[ProcessableClaimLineItem]:
    now = datetime.now()
    return [
        ProcessableClaimLineItem(id=10, claim_db_id=1, line_number=1, service_date=date.today(), procedure_code="99213", units=1, charge_amount=Decimal("100"), created_at=now, updated_at=now, rvu_total=None),
        ProcessableClaimLineItem(id=11, claim_db_id=1, line_number=2, service_date=date.today(), procedure_code="80053", units=2, charge_amount=Decimal("50"), created_at=now, updated_at=now, rvu_total=None),
        ProcessableClaimLineItem(id=12, claim_db_id=1, line_number=3, service_date=date.today(), procedure_code="UNKNOWN_PROC", units=1, charge_amount=Decimal("75"), created_at=now, updated_at=now, rvu_total=None), # Will use DEFAULT_RVU
    ]

@pytest.fixture
def sample_claim_for_rvu_tests(sample_line_items_for_rvu_tests) -> ProcessableClaim: # Renamed
    return ProcessableClaim(
        id=1, claim_id="RVUTEST001", facility_id="F001", patient_account_number="P001",
        service_from_date=date.today(), service_to_date=date.today(),
        total_charges=sum(li.charge_amount for li in sample_line_items_for_rvu_tests),
        processing_status="validation_passed", created_at=datetime.now(), updated_at=datetime.now(),
        line_items=sample_line_items_for_rvu_tests
    )

@pytest.mark.asyncio
async def test_calculate_rvu_cache_miss_then_hit(rvu_service: RVUService, mock_cache_manager: MagicMock, sample_claim_for_rvu_tests: ProcessableClaim):
    # First call: Cache miss for all items
    # mock_cache_manager.get is already configured to return None by default in the fixture
    await rvu_service.calculate_rvu_for_claim(sample_claim_for_rvu_tests, db_session=None)

    line1 = sample_claim_for_rvu_tests.line_items[0]
    assert line1.rvu_total == TEST_RVU_DATA_MAP["99213"] * Decimal(line1.units)
    mock_cache_manager.set.assert_any_call(f"rvu:{line1.procedure_code}", str(TEST_RVU_DATA_MAP[line1.procedure_code]), ttl=3600)

    # Second call: Cache hit for items set in the first call
    for line_item in sample_claim_for_rvu_tests.line_items:
        line_item.rvu_total = None # Reset for a clean second run calculation

    # Configure mock_cache_manager.get to simulate cache hits for existing items
    # This needs to be an async function if the real one is.
    # For MagicMock, assigning a function to side_effect that returns an awaitable (like a raw value)
    # or an async function itself is the way to go.
    async def get_side_effect_async(key_bytes): # aiomcache takes bytes
        key_str = key_bytes.decode('utf-8')
        if key_str == "rvu:99213": return str(TEST_RVU_DATA_MAP["99213"]).encode('utf-8')
        if key_str == "rvu:80053": return str(TEST_RVU_DATA_MAP["80053"]).encode('utf-8')
        if key_str == "rvu:UNKNOWN_PROC": return str(TEST_RVU_DATA_MAP["DEFAULT_RVU"]).encode('utf-8')
        return None # Default for any other key

    # If mcm.get was defined as `async def get(...)` in CacheManager, MagicMock might handle it.
    # If not, we need to ensure the mock's get is awaitable.
    # Let's assume direct assignment of an async func to .get works for this test context.
    # Or, if mcm.get is a MagicMock instance itself:
    mock_cache_manager.get.side_effect = get_side_effect_async

    mock_cache_manager.set.reset_mock()

    await rvu_service.calculate_rvu_for_claim(sample_claim_for_rvu_tests, db_session=None)

    line1_after_cache_hit = sample_claim_for_rvu_tests.line_items[0]
    assert line1_after_cache_hit.rvu_total == TEST_RVU_DATA_MAP["99213"] * Decimal(line1_after_cache_hit.units)

    # Verify cache_manager.set was NOT called for any of these items on the second run
    assert mock_cache_manager.set.call_count == 0, "Cache set should not have been called if all items were cache hits."

@pytest.mark.asyncio
async def test_calculate_rvu_for_claim_values_from_mocked_loader(rvu_service: RVUService, mock_cache_manager: MagicMock, sample_claim_for_rvu_tests: ProcessableClaim):
    mock_cache_manager.get.return_value = None # Ensure all are cache misses

    await rvu_service.calculate_rvu_for_claim(sample_claim_for_rvu_tests, db_session=None)

    line1 = sample_claim_for_rvu_tests.line_items[0]
    line2 = sample_claim_for_rvu_tests.line_items[1]
    line3 = sample_claim_for_rvu_tests.line_items[2]

    assert line1.rvu_total == TEST_RVU_DATA_MAP["99213"] * Decimal(line1.units)
    assert line2.rvu_total == TEST_RVU_DATA_MAP["80053"] * Decimal(line2.units)
    assert line3.rvu_total == TEST_RVU_DATA_MAP["DEFAULT_RVU"] * Decimal(line3.units)

    mock_cache_manager.set.assert_any_call(f"rvu:{line1.procedure_code}", str(TEST_RVU_DATA_MAP[line1.procedure_code]), ttl=3600)
    mock_cache_manager.set.assert_any_call(f"rvu:{line2.procedure_code}", str(TEST_RVU_DATA_MAP[line2.procedure_code]), ttl=3600)
    mock_cache_manager.set.assert_any_call(f"rvu:{line3.procedure_code}", str(TEST_RVU_DATA_MAP["DEFAULT_RVU"]), ttl=3600)

@pytest.mark.asyncio
async def test_calculate_rvu_no_line_items(rvu_service: RVUService, sample_claim_for_rvu_tests: ProcessableClaim):
    claim_copy = sample_claim_for_rvu_tests.model_copy(deep=True)
    claim_copy.line_items = []
    await rvu_service.calculate_rvu_for_claim(claim_copy, db_session=None)
    assert not claim_copy.line_items
