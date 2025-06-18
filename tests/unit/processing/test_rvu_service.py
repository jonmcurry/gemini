import pytest
from decimal import Decimal
from datetime import date, datetime
from claims_processor.src.processing.rvu_service import RVUService, MEDICARE_CONVERSION_FACTOR, MOCK_RVU_DATA
from claims_processor.src.api.models.claim_models import ProcessableClaim, ProcessableClaimLineItem

@pytest.fixture
def rvu_service() -> RVUService:
    return RVUService()

@pytest.fixture
def sample_line_items_for_rvu() -> list[ProcessableClaimLineItem]:
    # Using datetime.now() for created_at/updated_at for simplicity in test data
    now = datetime.now()
    return [
        ProcessableClaimLineItem(id=10, claim_db_id=1, line_number=1, service_date=date.today(), procedure_code="99213", units=1, charge_amount=Decimal("100"), created_at=now, updated_at=now, rvu_total=None),
        ProcessableClaimLineItem(id=11, claim_db_id=1, line_number=2, service_date=date.today(), procedure_code="80053", units=2, charge_amount=Decimal("50"), created_at=now, updated_at=now, rvu_total=None),
        ProcessableClaimLineItem(id=12, claim_db_id=1, line_number=3, service_date=date.today(), procedure_code="UNKNOWN_PROC", units=1, charge_amount=Decimal("75"), created_at=now, updated_at=now, rvu_total=None),
    ]

@pytest.fixture
def sample_claim_for_rvu(sample_line_items_for_rvu) -> ProcessableClaim:
    # A minimal ProcessableClaim needed for RVU testing
    return ProcessableClaim(
        id=1, claim_id="RVUTEST001", facility_id="F001", patient_account_number="P001",
        service_from_date=date.today(), service_to_date=date.today(),
        total_charges=sum(li.charge_amount for li in sample_line_items_for_rvu), # total_charges not directly used by RVU service but good to be consistent
        processing_status="validation_passed", created_at=datetime.now(), updated_at=datetime.now(),
        line_items=sample_line_items_for_rvu
    )


@pytest.mark.asyncio # RVU service method is async currently
async def test_calculate_rvu_for_claim_known_codes(rvu_service: RVUService, sample_claim_for_rvu: ProcessableClaim):
    await rvu_service.calculate_rvu_for_claim(sample_claim_for_rvu, db_session=None) # db_session not used by mock

    # Line 1: 99213, units 1
    assert sample_claim_for_rvu.line_items[0].procedure_code == "99213"
    expected_rvu_99213 = MOCK_RVU_DATA["99213"] * Decimal("1")
    assert sample_claim_for_rvu.line_items[0].rvu_total == expected_rvu_99213

    # Line 2: 80053, units 2
    assert sample_claim_for_rvu.line_items[1].procedure_code == "80053"
    expected_rvu_80053 = MOCK_RVU_DATA["80053"] * Decimal("2")
    assert sample_claim_for_rvu.line_items[1].rvu_total == expected_rvu_80053

    # Line 3: UNKNOWN_PROC, units 1 (should use default)
    assert sample_claim_for_rvu.line_items[2].procedure_code == "UNKNOWN_PROC"
    expected_rvu_unknown = MOCK_RVU_DATA["DEFAULT_RVU"] * Decimal("1")
    assert sample_claim_for_rvu.line_items[2].rvu_total == expected_rvu_unknown

@pytest.mark.asyncio
async def test_calculate_rvu_no_line_items(rvu_service: RVUService, sample_claim_for_rvu: ProcessableClaim):
    # Create a new claim instance or deepcopy for modification if fixtures are session-scoped and reused.
    # For function-scoped fixtures, direct modification is fine.
    claim_without_lines = sample_claim_for_rvu.model_copy(deep=True) # Pydantic v2 copy
    claim_without_lines.line_items = []

    await rvu_service.calculate_rvu_for_claim(claim_without_lines, db_session=None)
    # No error should occur, and no RVUs calculated.
    # The method logs a warning, which we can't easily check here without caplog fixture from pytest.
    # Just ensure it runs without error and line_items remain empty.
    assert not claim_without_lines.line_items

@pytest.mark.asyncio
async def test_calculate_rvu_line_item_units_decimal_conversion(rvu_service: RVUService):
    # Test if units are correctly handled (e.g. if they were string representations of numbers)
    # The RVUService currently uses Decimal(units) for multiplication.
    # ProcessableClaimLineItem defines units as int. If it were str, this test would be more relevant.
    # For now, this test confirms that multiplication with int units works as expected.
    now = datetime.now()
    line_item_test_units = ProcessableClaimLineItem(
        id=20, claim_db_id=2, line_number=1, service_date=date.today(),
        procedure_code="99213", units=3, # Integer units
        charge_amount=Decimal("300"), created_at=now, updated_at=now, rvu_total=None
    )
    claim_for_rvu_units_test = ProcessableClaim(
        id=2, claim_id="RVUUNITS001", facility_id="F002", patient_account_number="P002",
        service_from_date=date.today(), service_to_date=date.today(), total_charges=Decimal("300"),
        processing_status="validation_passed", created_at=now, updated_at=now,
        line_items=[line_item_test_units]
    )

    await rvu_service.calculate_rvu_for_claim(claim_for_rvu_units_test, db_session=None)

    expected_rvu = MOCK_RVU_DATA["99213"] * Decimal("3")
    assert claim_for_rvu_units_test.line_items[0].rvu_total == expected_rvu
    assert isinstance(claim_for_rvu_units_test.line_items[0].rvu_total, Decimal)
