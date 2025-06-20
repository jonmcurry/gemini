import pytest
from datetime import date, timedelta, datetime
from decimal import Decimal
from claims_processor.src.processing.validation.claim_validator import ClaimValidator
from claims_processor.src.api.models.claim_models import ProcessableClaim, ProcessableClaimLineItem

@pytest.fixture
def validator() -> ClaimValidator:
    return ClaimValidator()

@pytest.fixture
def basic_line_item_data() -> dict:
    return {
        "id": 101,
        "claim_db_id": 1,
        "line_number": 1,
        "service_date": date.today(),
        "procedure_code": "99213",
        "units": 1,
        "charge_amount": Decimal("150.00"),
        "rvu_total": None, # RVU not calculated yet
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
    }

@pytest.fixture
def basic_claim_data(basic_line_item_data) -> dict:
    today = date.today()
    return {
        "id": 1, # DB ID
        "claim_id": "VALID_CLAIM_001",
        "facility_id": "FAC001",
        "patient_account_number": "PAT001",
        "patient_first_name": "John",
        "patient_last_name": "Doe",
        "patient_date_of_birth": date(1980, 1, 1),
        "service_from_date": today,
        "service_to_date": today + timedelta(days=1),
        "total_charges": Decimal("150.00"),
        "processing_status": "pending",
        "batch_id": "BATCH001",
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        "processed_at": None,
        "line_items": [ProcessableClaimLineItem(**basic_line_item_data)]
    }

def test_valid_claim(validator: ClaimValidator, basic_claim_data: dict):
    claim = ProcessableClaim(**basic_claim_data)
    errors = validator.validate_claim(claim)
    assert not errors, f"Validation failed for a supposedly valid claim: {errors}"

def test_missing_claim_id(validator: ClaimValidator, basic_claim_data: dict):
    # Pydantic model ProcessableClaim has claim_id as non-optional.
    # To test the validator's specific check (if Pydantic check was bypassed or field was Optional):
    # Create a valid model, then forcibly make claim_id empty.
    valid_claim = ProcessableClaim(**basic_claim_data)
    valid_claim.claim_id = ""
    errors = validator.validate_claim(valid_claim)
    assert "Missing claim_id." in errors

def test_missing_facility_id(validator: ClaimValidator, basic_claim_data: dict):
    valid_claim = ProcessableClaim(**basic_claim_data)
    valid_claim.facility_id = ""
    errors = validator.validate_claim(valid_claim)
    assert "Missing facility_id." in errors

def test_service_from_date_after_to_date(validator: ClaimValidator, basic_claim_data: dict):
    basic_claim_data["service_from_date"] = date.today()
    basic_claim_data["service_to_date"] = date.today() - timedelta(days=1)
    claim = ProcessableClaim(**basic_claim_data)
    errors = validator.validate_claim(claim)
    assert any("Service-from date" in error and "cannot be after service-to date" in error for error in errors)

def test_total_charges_zero_or_negative(validator: ClaimValidator, basic_claim_data: dict):
    data_zero_charge = basic_claim_data.copy()
    data_zero_charge["total_charges"] = Decimal("0.00")
    # Need to update line item if sum is checked, but for now, only total_charges is checked directly.
    # data_zero_charge["line_items"] = [ProcessableClaimLineItem(**{**basic_line_item_data(), "charge_amount": Decimal("0.00")})]

    claim = ProcessableClaim(**data_zero_charge)
    errors = validator.validate_claim(claim)
    assert any("Total charges" in error and "must be positive" in error for error in errors)

    data_negative_charge = basic_claim_data.copy()
    data_negative_charge["total_charges"] = Decimal("-10.00")
    # data_negative_charge["line_items"] = [ProcessableClaimLineItem(**{**basic_line_item_data(), "charge_amount": Decimal("-10.00")})]
    claim2 = ProcessableClaim(**data_negative_charge)
    errors2 = validator.validate_claim(claim2)
    assert any("Total charges" in error and "must be positive" in error for error in errors2)

def test_no_line_items(validator: ClaimValidator, basic_claim_data: dict):
    basic_claim_data["line_items"] = []
    claim = ProcessableClaim(**basic_claim_data)
    errors = validator.validate_claim(claim)
    assert "Claim must have at least one line item." in errors

# Line Item Specific Tests
def test_line_item_missing_procedure_code(validator: ClaimValidator, basic_claim_data: dict, basic_line_item_data: dict):
    line_data = basic_line_item_data.copy()
    line_data["procedure_code"] = ""
    claim_data_copy = basic_claim_data.copy()
    claim_data_copy["line_items"] = [ProcessableClaimLineItem(**line_data)]

    # Ensure line item charge matches claim total if sum isn't validated, or adjust total_charges
    claim_data_copy["total_charges"] = line_data["charge_amount"]

    claim = ProcessableClaim(**claim_data_copy)
    errors = validator.validate_claim(claim)
    assert any("Line 1: Missing procedure_code." in error for error in errors)

def test_line_item_invalid_units(validator: ClaimValidator, basic_claim_data: dict, basic_line_item_data: dict):
    line_data = basic_line_item_data.copy()
    line_data["units"] = 0
    claim_data_copy = basic_claim_data.copy()
    claim_data_copy["line_items"] = [ProcessableClaimLineItem(**line_data)]
    claim_data_copy["total_charges"] = line_data["charge_amount"]

    claim = ProcessableClaim(**claim_data_copy)
    errors = validator.validate_claim(claim)
    assert any("Line 1: Units (0) must be positive." in error for error in errors)

def test_line_item_negative_charge(validator: ClaimValidator, basic_claim_data: dict, basic_line_item_data: dict):
    line_data = basic_line_item_data.copy()
    line_data["charge_amount"] = Decimal("-50.00")
    claim_data_copy = basic_claim_data.copy()
    claim_data_copy["line_items"] = [ProcessableClaimLineItem(**line_data)]
    claim_data_copy["total_charges"] = line_data["charge_amount"] # Adjust claim total to match line

    claim = ProcessableClaim(**claim_data_copy)
    errors = validator.validate_claim(claim)
    assert any("Line 1: Charge amount (-50.00) cannot be negative." in error for error in errors)

def test_line_item_service_date_outside_claim_dates(validator: ClaimValidator, basic_claim_data: dict, basic_line_item_data: dict):
    claim_from_date = date(2023, 1, 10)
    claim_to_date = date(2023, 1, 15)

    claim_data_copy = basic_claim_data.copy()
    claim_data_copy["service_from_date"] = claim_from_date
    claim_data_copy["service_to_date"] = claim_to_date

    line_data_before = basic_line_item_data.copy()
    line_data_before["service_date"] = date(2023, 1, 9) # Before claim service_from_date
    claim_data_copy["line_items"] = [ProcessableClaimLineItem(**line_data_before)]
    claim_data_copy["total_charges"] = line_data_before["charge_amount"]
    claim = ProcessableClaim(**claim_data_copy)
    errors = validator.validate_claim(claim)
    assert any(f"Line 1: Service date ({line_data_before['service_date']}) is outside of claim service period" in error for error in errors)

    line_data_after = basic_line_item_data.copy()
    line_data_after["service_date"] = date(2023, 1, 16) # After claim service_to_date
    claim_data_copy["line_items"] = [ProcessableClaimLineItem(**line_data_after)]
    claim_data_copy["total_charges"] = line_data_after["charge_amount"]
    claim2 = ProcessableClaim(**claim_data_copy)
    errors2 = validator.validate_claim(claim2)
    assert any(f"Line 1: Service date ({line_data_after['service_date']}) is outside of claim service period" in error for error in errors2)

def test_line_item_service_date_within_claim_dates_valid(validator: ClaimValidator, basic_claim_data: dict, basic_line_item_data: dict):
    claim_from_date = date(2023, 1, 10)
    claim_to_date = date(2023, 1, 15)

    claim_data_copy = basic_claim_data.copy()
    claim_data_copy["service_from_date"] = claim_from_date
    claim_data_copy["service_to_date"] = claim_to_date

    line_data = basic_line_item_data.copy()
    line_data["service_date"] = date(2023, 1, 12) # Within range
    claim_data_copy["line_items"] = [ProcessableClaimLineItem(**line_data)]
    claim_data_copy["total_charges"] = line_data["charge_amount"]

    claim = ProcessableClaim(**claim_data_copy)
    errors = validator.validate_claim(claim)
    assert not any(f"Line 1: Service date ({line_data['service_date']}) is outside of claim service period" in error for error in errors)


# --- New tests to be added ---

# Using a fixed UTC for tests if needed, or a library for timezone if not available by default
# For Python 3.9+ zoneinfo is standard.
try:
    from zoneinfo import ZoneInfo
    UTC = ZoneInfo("UTC")
except ImportError:
    UTC = timezone.utc

def create_base_test_claim(**kwargs) -> ProcessableClaim:
    # Helper to create a valid ProcessableClaim with sensible defaults
    now = datetime.now(tz=UTC)

    default_line_items_data = [
        {
            "id": 1, "claim_db_id": 1, "line_number": 1, "service_date": date(2024, 1, 15),
            "procedure_code": "P0001", "units": 1, "charge_amount": Decimal("50.00"),
            "rvu_total": None, "created_at": now, "updated_at": now
        },
        {
            "id": 2, "claim_db_id": 1, "line_number": 2, "service_date": date(2024, 1, 16),
            "procedure_code": "P0002", "units": 1, "charge_amount": Decimal("50.00"),
            "rvu_total": None, "created_at": now, "updated_at": now
        }
    ]

    line_items_input = kwargs.pop("line_items", None)

    final_line_items = []
    if line_items_input is not None:
        if all(isinstance(li, ProcessableClaimLineItem) for li in line_items_input):
            final_line_items = line_items_input
        else:
            for i, li_data in enumerate(line_items_input):
                base_li_data = {"created_at": now, "updated_at": now, "id": i+1, "claim_db_id": kwargs.get("id",1)}
                base_li_data.update(li_data)
                final_line_items.append(ProcessableClaimLineItem(**base_li_data))
    else:
        for li_data in default_line_items_data:
             final_line_items.append(ProcessableClaimLineItem(**li_data))

    calculated_total_charges = sum(li.charge_amount for li in final_line_items) if final_line_items else Decimal("0.00")

    data = {
        "id": 1, "claim_id": "C001", "facility_id": "F001", "patient_account_number": "ACC001",
        "medical_record_number": "MRN001", "patient_first_name": "John", "patient_last_name": "Doe",
        "patient_date_of_birth": date(1990, 1, 1), "insurance_type": "PPO", "insurance_plan_id": "PLAN01",
        "financial_class": "Personal", "service_from_date": date(2024, 1, 10),
        "service_to_date": date(2024, 1, 20),
        "total_charges": calculated_total_charges,
        "processing_status": "pending", "batch_id": None, "created_at": now, "updated_at": now,
        "processed_at": None, "ml_score": None, "ml_derived_decision": None,
        "processing_duration_ms": None,
        "line_items": final_line_items
    }

    if "total_charges" in kwargs: # Allow explicit override of total_charges for testing mismatch
        data["total_charges"] = kwargs.pop("total_charges")

    data.update(kwargs)

    return ProcessableClaim(**data)

# Test using the new helper
def test_valid_claim_with_helper(validator: ClaimValidator):
    claim = create_base_test_claim()
    errors = validator.validate_claim(claim)
    assert not errors, f"Validation errors for a supposedly valid claim (using helper): {errors}"

def test_sum_of_line_charges_mismatch(validator: ClaimValidator):
    # Default lines in create_base_test_claim sum to 100.00
    claim = create_base_test_claim(total_charges=Decimal("101.00"))
    errors = validator.validate_claim(claim)
    assert any("Sum of line item charges (100.00) does not match claim total charges (101.00)" in error for error in errors)

def test_sum_of_line_charges_match(validator: ClaimValidator):
    now = datetime.now(tz=UTC)
    lines_data_for_objects = [ # Data to create ProcessableClaimLineItem objects
        {"id":1, "claim_db_id":1, "line_number": 1, "service_date": date(2024,1,15), "procedure_code": "P1", "units": 1, "charge_amount": Decimal("25.50"), "created_at":now, "updated_at":now},
        {"id":2, "claim_db_id":1, "line_number": 2, "service_date": date(2024,1,16), "procedure_code": "P2", "units": 2, "charge_amount": Decimal("37.25"), "created_at":now, "updated_at":now}
    ] # Sum = 25.50 + 2*37.25 = 25.50 + 74.50 = 100.00

    line_items_obj = [ProcessableClaimLineItem(**li) for li in lines_data_for_objects]

    claim = create_base_test_claim(line_items=line_items_obj, total_charges=Decimal("100.00"))
    errors = validator.validate_claim(claim)
    assert not any("Sum of line item charges" in error for error in errors), f"Mismatch error found: {errors}"
