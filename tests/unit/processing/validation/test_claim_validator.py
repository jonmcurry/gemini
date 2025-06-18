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
