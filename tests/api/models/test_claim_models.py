import pytest
from pydantic import ValidationError
from datetime import date, datetime, timezone # Ensure timezone is imported
from decimal import Decimal
from typing import List, Optional

from claims_processor.src.api.models.claim_models import ProcessableClaim, ProcessableClaimLineItem

# Helper to create ProcessableClaimLineItem for ProcessableClaim
def create_minimal_processable_line_item_data(line_num=1, charge=Decimal("50.00")) -> dict:
    # Use timezone.utc for robust, timezone-aware datetime objects
    # Fallback for environments where zoneinfo might not be available (though pytest.importorskip handles it)
    utc_tz = timezone.utc
    try:
        from zoneinfo import ZoneInfo
        utc_tz = ZoneInfo("UTC")
    except ImportError:
        pass
    now = datetime.now(tz=utc_tz)

    return {
        "id": line_num, "claim_db_id": 1, "line_number": line_num,
        "service_date": date(2024, 1, 15), "procedure_code": f"P{line_num:03d}",
        "units": 1, "charge_amount": charge,
        "rvu_total": None, "created_at": now, "updated_at": now
    }

def get_minimal_valid_processable_claim_data():
    utc_tz = timezone.utc
    try:
        from zoneinfo import ZoneInfo
        utc_tz = ZoneInfo("UTC")
    except ImportError:
        pass
    now = datetime.now(tz=utc_tz)

    line_item_dict_data = create_minimal_processable_line_item_data()
    # For ProcessableClaim, line_items should be instances of ProcessableClaimLineItem
    line_item_model = ProcessableClaimLineItem(**line_item_dict_data)

    return {
        "id": 1, "claim_id": "C001", "facility_id": "F001", "patient_account_number": "ACC001",
        "service_from_date": date(2024, 1, 10), "service_to_date": date(2024, 1, 20),
        "total_charges": Decimal(line_item_model.charge_amount), # Ensure > 0 based on line item
        "processing_status": "pending", "created_at": now, "updated_at": now,
        "line_items": [line_item_model] # Pass as model instance
    }

def test_processable_claim_with_all_new_optional_fields():
    data = get_minimal_valid_processable_claim_data()
    # Ensure total_charges is positive and matches line items if any, or set a default
    if not data.get("line_items"):
        data["total_charges"] = Decimal("1.00") # Default positive value
    else:
        data["total_charges"] = sum(li.charge_amount for li in data["line_items"])
        if data["total_charges"] <= Decimal(0): # Ensure positive
             data["total_charges"] = Decimal("1.00")


    data.update({
        "patient_middle_name": "Middle",
        "admission_date": date(2024, 1, 1),
        "discharge_date": date(2024, 1, 2),
        "expected_reimbursement": Decimal("8.00"),
        "subscriber_id": "SUB001", # Assuming decrypted
        "billing_provider_npi": "BPNPI01",
        "billing_provider_name": "Billing Prov Name",
        "attending_provider_npi": "APNPI01",
        "attending_provider_name": "Attending Prov Name",
        "primary_diagnosis_code": "D001",
        "diagnosis_codes": ["D002", "D003"]
    })

    claim = ProcessableClaim(**data)
    assert claim.patient_middle_name == "Middle"
    assert claim.admission_date == date(2024, 1, 1)
    assert claim.discharge_date == date(2024, 1, 2)
    assert claim.expected_reimbursement == Decimal("8.00")
    assert claim.subscriber_id == "SUB001"
    assert claim.billing_provider_npi == "BPNPI01"
    assert claim.billing_provider_name == "Billing Prov Name"
    assert claim.attending_provider_npi == "APNPI01"
    assert claim.attending_provider_name == "Attending Prov Name"
    assert claim.primary_diagnosis_code == "D001"
    assert claim.diagnosis_codes == ["D002", "D003"]

def test_processable_claim_with_some_new_optional_fields_null():
    data = get_minimal_valid_processable_claim_data()
    data.update({
        "patient_middle_name": "Middle",
        "admission_date": None,
        "primary_diagnosis_code": "D001",
        "expected_reimbursement": None
    })
    claim = ProcessableClaim(**data)
    assert claim.patient_middle_name == "Middle"
    assert claim.admission_date is None
    assert claim.primary_diagnosis_code == "D001"
    assert claim.expected_reimbursement is None

    # Check a few fields that should have defaulted to None
    assert claim.discharge_date is None
    assert claim.subscriber_id is None
    assert claim.billing_provider_npi is None
    assert claim.diagnosis_codes is None

def test_processable_claim_required_fields_missing():
    data = get_minimal_valid_processable_claim_data()
    del data["claim_id"] # Remove a required field
    with pytest.raises(ValidationError):
        ProcessableClaim(**data)

def test_processable_claim_total_charges_validation():
    data = get_minimal_valid_processable_claim_data()
    # Test total_charges constraint (gt=Decimal(0))
    data["total_charges"] = Decimal("-1.00")
    with pytest.raises(ValidationError):
        ProcessableClaim(**data)

    data["total_charges"] = Decimal("0.00")
    with pytest.raises(ValidationError):
        ProcessableClaim(**data)

# ProcessableClaim doesn't use constr for most of its string fields currently,
# but if it did, tests similar to IngestionClaim's constr tests would apply.
# For example, if subscriber_id had length constraints in ProcessableClaim:
# def test_processable_claim_subscriber_id_constraints():
#     data = get_minimal_valid_processable_claim_data()
#     data["subscriber_id"] = "A" * 101 # If max_length=100
#     with pytest.raises(ValidationError):
#         ProcessableClaim(**data)
