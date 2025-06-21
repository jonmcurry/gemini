import pytest
from pydantic import ValidationError
from datetime import date
from decimal import Decimal

from claims_processor.src.api.models.ingestion_models import IngestionClaim, IngestionClaimLineItem

def get_minimal_valid_ingestion_line_item_data():
    return {
        "line_number": 1, "service_date": date(2024, 1, 1),
        "procedure_code": "P001", "units": 1, "charge_amount": Decimal("10.00")
    }

def get_minimal_valid_ingestion_claim_data():
    # Ensure line_items are IngestionClaimLineItem instances for validation
    line_item_model = IngestionClaimLineItem(**get_minimal_valid_ingestion_line_item_data())
    return {
        "claim_id": "C001", "facility_id": "F001", "patient_account_number": "ACC001",
        "service_from_date": date(2024, 1, 1), "service_to_date": date(2024, 1, 2),
        "total_charges": Decimal("10.00"), # Must be > 0
        "line_items": [line_item_model] # Pass as model instance
    }

def test_ingestion_claim_with_all_new_optional_fields():
    data = get_minimal_valid_ingestion_claim_data()
    data.update({
        "patient_middle_name": "Middle",
        "admission_date": date(2024, 1, 1),
        "discharge_date": date(2024, 1, 2),
        "expected_reimbursement": Decimal("8.00"),
        "subscriber_id": "SUB001",
        "billing_provider_npi": "BPNPI01",
        "billing_provider_name": "Billing Prov Name",
        "attending_provider_npi": "APNPI01",
        "attending_provider_name": "Attending Prov Name",
        "primary_diagnosis_code": "D001",
        "diagnosis_codes": ["D002", "D003"]
    })
    claim = IngestionClaim(**data)
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

def test_ingestion_claim_with_some_new_optional_fields_null():
    data = get_minimal_valid_ingestion_claim_data()
    data.update({
        "patient_middle_name": "Middle",
        "admission_date": None, # Explicitly None
        "subscriber_id": "SUB001",
        "expected_reimbursement": None
        # Other new fields will default to None as per Pydantic model definition
    })
    claim = IngestionClaim(**data)
    assert claim.patient_middle_name == "Middle"
    assert claim.admission_date is None
    assert claim.subscriber_id == "SUB001"
    assert claim.expected_reimbursement is None

    # Check a few fields that should have defaulted to None
    assert claim.discharge_date is None
    assert claim.primary_diagnosis_code is None
    assert claim.billing_provider_npi is None
    assert claim.diagnosis_codes is None

def test_ingestion_claim_required_fields_missing():
    data = get_minimal_valid_ingestion_claim_data()
    del data["claim_id"] # Remove a required field
    with pytest.raises(ValidationError):
        IngestionClaim(**data)

def test_ingestion_claim_total_charges_validation():
    data = get_minimal_valid_ingestion_claim_data()
    data["total_charges"] = Decimal("-1.00") # Must be > 0
    with pytest.raises(ValidationError):
        IngestionClaim(**data)

    data["total_charges"] = Decimal("0.00") # Must be > 0
    with pytest.raises(ValidationError):
        IngestionClaim(**data)

def test_ingestion_claim_line_items_validation():
    data = get_minimal_valid_ingestion_claim_data()
    data["line_items"] = [] # Must have at least one line item
    with pytest.raises(ValidationError):
        IngestionClaim(**data)

# Example test for a constr field if needed for new fields
# (Most new string fields are Optional, so presence of value is main test)
def test_ingestion_claim_subscriber_id_constraints():
    data = get_minimal_valid_ingestion_claim_data()
    # Assuming subscriber_id has max_length=100 from IngestionClaim model
    data["subscriber_id"] = "A" * 101
    with pytest.raises(ValidationError):
        IngestionClaim(**data)

    data["subscriber_id"] = "   ValidSubID  " # Check strip_whitespace
    claim = IngestionClaim(**data)
    assert claim.subscriber_id == "ValidSubID"
