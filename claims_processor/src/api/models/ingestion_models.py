from pydantic import BaseModel, Field, condecimal, constr
from datetime import date
from typing import Optional, List
from decimal import Decimal # Ensure Decimal is imported for condecimal constraints

class IngestionClaimLineItem(BaseModel):
    # No 'id' or 'claim_db_id' from caller for new line items
    line_number: int = Field(..., gt=0, description="Line item number, must be positive.")
    service_date: date = Field(..., description="Service date for the line item.")
    procedure_code: constr(strip_whitespace=True, min_length=1, max_length=20) = Field(..., description="Procedure code.")
    units: int = Field(..., gt=0, description="Number of units, must be positive.")
    charge_amount: condecimal(max_digits=15, decimal_places=2, ge=Decimal(0)) = Field(..., description="Charge amount for this line item, must be non-negative.")

    procedure_description: Optional[constr(strip_whitespace=True, max_length=500)] = Field(None, description="Optional: Procedure description.")
    rendering_provider_npi: Optional[constr(strip_whitespace=True, max_length=20)] = Field(None, description="Optional: Rendering provider NPI.")


class IngestionClaim(BaseModel):
    # No 'id' (PK) from caller for new claims
    claim_id: constr(strip_whitespace=True, min_length=1, max_length=100) = Field(..., description="Unique business claim identifier from source.")
    facility_id: constr(strip_whitespace=True, min_length=1, max_length=50) = Field(..., description="Facility identifier.")
    patient_account_number: constr(strip_whitespace=True, min_length=1, max_length=100) = Field(..., description="Patient account number.")

    medical_record_number: Optional[constr(strip_whitespace=True, max_length=150)] = Field(None, description="Optional: Patient's Medical Record Number. Will be encrypted if provided.")

    patient_first_name: Optional[constr(strip_whitespace=True, max_length=100)] = Field(None)
    patient_middle_name: Optional[constr(strip_whitespace=True, max_length=100)] = Field(None, description="Optional: Patient's middle name.")
    patient_last_name: Optional[constr(strip_whitespace=True, max_length=100)] = Field(None)
    patient_date_of_birth: Optional[date] = Field(None, description="Optional: Patient's date of birth. Will be encrypted if provided.")

    subscriber_id: Optional[constr(strip_whitespace=True, max_length=100)] = Field(None, description="Optional: Subscriber ID for insurance.")

    admission_date: Optional[date] = Field(None, description="Optional: Admission date for inpatient services.")
    discharge_date: Optional[date] = Field(None, description="Optional: Discharge date for inpatient services.")

    financial_class: Optional[constr(strip_whitespace=True, max_length=50)] = Field(None, description="Optional: Financial class.")
    insurance_type: Optional[constr(strip_whitespace=True, max_length=100)] = Field(None, description="Optional: Insurance type.")
    insurance_plan_id: Optional[constr(strip_whitespace=True, max_length=100)] = Field(None, description="Optional: Insurance plan ID.")

    expected_reimbursement: Optional[condecimal(max_digits=15, decimal_places=2)] = Field(None, description="Optional: Expected reimbursement amount.")

    service_from_date: date = Field(..., description="Service from date.")
    service_to_date: date = Field(..., description="Service to date.")

    total_charges: condecimal(max_digits=15, decimal_places=2, gt=Decimal(0)) = Field(..., description="Total charges for the claim, must be positive.")

    billing_provider_npi: Optional[constr(strip_whitespace=True, max_length=20)] = Field(None, description="Optional: Billing provider NPI.")
    billing_provider_name: Optional[constr(strip_whitespace=True, max_length=200)] = Field(None, description="Optional: Billing provider name.")
    attending_provider_npi: Optional[constr(strip_whitespace=True, max_length=20)] = Field(None, description="Optional: Attending provider NPI.")
    attending_provider_name: Optional[constr(strip_whitespace=True, max_length=200)] = Field(None, description="Optional: Attending provider name.")

    primary_diagnosis_code: Optional[constr(strip_whitespace=True, max_length=20)] = Field(None, description="Optional: Primary diagnosis code (ICD).")
    diagnosis_codes: Optional[List[str]] = Field(None, description="Optional: List of other diagnosis codes (ICD).")

    line_items: List[IngestionClaimLineItem] = Field(..., min_length=1, description="List of claim line items. Must have at least one.")

    ingestion_batch_id: Optional[constr(strip_whitespace=True, max_length=100)] = Field(None, description="Optional batch ID from the source system for this ingestion group.")

    # Pydantic model config can be added if needed, e.g., for extra='forbid'
    # class Config:
    #     extra = "forbid"
