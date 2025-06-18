from pydantic import BaseModel
from datetime import date, datetime # Import datetime
from typing import Optional

class ClaimBase(BaseModel): # Renamed original Claim to ClaimBase
    claim_id: str
    facility_id: str
    patient_account_number: str
    service_from_date: date
    service_to_date: date
    total_charges: float
    # Optional fields from ClaimModel that can be part of creation
    patient_first_name: Optional[str] = None
    patient_last_name: Optional[str] = None
    patient_date_of_birth: Optional[date] = None
    batch_id: Optional[str] = None

class ClaimCreate(ClaimBase):
    pass

class ClaimResponse(ClaimBase):
    id: int # Database ID
    processing_status: str
    created_at: datetime
    updated_at: datetime

    # For Pydantic V2
    model_config = {"from_attributes": True}
