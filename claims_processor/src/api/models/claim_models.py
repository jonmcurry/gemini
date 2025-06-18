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


# New models for internal processing, closely mirroring DB structure
# These will be populated from SQLAlchemy objects
from pydantic import constr, condecimal # Import constr and condecimal
from typing import List # Import List

class ProcessableClaimLineItem(BaseModel):
    id: int
    claim_db_id: int # Foreign key to the ClaimModel's id
    line_number: int
    service_date: date
    procedure_code: str # Consider constr(max_length=20) if strict
    units: int
    charge_amount: condecimal(max_digits=15, decimal_places=2)
    rvu_total: Optional[condecimal(max_digits=10, decimal_places=4)] = None
    created_at: datetime
    updated_at: datetime

    # If using Pydantic V2
    model_config = {"from_attributes": True}
    # class Config: # For Pydantic V1
    #     orm_mode = True


class ProcessableClaim(BaseModel):
    id: int # Primary Key from DB
    claim_id: str # Business claim ID
    facility_id: str
    patient_account_number: str

    patient_first_name: Optional[str] = None
    patient_last_name: Optional[str] = None
    patient_date_of_birth: Optional[date] = None

    service_from_date: date
    service_to_date: date

    total_charges: condecimal(max_digits=15, decimal_places=2)

    processing_status: str # This will reflect current DB status when fetched
    batch_id: Optional[str] = None

    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime] = None

    line_items: List[ProcessableClaimLineItem] = []

    # If using Pydantic V2
    model_config = {"from_attributes": True}
    # class Config: # For Pydantic V1
    #     orm_mode = True
