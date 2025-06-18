from pydantic import BaseModel
from datetime import date

class Claim(BaseModel):
    claim_id: str
    facility_id: str
    patient_account_number: str
    service_from_date: date
    service_to_date: date
    total_charges: float
