from sqlalchemy import Column, Integer, String, TEXT, TIMESTAMP, func, Date # Added Date
from sqlalchemy.dialects.postgresql import JSONB
# from sqlalchemy.orm import relationship # Not using ORM relationship for now
from ..db_session import Base

class FailedClaimModel(Base):
    __tablename__ = "failed_claims"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Reference to the original claim in the 'claims' (staging) table's 'id' field (which is part of its composite PK)
    original_claim_db_id = Column(Integer, index=True, nullable=True)

    # Key identifying fields from the original claim, duplicated for easier querying
    claim_id = Column(String(100), index=True, nullable=True) # Business claim ID from original claim
    facility_id = Column(String(50), index=True, nullable=True) # From original claim
    patient_account_number = Column(String(100), index=True, nullable=True) # From original claim
    # service_from_date from original claim might also be useful for partitioning this table later if it grows large
    # original_service_from_date = Column(Date, index=True, nullable=True)


    failed_at_stage = Column(String(50), nullable=False, index=True) # e.g., 'VALIDATION', 'ML_PREDICTION', 'RVU_CALCULATION', 'CONVERSION_ERROR'
    failure_reason = Column(TEXT, nullable=False) # Detailed error messages or list of errors (can be JSON string)
    failure_timestamp = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False, index=True)

    # Store a snapshot of the ProcessableClaim Pydantic model (as dict/JSON) as it was when it failed.
    original_claim_data = Column(JSONB, nullable=True)

    def __repr__(self):
        return f"<FailedClaimModel(id={self.id}, claim_id='{self.claim_id}', failed_at_stage='{self.failed_at_stage}')>"
