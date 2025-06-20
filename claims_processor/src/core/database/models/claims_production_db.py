from sqlalchemy import Column, Integer, String, Date, Numeric, TIMESTAMP, func, PrimaryKeyConstraint
from sqlalchemy.dialects.postgresql import JSONB # In case details are ever needed, though not specified for this table
from ..db_session import Base # Assuming Base is in db_session.py

class ClaimsProductionModel(Base):
    __tablename__ = "claims_production"

    # Core fields mirroring ClaimModel, plus analytics fields
    # id should be unique in the context of service_from_date for PK, but might map to original claims.id
    # For simplicity, let's assume it's populated with the same 'id' from the 'claims' table.
    id = Column(Integer, index=True, nullable=False)
    claim_id = Column(String(100), unique=True, index=True, nullable=False) # Business claim ID

    # Copied fields from ClaimModel (subset needed for identification and partitioning)
    facility_id = Column(String(50), nullable=False, index=True)
    patient_account_number = Column(String(100), nullable=False, index=True)

    patient_first_name = Column(String(100), nullable=True) # Or omit if not needed for analytics table
    patient_last_name = Column(String(100), nullable=True)  # Or omit
    patient_date_of_birth = Column(Date, nullable=True)    # Or omit

    service_from_date = Column(Date, nullable=False, index=True) # Partition Key
    service_to_date = Column(Date, nullable=False, index=True)

    total_charges = Column(Numeric(15, 2), nullable=False)

    # Fields from original ClaimModel that might be useful for context in analytics table
    # Or these could be joined from the staging 'claims' table if 'id' is a reliable FK.
    # For a denormalized analytics table, some duplication is common.
    # processing_status = Column(String(50), index=True) # Status at time of transfer
    # batch_id = Column(String(100), index=True, nullable=True)

    # Analytics-specific columns (as per REQUIREMENTS.md for claims_production)
    ml_prediction_score = Column(Numeric(5, 4), nullable=True) # e.g., 0.xxxx
    risk_category = Column(String(50), nullable=True) # Increased length from 20 in reqs for flexibility
    processing_duration_ms = Column(Integer, nullable=True)
    throughput_achieved = Column(Numeric(10, 2), nullable=True) # e.g., claims/sec during its batch

    # Timestamps for the production record itself
    created_at_prod = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False, comment="Timestamp of insertion into production table")
    # updated_at_prod = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False) # If records here get updated

    # No direct relationship to line_items here, analytics table is often flatter.
    # Line items would be in their own analytics table or joined from staging if needed.

    __table_args__ = (
        PrimaryKeyConstraint('id', 'service_from_date', name='pk_claims_production'),
        # Add any unique constraints if necessary for this table, e.g. on claim_id if it's the main business key.
        # The 'claim_id' column already has unique=True, which creates a unique constraint.
        {'postgresql_partition_by': 'RANGE (service_from_date)'}
    )

    def __repr__(self):
        return f"<ClaimsProductionModel(id={self.id}, claim_id='{self.claim_id}', service_from_date='{self.service_from_date}')>"
