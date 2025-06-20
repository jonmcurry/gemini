from sqlalchemy import Column, Integer, String, Date, Numeric, ForeignKey, TIMESTAMP, func, PrimaryKeyConstraint, UniqueConstraint
from sqlalchemy.orm import relationship
from ..db_session import Base

class ClaimModel(Base):
    __tablename__ = "claims"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    claim_id = Column(String(100), unique=True, index=True, nullable=False) # Business key
    facility_id = Column(String(50), nullable=False, index=True)
    patient_account_number = Column(String(100), nullable=False, index=True)

    patient_first_name = Column(String(100), nullable=True)
    patient_last_name = Column(String(100), nullable=True)
    patient_date_of_birth = Column(String(255), nullable=True) # Changed for encryption
    medical_record_number = Column(String(150), nullable=True, index=True)

    financial_class = Column(String(50), nullable=True, index=True)
    insurance_type = Column(String(100), nullable=True, index=True)
    insurance_plan_id = Column(String(100), nullable=True, index=True)

    # service_from_date is part of composite PK and partition key
    service_from_date = Column(Date, nullable=False, index=True)
    service_to_date = Column(Date, nullable=False, index=True)

    total_charges = Column(Numeric(15, 2), nullable=False)

    processing_status = Column(String(50), default='pending', index=True)
    batch_id = Column(String(100), index=True, nullable=True)
    priority = Column(Integer, server_default='1', nullable=False, index=True) # New priority field

    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    processed_at = Column(TIMESTAMP(timezone=True), nullable=True)

    # New fields for tracking transfer and ML results that might influence transfer/analytics
    transferred_to_prod_at = Column(TIMESTAMP(timezone=True), nullable=True, index=True)
    processing_duration_ms = Column(Integer, nullable=True)
    ml_score = Column(Numeric(5, 4), nullable=True)
    ml_derived_decision = Column(String(50), nullable=True)

    line_items = relationship("ClaimLineItemModel", back_populates="claim", cascade="all, delete-orphan")

    __table_args__ = (
        # PrimaryKeyConstraint removed as 'id' is now the sole PK.
        # 'claim_id' is already marked unique=True, creating its own unique constraint.
        # Add other composite unique constraints if necessary. Example:
        # UniqueConstraint('claim_id', 'facility_id', name='uq_claim_facility'),
        {'postgresql_partition_by': 'RANGE (service_from_date)'}
    )

class ClaimLineItemModel(Base):
    __tablename__ = "claim_line_items"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    claim_db_id = Column(Integer, ForeignKey("claims.id"), nullable=False, index=True) # Refers to ClaimModel.id

    line_number = Column(Integer, nullable=False)

    service_date = Column(Date, index=True, nullable=False) # Partition Key

    procedure_code = Column(String(20), nullable=False, index=True)
    units = Column(Integer, default=1, nullable=False)
    charge_amount = Column(Numeric(15, 2), nullable=False)

    rvu_total = Column(Numeric(10, 4), nullable=True)

    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    claim = relationship("ClaimModel", back_populates="line_items")

    __table_args__ = (
        # PrimaryKeyConstraint removed
        UniqueConstraint('claim_db_id', 'line_number', 'service_date', name='uq_claimitem_claim_line_servicedate'), # Ensure line_number is unique per claim per service_date (if service_date varies within a claim for lines)
        # If service_date on line item is always same as claim's service_from_date for partitioning, then 'service_date' might not be needed in UQ.
        # However, line items can have different service dates. Partitioning by line's service_date is correct.
        # The uniqueness of line_number should be per claim.
        # The old PK was (id, service_date). New PK is (id).
        # We need line_number to be unique for a given claim_db_id.
        UniqueConstraint('claim_db_id', 'line_number', name='uq_claimitem_claim_line'),
        {'postgresql_partition_by': 'RANGE (service_date)'}
    )
