from sqlalchemy import Column, Integer, String, Date, Numeric, ForeignKey, TIMESTAMP, func, PrimaryKeyConstraint
from sqlalchemy.orm import relationship
from ..db_session import Base

class ClaimModel(Base):
    __tablename__ = "claims"

    # id is part of composite PK
    id = Column(Integer, index=True, nullable=False)
    claim_id = Column(String(100), unique=True, index=True, nullable=False)
    facility_id = Column(String(50), nullable=False, index=True)
    patient_account_number = Column(String(100), nullable=False, index=True)

    patient_first_name = Column(String(100), nullable=True)
    patient_last_name = Column(String(100), nullable=True)
    patient_date_of_birth = Column(Date, nullable=True)

    # service_from_date is part of composite PK and partition key
    service_from_date = Column(Date, nullable=False, index=True)
    service_to_date = Column(Date, nullable=False, index=True)

    total_charges = Column(Numeric(15, 2), nullable=False)

    processing_status = Column(String(50), default='pending', index=True)
    batch_id = Column(String(100), index=True, nullable=True)

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
        PrimaryKeyConstraint('id', 'service_from_date', name='pk_claims'),
        {'postgresql_partition_by': 'RANGE (service_from_date)'}
    )

class ClaimLineItemModel(Base):
    __tablename__ = "claim_line_items"

    id = Column(Integer, index=True, nullable=False) # Will be part of composite PK.
    claim_db_id = Column(Integer, ForeignKey("claims.id"), nullable=False, index=True)

    line_number = Column(Integer, nullable=False)

    service_date = Column(Date, index=True, nullable=False) # Partition Key, part of Composite PK

    procedure_code = Column(String(20), nullable=False, index=True)
    units = Column(Integer, default=1, nullable=False)
    charge_amount = Column(Numeric(15, 2), nullable=False)

    rvu_total = Column(Numeric(10, 4), nullable=True)

    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    claim = relationship("ClaimModel", back_populates="line_items")

    __table_args__ = (
        PrimaryKeyConstraint('id', 'service_date', name='pk_claim_line_items'),
        {'postgresql_partition_by': 'RANGE (service_date)'}
    )
