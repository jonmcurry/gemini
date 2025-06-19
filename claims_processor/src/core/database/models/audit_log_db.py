from sqlalchemy import Column, Integer, String, TIMESTAMP, Boolean, Text
from sqlalchemy.dialects.postgresql import JSONB # For PostgreSQL specific JSONB
from sqlalchemy.sql import func # For server_default=func.now()
from ..db_session import Base # Assuming Base is in db_session at claims_processor/src/core/database/db_session.py

class AuditLogModel(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False, index=True)

    user_id = Column(String(100), nullable=True, index=True) # Nullable for system events
    action = Column(String(255), nullable=False, index=True) # e.g., CREATE_CLAIM, PROCESS_BATCH_TRIGGER
    resource = Column(String(255), nullable=True, index=True) # e.g., Claim, ClaimBatch
    resource_id = Column(String(255), nullable=True, index=True) # ID of the affected resource

    # For HIPAA compliance, patient identifiers should be handled carefully.
    # Storing a hash is better than raw ID. For now, field is placeholder.
    patient_id_hash = Column(String(255), nullable=True, index=True)

    ip_address = Column(String(100), nullable=True)
    user_agent = Column(Text, nullable=True) # User-Agent string can be long
    session_id = Column(String(255), nullable=True, index=True) # If session management is implemented

    success = Column(Boolean, nullable=False)
    failure_reason = Column(Text, nullable=True) # If success is False

    details = Column(JSONB, nullable=True) # For any additional structured context (PostgreSQL JSONB)
                                        # For other DBs, use sa.JSON

    def __repr__(self):
        return f"<AuditLogModel(id={self.id}, action='{self.action}', user='{self.user_id}', resource='{self.resource}/{self.resource_id}', success={self.success})>"
