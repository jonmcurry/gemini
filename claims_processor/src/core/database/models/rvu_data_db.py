from sqlalchemy import Column, Integer, String, Numeric, Text, UniqueConstraint
from sqlalchemy.orm import relationship # Not strictly needed for this model if no FKs to it yet
from ..db_session import Base

class RVUDataModel(Base):
    __tablename__ = "rvu_data"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    procedure_code = Column(String(50), nullable=False, unique=True, index=True)
    # Per REQUIREMENTS.md, some procedure codes are like "99213", some are "DEFAULT_RVU"
    # String(50) should be sufficient. unique=True and index=True are important.

    rvu_value = Column(Numeric(10, 4), nullable=False)
    # Numeric for precision, e.g., 2.11, 0.8500. (10 total digits, 4 after decimal)

    description = Column(Text, nullable=True)

    # Modifiers are not part of this simplified model as per plan decision.
    # If modifiers were needed, a composite unique constraint on (procedure_code, modifier_code)
    # would be necessary, and procedure_code alone would not be unique.
    # Example: __table_args__ = (UniqueConstraint('procedure_code', 'modifier_code', name='uq_proc_mod'),)

    def __repr__(self):
        return f"<RVUDataModel(id={self.id}, procedure_code='{self.procedure_code}', rvu_value={self.rvu_value})>"
