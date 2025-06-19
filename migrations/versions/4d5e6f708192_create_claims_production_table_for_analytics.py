"""Create claims_production table for analytics

Revision ID: 4d5e6f708192
Revises: 3c4d5e6f7081
Create Date: 2024-07-16 14:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '4d5e6f708192'
down_revision = '3c4d5e6f7081' # Points to the audit_logs table migration
branch_labels = None
depends_on = None

def upgrade():
    op.create_table('claims_production',
        # Core fields from ClaimModel
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('claim_id', sa.String(length=100), nullable=False),
        sa.Column('facility_id', sa.String(length=50), nullable=False),
        sa.Column('patient_account_number', sa.String(length=100), nullable=False),
        sa.Column('patient_first_name', sa.String(length=100), nullable=True),
        sa.Column('patient_last_name', sa.String(length=100), nullable=True),
        sa.Column('patient_date_of_birth', sa.Date(), nullable=True),
        sa.Column('service_from_date', sa.Date(), nullable=False), # Partition Key
        sa.Column('service_to_date', sa.Date(), nullable=False),
        sa.Column('total_charges', sa.Numeric(precision=15, scale=2), nullable=False),

        # Analytics-specific columns
        sa.Column('ml_prediction_score', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('risk_category', sa.String(length=50), nullable=True),
        sa.Column('processing_duration_ms', sa.Integer(), nullable=True),
        sa.Column('throughput_achieved', sa.Numeric(precision=10, scale=2), nullable=True),

        sa.Column('created_at_prod', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False, comment="Timestamp of insertion into production table"),

        sa.PrimaryKeyConstraint('id', 'service_from_date', name=op.f('pk_claims_production')),
        sa.UniqueConstraint('claim_id', name=op.f('uq_claims_production_claim_id')), # Ensure claim_id is unique here too

        postgresql_partition_by='RANGE (service_from_date)' # Partitioning clause
    )

    # Create indexes (some might be covered by PK/UQ)
    # op.create_index(op.f('ix_claims_production_id'), 'claims_production', ['id'], unique=False) # Index on 'id' is part of PK
    # op.create_index(op.f('ix_claims_production_claim_id'), 'claims_production', ['claim_id'], unique=True) # Index covered by UQ
    op.create_index(op.f('ix_claims_production_facility_id'), 'claims_production', ['facility_id'], unique=False)
    op.create_index(op.f('ix_claims_production_patient_account_number'), 'claims_production', ['patient_account_number'], unique=False)
    op.create_index(op.f('ix_claims_production_service_from_date'), 'claims_production', ['service_from_date'], unique=False) # Index on partition key

    # Create initial partitions for 'claims_production' table
    op.execute("CREATE TABLE claims_production_y2023 PARTITION OF claims_production FOR VALUES FROM ('2023-01-01') TO ('2024-01-01')")
    op.execute("CREATE TABLE claims_production_y2024 PARTITION OF claims_production FOR VALUES FROM ('2024-01-01') TO ('2025-01-01')")
    op.execute("CREATE TABLE claims_production_y2025 PARTITION OF claims_production FOR VALUES FROM ('2025-01-01') TO ('2026-01-01')")

def downgrade():
    # Drop partitions first
    op.execute("DROP TABLE IF EXISTS claims_production_y2023")
    op.execute("DROP TABLE IF EXISTS claims_production_y2024")
    op.execute("DROP TABLE IF EXISTS claims_production_y2025")

    # Then drop indexes that are not automatically dropped with table/constraints
    op.drop_index(op.f('ix_claims_production_service_from_date'), table_name='claims_production')
    op.drop_index(op.f('ix_claims_production_patient_account_number'), table_name='claims_production')
    op.drop_index(op.f('ix_claims_production_facility_id'), table_name='claims_production')
    # UQ and PK constraints (and their implicit indexes) are dropped with the table.
    # Explicitly created separate indexes on PK/UQ columns would need explicit drop if not op.f() named with constraint.

    op.drop_table('claims_production')
