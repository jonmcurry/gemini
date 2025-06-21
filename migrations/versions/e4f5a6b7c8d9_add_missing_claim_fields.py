"""Add missing claim fields to claims table

Revision ID: e4f5a6b7c8d9
Revises: d3e4f5a6b7c8
Create Date: 2024-03-15 10:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'e4f5a6b7c8d9'
down_revision = 'd3e4f5a6b7c8'
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.add_column('claims', sa.Column('patient_middle_name', sa.String(length=100), nullable=True))
    op.add_column('claims', sa.Column('admission_date', sa.Date(), nullable=True))
    op.create_index(op.f('ix_claims_admission_date'), 'claims', ['admission_date'], unique=False)
    op.add_column('claims', sa.Column('discharge_date', sa.Date(), nullable=True))
    op.create_index(op.f('ix_claims_discharge_date'), 'claims', ['discharge_date'], unique=False)
    op.add_column('claims', sa.Column('expected_reimbursement', sa.Numeric(precision=15, scale=2), nullable=True))
    op.add_column('claims', sa.Column('subscriber_id', sa.String(length=100), nullable=True))
    op.create_index(op.f('ix_claims_subscriber_id'), 'claims', ['subscriber_id'], unique=False)
    op.add_column('claims', sa.Column('billing_provider_npi', sa.String(length=20), nullable=True))
    op.create_index(op.f('ix_claims_billing_provider_npi'), 'claims', ['billing_provider_npi'], unique=False)
    op.add_column('claims', sa.Column('billing_provider_name', sa.String(length=200), nullable=True))
    op.add_column('claims', sa.Column('attending_provider_npi', sa.String(length=20), nullable=True))
    op.create_index(op.f('ix_claims_attending_provider_npi'), 'claims', ['attending_provider_npi'], unique=False)
    op.add_column('claims', sa.Column('attending_provider_name', sa.String(length=200), nullable=True))
    op.add_column('claims', sa.Column('primary_diagnosis_code', sa.String(length=20), nullable=True))
    op.create_index(op.f('ix_claims_primary_diagnosis_code'), 'claims', ['primary_diagnosis_code'], unique=False)
    op.add_column('claims', sa.Column('diagnosis_codes', postgresql.JSONB(astext_fallback=True), nullable=True))

def downgrade() -> None:
    op.drop_column('claims', 'diagnosis_codes')
    op.drop_index(op.f('ix_claims_primary_diagnosis_code'), table_name='claims')
    op.drop_column('claims', 'primary_diagnosis_code')
    op.drop_column('claims', 'attending_provider_name')
    op.drop_index(op.f('ix_claims_attending_provider_npi'), table_name='claims')
    op.drop_column('claims', 'attending_provider_npi')
    op.drop_column('claims', 'billing_provider_name')
    op.drop_index(op.f('ix_claims_billing_provider_npi'), table_name='claims')
    op.drop_column('claims', 'billing_provider_npi')
    op.drop_index(op.f('ix_claims_subscriber_id'), table_name='claims')
    op.drop_column('claims', 'subscriber_id')
    op.drop_column('claims', 'expected_reimbursement')
    op.drop_index(op.f('ix_claims_discharge_date'), table_name='claims')
    op.drop_column('claims', 'discharge_date')
    op.drop_index(op.f('ix_claims_admission_date'), table_name='claims')
    op.drop_column('claims', 'admission_date')
    op.drop_column('claims', 'patient_middle_name')
