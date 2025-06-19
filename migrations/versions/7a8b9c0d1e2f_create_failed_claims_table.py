"""Create failed_claims table

Revision ID: 7a8b9c0d1e2f
Revises: 6f708192a3b4
Create Date: 2024-07-16 17:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# Revision identifiers, used by Alembic.
revision = '7a8b9c0d1e2f'
down_revision = '6f708192a3b4'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'failed_claims',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False, autoincrement=True), # Added nullable=False for PK
        sa.Column('original_claim_db_id', sa.Integer(), nullable=True),
        sa.Column('claim_id', sa.String(length=100), nullable=True),
        sa.Column('facility_id', sa.String(length=50), nullable=True),
        sa.Column('patient_account_number', sa.String(length=100), nullable=True),
        sa.Column('failed_at_stage', sa.String(length=50), nullable=False),
        sa.Column('failure_reason', sa.TEXT(), nullable=False),
        sa.Column('failure_timestamp', sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('original_claim_data', JSONB, nullable=True)
    )
    # Create indexes for all indexed columns as defined in the model
    # index=True on Column definition is a hint for autogenerate. For manual, create them explicitly.
    op.create_index(op.f('ix_failed_claims_id'), 'failed_claims', ['id'], unique=False) # Index on PK is usually automatic, but explicit for op.f() naming
    op.create_index(op.f('ix_failed_claims_original_claim_db_id'), 'failed_claims', ['original_claim_db_id'], unique=False)
    op.create_index(op.f('ix_failed_claims_claim_id'), 'failed_claims', ['claim_id'], unique=False)
    op.create_index(op.f('ix_failed_claims_facility_id'), 'failed_claims', ['facility_id'], unique=False)
    op.create_index(op.f('ix_failed_claims_patient_account_number'), 'failed_claims', ['patient_account_number'], unique=False)
    op.create_index(op.f('ix_failed_claims_failed_at_stage'), 'failed_claims', ['failed_at_stage'], unique=False)
    op.create_index(op.f('ix_failed_claims_failure_timestamp'), 'failed_claims', ['failure_timestamp'], unique=False)


def downgrade() -> None:
    # Drop indexes first
    op.drop_index(op.f('ix_failed_claims_failure_timestamp'), table_name='failed_claims')
    op.drop_index(op.f('ix_failed_claims_failed_at_stage'), table_name='failed_claims')
    op.drop_index(op.f('ix_failed_claims_patient_account_number'), table_name='failed_claims')
    op.drop_index(op.f('ix_failed_claims_facility_id'), table_name='failed_claims')
    op.drop_index(op.f('ix_failed_claims_claim_id'), table_name='failed_claims')
    op.drop_index(op.f('ix_failed_claims_original_claim_db_id'), table_name='failed_claims')
    op.drop_index(op.f('ix_failed_claims_id'), table_name='failed_claims')

    op.drop_table('failed_claims')
